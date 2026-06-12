"""
Parallel Retrieve: Run retrieval on multiple videos in parallel.

Usage:
    # Multi-GPU via torchrun
    torchrun --nproc_per_node=7 -m retrieve.run --video-keys KEY1 KEY2 ... [options]

    # Single process (no torchrun needed)
    python -m retrieve.run --video-keys KEY1 --memory-root <MEMORY_ROOT> [options]
"""

import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from .core.memory_base import MemoryBase
from .core.agent import RetrieveAgent
from .core.prompts import REASONER_SYSTEM_PROMPT, ANALYZER_PROMPT
from .client.llm_client import create_llm_client
from .client.embedder import QwenEmbedder


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(_encoder.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(text) // 4


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def get_video_questions(data_path: str, video_keys: List[str]) -> Dict[str, List[Dict]]:
    """Load questions for specified videos from a JSONL file."""
    videos_questions = {}
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)
            if data["key"] in video_keys:
                videos_questions[data["key"]] = data["qa"]
    return videos_questions


def load_progress(output_dir: Path, video_key: str) -> set:
    """Load already completed question UIDs for a video (checkpoint/resume)."""
    summary_file = output_dir / video_key / "summary.json"
    if not summary_file.exists():
        return set()
    try:
        with open(summary_file) as f:
            data = json.load(f)
            return {
                r["uid"] for r in data.get("results", [])
                if r.get("result", {}).get("success") or r.get("result", {}).get("error")
            }
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Agent logging wrapper
# ---------------------------------------------------------------------------
def run_agent_with_logging(agent: RetrieveAgent, query: str, log_dir: Path, options: str = "") -> Any:
    """Run agent and save detailed per-round logs (reasoner, analyzer, tool calls)."""
    original_chat = agent.llm.chat
    raw_responses = {"reasoner": [], "analyzer": []}

    def logged_chat(messages, images=None, **kwargs):
        response = original_chat(messages, images=images, **kwargs)
        if messages and len(messages) > 0:
            last_content = messages[-1].get("content", "")
            if isinstance(last_content, str):
                if "Current Context" in last_content or "Question" in last_content:
                    raw_responses["reasoner"].append(response)
                elif "Reasoner Output" in last_content:
                    raw_responses["analyzer"].append(response)
        return response

    agent.llm.chat = logged_chat

    original_call_reasoner = agent._call_reasoner
    original_call_analyzer = agent._call_analyzer
    original_parse_json = agent._parse_json

    logs = {"rounds": [], "final_result": None}

    def logged_parse_json(text):
        result = original_parse_json(text)
        if not result and text and text.strip():
            print(f"[WARN] JSON parse failed. Raw text length: {len(text)}")
            print(f"[WARN] First 500 chars: {text[:500]}")
        return result

    def logged_call_reasoner(context, images=None):
        round_idx = len(logs["rounds"])
        result = original_call_reasoner(context, images)

        raw_idx = len(raw_responses["reasoner"]) - 1
        raw_response = raw_responses["reasoner"][raw_idx] if raw_idx >= 0 else ""

        if round_idx >= len(logs["rounds"]):
            logs["rounds"].append({"reasoner": {}, "analyzer": {}, "tool": {}})

        full_messages = [
            {"role": "system", "content": REASONER_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        api_usage = result.get("_usage") if isinstance(result, dict) else None
        system_tokens_est = count_tokens(REASONER_SYSTEM_PROMPT)
        user_tokens_est = count_tokens(context)
        token_info = {
            "system_prompt_estimate": system_tokens_est,
            "user_content_estimate": user_tokens_est,
            "tiktoken_total_estimate": system_tokens_est + user_tokens_est,
        }
        if api_usage:
            token_info["api_actual"] = api_usage
            token_info["input_tokens"] = api_usage.get("input_tokens", 0)
            token_info["output_tokens"] = api_usage.get("output_tokens", 0)
            token_info["total_tokens"] = api_usage.get("total_tokens", 0)

        logs["rounds"][round_idx]["reasoner"] = {
            "has_images": images is not None,
            "num_images": len(images) if images else 0,
            "context_length": len(context),
            "tokens": token_info,
            "is_compressed": round_idx > 0,
            "output": {k: v for k, v in result.items() if k != "_usage"} if isinstance(result, dict) else result,
            "raw_response": raw_response,
            "api_usage": api_usage,
        }

        with open(log_dir / f"round_{round_idx}_reasoner.json", "w") as f:
            json.dump({
                "round": round_idx,
                "has_images": images is not None,
                "num_images": len(images) if images else 0,
                "context_length": len(context),
                "tokens": token_info,
                "is_compressed": round_idx > 0,
                "system_prompt": REASONER_SYSTEM_PROMPT,
                "user_content": context,
                "messages": full_messages,
                "images": images if images else [],
                "output": {k: v for k, v in result.items() if k != "_usage"} if isinstance(result, dict) else result,
                "api_usage": api_usage,
                "raw_response": raw_response,
            }, f, indent=2, ensure_ascii=False)

        return result

    def logged_call_analyzer(reasoner_output):
        round_idx = len(logs["rounds"]) - 1
        result = original_call_analyzer(reasoner_output)

        raw_idx = len(raw_responses["analyzer"]) - 1
        raw_response = raw_responses["analyzer"][raw_idx] if raw_idx >= 0 else ""

        tool_descriptions = agent.tools.get_tool_descriptions()
        full_prompt = ANALYZER_PROMPT.format(
            reasoner_output=json.dumps(reasoner_output, ensure_ascii=False, indent=2),
            tool_descriptions=tool_descriptions,
        )

        api_usage = result.get("_usage") if isinstance(result, dict) else None
        prompt_tokens_est = count_tokens(full_prompt)
        token_info = {"prompt_estimate": prompt_tokens_est}
        if api_usage:
            token_info["api_actual"] = api_usage
            token_info["input_tokens"] = api_usage.get("input_tokens", 0)
            token_info["output_tokens"] = api_usage.get("output_tokens", 0)
            token_info["total_tokens"] = api_usage.get("total_tokens", 0)

        logs["rounds"][round_idx]["analyzer"] = {
            "input": reasoner_output,
            "tokens": token_info,
            "output": {k: v for k, v in result.items() if k != "_usage"} if isinstance(result, dict) else result,
            "raw_response": raw_response,
            "api_usage": api_usage,
        }

        with open(log_dir / f"round_{round_idx}_analyzer.json", "w") as f:
            json.dump({
                "reasoner_output": reasoner_output,
                "tool_descriptions": tool_descriptions,
                "tokens": token_info,
                "full_prompt": full_prompt,
                "messages": [{"role": "user", "content": full_prompt}],
                "output": {k: v for k, v in result.items() if k != "_usage"} if isinstance(result, dict) else result,
                "api_usage": api_usage,
                "raw_response": raw_response,
            }, f, indent=2, ensure_ascii=False)

        return result

    # Patch agent methods
    agent._call_reasoner = logged_call_reasoner
    agent._call_analyzer = logged_call_analyzer
    agent._parse_json = logged_parse_json

    # Log tool calls
    original_execute = agent.tools.execute

    def logged_execute(tool_name, params):
        round_idx = len(logs["rounds"]) - 1
        result = original_execute(tool_name, params)

        logs["rounds"][round_idx]["tool"] = {
            "tool_name": tool_name,
            "params": params,
            "success": result.success,
            "result_preview": str(result.result)[:500] if result.result else None,
        }

        with open(log_dir / f"round_{round_idx}_tool.json", "w") as f:
            json.dump({
                "tool_name": tool_name,
                "params": params,
                "success": result.success,
                "result": result.result if result.success else {"error": result.error},
            }, f, indent=2, ensure_ascii=False, default=str)

        return result

    agent.tools.execute = logged_execute

    # Capture initial context
    initial_context = agent._build_initial_context(query, options)
    initial_tokens = count_tokens(initial_context)
    system_tokens = count_tokens(REASONER_SYSTEM_PROMPT)

    logs["initial_context"] = {
        "query": query,
        "options": options,
        "context": initial_context,
        "context_length": len(initial_context),
        "tokens": {
            "system_prompt": system_tokens,
            "initial_context": initial_tokens,
            "total_first_round": system_tokens + initial_tokens,
        },
    }

    with open(log_dir / "initial_context.json", "w") as f:
        json.dump({
            "query": query,
            "options": options,
            "system_prompt": REASONER_SYSTEM_PROMPT,
            "initial_context": initial_context,
            "context_length": len(initial_context),
            "tokens": {
                "system_prompt": system_tokens,
                "initial_context": initial_tokens,
                "total_first_round": system_tokens + initial_tokens,
            },
        }, f, indent=2, ensure_ascii=False)

    # Run agent
    try:
        result = agent.run(query, options)
        logs["final_result"] = {
            "success": result.success,
            "answer": result.answer,
            "reasoning": result.reasoning,
            "rounds": result.rounds,
            "error": result.error,
            "context_history": result.context_history,
            "images_used": result.images_used,
        }
    finally:
        agent._call_reasoner = original_call_reasoner
        agent._call_analyzer = original_call_analyzer
        agent._parse_json = original_parse_json
        agent.tools.execute = original_execute
        agent.llm.chat = original_chat

    with open(log_dir / "complete_log.json", "w") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False, default=str)

    return result


# ---------------------------------------------------------------------------
# Single-question runner
# ---------------------------------------------------------------------------
def run_single_question(
    agent: RetrieveAgent,
    question: Dict,
    save_dir: Path,
    round_logger=None,
) -> Dict[str, Any]:
    """Run retrieval for a single question and save detailed logs."""
    uid = question["uid"]
    query = question["question"]
    answer = question["answer"]

    if round_logger:
        round_logger(f"\n{'='*60}")
        round_logger(f"UID: {uid}")
        round_logger(f"Question: {query.split(chr(10))[0][:80]}...")
        round_logger(f"Ground Truth: {answer}")
        round_logger(f"{'='*60}")

    question_dir = save_dir / f"q_{uid}"
    question_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_agent_with_logging(agent, query, question_dir, options=answer)
        success = result.success
        result_answer = result.answer
        result_reasoning = result.reasoning
        result_rounds = result.rounds
        result_error = result.error
    except Exception as e:
        if round_logger:
            round_logger(f"[ERROR] Exception: {e}")
        with open(question_dir / "error.log", "w") as f:
            f.write(traceback.format_exc())
        success = False
        result_answer = None
        result_reasoning = None
        result_rounds = 0
        result_error = str(e)

    final_result = {
        "uid": uid,
        "question": query,
        "ground_truth": answer,
        "question_type": question.get("question_type", []),
        "time_reference": question.get("time_reference", ""),
        "result": {
            "success": success,
            "answer": result_answer,
            "reasoning": result_reasoning,
            "rounds": result_rounds,
            "error": result_error,
        },
        "correct": result_answer == answer if result_answer else False,
    }

    with open(question_dir / "final_result.json", "w") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    return final_result


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------
def process_video(
    video_key: str,
    questions: List[Dict],
    phase2_dir: Path,
    phase3_dir: Path,
    output_dir: Path,
    llm_client,
    embedder,
    rank: int,
    max_rounds: int = 8,
    default_top_k: int = 10,
    precomputed_embeddings: str = None,
) -> Dict[str, Any]:
    """Process all questions for a single video."""
    print(f"[Rank {rank}] Processing video: {video_key} ({len(questions)} questions)")

    # Checkpoint/resume
    completed_uids = load_progress(output_dir, video_key)
    if completed_uids:
        print(f"[Rank {rank}] Found {len(completed_uids)} completed questions, resuming...")

    video_output_dir = output_dir / video_key
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Load MemoryBase
    print(f"[Rank {rank}] Loading MemoryBase for {video_key}...")
    try:
        mb = MemoryBase()
        mb.load(phase2_dir=str(phase2_dir), phase3_dir=str(phase3_dir))
        stats = mb.stats()
        print(f"[Rank {rank}] Loaded: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

        if precomputed_embeddings and Path(precomputed_embeddings).exists():
            loaded = mb.load_embeddings(precomputed_embeddings)
            print(f"[Rank {rank}] Loaded {loaded} precomputed embeddings")
        else:
            mb.set_embedder(embedder)
            mb.compute_embeddings()

        # Set embedder for runtime query embedding
        mb.set_embedder(embedder)
        emb_stats = mb.stats()
        print(f"[Rank {rank}] Embedded: {emb_stats.get('embedded_nodes', 0)} nodes")
    except Exception as e:
        print(f"[Rank {rank}] ERROR: Failed to load graph: {e}")
        return {
            "video_key": video_key,
            "status": "failed",
            "error": f"Failed to load graph: {str(e)}",
            "total": len(questions),
            "completed": 0,
            "correct": 0,
        }

    # Create agent
    agent = RetrieveAgent(
        mb, llm_client,
        max_rounds=max_rounds,
        default_top_k=default_top_k,
    )

    # Process questions
    results = []
    correct = 0
    total = 0
    skipped = 0

    for i, question in enumerate(questions):
        uid = question["uid"]

        if uid in completed_uids:
            skipped += 1
            result_file = video_output_dir / f"q_{uid}" / "final_result.json"
            if result_file.exists():
                with open(result_file) as f:
                    existing_result = json.load(f)
                    results.append(existing_result)
                    if existing_result.get("correct"):
                        correct += 1
                    total += 1
            continue

        print(f"[Rank {rank}] [{i+1}/{len(questions)}] Processing Q{uid}...")

        try:
            result = run_single_question(
                agent, question, video_output_dir,
                round_logger=lambda msg: print(f"[Rank {rank}] {msg}"),
            )
        except Exception as e:
            print(f"[Rank {rank}] ERROR on Q{uid}: {e}")
            result = {
                "uid": uid,
                "question": question["question"],
                "ground_truth": question["answer"],
                "result": {"success": False, "error": str(e)},
                "correct": False,
            }
            question_dir = video_output_dir / f"q_{uid}"
            question_dir.mkdir(parents=True, exist_ok=True)
            with open(question_dir / "final_result.json", "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        results.append(result)
        if result.get("correct"):
            correct += 1
        total += 1

        # Save intermediate summary after each question
        with open(video_output_dir / "summary.json", "w") as f:
            json.dump({
                "video_key": video_key,
                "timestamp": datetime.now().isoformat(),
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else 0,
                "skipped": skipped,
                "results": results,
            }, f, indent=2, ensure_ascii=False)

    # Final summary
    accuracy = correct / total if total > 0 else 0
    print(f"[Rank {rank}] Video {video_key} completed: {correct}/{total} = {accuracy:.2%}")

    summary = {
        "video_key": video_key,
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "skipped": skipped,
        "results": results,
    }

    with open(video_output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------
def build_llm_client(args) -> Any:
    """Create LLM client from CLI arguments."""
    model = args.model or "gemini-2.5-pro"
    kwargs = {"model": model}
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if getattr(args, "base_url", None):
        kwargs["base_url"] = args.base_url
    return create_llm_client("openai", **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Parallel retrieval on multiple videos")
    parser.add_argument("--video-keys", type=str, nargs="+", required=True,
                        help="Video keys to process")
    parser.add_argument("--memory-root", type=str, default="",
                        help="Root directory containing memory graphs")
    parser.add_argument("--data-path", type=str, default="",
                        help="Path to video_info.meta.jsonl")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g., gemini-2.5-pro, gpt-4o)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (or set LLM_API_KEY env var)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="LLM API base URL (OpenAI-compatible endpoint)")
    parser.add_argument("--embed-model-path", type=str,
                        default=None,
                        help="Embedding model path (e.g., Qwen/Qwen3-Embedding-8B)")
    parser.add_argument("--precomputed-embeddings", type=str, default=None,
                        help="Path to precomputed embeddings JSON (skip compute)")
    parser.add_argument("--embed-server-urls", type=str, nargs="*", default=None,
                        help="Remote embedding server URLs for runtime query embedding")
    parser.add_argument("--max-rounds", type=int, default=12,
                        help="Maximum iteration rounds per question")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Default top_k for search_nodes and search_ocr_text")
    parser.add_argument("--phase2-name", type=str, default="phase2_topdown",
                        help="Phase 2 subdirectory name under memory_root/<video>/")
    parser.add_argument("--phase3-name", type=str, default="phase4",
                        help="Phase 3 subdirectory name under memory_root/<video>/")
    parser.add_argument("--questions", type=str, nargs="*",
                        help="Specific question UIDs to run (default: all)")
    parser.add_argument("--max-questions", type=int,
                        help="Maximum number of questions to run per video")
    args = parser.parse_args()

    my_videos = args.video_keys
    print(f"Processing videos: {my_videos}")

    if not my_videos:
        print("No videos specified, exiting...")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    all_questions = get_video_questions(args.data_path, args.video_keys)

    # Initialize LLM client
    print(f"Initializing LLM client (model={args.model})...")
    llm_client = build_llm_client(args)
    print(f"LLM client ready.")

    # Initialize embedder (for runtime query embedding)
    if args.embed_server_urls:
        from .client.remote_embedder import RemoteEmbedder
        print(f"Using remote embedding servers: {args.embed_server_urls}")
        embedder = RemoteEmbedder(args.embed_server_urls)
    else:
        print(f"Initializing local embedder...")
        embedder = QwenEmbedder(model_path=args.embed_model_path)
    print(f"Embedder ready.")

    # Process assigned videos
    all_results = []

    try:
        for video_key in my_videos:
            questions = all_questions.get(video_key, [])
            if not questions:
                print(f"WARNING: No questions found for {video_key}")
                continue

            # Filter questions if specified
            if args.questions:
                questions = [q for q in questions if q["uid"] in args.questions]
            if args.max_questions:
                questions = questions[:args.max_questions]

            # Determine phase2 and phase3 directories
            phase2_dir = Path(args.memory_root) / video_key / args.phase2_name
            phase3_dir = Path(args.memory_root) / video_key / args.phase3_name

            if not phase2_dir.exists() or not phase3_dir.exists():
                print(f"WARNING: Memory data not found for {video_key}")
                all_results.append({
                    "video_key": video_key,
                    "status": "failed",
                    "error": "Memory data not found",
                })
                continue

            try:
                # Determine precomputed embeddings path for this video
                emb_path = None
                if args.precomputed_embeddings:
                    emb_path = args.precomputed_embeddings
                else:
                    candidate = Path(args.memory_root) / video_key / "embeddings.json"
                    if candidate.exists():
                        emb_path = str(candidate)

                result = process_video(
                    video_key=video_key,
                    questions=questions,
                    phase2_dir=phase2_dir,
                    phase3_dir=phase3_dir,
                    output_dir=output_dir,
                    llm_client=llm_client,
                    embedder=embedder,
                    rank=0,
                    max_rounds=args.max_rounds,
                    default_top_k=args.top_k,
                    precomputed_embeddings=emb_path,
                )
                all_results.append(result)
            except Exception as e:
                print(f"ERROR processing {video_key}: {e}")
                traceback.print_exc()
                all_results.append({
                    "video_key": video_key,
                    "status": "failed",
                    "error": str(e),
                })

    finally:
        # Save overall summary
        summary = {
            "videos": my_videos,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }
        with open(output_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"All done. Summary saved to run_summary.json")


if __name__ == "__main__":
    main()