# Agentic Retrieval

This directory contains the **agentic tool-augmented retrieval** module of MemDreamer. Given a pre-built Hierarchical Graph Memory and a question, the system iteratively queries the memory through tool calls within an Observation–Reason–Action loop to arrive at an answer.

---

## 1. Reproduce Paper Results

We release pre-built memory files and full inference trajectories on HuggingFace:

```
🤗 https://huggingface.co/inclusionAI/MemDreamer
```

To reproduce the accuracy numbers reported in our paper, simply run:

```bash
# LVBench
python retrieve/scripts/check_progress.py <trajectory_dir> \
    --data data/LVBench/video_info.meta.jsonl

# LongVideoBench
python retrieve/scripts/check_progress.py <trajectory_dir> \
    --data data/LongVideoBench/video_info.meta.jsonl
```

where `<trajectory_dir>` is the path to the downloaded inference trajectories (e.g., `trajectories/lvbench_gemini31pro/`). The script will report overall accuracy and per-category breakdowns.

---

## 2. Run Retrieval from Scratch

### Prerequisites

```bash
pip install -r requirements.txt
```

You will need:
- **Memory files**: Download from HuggingFace (or build your own). Each video has `phase2_topdown/` (subgraphs) and `phase4/` (hierarchical graph).
- **Embedding model**: [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) for semantic search.
- **LLM API access**: Any OpenAI-compatible endpoint (Gemini, GPT, vLLM, etc.).

### Step 1: Precompute Embeddings

Although the released memory already includes `embeddings.json`, you can recompute them:

```bash
python retrieve/scripts/precompute_embeddings.py \
    --memory-dir /path/to/memory_root \
    --embed-model-path Qwen/Qwen3-Embedding-8B \
    --num-workers 8
```

This produces `embeddings.json` under each `<memory_root>/<video>/` directory.

### Step 2: Start Embedding Servers

The retrieval agent needs to embed queries at runtime. Launch embedding servers (one per GPU):

```bash
export EMBED_MODEL_PATH="Qwen/Qwen3-Embedding-8B"
bash retrieve/scripts/start_embedding_servers.sh 8 8001
```

This starts 8 servers on ports 8001–8008, each on a separate GPU.

### Step 3: Run Retrieval

Edit `retrieve/run.sh` with your configuration:

```bash
MEMORY_DIR="/path/to/memory_root"          # Contains <video>/phase2_topdown/ and <video>/phase4/
DATA_FILE="/path/to/video_info.meta.jsonl"  # Question file (LVBench, LongVideoBench, etc.)
MODEL="gemini-2.5-pro"                     # Any model name supported by your endpoint
API_KEY="your-api-key"                     # Or set LLM_API_KEY env var
BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"  # OpenAI-compatible endpoint
EMBED_MODEL_PATH="Qwen/Qwen3-Embedding-8B"
```

Then launch:

```bash
bash retrieve/run.sh results/my_experiment
```

This processes all videos concurrently (up to `MAX_CONCURRENT=10`) and saves per-question logs and aggregated results to the output directory.

#### Single Video (for debugging)

```bash
python -m retrieve.run \
    --video-keys "your_video_key" \
    --memory-root /path/to/memory_root \
    --data-path /path/to/video_info.meta.jsonl \
    --output-dir results/debug \
    --model gemini-2.5-pro \
    --api_key "your-api-key" \
    --base-url "https://generativelanguage.googleapis.com/v1beta/openai" \
    --embed-server-urls http://localhost:8001 \
    --max-rounds 12 \
    --top-k 10
```

---

## 3. Using a Custom LLM Backend

To integrate your own LLM, subclass `BaseLLMClient` in `retrieve/client/llm_client.py`:

```python
from retrieve.client.llm_client import BaseLLMClient

class MyCustomClient(BaseLLMClient):
    def chat(self, messages, images=None):
        # Call your model here
        response_text = my_model.generate(messages)
        return {
            "content": response_text,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
```

Then pass it directly to the agent:

```python
from retrieve.core.memory_base import MemoryBase
from retrieve.core.agent import RetrieveAgent

mb = MemoryBase()
mb.load(phase2_dir="...", phase3_dir="...")

client = MyCustomClient()
agent = RetrieveAgent(mb, client, max_rounds=12)
result = agent.run(query="...", options="A) ... B) ...")
print(result.answer)
```

The built-in `OpenAICompatibleClient` already works with any OpenAI-compatible endpoint (Gemini, GPT, vLLM, Together, OpenRouter, etc.) — see the `--base-url` parameter in `run.sh`.

---

## 4. Output Structure

Per-question detailed logs are saved for reproducibility and analysis:

```
<output_dir>/
├── <video_key>/
│   ├── summary.json                 # Per-video accuracy summary
│   └── q_<uid>/
│       ├── initial_context.json     # First-round full context + token counts
│       ├── round_0_reasoner.json    # Reasoner input/output per round
│       ├── round_0_analyzer.json    # Analyzer input/output per round
│       ├── round_0_tool.json        # Tool call and result per round
│       ├── final_result.json        # Answer, correctness, total rounds
│       └── complete_log.json        # Full execution trace
└── run_summary.json                 # Overall run metadata
```

---

## 5. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-rounds` | 12 | Maximum Observation-Reason-Action loop iterations |
| `--top-k` | 10 | Number of results returned by `search_nodes` |
| `--phase2-name` | `phase2_topdown` | Subdirectory name for Phase 2 data |
| `--phase3-name` | `phase4` | Subdirectory name for Phase 3 data |
