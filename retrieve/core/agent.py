"""
Agent: LLM Reasoning driven retrieval loop

Core flow:
1. Initialize Context (Query + Options + Summary + Super Events)
2. Reasoner inference -> output can_answer / need_info
3. Analyzer analysis -> decide tool call
4. Tool Executor -> get data
5. Update Context -> loop
"""

import json
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from .memory_base import MemoryBase
from .tools import ToolRegistry, ToolResult
from .prompts import (
    REASONER_SYSTEM_PROMPT,
    ANALYZER_PROMPT,
    TOOL_RESULT_TEMPLATE,
)


class ExplorationHistory:
    """Track tool calls across rounds to prevent repetitive searches and inform the Reasoner
    about what has been explored vs. what remains."""

    def __init__(self):
        # (round, query, macro_ids_found) for each search_nodes call
        self.searches: List[Tuple[int, str, List[str]]] = []
        # (round, query) for each search_ocr_text call
        self.ocr_searches: List[Tuple[int, str]] = []
        # (round, start_sec, end_sec, macro_ids_found) for each search_by_time call
        self.time_searches: List[Tuple[int, float, float, List[str]]] = []
        # (round, super_id) for get_macro_events calls
        self.macro_browsed: List[Tuple[int, str]] = []
        # (round, macro_id, label) for get_subgraph calls
        self.subgraphs_viewed: List[Tuple[int, str, str]] = []
        # All macro_ids ever returned by any search (for "found but not explored" tracking)
        self.found_macro_ids: Set[str] = set()

    def record_search(self, round_num: int, query: str, result_text: str):
        """Record a search_nodes call and extract macro_ids from result."""
        macro_ids = self._extract_macro_ids(result_text)
        self.searches.append((round_num, query, macro_ids))
        self.found_macro_ids.update(macro_ids)

    def record_ocr_search(self, round_num: int, query: str):
        self.ocr_searches.append((round_num, query))

    def record_time_search(self, round_num: int, start_sec: float, end_sec: float, result_text: str):
        macro_ids = self._extract_macro_ids(result_text)
        self.time_searches.append((round_num, start_sec, end_sec, macro_ids))
        self.found_macro_ids.update(macro_ids)

    def record_macro_events(self, round_num: int, super_id: str):
        self.macro_browsed.append((round_num, super_id))

    def record_subgraph(self, round_num: int, macro_id: str, label: str = ""):
        self.subgraphs_viewed.append((round_num, macro_id, label))

    def record_tool_call(self, round_num: int, tool_name: str, params: Dict, result_text: str):
        """Auto-record any tool call based on its type."""
        if tool_name == "search_nodes":
            query = params.get("query", "")
            self.record_search(round_num, query, result_text)
        elif tool_name == "search_ocr_text":
            query = params.get("query", "")
            self.record_ocr_search(round_num, query)
        elif tool_name == "search_by_time":
            start = params.get("start_sec", 0)
            end = params.get("end_sec", 0)
            self.record_time_search(round_num, start, end, result_text)
        elif tool_name == "get_macro_events":
            super_id = params.get("super_id", "all")
            self.record_macro_events(round_num, super_id)
        elif tool_name == "get_subgraph":
            macro_id = params.get("macro_id", "")
            label = self._extract_subgraph_label(result_text)
            self.record_subgraph(round_num, macro_id, label)

    def get_viewed_macro_ids(self) -> Set[str]:
        return {m[1] for m in self.subgraphs_viewed}

    def get_unexplored_macros(self) -> Set[str]:
        """Macro IDs found by searches but not yet viewed via get_subgraph."""
        return self.found_macro_ids - self.get_viewed_macro_ids()

    def get_used_tool_types(self) -> Set[str]:
        """Set of tool types that have been used at least once."""
        used = set()
        if self.searches:
            used.add("search_nodes")
        if self.ocr_searches:
            used.add("search_ocr_text")
        if self.time_searches:
            used.add("search_by_time")
        if self.macro_browsed:
            used.add("get_macro_events")
        if self.subgraphs_viewed:
            used.add("get_subgraph")
        return used

    def format_for_context(self) -> str:
        """Render exploration history as a structured section for the compressed context."""
        lines = []

        # Searches Done
        if self.searches:
            lines.append("### Searches Done (semantic)")
            for rnd, query, macros in self.searches:
                if macros:
                    lines.append(f"R{rnd}: search_nodes(\"{query}\") → {', '.join(macros)}")
                else:
                    lines.append(f"R{rnd}: search_nodes(\"{query}\") → no matches")

        if self.ocr_searches:
            lines.append("### Searches Done (OCR)")
            for rnd, query in self.ocr_searches:
                lines.append(f"R{rnd}: search_ocr_text(\"{query}\")")

        if self.time_searches:
            lines.append("### Searches Done (time)")
            for rnd, start, end, macros in self.time_searches:
                mins_s = int(start) // 60
                secs_s = int(start) % 60
                mins_e = int(end) // 60
                secs_e = int(end) % 60
                if macros:
                    lines.append(f"R{rnd}: search_by_time({mins_s}:{secs_s:02d}-{mins_e}:{secs_e:02d}) → {', '.join(macros)}")
                else:
                    lines.append(f"R{rnd}: search_by_time({mins_s}:{secs_s:02d}-{mins_e}:{secs_e:02d}) → no matches")

        # Macro Events Browsed
        if self.macro_browsed:
            lines.append("### Super Events Browsed (get_macro_events)")
            for rnd, super_id in self.macro_browsed:
                lines.append(f"R{rnd}: {super_id}")

        # Subgraphs Viewed
        if self.subgraphs_viewed:
            lines.append("### Macros Explored (get_subgraph)")
            for rnd, macro_id, label in self.subgraphs_viewed:
                label_short = label[:60] if label else ""
                if label_short:
                    lines.append(f"R{rnd}: {macro_id} — {label_short}")
                else:
                    lines.append(f"R{rnd}: {macro_id}")

        # Unexplored Macros
        unexplored = self.get_unexplored_macros()
        if unexplored:
            lines.append(f"### Macros Found But NOT Explored: {', '.join(sorted(unexplored))}")

        # Tools Not Yet Used
        all_tools = {"search_nodes", "search_ocr_text", "search_by_time", "get_macro_events", "get_subgraph"}
        unused = all_tools - self.get_used_tool_types()
        if unused and (self.searches or self.ocr_searches or self.time_searches):
            lines.append(f"### Tools Not Yet Used: {', '.join(sorted(unused))}")

        if not lines:
            return ""

        return "## Exploration History\n" + "\n".join(lines)

    @staticmethod
    def _extract_macro_ids(text: str) -> List[str]:
        """Extract macro_XXXX IDs from tool result text."""
        if not text:
            return []
        return list(set(re.findall(r'macro_\d{4}', text)))

    @staticmethod
    def _extract_subgraph_label(text: str) -> str:
        """Extract the subgraph title/label from get_subgraph result."""
        if not text:
            return ""
        # Format: "=== Subgraph: <label> ==="
        m = re.search(r'===\s*Subgraph:\s*(.+?)\s*===', text)
        if m:
            return m.group(1)
        return ""


@dataclass
class AgentResult:
    """Agent execution result"""
    success: bool
    answer: Optional[str] = None
    reasoning: Optional[str] = None
    rounds: int = 0
    error: Optional[str] = None
    context_history: List[str] = field(default_factory=list)
    images_used: List[str] = field(default_factory=list)


class RetrieveAgent:
    """Retrieval Agent with visual reasoning support"""

    def __init__(
        self,
        memory_base: MemoryBase,
        llm_client,
        max_rounds: int = 8,
        default_top_k: int = 10,
    ):
        """
        Initialize Agent

        Args:
            memory_base: MemoryBase instance (loaded with Phase 2 + Phase 3 data)
            llm_client: LLM client with chat(messages, images) method
            max_rounds: Maximum iteration rounds
            default_top_k: Default top_k for search_nodes and search_ocr_text
        """
        self.mb = memory_base
        self.llm = llm_client
        self.max_rounds = max_rounds
        self.default_top_k = default_top_k
        self.tools = ToolRegistry(memory_base, default_top_k=default_top_k)

    def run(
        self,
        query: str,
        options: str,
    ) -> AgentResult:
        """
        Execute retrieval with compressed context across rounds.

        Context strategy:
        - Round 0: full initial context (Video Summary + Super Events + Query + Options)
        - Round N (N>=1): Query + Options + accumulated useful_info + current tool result
        """
        initial_context = self._build_initial_context(query, options)
        context_history = [initial_context]

        # Accumulated useful_info across rounds
        accumulated_info = []
        pending_images = []
        all_images_used = []
        rounds = 0

        last_tool_result_raw = ""
        last_tool_result_text = ""
        last_tool_name = ""

        # Strategy guard: track tool history to prevent repetitive patterns
        tool_history: List[str] = []  # list of tool names called
        search_queries: List[str] = []  # queries used in search_nodes
        visited_macros: List[str] = []  # macro_ids explored via get_subgraph

        # Exploration history for structured context
        exploration = ExplorationHistory()

        while rounds < self.max_rounds:
            rounds += 1
            print(f"\n========== Round {rounds} ==========")

            # Build context for this round
            if rounds == 1:
                context = initial_context
            else:
                context = self._build_compressed_context(
                    query, options, accumulated_info, last_tool_result_text, last_tool_name,
                    exploration=exploration,
                )

            context_history.append(context)

            # Step 1: Call Reasoner
            print("[Reasoner] Thinking...")
            reasoner_output = self._call_reasoner(context, pending_images)
            print(f"[Reasoner] Output: can_answer={reasoner_output.get('can_answer')}")

            if pending_images:
                all_images_used.extend(pending_images)
                pending_images = []

            # Collect useful_info from reasoner
            useful = reasoner_output.get("useful_info", "")
            if useful:
                accumulated_info.append(f"### Round {rounds} findings\n{useful}")
                print(f"[Reasoner] useful_info: {useful[:100]}...")

            # Check if can answer
            if reasoner_output.get("can_answer", False):
                return AgentResult(
                    success=True,
                    answer=reasoner_output.get("answer"),
                    reasoning=reasoner_output.get("reasoning"),
                    rounds=rounds,
                    context_history=context_history,
                    images_used=all_images_used,
                )

            # Step 2: Call Analyzer
            print("[Analyzer] Analyzing...")
            analyzer_output = self._call_analyzer(reasoner_output)
            tool_name = analyzer_output.get("tool")
            print(f"[Analyzer] Decided: {tool_name}")

            # Check if cannot answer
            if tool_name == "cannot_answer":
                return AgentResult(
                    success=False,
                    error=analyzer_output.get("params", {}).get("reason", "Cannot answer"),
                    rounds=rounds,
                    context_history=context_history,
                    images_used=all_images_used,
                )

            # Check if direct answer
            if tool_name == "answer":
                answer_val = analyzer_output.get("params", {}).get("answer")
                return AgentResult(
                    success=True,
                    answer=answer_val,
                    reasoning=reasoner_output.get("reasoning"),
                    rounds=rounds,
                    context_history=context_history,
                    images_used=all_images_used,
                )

            # Step 3: Execute tool
            tool_params = analyzer_output.get("params", {})
            print(f"[Tool] Executing: {tool_name}({tool_params})")
            tool_result = self.tools.execute(tool_name, tool_params)

            if not tool_result.success:
                print(f"[Tool] Error: {tool_result.error}")
                last_tool_result_text = f"Error: Tool {tool_name} failed: {tool_result.error}"
                last_tool_result_raw = last_tool_result_text
                last_tool_name = tool_name
                continue

            # Record exploration history
            if tool_result.success:
                exploration.record_tool_call(rounds, tool_name, tool_params, tool_result.result or "")

            # Step 4: Prepare tool result for next round's context
            last_tool_name = tool_name
            if tool_result.success:
                last_tool_result_text = TOOL_RESULT_TEMPLATE.format(
                    tool_name=tool_name,
                    tool_params=json.dumps(tool_params, ensure_ascii=False),
                    tool_result=tool_result.result,
                )
                last_tool_result_raw = tool_result.result or ""
            else:
                last_tool_result_text = f"Error: {tool_result.error}"
                last_tool_result_raw = last_tool_result_text

            # Collect images
            new_images = tool_result.metadata.get("images", []) if tool_result.success else []
            if new_images:
                pending_images = new_images
                print(f"[Tool] Collected {len(new_images)} keyframe images for visual analysis")

        # Exceeded max rounds
        return AgentResult(
            success=False,
            error="Exceeded maximum iteration rounds",
            rounds=rounds,
            context_history=context_history,
            images_used=all_images_used,
        )

    def _build_initial_context(self, query: str, options: str) -> str:
        summary = self.mb.get_context_summary()
        return f"""{summary}

## Question
{query}

## Options
{options}"""

    def _build_compressed_context(
        self,
        query: str,
        options: str,
        accumulated_info: List[str],
        tool_result_text: str,
        tool_name: str,
        exploration: ExplorationHistory = None,
    ) -> str:
        """Build compressed context: Query + Options + exploration history + accumulated useful_info + current tool result"""
        info_str = "\n\n".join(accumulated_info) if accumulated_info else "None yet."

        # Build exploration history section
        exploration_section = ""
        if exploration:
            exploration_section = exploration.format_for_context()

        return f"""## Question
{query}

## Options
{options}

{exploration_section}
## Findings from Previous Rounds
{info_str}
## Current Round Tool Result ({tool_name})
{tool_result_text}"""

    def _call_reasoner(
        self,
        context: str,
        images: Optional[List[str]] = None,
        max_retries: int = 5,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": REASONER_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        if images:
            print(f"[Reasoner] Analyzing {len(images)} keyframe images...")

        import time as _time
        for attempt in range(max_retries):
            response = self.llm.chat(messages, images=images)
            # response is now a dict with "content" and "usage"
            response_text = response.get("content", "") if isinstance(response, dict) else response
            usage = response.get("usage") if isinstance(response, dict) else None
            result = self._parse_output(response_text)

            if result:
                if attempt > 0:
                    print(f"[Reasoner] Got valid output on attempt {attempt + 1}")
                result["_usage"] = usage  # Add usage info to result
                return result

            print(f"[Reasoner] Empty output on attempt {attempt + 1}/{max_retries}, retrying...")
            if attempt < max_retries - 1:
                _time.sleep(3)

        print(f"[Reasoner] WARNING: All {max_retries} attempts returned empty output")
        return {}

    def _call_analyzer(self, reasoner_output: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
        tool_descriptions = self.tools.get_tool_descriptions()

        prompt = ANALYZER_PROMPT.format(
            reasoner_output=json.dumps(reasoner_output, ensure_ascii=False, indent=2),
            tool_descriptions=tool_descriptions,
        )

        messages = [{"role": "user", "content": prompt}]

        import time as _time
        for attempt in range(max_retries):
            response = self.llm.chat(messages)
            # response is now a dict with "content" and "usage"
            response_text = response.get("content", "") if isinstance(response, dict) else response
            usage = response.get("usage") if isinstance(response, dict) else None
            result = self._parse_output(response_text)

            if result:
                if attempt > 0:
                    print(f"[Analyzer] Got valid output on attempt {attempt + 1}")
                result["_usage"] = usage  # Add usage info to result
                return result

            print(f"[Analyzer] Empty output on attempt {attempt + 1}/{max_retries}, retrying...")
            if attempt < max_retries - 1:
                _time.sleep(3)

        print(f"[Analyzer] WARNING: All {max_retries} attempts returned empty output")
        return {}

    def _parse_output(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {}

        text = text.strip()
        result = self._parse_json(text)
        if result:
            return result

        result = self._parse_yaml(text)
        if result:
            return result

        return self._parse_fallback(text)

    def _parse_yaml(self, text: str) -> Dict[str, Any]:
        text = re.sub(r'```(?:yaml)?\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = text.strip()

        try:
            import yaml
            result = yaml.safe_load(text)
            if isinstance(result, dict):
                return result
        except ImportError:
            pass
        except Exception:
            pass

        return {}

    def _parse_json(self, text: str) -> Dict[str, Any]:
        # Strip reasoning model tags (MiniMax <thought>, DeepSeek <think>, etc.)
        # Case 1: with closing tag </thought> or
        # Case 2: without closing tag, followed by JSON (MiniMax style: <thought>...\n\n{...})
        text = re.sub(r'<(?:thought|think)>[\s\S]*?(?:</(?:thought|think)>|(?=\n\s*\{))', '', text)

        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        # Handle multiple concatenated JSON objects (e.g., GPT-5.4 outputs two copies).
        # Strategy: parse all JSON objects, prefer the one with can_answer=true.
        for split_pattern in ['}\n{', '}\r\n{', '}{']:
            if split_pattern in text:
                parts = text.split(split_pattern)
                parsed_list = []
                for i, part in enumerate(parts):
                    fragment = part if i == 0 else part
                    # Re-add the closing brace removed by split (except last part already has it)
                    if i < len(parts) - 1:
                        fragment = fragment.rstrip() + '}'
                    # Non-first parts are missing the opening brace
                    if i > 0 and not fragment.lstrip().startswith('{'):
                        fragment = '{' + fragment
                    try:
                        obj = json.loads(fragment)
                        if isinstance(obj, dict):
                            parsed_list.append(obj)
                    except Exception:
                        # Fragment may have garbage prefix (e.g., GPT outputs non-JSON text before JSON)
                        # Try to extract the first valid JSON object from the fragment
                        json_match = re.search(r'\{[\s\S]*\}', fragment)
                        if json_match:
                            try:
                                obj = json.loads(json_match.group())
                                if isinstance(obj, dict):
                                    parsed_list.append(obj)
                            except Exception:
                                pass
                if parsed_list:
                    # Prefer the one with can_answer=true (final answer over exploration)
                    for obj in parsed_list:
                        if obj.get('can_answer') is True:
                            return obj
                    # Otherwise take the first valid one
                    return parsed_list[0]

        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except Exception:
                pass

        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                json_str = json_match.group()
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                return json.loads(json_str)
            except Exception:
                pass

        return {}

    def _parse_fallback(self, text: str) -> Dict[str, Any]:
        result = {}

        # can_answer
        match = re.search(r'can_answer[:\s]+(?:["\']?)(true|false)(?:["\']?)', text, re.IGNORECASE)
        if match:
            result['can_answer'] = match.group(1).lower() == 'true'

        # answer
        match = re.search(r'answer[:\s]+(?:["\']?)([ABCD]|null)(?:["\']?)', text, re.IGNORECASE)
        if match:
            val = match.group(1).upper()
            result['answer'] = None if val == 'NULL' else val

        # reasoning
        match = re.search(r'reasoning[:\s]+["\']?(.+?)(?=["\']?\n(?:can_answer|answer|useful_info|need_info|tool|params|information_need)[:\s]|$)', text, re.DOTALL | re.IGNORECASE)
        if match:
            result['reasoning'] = match.group(1).strip().strip('"\'')

        # useful_info
        match = re.search(r'useful_info[:\s]+["\']?(.+?)(?=["\']?\n(?:can_answer|answer|reasoning|need_info|tool|params|information_need)[:\s]|$)', text, re.DOTALL | re.IGNORECASE)
        if match:
            result['useful_info'] = match.group(1).strip().strip('"\'')

        # need_info (legacy)
        match = re.search(r'need_info[:\s]+["\']?(.+?)(?=["\']?\n(?:can_answer|answer|reasoning|useful_info|tool|params|information_need)[:\s]|$)', text, re.DOTALL | re.IGNORECASE)
        if match:
            result['need_info'] = match.group(1).strip().strip('"\'')

        # information_need
        info_need = {}
        match = re.search(r'information_need[:\s]*\{[^}]*type[:\s]+["\']?(\w+)["\']?', text, re.DOTALL | re.IGNORECASE)
        if match:
            info_need['type'] = match.group(1)

        match = re.search(r'information_need[:\s]*\{[^}]*target[:\s]+["\']?([\w\d_]+)["\']?', text, re.DOTALL | re.IGNORECASE)
        if match:
            info_need['target'] = match.group(1)

        match = re.search(r'information_need[:\s]*\{[^}]*search_hints[:\s]+\[(.+?)\]', text, re.DOTALL | re.IGNORECASE)
        if match:
            hints = re.findall(r'["\']([^"\']+)["\']', match.group(1))
            if hints:
                info_need['search_hints'] = hints

        match = re.search(r'information_need[:\s]*\{[^}]*reason[:\s]+["\']?(.+?)["\']?(?=\s*\}|$)', text, re.DOTALL | re.IGNORECASE)
        if match:
            info_need['reason'] = match.group(1).strip()

        if info_need:
            result['information_need'] = info_need

        # tool (for analyzer output)
        match = re.search(r'tool[:\s]+(?:["\']?)(\w+)(?:["\']?)', text, re.IGNORECASE)
        if match:
            result['tool'] = match.group(1).strip()

        # params
        params = {}

        # super_id
        match = re.search(r'super[_]?id[:\s]+(?:["\']?)(super[_]?\d{1,2})(?:["\']?)', text, re.IGNORECASE)
        if match:
            raw_id = match.group(1).strip()
            digits = re.search(r'\d+', raw_id)
            if digits:
                params['super_id'] = f"super_{int(digits.group()):02d}"

        # macro_id
        match = re.search(r'macro[_]?id[:\s]+(?:["\']?)(macro[_]?\d{1,4})(?:["\']?)', text, re.IGNORECASE)
        if match:
            raw_id = match.group(1).strip()
            digits = re.search(r'\d+', raw_id)
            if digits:
                params['macro_id'] = f"macro_{int(digits.group()):04d}"

        # node_id
        match = re.search(r'node[_]?id[:\s]+(?:["\']?)([\w]+(?:_[\w]+)*)(?:["\']?)', text, re.IGNORECASE)
        if match:
            params['node_id'] = match.group(1).strip()

        # query
        match = re.search(r'query[:\s]+["\'](.+?)["\']', text, re.DOTALL)
        if match:
            params['query'] = match.group(1).strip()

        # name (for search_entity)
        match = re.search(r'name[:\s]+["\'](.+?)["\']', text, re.DOTALL)
        if match:
            params['name'] = match.group(1).strip()

        # slot_ids (list)
        match = re.search(r'slot_ids[:\s]+\[(.+?)\]', text)
        if match:
            ids = re.findall(r'\d+', match.group(1))
            if ids:
                params['slot_ids'] = [int(i) for i in ids]
        else:
            match = re.search(r'slot_ids[:\s]+\n((?:\s+-\s*\d+\n?)+)', text)
            if match:
                ids = re.findall(r'\d+', match.group(1))
                if ids:
                    params['slot_ids'] = [int(i) for i in ids]

        # entity_hints (list)
        match = re.search(r'entity_hints[:\s]+\[(.+?)\]', text)
        if match:
            hints = re.findall(r'["\']([^"\']+)["\']', match.group(1))
            if hints:
                params['entity_hints'] = hints

        # start_sec / end_sec
        match = re.search(r'start[_]?sec[:\s]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            params['start_sec'] = float(match.group(1))
        match = re.search(r'end[_]?sec[:\s]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            params['end_sec'] = float(match.group(1))

        # top_k
        match = re.search(r'top[_]?k[:\s]+(\d+)', text, re.IGNORECASE)
        if match:
            params['top_k'] = int(match.group(1))

        # node_types
        match = re.search(r'node_types[:\s]+\[(.+?)\]', text)
        if match:
            types = re.findall(r'["\']([^"\']+)["\']', match.group(1))
            if types:
                params['node_types'] = types

        # direction
        match = re.search(r'direction[:\s]+["\']?(outgoing|incoming|both)["\']?', text, re.IGNORECASE)
        if match:
            params['direction'] = match.group(1).lower()

        # answer (in params)
        match = re.search(r'params[:\s]+.*?answer[:\s]+(?:["\']?)([ABCD])(?:["\']?)', text, re.DOTALL | re.IGNORECASE)
        if match and result.get('tool') == 'answer':
            params['answer'] = match.group(1).upper()

        # reason (in params for cannot_answer)
        match = re.search(r'reason[:\s]+["\']?(.+?)(?=["\']?\n\w+[:\s]|$)', text, re.DOTALL | re.IGNORECASE)
        if match and result.get('tool') == 'cannot_answer':
            params['reason'] = match.group(1).strip().strip('"\'')

        # super_ids (list for get_super_events)
        match = re.search(r'super_ids[:\s]+\[(.+?)\]', text)
        if match:
            ids = re.findall(r'["\']([^"\']+)["\']', match.group(1))
            if ids:
                params['super_ids'] = ids

        # macro_ids (list for get_macro_events)
        match = re.search(r'macro_ids[:\s]+\[(.+?)\]', text)
        if match:
            ids = re.findall(r'["\']([^"\']+)["\']', match.group(1))
            if ids:
                params['macro_ids'] = ids

        if params:
            result['params'] = params

        return result