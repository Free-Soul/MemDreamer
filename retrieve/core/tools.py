"""
Tools: Retrieval tool definitions and implementations (MemoryBase version)

Each tool returns LLM-friendly formatted text in ToolResult.result,
and puts side-effect data (e.g., image paths) in ToolResult.metadata.
"""

import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from .memory_base import MemoryBase


def _fmt_time(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _fmt_time_range(tr) -> str:
    """Convert time range to [M:SS-M:SS] format."""
    if not tr or len(tr) < 2:
        return ""
    return f"[{_fmt_time(tr[0])}-{_fmt_time(tr[1])}]"


@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    result: str  # Formatted text for LLM context
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # Side-effect data (e.g., image paths)


class ToolRegistry:
    """Tool registry backed by MemoryBase"""

    def __init__(self, memory_base: MemoryBase, default_top_k: int = 10):
        self.mb = memory_base
        self.default_top_k = default_top_k
        self.tools: Dict[str, Callable] = {}
        self._register_tools()

    def _register_tools(self):
        self.tools = {
            "get_video_summary": self._get_video_summary,
            "get_super_events": self._get_super_events,
            "get_macro_events": self._get_macro_events,
            "get_subgraph": self._get_subgraph,
            "get_keyframes": self._get_keyframes,
            "search_nodes": self._search_nodes,
            "search_ocr_text": self._search_ocr_text,
            "search_by_time": self._search_by_time,
            "search_entity": self._search_entity,
            "get_relations": self._get_relations,
        }

    def get_tool_descriptions(self) -> str:
        return """Available tools:

1. get_video_summary
   - Description: Get overall video summary
   - Parameters: {} (no parameters needed)

2. get_super_events
   - Description: Get Super Event nodes (high-level video segments)
   - Parameters:
     - super_ids (optional): list of specific super node IDs, e.g., ["super_01", "super_02"]
   - Returns: Each super event contains time_range, key_entities, description, and child macro_ids

3. get_macro_events
   - Description: Get Macro Event nodes (detailed events within a Super Event)
   - Parameters:
     - super_id (optional): filter by super event ID, e.g., "super_01"
     - macro_ids (optional): list of specific macro event IDs, e.g., ["macro_0001", "macro_0002"]
   - Returns: Each macro event has slot_id, time_range, label, key_entity_names, description

4. get_subgraph
   - Description: Get detailed subgraph for a specific Macro Event. CONTAINS RICH STRUCTURED INFO!
   - Parameters: macro_id (required): the macro event ID (e.g., "macro_0001")
   - Returns:
     - Entities: Each entity has name, description, attributes, visual_grounding
     - Events: Each event has label, description, time_range, subject/object entity IDs
     - OCR texts: On-screen text with time ranges
     - Key Relations: CAUSAL, SPATIAL, ATTRIBUTIVE edges between entities and events

5. search_nodes
   - Description: Semantic search over Entity and Event nodes using embedding similarity. Excludes OCRText.
   - Parameters:
     - query (required): search query text, e.g., "three-point shot in the third quarter"
     - top_k (optional): number of results (default: 10)
     - node_types (optional): list of node types to filter, e.g., ["Event", "Entity"]
   - Returns: List of matching Entity/Event nodes with similarity scores

6. search_ocr_text
   - Description: Semantic search over OCR text nodes only. Use this to find on-screen text like scores, names, graphics.
   - Parameters:
     - query (required): search query text, e.g., "scoreboard", "USA score", "third quarter"
     - top_k (optional): number of results (default: 10)
   - Returns: List of matching OCRText nodes with similarity scores

7. search_by_time
   - Description: Find macro events covering a time range (in seconds from video start). Returns macro event IDs with parent super event, so you can then call get_subgraph for details.
   - Parameters:
     - start_sec (required): start time in seconds
     - end_sec (required): end time in seconds
   - Returns: List of macro events within the time range, with parent super event label

8. search_entity
   - Description: Search entity by name, using entity unification from Phase 3 Step 2.5
   - Parameters:
     - name (required): entity name to search, e.g., "LeBron James", "Team USA"
   - Returns: Entity occurrences across macro/super events with entity_details

9. get_relations
   - Description: Get relations for a specific node (outgoing and incoming edges)
   - Parameters:
     - node_id (required): the node ID (e.g., "super_01", "macro_0001", "slot0_ent_001")
     - direction (optional): "outgoing", "incoming", or "both" (default: "both")
   - Returns: List of relations with source, target, type, and label
"""

    def execute(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        if tool_name not in self.tools:
            return ToolResult(success=False, result="", error=f"Unknown tool: {tool_name}")

        try:
            return self.tools[tool_name](**params)
        except Exception as e:
            return ToolResult(success=False, result="", error=str(e))

    # =========================================================================
    # Tool implementations
    # =========================================================================

    def _get_video_summary(self) -> ToolResult:
        text = self.mb.get_video_summary()
        return ToolResult(success=True, result=text)

    def _get_super_events(self, super_ids: Optional[List[str]] = None) -> ToolResult:
        events = self.mb.get_super_events(ids=super_ids)
        if not events:
            return ToolResult(success=True, result="No super events found.")

        lines = [f"Found {len(events)} super events:\n"]
        for e in events:
            tr = _fmt_time_range(e.get("time_range"))
            children = e.get("child_ids", [])
            keys = e.get("key_entities", [])
            lines.append(f"[{e['node_id']}] {e['label']} {tr}")
            lines.append(f"  {e.get('macro_count', 0)} macro events | Key entities: {', '.join(keys)}")
            if e.get("description"):
                lines.append(f"  {e['description']}")
            lines.append("")

        return ToolResult(success=True, result="\n".join(lines))

    def _get_macro_events(
        self,
        super_id: Optional[str] = None,
        macro_ids: Optional[List[str]] = None,
    ) -> ToolResult:
        events = self.mb.get_macro_events(super_id=super_id, ids=macro_ids)
        if not events:
            return ToolResult(success=True, result="No macro events found.")

        lines = [f"Found {len(events)} macro events:\n"]
        for e in events:
            tr = _fmt_time_range(e.get("time_range"))
            slot = e.get("slot_id", "")
            keys = e.get("key_entity_names", [])
            slot_str = f"slot={slot}" if slot != "" else ""
            lines.append(f"[{e['node_id']}] {e['label']} {tr} {slot_str}")
            if keys:
                lines.append(f"  Key entities: {', '.join(keys)}")
            if e.get("description"):
                lines.append(f"  {e['description']}")
            lines.append("")

        return ToolResult(success=True, result="\n".join(lines))

    def _get_subgraph(self, macro_id: str) -> ToolResult:
        sg = self.mb.get_subgraph(macro_id)
        if not sg:
            return ToolResult(success=True, result=f"No subgraph found for {macro_id}.")

        text = _format_subgraph(sg)
        return ToolResult(success=True, result=text)

    def _get_keyframes(
        self,
        entity_ids: List[str],
    ) -> ToolResult:
        image_paths = []
        not_found = []
        lines = []

        for eid in entity_ids:
            node = self.mb.nodes.get(eid)
            if not node:
                not_found.append(eid)
                continue

            if node.node_type != "Entity":
                not_found.append(eid)
                continue

            #  keyframe
            kf_path = self.mb.get_keyframe_path(eid)
            if not kf_path:
                not_found.append(eid)
                continue

            lines.append(f"--- {eid}: {node.label} ({node.entity_type}) ---")
            if node.description:
                lines.append(f"  {node.description.replace(chr(10), ' ')}")
            if node.attributes:
                attrs = ", ".join(f"{k}: {v}" for k, v in node.attributes.items())
                lines.append(f"  Attributes: {attrs}")
            lines.append(f"  Keyframe: {kf_path}")
            lines.append("")
            image_paths.append(kf_path)

        if not image_paths:
            return ToolResult(
                success=True,
                result=f"No keyframe images found for the given entity IDs. Not found: {not_found}",
            )

        result_lines = [f"Found {len(image_paths)} keyframe images:\n"]
        result_lines.extend(lines)
        if not_found:
            result_lines.append(f"Not found / not Entity type: {not_found}")

        return ToolResult(
            success=True,
            result="\n".join(result_lines),
            metadata={"images": image_paths},
        )

    def _search_nodes(
        self,
        query: str,
        top_k: Optional[int] = None,
        node_types: Optional[List[str]] = None,
    ) -> ToolResult:
        top_k = top_k if top_k is not None else self.default_top_k
        # exclude_ocr=True:  OCRText
        # If node_types explicitly includes "OCRText", don't exclude it
        if node_types and "OCRText" in node_types:
            results = self.mb.search_by_embedding(query=query, top_k=top_k, node_types=node_types, exclude_ocr=False)
        else:
            results = self.mb.search_by_embedding(query=query, top_k=top_k, node_types=node_types, exclude_ocr=True)
        if not results:
            return ToolResult(success=True, result="No matching nodes found.")

        lines = [f"Found {len(results)} matching nodes for \"{query}\":\n"]
        for i, (node, score) in enumerate(results):
            tr = _fmt_time_range(node.time_range)

            # Parent macro
            ego = self.mb.get_node_ego_graph(node.node_id)
            if ego.get("parent_macro"):
                pm = ego["parent_macro"]
                lines.append(f"In <Macro Event> {pm['macro_id']} \"{pm['label']}\"")

            # Match node
            lines.append(f"<Match Node {i}> {node.node_id} ({node.node_type}) {tr}")
            if node.label:
                lines.append(f"  {node.label.replace(chr(10), ' ')}")
            if node.description and node.description != node.label:
                lines.append(f"  {node.description.replace(chr(10), ' ')}")

            # Related events and entities
            neighbors = ego.get("neighbors", [])
            if neighbors:
                lines.append("  Related events and entities:")
                for nb in neighbors:
                    direction = "->" if nb["direction"] == "outgoing" else "<-"
                    nb_label = nb["label"].replace("\n", " ")
                    nb_desc = nb.get("description", "")
                    if nb_desc and nb_desc != nb_label:
                        nb_desc = nb_desc.replace("\n", " ")
                        lines.append(f"    {direction}[{nb['relation_type']}:{nb['relation_label']}] {nb['node_id']} ({nb['node_type']}) \"{nb_label}\": {nb_desc}")
                    else:
                        lines.append(f"    {direction}[{nb['relation_type']}:{nb['relation_label']}] {nb['node_id']} ({nb['node_type']}) \"{nb_label}\"")

            lines.append("")

        return ToolResult(success=True, result="\n".join(lines))

    def _search_ocr_text(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> ToolResult:
        top_k = top_k if top_k is not None else self.default_top_k
        results = self.mb.search_by_embedding(query=query, top_k=top_k, node_types=["OCRText"], exclude_ocr=False)
        if not results:
            return ToolResult(success=True, result="No matching OCR text found.")

        lines = [f"Found {len(results)} OCR text matches for \"{query}\":\n"]
        for node, score in results:
            tr = _fmt_time_range(node.time_range)
            slot_str = f" slot={node.slot_id}" if node.slot_id >= 0 else ""
            lines.append(f"[{score:.3f}] {node.node_id} {tr}{slot_str}")
            if node.label:
                lines.append(f"  Text: {node.label.replace(chr(10), ' ')}")
            if node.description:
                lines.append(f"  {node.description.replace(chr(10), ' ')}")
            lines.append("")

        return ToolResult(success=True, result="\n".join(lines))

    def _search_by_time(
        self,
        start_sec: float,
        end_sec: float,
    ) -> ToolResult:
        nodes = self.mb.search_by_time(start=start_sec, end=end_sec, node_types=["MacroEvent"])
        if not nodes:
            return ToolResult(
                success=True,
                result=f"No macro events found in time range {_fmt_time(start_sec)}-{_fmt_time(end_sec)}.",
            )

        lines = [f"Found {len(nodes)} macro events in {_fmt_time(start_sec)}-{_fmt_time(end_sec)}:\n"]
        for n in nodes:
            tr = _fmt_time_range(n.time_range)
            #  super event
            parent_ids = [eid for eid in n.parent_ids if eid.startswith("super_")]
            parent_str = ""
            if parent_ids:
                parent = self.mb.nodes.get(parent_ids[0])
                if parent:
                    parent_str = f" (in {parent.label})"
            lines.append(f"[{n.node_id}] {n.label} {tr}{parent_str}")
            if n.key_entity_names:
                lines.append(f"  Key entities: {', '.join(n.key_entity_names)}")
            lines.append("")

        return ToolResult(success=True, result="\n".join(lines))

    def _search_entity(self, name: str) -> ToolResult:
        occurrences = self.mb.get_entity_occurrences(name)
        entity_nodes = self.mb.search_by_name(name, node_types=["Entity"])

        lines = [f"Entity \"{name}\" appears in {len(occurrences)} macro events:\n"]

        #  macro
        sorted_occ = sorted(occurrences, key=lambda o: o.get("time_range", [0])[0])
        for o in sorted_occ:
            tr = _fmt_time_range(o.get("time_range"))
            macro_label = o.get("macro_label", "")
            super_label = o.get("super_label", "")
            parent_str = f" (in {super_label})" if super_label else ""
            lines.append(f"  [{o['macro_id']}] {macro_label} {tr}{parent_str}")

        if entity_nodes:
            lines.append(f"\nEntity nodes ({len(entity_nodes)}):")
            for n in entity_nodes:
                tr = _fmt_time_range(n.time_range)
                lines.append(f"  [{n.node_id}] {n.label} ({n.entity_type}) slot={n.slot_id} {tr}")
                if n.description:
                    lines.append(f"    {n.description.replace(chr(10), ' ')}")

        # If only one macro, include entity_details inline
        if len(occurrences) == 1 and occurrences[0].get("entity_details"):
            lines.append("\nEntity details in this macro:")
            for d in occurrences[0]["entity_details"]:
                lines.append(f"  [{d['node_id']}] {d.get('description', '')}")
                if d.get("attributes"):
                    attrs = ", ".join(f"{k}: {v}" for k, v in d["attributes"].items())
                    lines.append(f"    Attributes: {attrs}")

        return ToolResult(success=True, result="\n".join(lines))

    def _get_relations(
        self,
        node_id: str,
        direction: str = "both",
    ) -> ToolResult:
        relations = self.mb.get_relations(node_id, direction=direction)
        if not relations:
            return ToolResult(success=True, result=f"No relations found for {node_id}.")

        # Separate outgoing vs incoming
        outgoing = [r for r in relations if r.get("source_id") == node_id]
        incoming = [r for r in relations if r.get("target_id") == node_id]

        lines = [f"Relations for {node_id} ({len(relations)} total):\n"]

        if outgoing:
            lines.append(f"--- Outgoing ({len(outgoing)}) ---")
            for r in outgoing:
                rtype = r.get("relation_type", "")
                rlabel = r.get("relation_label", "")
                target = r.get("target_id", "")
                target_label = r.get("target_label", "")
                target_str = f'{target} "{target_label}"' if target_label else target
                reason = r.get("reason")
                lines.append(f"  -[{rtype}:{rlabel}]-> {target_str}")
                if reason:
                    lines.append(f"    Reason: {reason}")
            lines.append("")

        if incoming:
            lines.append(f"--- Incoming ({len(incoming)}) ---")
            for r in incoming:
                rtype = r.get("relation_type", "")
                rlabel = r.get("relation_label", "")
                source = r.get("source_id", "")
                source_label = r.get("source_label", "")
                source_str = f'{source} "{source_label}"' if source_label else source
                lines.append(f"  <-[{rtype}:{rlabel}]- {source_str}")
            lines.append("")

        return ToolResult(success=True, result="\n".join(lines))


# =============================================================================
# Subgraph formatter
# =============================================================================

def _format_subgraph(sg: Dict) -> str:
    """Format subgraph data for LLM consumption"""
    lines = [f"=== Subgraph: {sg['macro_label']} ==="]
    lines.append(f"Time: {_fmt_time(sg['time_range'][0])}-{_fmt_time(sg['time_range'][1])}")
    if sg.get("macro_description"):
        lines.append(f"Description: {sg['macro_description']}")
    lines.append("")

    # Entities
    entities = sg.get("entities", [])
    lines.append(f"--- Entities ({len(entities)}) ---")
    for ent in entities:
        etype = ent.get("entity_type", "Unknown")
        lines.append(f"[{ent['node_id']}] {ent['name']} ({etype})")
        if ent.get("description"):
            lines.append(f"  {ent['description']}")
        if ent.get("attributes"):
            attrs = ", ".join(f"{k}: {v}" for k, v in ent["attributes"].items())
            lines.append(f"  Attributes: {attrs}")
        if ent.get("visual_grounding", {}).get("distinctive_features"):
            feats = ent["visual_grounding"]["distinctive_features"]
            lines.append(f"  Visual: {', '.join(feats)}")
    lines.append("")

    # Events
    events = sg.get("events", [])
    lines.append(f"--- Events ({len(events)}) ---")
    for evt in events:
        tr = _fmt_time_range(evt.get("time_range"))
        lines.append(f"[{evt['node_id']}] {evt['label']} {tr}")
        if evt.get("description"):
            lines.append(f"  {evt['description']}")
    lines.append("")

    # OCR texts
    ocr_texts = sg.get("ocr_texts", [])
    if ocr_texts:
        lines.append(f"--- OCR Texts ({len(ocr_texts)}) ---")
        for ocr in ocr_texts:
            tr = _fmt_time_range(ocr.get("time_range"))
            # Collapse OCR text to single line
            ocr_text = ocr["text"].replace("\n", " ")
            lines.append(f"[{ocr['node_id']}] \"{ocr_text}\" {tr}")
        lines.append("")

    # Edges — filter out common PERFORMS / HAPPENS_IN / TEMPORAL, show high-value ones
    edges = sg.get("edges", [])
    HIGH_VALUE_TYPES = {"CAUSAL", "SPATIAL", "ATTRIBUTIVE"}
    HIGH_VALUE_LABELS = {"USES_TOOL", "RECEIVES", "CONTEXT_FOR"}
    important_edges = []
    other_edges = []
    for edge in edges:
        rtype = edge.get("relation_type", "")
        rlabel = edge.get("relation_label", "")
        if rtype in HIGH_VALUE_TYPES or rlabel in HIGH_VALUE_LABELS:
            important_edges.append(edge)
        else:
            other_edges.append(edge)

    if important_edges:
        lines.append(f"--- Key Relations ({len(important_edges)}) ---")
        for edge in important_edges:
            src_label = edge.get("source_label", "").replace("\n", " ")
            tgt_label = edge.get("target_label", "").replace("\n", " ")
            src_str = f"{edge['source_id']} ({src_label})" if src_label else edge['source_id']
            tgt_str = f"{edge['target_id']} ({tgt_label})" if tgt_label else edge['target_id']
            lines.append(f"  {src_str} -[{edge['relation_type']}:{edge['relation_label']}]-> {tgt_str}")
        lines.append("")

    return "\n".join(lines)