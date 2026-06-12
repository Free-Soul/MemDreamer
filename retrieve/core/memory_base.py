"""
Memory Base: Unified memory loader + retrieval API

Loads Phase 2 + Phase 3 data, builds multi-dimensional indexes, provides query APIs.

Data sources:
  - Phase 3: hierarchical_graph_final.json (hierarchy + relations)
  - Phase 3: step2p5_entity_unification.json (entity name normalization)
  - Phase 2: step5_subgraph.json x N (Entity/Event/OCRText + edges)
  - Phase 2: features/entity_id/best_frame.jpg (visual frames)

Four-level nested graph:
  Root -> SuperEvent -> MacroEvent -> Subgraph (H-STSG)
"""

import json
import logging
import os
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MemoryNode:
    """Unified node representation."""
    node_id: str
    node_type: str  # RootEvent | SuperEvent | MacroEvent | Entity | Event | OCRText
    label: str
    description: str
    time_range: Tuple[float, float]  # (start_sec, end_sec)
    embedding: Optional[List[float]] = None

    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)

    attributes: Dict[str, Any] = field(default_factory=dict)
    slot_id: int = -1

    # Entity
    entity_type: str = ""
    visual_grounding: Dict[str, Any] = field(default_factory=dict)
    original_ids: List[str] = field(default_factory=list)

    # Event
    event_type: str = ""
    subject_entity_ids: List[str] = field(default_factory=list)
    object_entity_ids: List[str] = field(default_factory=list)

    # OCRText
    ocr_type: str = ""
    text: str = ""

    # SuperEvent
    key_entities: List[str] = field(default_factory=list)

    # MacroEvent
    key_entity_names: List[str] = field(default_factory=list)
    subevent_ids: List[str] = field(default_factory=list)

    # RootEvent
    title: str = ""
    themes: List[str] = field(default_factory=list)
    emotional_tone: str = ""


@dataclass
class MemoryEdge:
    """Unified edge representation."""
    source_id: str
    target_id: str
    relation_type: str   # HIERARCHICAL | PROGRESSION | CAUSAL | TEMPORAL | SEMANTIC | SPATIAL | ATTRIBUTIVE
    relation_label: str  # SUBEVENT_OF | LEADS_TO | CAUSES | BEFORE | PERFORMS | ...
    source_label: str = ""
    target_label: str = ""
    reason: Optional[str] = None


# =============================================================================
# MemoryBase
# =============================================================================

class MemoryBase:
    """Unified memory loader + retrieval API."""

    def __init__(self):
        # -- Primary storage --
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[MemoryEdge] = []

        # -- Indexes --
        self.nodes_by_type: Dict[str, List[str]] = {}
        self.edges_by_source: Dict[str, List[MemoryEdge]] = {}
        self.edges_by_target: Dict[str, List[MemoryEdge]] = {}
        self.entity_name_to_ids: Dict[str, List[str]] = {}
        self.entity_name_to_macro_ids: Dict[str, List[str]] = {}
        self.time_index: List[Tuple[float, float, str]] = []

        # -- Cached subgraphs (macro_id -> subgraph dict) --
        self._subgraphs: Dict[str, Dict] = {}

        # -- Entity unification --
        self._entity_canonical: Dict[str, str] = {}  # variant_name -> canonical_name
        self._entity_groups: List[Dict] = []

        # -- Keyframe directory --
        self._phase2_dir: Optional[Path] = None

        # -- Embedding --
        self._embedder = None
        self._emb_matrix = None       # numpy matrix for fast search
        self._emb_node_ids = []       # aligned node IDs for matrix rows

    # =========================================================================
    # Loading
    # =========================================================================

    def load(
        self,
        phase2_dir: str,
        phase3_dir: str,
    ) -> "MemoryBase":
        """
        Load all Phase 2 + Phase 3 data

        Args:
            phase2_dir: Phase 2 TopDown Pipeline output directory
            phase3_dir: Phase 3 output directory

        Returns:
            self (for chaining)
        """
        self._phase2_dir = Path(phase2_dir)

        # 1. Load Phase 3 (hierarchical graph)
        self._load_hierarchical_graph(phase3_dir)

        # 2. Load Phase 2 subgraphs
        self._load_subgraphs(phase2_dir)

        # 3. Load entity unification
        self._load_entity_unification(phase3_dir)

        # 4. Build indexes
        self._build_indexes()

        # 5. Link hierarchy (parent/child)
        self._link_hierarchy()

        logger.info(
            f"MemoryBase loaded: "
            f"{len(self.nodes)} nodes, "
            f"{len(self.edges)} edges, "
            f"{len(self.entity_name_to_ids)} entity names, "
            f"{len(self.time_index)} time-indexed nodes"
        )

        return self

    def _load_hierarchical_graph(self, phase3_dir: str):
        """Load Phase 3 hierarchical graph (Root + Super + Macro + relations)."""
        phase3_path = Path(phase3_dir)

        # Try final graph, fallback to intermediate graph
        graph_file = phase3_path / "hierarchical_graph_final.json"
        if not graph_file.exists():
            graph_file = phase3_path / "hierarchical_graph.json"
        if not graph_file.exists():
            raise FileNotFoundError(f"Phase 3 graph not found in {phase3_dir}")

        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Root Event
        root_data = data.get("root_event")
        if root_data:
            time_range = root_data.get("time_range", {})
            start = time_range.get("start", 0) if isinstance(time_range, dict) else time_range[0]
            end = time_range.get("end", 0) if isinstance(time_range, dict) else time_range[1]
            root_node = MemoryNode(
                node_id=root_data.get("node_id", "root_001"),
                node_type="RootEvent",
                label=root_data.get("title", ""),
                description=root_data.get("description", ""),
                time_range=(float(start), float(end)),
                title=root_data.get("title", ""),
                themes=root_data.get("themes", []),
                emotional_tone=root_data.get("emotional_tone", ""),
                key_entities=root_data.get("key_entities", []),
            )
            self.nodes[root_node.node_id] = root_node

        # Super Events
        for se in data.get("super_events", []):
            tr = se.get("time_range", [0, 0])
            node = MemoryNode(
                node_id=se["node_id"],
                node_type="SuperEvent",
                label=se.get("label", ""),
                description=se.get("description", ""),
                time_range=(float(tr[0]), float(tr[1])),
                key_entities=se.get("key_entities", []),
                child_ids=se.get("sub_macro_ids", []),
            )
            self.nodes[node.node_id] = node

        # Macro Events
        for me in data.get("macro_events", []):
            tr = me.get("time_range", [0, 0])
            slot_ids = me.get("anchors", {}).get("slot_ids", [])
            slot_id = slot_ids[0] if slot_ids else -1
            label = me.get("label", "").replace("_", " ")
            node = MemoryNode(
                node_id=me["node_id"],
                node_type="MacroEvent",
                label=label,
                description=me.get("summary", ""),
                time_range=(float(tr[0]), float(tr[1])),
                slot_id=slot_id,
                key_entity_names=me.get("key_entity_names", me.get("key_entities", [])),
                subevent_ids=me.get("subevent_ids", []),
            )
            self.nodes[node.node_id] = node

        # Edges — attach source/target labels
        for edge_data in data.get("super_relations", []):
            src_id = edge_data["source_id"]
            tgt_id = edge_data["target_id"]
            src_node = self.nodes.get(src_id)
            tgt_node = self.nodes.get(tgt_id)
            self.edges.append(MemoryEdge(
                source_id=src_id,
                target_id=tgt_id,
                relation_type=edge_data.get("relation_type", "PROGRESSION"),
                relation_label=edge_data.get("relation_label", ""),
                source_label=src_node.label if src_node else "",
                target_label=tgt_node.label if tgt_node else "",
                reason=edge_data.get("reason"),
            ))

        for edge_data in data.get("macro_relations", []):
            src_id = edge_data["source_id"]
            tgt_id = edge_data["target_id"]
            src_node = self.nodes.get(src_id)
            tgt_node = self.nodes.get(tgt_id)
            self.edges.append(MemoryEdge(
                source_id=src_id,
                target_id=tgt_id,
                relation_type=edge_data.get("relation_type", "CAUSAL"),
                relation_label=edge_data.get("relation_label", ""),
                source_label=src_node.label if src_node else "",
                target_label=tgt_node.label if tgt_node else "",
                reason=edge_data.get("reason"),
            ))

        for edge_data in data.get("subevent_relations", []):
            src_id = edge_data["source_id"]
            tgt_id = edge_data["target_id"]
            src_node = self.nodes.get(src_id)
            tgt_node = self.nodes.get(tgt_id)
            self.edges.append(MemoryEdge(
                source_id=src_id,
                target_id=tgt_id,
                relation_type="HIERARCHICAL",
                relation_label=edge_data.get("relation_label", "SUBEVENT_OF"),
                source_label=src_node.label if src_node else "",
                target_label=tgt_node.label if tgt_node else "",
            ))

        for edge_data in data.get("root_relations", []):
            src_id = edge_data["source_id"]
            tgt_id = edge_data["target_id"]
            src_node = self.nodes.get(src_id)
            tgt_node = self.nodes.get(tgt_id)
            self.edges.append(MemoryEdge(
                source_id=src_id,
                target_id=tgt_id,
                relation_type="HIERARCHICAL",
                relation_label=edge_data.get("relation_label", "SUBEVENT_OF"),
                source_label=src_node.label if src_node else "",
                target_label=tgt_node.label if tgt_node else "",
            ))

        logger.info(
            f"Loaded Phase 3: "
            f"{1 if root_data else 0} Root, "
            f"{len(data.get('super_events', []))} Super, "
            f"{len(data.get('macro_events', []))} Macro, "
            f"{len(self.edges)} edges"
        )

    def _load_subgraphs(self, phase2_dir: str):
        """Load Phase 2 subgraphs for all episodes."""
        phase2_path = Path(phase2_dir)
        episode_dirs = sorted(phase2_path.glob("episode_*"))

        if not episode_dirs:
            raise FileNotFoundError(f"No episode directories found in {phase2_dir}")

        total_entities = 0
        total_events = 0
        total_ocr = 0
        total_edges = 0

        for episode_dir in episode_dirs:
            # Try nested structure first, then flat
            subgraph_file = episode_dir / episode_dir.name / "step5_subgraph.json"
            if not subgraph_file.exists():
                subgraph_file = episode_dir / "step5_subgraph.json"
            if not subgraph_file.exists():
                continue

            with open(subgraph_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            slot_id = data.get("slot_id", -1)
            ep_start = data.get("start_sec", 0.0)
            ep_end = data.get("end_sec", 0.0)

            #  macro_id
            macro_id = self._find_macro_by_slot(slot_id)

            #  get_subgraph
            if macro_id:
                self._subgraphs[macro_id] = data

            for node_data in data.get("nodes", []):
                node_type = node_data.get("node_type", "")

                if node_type == "MacroEvent":
                    # MacroEvent already loaded from Phase 3, skip
                    continue

                #  subgraph  slot  episode ID
                original_id = str(node_data.get("node_id", node_data.get("entity_id", node_data.get("event_id", node_data.get("ocr_id", "")))))
                global_id = f"slot{slot_id}_{original_id}"

                node = self._parse_subgraph_node(node_data, global_id, slot_id, ep_start, ep_end)
                if node:
                    #  macro
                    if macro_id:
                        node.parent_ids.append(macro_id)
                        # Add HIERARCHICAL edge: subgraph node -> macro
                        macro_node = self.nodes.get(macro_id)
                        self.edges.append(MemoryEdge(
                            source_id=global_id,
                            target_id=macro_id,
                            relation_type="HIERARCHICAL",
                            relation_label="SUBEVENT_OF",
                            source_label=node.label,
                            target_label=macro_node.label if macro_node else "",
                        ))
                    self.nodes[node.node_id] = node

                    if node_type == "Entity":
                        total_entities += 1
                    elif node_type == "Event":
                        total_events += 1
                    elif node_type == "OCRText":
                        total_ocr += 1

            for edge_data in data.get("edges", []):
                source_id = str(edge_data.get("source_id", ""))
                target_id = str(edge_data.get("target_id", ""))

                #  slot
                if not source_id.startswith(f"slot{slot_id}_"):
                    source_id = f"slot{slot_id}_{source_id}"
                if not target_id.startswith(f"slot{slot_id}_"):
                    target_id = f"slot{slot_id}_{target_id}"

                # Index by label
                src_node = self.nodes.get(source_id)
                tgt_node = self.nodes.get(target_id)

                self.edges.append(MemoryEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=edge_data.get("relation_type", "SEMANTIC"),
                    relation_label=edge_data.get("relation_label", ""),
                    source_label=src_node.label if src_node else "",
                    target_label=tgt_node.label if tgt_node else "",
                ))
                total_edges += 1

        logger.info(
            f"Loaded Phase 2 subgraphs: "
            f"{total_entities} Entities, {total_events} Events, {total_ocr} OCR, "
            f"{total_edges} edges"
        )

    def _parse_subgraph_node(
        self,
        node_data: Dict,
        global_id: str,
        slot_id: int,
        ep_start: float,
        ep_end: float,
    ) -> Optional[MemoryNode]:
        """Parse a single node from subgraph data."""
        node_type = node_data.get("node_type", "")

        if node_type == "Entity":
            name = node_data.get("name", "") or ""
            # If name is None, fallback to entity_type + description
            if not name or name == "None":
                name = node_data.get("entity_type", "Entity")
                if desc := node_data.get("description", ""):
                    name = f"{name}: {desc}"
            etype = node_data.get("entity_type", "")
            desc = node_data.get("description", "")
            attrs = node_data.get("attributes", {})
            vg = node_data.get("visual_grounding", {})
            anchors = node_data.get("anchors", {}).get("slot_ids", [])
            slot_ids = anchors if anchors else [slot_id]

            # Use visual_grounding time if available
            vg_time = vg.get("primary_time_sec", None)
            if vg_time is not None:
                tr = (ep_start + float(vg_time), ep_start + float(vg_time) + 5)
            else:
                tr = (ep_start, ep_end)

            return MemoryNode(
                node_id=global_id,
                node_type="Entity",
                label=name,
                description=desc,
                time_range=tr,
                slot_id=slot_id,
                attributes=attrs,
                entity_type=etype,
                visual_grounding=vg,
                original_ids=[node_data.get("entity_id", node_data.get("node_id", ""))],
            )

        elif node_type == "Event":
            evt_type = node_data.get("event_type", "")
            desc = node_data.get("description", "")
            label = node_data.get("action", desc if desc else "")
            tr_data = node_data.get("time_range", [0, 0])
            local_start = float(tr_data[0]) if tr_data else 0
            local_end = float(tr_data[1]) if tr_data and len(tr_data) > 1 else local_start
            tr = (ep_start + local_start, ep_start + local_end)

            return MemoryNode(
                node_id=global_id,
                node_type="Event",
                label=label,
                description=desc,
                time_range=tr,
                slot_id=slot_id,
                event_type=evt_type,
                subject_entity_ids=node_data.get("subject_entity_ids", []),
                object_entity_ids=node_data.get("object_entity_ids", []),
            )

        elif node_type == "OCRText":
            text = node_data.get("text", "")
            ocr_type = node_data.get("ocr_type", "")
            desc = node_data.get("description", "")
            tr_data = node_data.get("time_range", [0, 0])
            local_start = float(tr_data[0]) if tr_data else 0
            local_end = float(tr_data[1]) if tr_data and len(tr_data) > 1 else local_start
            tr = (ep_start + local_start, ep_start + local_end)

            return MemoryNode(
                node_id=global_id,
                node_type="OCRText",
                label=text,
                description=desc,
                time_range=tr,
                slot_id=slot_id,
                ocr_type=ocr_type,
                text=text,
            )

        return None

    def _find_macro_by_slot(self, slot_id: int) -> Optional[str]:
        """Find the macro_id corresponding to a slot_id."""
        for node_id, node in self.nodes.items():
            if node.node_type == "MacroEvent" and node.slot_id == slot_id:
                return node_id
        return None

    def _load_entity_unification(self, phase3_dir: str):
        """Load entity name normalization mapping."""
        phase3_path = Path(phase3_dir)
        unif_file = phase3_path / "intermediate" / "step2p5_entity_unification.json"

        if not unif_file.exists():
            logger.info("No entity unification file found, skipping")
            return

        with open(unif_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for mapping in data.get("entity_mappings", []):
            canonical = mapping.get("canonical_name", "")
            canonical_type = mapping.get("canonical_type", "")
            variants = mapping.get("variants", [])
            self._entity_groups.append({
                "canonical_name": canonical,
                "canonical_type": canonical_type,
                "variants": variants,
            })
            for variant in variants:
                self._entity_canonical[variant] = canonical

        logger.info(f"Loaded {len(self._entity_groups)} entity unification groups, "
                     f"{len(self._entity_canonical)} variant mappings")

    def _build_indexes(self):
        """Build all indexes."""
        # nodes_by_type
        self.nodes_by_type = {}
        for node_id, node in self.nodes.items():
            nt = node.node_type
            if nt not in self.nodes_by_type:
                self.nodes_by_type[nt] = []
            self.nodes_by_type[nt].append(node_id)

        # edges_by_source / edges_by_target
        self.edges_by_source = {}
        self.edges_by_target = {}
        for edge in self.edges:
            self.edges_by_source.setdefault(edge.source_id, []).append(edge)
            self.edges_by_target.setdefault(edge.target_id, []).append(edge)

        # entity_name_to_ids (using canonical names + variants)
        self.entity_name_to_ids = {}
        self.entity_name_to_macro_ids = {}

        for node_id, node in self.nodes.items():
            if node.node_type == "Entity" and node.label:
                # Index by label
                name_lower = node.label.lower()
                self.entity_name_to_ids.setdefault(name_lower, []).append(node_id)
                canonical = self._entity_canonical.get(node.label)
                if canonical:
                    canonical_lower = canonical.lower()
                    if canonical_lower != name_lower:
                        self.entity_name_to_ids.setdefault(canonical_lower, []).append(node_id)
                    for variant in self._entity_canonical:
                        if self._entity_canonical[variant] == canonical:
                            variant_lower = variant.lower()
                            if variant_lower != name_lower and variant_lower != canonical_lower:
                                self.entity_name_to_ids.setdefault(variant_lower, []).append(node_id)

        # entity_name_to_macro_ids (from SuperEvent.key_entities and MacroEvent.key_entity_names)
        for node_id, node in self.nodes.items():
            if node.node_type == "SuperEvent":
                for ent_name in node.key_entities:
                    name_lower = ent_name.lower()
                    for child_id in node.child_ids:
                        self.entity_name_to_macro_ids.setdefault(name_lower, []).append(child_id)
            elif node.node_type == "MacroEvent":
                for ent_name in node.key_entity_names:
                    name_lower = ent_name.lower()
                    self.entity_name_to_macro_ids.setdefault(name_lower, []).append(node_id)
                # Subgraph Entity nodes are already in entity_name_to_ids

        for k in self.entity_name_to_macro_ids:
            self.entity_name_to_macro_ids[k] = list(set(self.entity_name_to_macro_ids[k]))

        # time_index
        self.time_index = []
        for node_id, node in self.nodes.items():
            if node.time_range and node.time_range[0] is not None and node.time_range[1] is not None:
                self.time_index.append((node.time_range[0], node.time_range[1], node_id))
        self.time_index.sort(key=lambda x: x[0])

        # Assign slot_id to Super/Macro nodes from child macros
        for node_id, node in self.nodes.items():
            if node.node_type == "SuperEvent" and node.slot_id < 0:
                # Inherit slot_id from first child macro
                for child_id in node.child_ids:
                    child = self.nodes.get(child_id)
                    if child and child.slot_id >= 0:
                        node.slot_id = child.slot_id
                        break

    def _link_hierarchy(self):
        """Link hierarchy relations (parent_ids, child_ids)."""
        # Macro -> Super (via SUBEVENT_OF edges)
        for node_id, node in self.nodes.items():
            if node.node_type == "MacroEvent":
                # Find parent SuperEvent
                for edge in self.edges_by_target.get(node_id, []):
                    if edge.relation_label == "SUBEVENT_OF":
                        parent_id = edge.source_id
                        if parent_id in self.nodes and parent_id not in node.parent_ids:
                            node.parent_ids.append(parent_id)

                # Fallback: find parent from super.child_ids
                if not node.parent_ids:
                    for sid, snode in self.nodes.items():
                        if snode.node_type == "SuperEvent" and node_id in snode.child_ids:
                            if sid not in node.parent_ids:
                                node.parent_ids.append(sid)

        # Sync Super.child_ids from Macro parent_ids
        for node_id, node in self.nodes.items():
            if node.node_type == "MacroEvent":
                for parent_id in node.parent_ids:
                    parent = self.nodes.get(parent_id)
                    if parent and node_id not in parent.child_ids:
                        parent.child_ids.append(node_id)

        # Subgraph nodes (Entity/Event/OCRText) -> Macro
        # parent_ids set in _load_subgraphs; now sync Macro.child_ids
        for node_id, node in self.nodes.items():
            if node.node_type in ("Entity", "Event", "OCRText"):
                for parent_id in node.parent_ids:
                    parent = self.nodes.get(parent_id)
                    if parent and node_id not in parent.child_ids:
                        parent.child_ids.append(node_id)

        # Super -> Root (via root_relations)
        for edge in self.edges:
            if edge.relation_label == "SUBEVENT_OF":
                target = self.nodes.get(edge.target_id)
                source = self.nodes.get(edge.source_id)
                if target and target.node_type == "RootEvent" and source:
                    if edge.source_id not in target.child_ids:
                        target.child_ids.append(edge.source_id)
                    if edge.target_id not in source.parent_ids:
                        source.parent_ids.append(edge.target_id)

        # Root -> Slot_id (inherit from child Super)
        for node_id, node in self.nodes.items():
            if node.node_type == "RootEvent" and node.slot_id < 0:
                for super_node_id in node.child_ids:
                    super_node = self.nodes.get(super_node_id)
                    if super_node and super_node.slot_id >= 0:
                        node.slot_id = super_node.slot_id
                        break

    # =========================================================================
    # Query API
    # =========================================================================

    def get_video_summary(self) -> str:
        """Get video summary."""
        for node in self.nodes.values():
            if node.node_type == "RootEvent":
                parts = []
                if node.title:
                    parts.append(f"Title: {node.title}")
                if node.description:
                    parts.append(node.description)
                if node.themes:
                    parts.append(f"Themes: {', '.join(node.themes)}")
                if node.key_entities:
                    parts.append(f"Key Entities: {', '.join(node.key_entities[:10])}")
                return "\n".join(parts)
        return ""

    def get_super_events(
        self,
        ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get SuperEvent list."""
        if ids:
            nodes = [self.nodes[sid] for sid in ids if sid in self.nodes]
        else:
            nodes = [self.nodes[sid] for sid in self.nodes_by_type.get("SuperEvent", [])]

        return [
            {
                "node_id": n.node_id,
                "label": n.label,
                "description": n.description,
                "time_range": list(n.time_range),
                "key_entities": n.key_entities,
                "macro_count": len(n.child_ids),
                "parent_ids": n.parent_ids,
            }
            for n in sorted(nodes, key=lambda x: x.time_range[0])
        ]

    def get_macro_events(
        self,
        super_id: Optional[str] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get MacroEvent list."""
        if ids:
            nodes = [self.nodes[mid] for mid in ids if mid in self.nodes]
        elif super_id and super_id in self.nodes:
            parent = self.nodes[super_id]
            nodes = [self.nodes[cid] for cid in parent.child_ids if cid in self.nodes]
        else:
            nodes = [self.nodes[mid] for mid in self.nodes_by_type.get("MacroEvent", [])]

        return [
            {
                "node_id": n.node_id,
                "label": n.label,
                "description": n.description,
                "time_range": list(n.time_range),
                "slot_id": n.slot_id,
                "key_entity_names": n.key_entity_names,
                "parent_ids": n.parent_ids,
            }
            for n in sorted(nodes, key=lambda x: x.time_range[0])
        ]

    def get_subgraph(self, macro_id: str) -> Optional[Dict[str, Any]]:
        """Get the episode subgraph for a MacroEvent."""
        if macro_id not in self.nodes:
            return None

        macro = self.nodes[macro_id]
        if macro.slot_id < 0:
            return None

        #  subgraph
        if macro_id in self._subgraphs:
            raw = self._subgraphs[macro_id]
            return self._format_subgraph(macro, raw)

        # fallback:  nodes
        return self._build_subgraph_from_nodes(macro)

    def _format_subgraph(self, macro: MemoryNode, raw: Dict) -> Dict[str, Any]:
        """Format raw subgraph data into return format."""
        slot_id = raw.get("slot_id", macro.slot_id)
        prefix = f"slot{slot_id}_"

        entities = []
        events = []
        ocr_texts = []

        for node_data in raw.get("nodes", []):
            nt = node_data.get("node_type", "")
            original_id = node_data.get("node_id", node_data.get("entity_id", node_data.get("event_id", node_data.get("ocr_id", ""))))
            global_id = f"{prefix}{original_id}"

            if nt == "Entity":
                raw_name = node_data.get("name", "") or ""
                if not raw_name or raw_name == "None":
                    raw_name = node_data.get("entity_type", "Entity")
                    if node_data.get("description", ""):
                        raw_name = f"{raw_name}: {node_data['description']}"
                entities.append({
                    "node_id": global_id,
                    "name": raw_name,
                    "entity_type": node_data.get("entity_type", ""),
                    "description": node_data.get("description", ""),
                    "attributes": node_data.get("attributes", {}),
                    "visual_grounding": node_data.get("visual_grounding", {}),
                })
            elif nt == "Event":
                events.append({
                    "node_id": global_id,
                    "event_type": node_data.get("event_type", ""),
                    "label": node_data.get("action", ""),
                    "description": node_data.get("description", ""),
                    "time_range": node_data.get("time_range"),
                    "subject_entity_ids": node_data.get("subject_entity_ids", []),
                    "object_entity_ids": node_data.get("object_entity_ids", []),
                })
            elif nt == "OCRText":
                ocr_texts.append({
                    "node_id": global_id,
                    "text": node_data.get("text", ""),
                    "ocr_type": node_data.get("ocr_type", ""),
                    "description": node_data.get("description", ""),
                    "time_range": node_data.get("time_range"),
                })

        # Build node_id -> label mapping for subgraph edges
        node_ids = {e["node_id"] for e in entities} | {e["node_id"] for e in events} | {e["node_id"] for e in ocr_texts}
        # node_id -> label lookup
        id_to_label = {}
        for e in entities:
            id_to_label[e["node_id"]] = e["name"]
        for e in events:
            id_to_label[e["node_id"]] = e.get("label", "")
        for e in ocr_texts:
            id_to_label[e["node_id"]] = e["text"]

        edges = []
        for edge_data in raw.get("edges", []):
            src = edge_data.get("source_id", "")
            tgt = edge_data.get("target_id", "")
            src_g = f"{prefix}{src}" if not src.startswith(prefix) else src
            tgt_g = f"{prefix}{tgt}" if not tgt.startswith(prefix) else tgt
            if src_g in node_ids or tgt_g in node_ids:
                edges.append({
                    "source_id": src_g,
                    "source_label": id_to_label.get(src_g, ""),
                    "target_id": tgt_g,
                    "target_label": id_to_label.get(tgt_g, ""),
                    "relation_type": edge_data.get("relation_type", ""),
                    "relation_label": edge_data.get("relation_label", ""),
                })

        return {
            "macro_id": macro.node_id,
            "macro_label": macro.label,
            "macro_description": macro.description,
            "time_range": list(macro.time_range),
            "slot_id": slot_id,
            "entities": entities,
            "events": events,
            "ocr_texts": ocr_texts,
            "edges": edges,
        }

    def _build_subgraph_from_nodes(self, macro: MemoryNode) -> Dict[str, Any]:
        """Build subgraph from nodes index (fallback)."""
        entities = []
        events = []
        ocr_texts = []
        edge_prefix = f"slot{macro.slot_id}_"

        for child_id in macro.child_ids:
            node = self.nodes.get(child_id)
            if not node:
                continue

            if node.node_type == "Entity":
                entities.append({
                    "node_id": node.node_id,
                    "name": node.label,
                    "entity_type": node.entity_type,
                    "description": node.description,
                    "attributes": node.attributes,
                    "visual_grounding": node.visual_grounding,
                })
            elif node.node_type == "Event":
                events.append({
                    "node_id": node.node_id,
                    "event_type": node.event_type,
                    "label": node.label,
                    "description": node.description,
                    "time_range": list(node.time_range),
                    "subject_entity_ids": node.subject_entity_ids,
                    "object_entity_ids": node.object_entity_ids,
                })
            elif node.node_type == "OCRText":
                ocr_texts.append({
                    "node_id": node.node_id,
                    "text": node.text,
                    "ocr_type": node.ocr_type,
                    "description": node.description,
                    "time_range": list(node.time_range),
                })

        #  slot
        node_ids = {e["node_id"] for e in entities} | {e["node_id"] for e in events} | {e["node_id"] for e in ocr_texts}
        edges = []
        for nid in node_ids:
            for edge in self.edges_by_source.get(nid, []):
                if edge.target_id in node_ids or edge.target_id == macro.node_id:
                    edges.append({
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "relation_type": edge.relation_type,
                        "relation_label": edge.relation_label,
                    })

        return {
            "macro_id": macro.node_id,
            "macro_label": macro.label,
            "macro_description": macro.description,
            "time_range": list(macro.time_range),
            "slot_id": macro.slot_id,
            "entities": entities,
            "events": events,
            "ocr_texts": ocr_texts,
            "edges": edges,
        }

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a single node by ID."""
        return self.nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        direction: str = "both",
    ) -> List[MemoryNode]:
        """Get neighbor nodes."""
        neighbor_ids = set()

        if direction in ("both", "outgoing"):
            for edge in self.edges_by_source.get(node_id, []):
                if edge_types is None or edge.relation_type in edge_types:
                    neighbor_ids.add(edge.target_id)

        if direction in ("both", "incoming"):
            for edge in self.edges_by_target.get(node_id, []):
                if edge_types is None or edge.relation_type in edge_types:
                    neighbor_ids.add(edge.source_id)

        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def get_relations(
        self,
        node_id: str,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        """Get all relations for a node."""
        result = []

        if direction in ("both", "outgoing"):
            for edge in self.edges_by_source.get(node_id, []):
                target = self.nodes.get(edge.target_id)
                result.append({
                    "source_id": edge.source_id,
                    "source_label": edge.source_label,
                    "target_id": edge.target_id,
                    "target_label": edge.target_label or (target.label if target else ""),
                    "relation_type": edge.relation_type,
                    "relation_label": edge.relation_label,
                    "reason": edge.reason,
                    "target_type": target.node_type if target else "",
                })

        if direction in ("both", "incoming"):
            for edge in self.edges_by_target.get(node_id, []):
                source = self.nodes.get(edge.source_id)
                result.append({
                    "source_id": edge.source_id,
                    "source_label": edge.source_label or (source.label if source else ""),
                    "target_id": edge.target_id,
                    "target_label": edge.target_label,
                    "relation_type": edge.relation_type,
                    "relation_label": edge.relation_label,
                    "reason": edge.reason,
                    "source_type": source.node_type if source else "",
                })

        return result

    def search_by_name(
        self,
        name: str,
        node_types: Optional[List[str]] = None,
    ) -> List[MemoryNode]:
        """Search nodes by name (supports entity name normalization)."""
        name_lower = name.lower().strip()
        results = []

        #  ()
        matched_ids = set()
        for key, ids in self.entity_name_to_ids.items():
            if name_lower == key:
                matched_ids.update(ids)

        #  label/description
        if not matched_ids:
            for node_id, node in self.nodes.items():
                if node_types and node.node_type not in node_types:
                    continue
                if name_lower in node.label.lower() or name_lower in node.description.lower():
                    matched_ids.add(node_id)

        canonical = self._entity_canonical.get(name)
        if canonical:
            canonical_lower = canonical.lower()
            for key, ids in self.entity_name_to_ids.items():
                if canonical_lower == key:
                    matched_ids.update(ids)

        results = [self.nodes[nid] for nid in matched_ids if nid in self.nodes]

        #  node_type : Entity > Event > OCRText > MacroEvent > SuperEvent
        type_order = {"Entity": 0, "Event": 1, "OCRText": 2, "MacroEvent": 3, "SuperEvent": 4, "RootEvent": 5}
        results.sort(key=lambda n: type_order.get(n.node_type, 9))

        return results

    def search_by_time(
        self,
        start: float,
        end: float,
        node_types: Optional[List[str]] = None,
    ) -> List[MemoryNode]:
        """Search nodes by time range (returns all nodes overlapping with [start, end]).

        Note: time_index is sorted by start; end is not guaranteed monotonically increasing,
        so we cannot early-terminate with end < start.
        For searches specifying node_types (e.g., MacroEvent), iterate target type nodes directly.
        """
        if node_types:
            # For typed searches, iterate over target type nodes directly
            type_sets = [set(self.nodes_by_type.get(t, [])) for t in node_types]
            target_ids = set()
            for s in type_sets:
                target_ids |= s
            results = []
            for node_id in target_ids:
                node = self.nodes.get(node_id)
                if node and node.time_range and len(node.time_range) >= 2:
                    if node.time_range[0] <= end and node.time_range[1] >= start:
                        results.append(node)
        else:
            # Use time_index with binary search
            results = []
            idx = bisect_left(self.time_index, (start,))
            for i in range(idx - 1, -1, -1):
                _, node_end, _ = self.time_index[i]
                if node_end < start:
                    break
                results.append(self.nodes.get(self.time_index[i][2]))
            for i in range(idx, len(self.time_index)):
                node_start, node_end, node_id = self.time_index[i]
                if node_start > end:
                    break
                if node_end >= start:
                    results.append(self.nodes.get(node_id))
            results = [n for n in results if n is not None]

        results.sort(key=lambda n: n.time_range[0] if n.time_range else 0)
        return results

    def search_by_embedding(
        self,
        query: str,
        top_k: int = 10,
        node_types: Optional[List[str]] = None,
        exclude_ocr: bool = True,
    ) -> List[Tuple["MemoryNode", float]]:
        """Search nodes by embedding semantic similarity (only Entity/Event/OCRText have embeddings).

        Uses pre-built embedding matrix for fast batch cosine similarity.
        Falls back to per-node computation if matrix is not available.

        Args:
            exclude_ocr: Exclude OCRText nodes by default to avoid short-text OCR noise.
                         Set to False or pass node_types=["OCRText"] to include OCR.
        """
        import numpy as np

        if self._embedder is None:
            logger.warning("No embedder set, call set_embedder() first")
            return []

        query_emb = self._embedder.embed(query)
        if query_emb is None:
            return []

        query_vec = np.array(query_emb, dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)

        # Fast path: use pre-built matrix
        if self._emb_matrix is not None:
            scores = self._emb_matrix @ query_vec
            indices = np.argsort(scores)[::-1]

            results = []
            for idx in indices:
                nid = self._emb_node_ids[idx]
                node = self.nodes[nid]
                # Filter by node_types
                if node_types and node.node_type not in node_types:
                    continue
                # Filter by exclude_ocr
                if exclude_ocr and node.node_type == "OCRText" and not node_types:
                    continue
                results.append((node, float(scores[idx])))
                if len(results) >= top_k:
                    break
            return results

        # Slow fallback: per-node computation
        candidates = []
        if node_types:
            for nt in node_types:
                for nid in self.nodes_by_type.get(nt, []):
                    node = self.nodes[nid]
                    if node.embedding is not None:
                        candidates.append(node)
        else:
            for node in self.nodes.values():
                if node.embedding is not None:
                    if exclude_ocr and node.node_type == "OCRText":
                        continue
                    candidates.append(node)

        if not candidates:
            return []

        scored = []
        for node in candidates:
            node_emb = np.array(node.embedding)
            node_norm = node_emb / (np.linalg.norm(node_emb) + 1e-10)
            score = float(np.dot(query_vec, node_norm))
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_node_ego_graph(self, node_id: str) -> Dict[str, Any]:
        """Get ego-graph (depth=1) for a node, including hierarchy info and filtered neighbors.

        Returns:
            {
                "node_id": str,
                "parent_macro": {"macro_id": str, "label": str} or None,
                "neighbors": [
                    {"node_id": str, "node_type": str, "label": str,
                     "direction": "outgoing"|"incoming",
                     "relation_type": str, "relation_label": str}
                ]
            }
        """
        node = self.nodes.get(node_id)
        if not node:
            return {"node_id": node_id, "parent_macro": None, "neighbors": []}

        # Find parent macro
        parent_macro = None
        for pid in node.parent_ids:
            pnode = self.nodes.get(pid)
            if pnode and pnode.node_type == "MacroEvent":
                parent_macro = {"macro_id": pid, "label": pnode.label}
                break

        # Collect depth=1 neighbors
        SHOW_TYPES = {"CAUSAL", "SPATIAL", "ATTRIBUTIVE", "HIERARCHICAL"}
        SHOW_LABELS = {"PERFORMS"}

        # TEMPORAL
        TEMPORAL_MAX_PER_DIRECTION = 2

        neighbors = []
        temporal_out_count = 0
        temporal_in_count = 0

        # Outgoing edges
        for edge in self.edges_by_source.get(node_id, []):
            rtype = edge.relation_type
            rlabel = edge.relation_label

            # Skip HIERARCHICAL:SUBEVENT_OF (redundant with hierarchy)
            if rtype == "HIERARCHICAL":
                continue

            if rtype not in SHOW_TYPES and rlabel not in SHOW_LABELS:
                continue

            # TEMPORAL
            if rtype == "TEMPORAL":
                temporal_out_count += 1
                if temporal_out_count > TEMPORAL_MAX_PER_DIRECTION:
                    continue

            tgt_node = self.nodes.get(edge.target_id)
            if tgt_node:
                neighbors.append({
                    "node_id": edge.target_id,
                    "node_type": tgt_node.node_type,
                    "label": tgt_node.label,
                    "description": tgt_node.description,
                    "direction": "outgoing",
                    "relation_type": rtype,
                    "relation_label": rlabel,
                })

        # Incoming edges
        for edge in self.edges_by_target.get(node_id, []):
            rtype = edge.relation_type
            rlabel = edge.relation_label

            if rtype == "HIERARCHICAL":
                continue

            if rtype not in SHOW_TYPES and rlabel not in SHOW_LABELS:
                continue

            if rtype == "TEMPORAL":
                temporal_in_count += 1
                if temporal_in_count > TEMPORAL_MAX_PER_DIRECTION:
                    continue

            src_node = self.nodes.get(edge.source_id)
            if src_node:
                neighbors.append({
                    "node_id": edge.source_id,
                    "node_type": src_node.node_type,
                    "label": src_node.label,
                    "description": src_node.description,
                    "direction": "incoming",
                    "relation_type": rtype,
                    "relation_label": rlabel,
                })

        return {
            "node_id": node_id,
            "parent_macro": parent_macro,
            "neighbors": neighbors,
        }

    def get_entity_occurrences(self, entity_name: str) -> List[Dict[str, Any]]:
        """Query which macro events / super events an entity appears in."""
        name_lower = entity_name.lower().strip()
        results = []

        canonical = self._entity_canonical.get(entity_name, entity_name)

        #  entity_name_to_macro_ids
        macro_ids = set()
        for key, mids in self.entity_name_to_macro_ids.items():
            if name_lower == key.lower() or canonical.lower() == key.lower():
                macro_ids.update(mids)

        for variant, canon in self._entity_canonical.items():
            if canon == canonical:
                for key, mids in self.entity_name_to_macro_ids.items():
                    if variant.lower() == key.lower():
                        macro_ids.update(mids)

        for macro_id in sorted(macro_ids):
            macro = self.nodes.get(macro_id)
            if not macro:
                continue
            #  parent super
            super_ids = macro.parent_ids
            super_label = ""
            if super_ids and super_ids[0] in self.nodes:
                super_label = self.nodes[super_ids[0]].label

            results.append({
                "entity_name": entity_name,
                "macro_id": macro_id,
                "macro_label": macro.label,
                "time_range": list(macro.time_range),
                "super_id": super_ids[0] if super_ids else "",
                "super_label": super_label,
            })

        # Also find Entity nodes in subgraphs for entity_detail
        name_lower_check = entity_name.lower()
        result_macro_ids = {r["macro_id"] for r in results}
        entity_details_by_macro: Dict[str, List[Dict]] = {}
        for node_id, node in self.nodes.items():
            if node.node_type == "Entity" and node.label and node.label.lower() == name_lower_check:
                for parent_id in node.parent_ids:
                    parent = self.nodes.get(parent_id)
                    if parent and parent.node_type == "MacroEvent":
                        detail = {
                            "node_id": node_id,
                            "description": node.description,
                            "attributes": node.attributes,
                        }
                        if parent.node_id not in result_macro_ids:
                            result_macro_ids.add(parent.node_id)
                            super_ids = parent.parent_ids
                            super_label = ""
                            if super_ids and super_ids[0] in self.nodes:
                                super_label = self.nodes[super_ids[0]].label
                            results.append({
                                "entity_name": entity_name,
                                "macro_id": parent.node_id,
                                "macro_label": parent.label,
                                "time_range": list(parent.time_range),
                                "super_id": super_ids[0] if super_ids else "",
                                "super_label": super_label,
                                "entity_details": [detail],
                            })
                        else:
                            #  entity_detail
                            for r in results:
                                if r["macro_id"] == parent.node_id:
                                    if "entity_details" not in r:
                                        r["entity_details"] = []
                                    r["entity_details"].append(detail)
                                    break

        return results

    def get_keyframes(
        self,
        slot_id: int,
        entity_hints: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get entity keyframe paths for a given episode."""
        if not self._phase2_dir:
            return []

        features_dir = self._phase2_dir / f"episode_{slot_id:04d}" / f"episode_{slot_id:04d}" / "features"
        if not features_dir.exists():
            return []

        keyframes = []
        entity_dirs = sorted([d for d in features_dir.iterdir() if d.is_dir()])

        for entity_dir in entity_dirs:
            best_frame = entity_dir / "best_frame.jpg"
            if not best_frame.exists():
                continue

            entity_id = entity_dir.name
            global_id = f"slot{slot_id}_{entity_id}"

            #  nodes  entity
            node = self.nodes.get(global_id)
            if entity_hints and node:
                hint_match = False
                label_lower = node.label.lower()
                desc_lower = node.description.lower()
                for hint in entity_hints:
                    if hint.lower() in label_lower or hint.lower() in desc_lower:
                        hint_match = True
                        break
                if not hint_match:
                    continue

            result = {
                "entity_id": global_id,
                "keyframe_path": str(best_frame),
                "slot_id": slot_id,
            }

            if node:
                result["entity_label"] = node.label
                result["entity_type"] = node.entity_type
                result["description"] = node.description
                result["attributes"] = node.attributes

            keyframes.append(result)

        return keyframes

    def get_keyframe_path(self, entity_id: str) -> Optional[str]:
        """Get keyframe image path for a single entity."""
        node = self.nodes.get(entity_id)
        if not node or node.node_type != "Entity" or node.slot_id < 0:
            return None

        if not self._phase2_dir:
            return None

        # entity_id : "slot{slot_id}_ent_{xxx}"
        #  "ent_{xxx}"
        parts = entity_id.split("_", 2)  # ["slot{N}", "ent", "{xxx}"]
        if len(parts) < 3:
            return None
        local_id = f"{parts[1]}_{parts[2]}"

        features_dir = self._phase2_dir / f"episode_{node.slot_id:04d}" / f"episode_{node.slot_id:04d}" / "features"
        best_frame = features_dir / local_id / "best_frame.jpg"
        if best_frame.exists():
            return str(best_frame)
        return None

    def get_context_summary(self, include_descriptions: bool = True) -> str:
        """Build initial context summary (first-round input for the Reasoner)."""
        lines = []

        root_nodes = [n for n in self.nodes.values() if n.node_type == "RootEvent"]
        if root_nodes:
            root = root_nodes[0]
            lines.append("## Video Summary")
            if root.title:
                lines.append(f"Title: {root.title}")
            if root.description:
                lines.append(root.description)
            if root.themes:
                lines.append(f"Themes: {', '.join(root.themes)}")
            lines.append("")

        # Super Events via ToolRegistry
        from .tools import ToolRegistry
        registry = ToolRegistry(self)
        tool_result = registry._get_super_events()
        lines.append("## Super Events (High-Level Time Segments)")
        lines.append(tool_result.result)

        return "\n".join(lines)

    # =========================================================================
    # Embedding
    # =========================================================================

    def set_embedder(self, embedder):
        """Set the embedding model."""
        self._embedder = embedder

    # Only Entity/Event/OCRText have embeddings (hierarchy nodes do not)
    EMBEDDABLE_TYPES = {"Entity", "Event", "OCRText"}

    def compute_embeddings(self, embedder=None):
        """Compute embeddings for Entity/Event/OCRText nodes."""
        if embedder:
            self._embedder = embedder
        if self._embedder is None:
            logger.warning("No embedder set, skipping embedding computation")
            return

        # Collect texts for batch encoding
        texts = []
        node_ids = []
        for node_id, node in self.nodes.items():
            if node.node_type not in self.EMBEDDABLE_TYPES:
                continue
            text_parts = []
            if node.label:
                text_parts.append(node.label)
            if node.description:
                text_parts.append(node.description)
            if node.node_type == "Entity":
                if node.entity_type:
                    text_parts.append(f"Type: {node.entity_type}")
                distinctive = node.visual_grounding.get("distinctive_features", [])
                if distinctive:
                    text_parts.append(f"Visual: {', '.join(distinctive)}")
            elif node.node_type == "Event":
                if node.event_type:
                    text_parts.append(f"Type: {node.event_type}")
            elif node.node_type == "OCRText":
                # For OCR, include raw text as keywords
                if node.text:
                    text_parts.append(f"Keywords: {node.text}")
            if not text_parts:
                continue
            texts.append(" | ".join(text_parts))
            node_ids.append(node_id)

        logger.info(f"Computing embeddings for {len(texts)} nodes...")

        #  embedding
        embeddings = self._embedder.embed_batch(texts)
        for node_id, emb in zip(node_ids, embeddings):
            if emb is not None:
                self.nodes[node_id].embedding = emb

        embedded_count = sum(1 for n in self.nodes.values() if n.embedding is not None)
        logger.info(f"Computed embeddings: {embedded_count}/{len(self.nodes)} nodes")
        self._rebuild_embedding_matrix()

    # =========================================================================
    # Stats
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """Return statistics."""
        return {
            "total_nodes": len(self.nodes),
            "nodes_by_type": {nt: len(ids) for nt, ids in self.nodes_by_type.items()},
            "total_edges": len(self.edges),
            "edges_by_type": self._count_edges_by_type(),
            "entity_names": len(self.entity_name_to_ids),
            "entity_macro_links": len(self.entity_name_to_macro_ids),
            "entity_unification_groups": len(self._entity_groups),
            "subgraphs_loaded": len(self._subgraphs),
            "time_index_size": len(self.time_index),
            "embedded_nodes": sum(1 for n in self.nodes.values() if n.embedding is not None),
        }

    def _count_edges_by_type(self) -> Dict[str, int]:
        counts = {}
        for edge in self.edges:
            counts[edge.relation_type] = counts.get(edge.relation_type, 0) + 1
        return counts

    # =========================================================================
    # Save / Load Embeddings
    # =========================================================================

    def save_embeddings(self, path: str):
        """Save all node embeddings to a JSON file.

        Format: {"node_id": [float, ...], ...}
        Only nodes with non-None embeddings are saved.
        """
        data = {}
        for node_id, node in self.nodes.items():
            if node.embedding is not None:
                data[node_id] = node.embedding
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"Saved {len(data)} embeddings to {path}")

    def load_embeddings(self, path: str) -> int:
        """Load embeddings from a JSON file into nodes.

        Returns the number of embeddings loaded.
        """
        if not Path(path).exists():
            logger.warning(f"Embeddings file not found: {path}")
            return 0
        with open(path) as f:
            data = json.load(f)
        loaded = 0
        for node_id, emb in data.items():
            if node_id in self.nodes:
                self.nodes[node_id].embedding = emb
                loaded += 1
        # Rebuild embedding matrix for search
        self._rebuild_embedding_matrix()
        logger.info(f"Loaded {loaded} embeddings from {path}")
        return loaded

    def _rebuild_embedding_matrix(self):
        """Rebuild numpy embedding matrix after loading embeddings."""
        import numpy as np
        emb_nodes = [(nid, self.nodes[nid]) for nid in self._embeddable_ids()
                     if self.nodes[nid].embedding is not None]
        if not emb_nodes:
            self._emb_matrix = None
            self._emb_node_ids = []
            return
        self._emb_node_ids = [nid for nid, _ in emb_nodes]
        self._emb_matrix = np.array(
            [self.nodes[nid].embedding for nid in self._emb_node_ids],
            dtype=np.float32,
        )
        norms = np.linalg.norm(self._emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._emb_matrix = self._emb_matrix / norms
        logger.info(f"Rebuilt embedding matrix: {self._emb_matrix.shape}")

    def _embeddable_ids(self) -> List[str]:
        """Return node IDs of embeddable types (Entity/Event/OCRText)."""
        ids = []
        for nt in self.EMBEDDABLE_TYPES:
            ids.extend(self.nodes_by_type.get(nt, []))
        return ids