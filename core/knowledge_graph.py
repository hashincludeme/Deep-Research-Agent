"""
ariadne/core/knowledge_graph.py

In-memory directed knowledge graph for Thena's research memory.

Node types:
  finding    — a discrete claim extracted from a source
  source     — a paper, URL, or document
  topic      — a research concept or theme
  entity     — a named entity (drug, gene, institution, person)

Edge types:
  supports    — A provides evidence that strengthens claim B
  contradicts — A and B make incompatible factual claims
  extends     — A refines, qualifies, or builds directly on B
  cites       — source/finding A references source B
  about       — finding/entity A relates to topic B
  tagged_with — finding A is tagged with entity node B

The graph is session-scoped: each research session creates its own
ThenaKnowledgeGraph instance, keyed by session_id in memory.py.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import networkx as nx

NodeType = Literal["finding", "source", "topic", "entity"]
EdgeType = Literal["supports", "contradicts", "extends", "cites", "about", "tagged_with"]


@dataclass
class KGNode:
    node_id: str
    node_type: NodeType
    label: str
    payload: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def to_dict(self, include_embedding: bool = True) -> dict:
        d = {
            "node_id":   self.node_id,
            "node_type": self.node_type,
            "label":     self.label,
            "payload":   self.payload,
        }
        if include_embedding:
            d["embedding"] = self.embedding
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "KGNode":
        return cls(
            node_id=d["node_id"],
            node_type=d["node_type"],
            label=d["label"],
            payload=d.get("payload", {}),
            embedding=d.get("embedding"),
        )


class ThenaKnowledgeGraph:
    """
    Directed, typed knowledge graph for one research session.

    The graph answers questions a flat list cannot:
      - "Which findings contradict each other?" → query contradicts edges
      - "What evidence supports this claim?" → traverse supports edges
      - "What findings are about SGLT2 inhibitors?" → traverse about edges from topic node
      - "Which findings are near-duplicates?" → cosine similarity on stored embeddings
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    # ── Node operations ───────────────────────────────────────────────────────────

    def add_node(self, node: KGNode) -> str:
        """Idempotent — no-op if node_id already present. Returns node_id."""
        if node.node_id not in self._g:
            self._g.add_node(node.node_id, **node.to_dict())
        return node.node_id

    def get_node(self, node_id: str) -> Optional[KGNode]:
        if node_id not in self._g:
            return None
        attrs = dict(self._g.nodes[node_id])
        return KGNode(
            node_id=attrs.get("node_id", node_id),
            node_type=attrs["node_type"],
            label=attrs.get("label", ""),
            payload=attrs.get("payload", {}),
            embedding=attrs.get("embedding"),
        )

    def all_nodes_of_type(self, node_type: NodeType) -> list[KGNode]:
        return [
            self.get_node(nid)
            for nid, attrs in self._g.nodes(data=True)
            if attrs.get("node_type") == node_type
        ]

    def node_count(self, node_type: Optional[NodeType] = None) -> int:
        if node_type is None:
            return self._g.number_of_nodes()
        return sum(
            1 for _, attrs in self._g.nodes(data=True)
            if attrs.get("node_type") == node_type
        )

    def edge_count(self) -> int:
        return self._g.number_of_edges()

    # ── Edge operations ───────────────────────────────────────────────────────────

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: EdgeType,
        reason: str = "",
        **attrs: Any,
    ) -> None:
        missing = [n for n in (src_id, dst_id) if n not in self._g]
        if missing:
            raise ValueError(
                f"Both nodes must exist before adding edge. Missing: {missing}"
            )
        self._g.add_edge(src_id, dst_id, edge_type=edge_type, reason=reason, **attrs)

    def edges_of_type(self, edge_type: EdgeType) -> list[tuple[KGNode, KGNode, dict]]:
        result = []
        for src, dst, attrs in self._g.edges(data=True):
            if attrs.get("edge_type") == edge_type:
                s, d = self.get_node(src), self.get_node(dst)
                if s and d:
                    result.append((s, d, dict(attrs)))
        return result

    def neighbors_by_edge_type(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        node_type: Optional[NodeType] = None,
        direction: Literal["out", "in", "both"] = "out",
    ) -> list[KGNode]:
        """
        Get neighbors reachable via edges of the given type.
        direction: "out" follows outgoing edges, "in" incoming, "both" either.
        """
        candidates: set[str] = set()
        if direction in ("out", "both"):
            for _, dst, attrs in self._g.out_edges(node_id, data=True):
                if edge_type is None or attrs.get("edge_type") == edge_type:
                    candidates.add(dst)
        if direction in ("in", "both"):
            for src, _, attrs in self._g.in_edges(node_id, data=True):
                if edge_type is None or attrs.get("edge_type") == edge_type:
                    candidates.add(src)
        nodes = [self.get_node(nid) for nid in candidates]
        nodes = [n for n in nodes if n is not None]
        if node_type:
            nodes = [n for n in nodes if n.node_type == node_type]
        return nodes

    # ── Topic / entity helpers ────────────────────────────────────────────────────

    def get_or_create_topic(self, label: str) -> str:
        """Case-insensitive lookup. Creates a topic node if absent. Returns node_id."""
        norm = label.lower().strip()
        for nid, attrs in self._g.nodes(data=True):
            if attrs.get("node_type") == "topic" and attrs.get("label", "").lower() == norm:
                return nid
        nid = str(uuid.uuid4())
        self.add_node(KGNode(node_id=nid, node_type="topic", label=label.strip()))
        return nid

    def get_or_create_entity(self, label: str, entity_type: str = "") -> str:
        norm = label.lower().strip()
        for nid, attrs in self._g.nodes(data=True):
            if (
                attrs.get("node_type") == "entity"
                and attrs.get("label", "").lower() == norm
            ):
                return nid
        nid = str(uuid.uuid4())
        self.add_node(KGNode(
            node_id=nid,
            node_type="entity",
            label=label.strip(),
            payload={"entity_type": entity_type},
        ))
        return nid

    # ── Contradiction detection ───────────────────────────────────────────────────

    def find_contradictions(
        self,
        topic_node_id: Optional[str] = None,
    ) -> list[tuple[KGNode, KGNode, dict]]:
        """
        Return all (A, B, attrs) triplets where A --[contradicts]--> B.
        If topic_node_id is given, filter to finding nodes connected to that topic.
        """
        all_pairs = self.edges_of_type("contradicts")
        if topic_node_id is None:
            return all_pairs

        # Filter: both findings must be connected to the given topic via 'about'
        topic_findings = {
            n.node_id
            for n in self.neighbors_by_edge_type(
                topic_node_id, edge_type="about", direction="in"
            )
        }
        return [
            (a, b, attrs)
            for a, b, attrs in all_pairs
            if a.node_id in topic_findings or b.node_id in topic_findings
        ]

    # ── Node merge (for semantic deduplication) ───────────────────────────────────

    def merge_nodes(self, keep_id: str, remove_id: str) -> None:
        """
        Absorb remove_id into keep_id.

        All edges pointing to/from remove_id are rewired to keep_id.
        Edges that would create a self-loop (src == dst after rewire) are dropped.
        The keep_id node absorbs remove_id's payload fields (keep_id wins on conflict).
        remove_id is deleted from the graph.
        """
        if keep_id == remove_id:
            return
        missing = [n for n in (keep_id, remove_id) if n not in self._g]
        if missing:
            raise ValueError(f"Cannot merge — nodes not in graph: {missing}")

        for src, _, attrs in list(self._g.in_edges(remove_id, data=True)):
            if src != keep_id:
                self._g.add_edge(src, keep_id, **attrs)

        for _, dst, attrs in list(self._g.out_edges(remove_id, data=True)):
            if dst != keep_id:
                self._g.add_edge(keep_id, dst, **attrs)

        remove_payload = self._g.nodes[remove_id].get("payload", {})
        keep_payload   = self._g.nodes[keep_id].get("payload", {})
        self._g.nodes[keep_id]["payload"] = {**remove_payload, **keep_payload}

        self._g.remove_node(remove_id)

    # ── Topic-based clustering ────────────────────────────────────────────────────

    def cluster_by_topics(self) -> dict[str, list[KGNode]]:
        """
        Groups finding nodes by their connected topic nodes (via outgoing 'about' edges).
        A finding with multiple topics appears in all relevant clusters.
        Findings with no topic connection go into '_uncategorized'.
        """
        clusters: dict[str, list[KGNode]] = {}
        for finding in self.all_nodes_of_type("finding"):
            topics = self.neighbors_by_edge_type(
                finding.node_id, edge_type="about", direction="out"
            )
            if not topics:
                clusters.setdefault("_uncategorized", []).append(finding)
            else:
                for t in topics:
                    clusters.setdefault(t.label, []).append(finding)
        return clusters

    # ── Serialization ─────────────────────────────────────────────────────────────

    def to_dict(self, include_embeddings: bool = True) -> dict:
        nodes = []
        for nid, attrs in self._g.nodes(data=True):
            n: dict[str, Any] = {
                "node_id":   attrs.get("node_id", nid),
                "node_type": attrs.get("node_type"),
                "label":     attrs.get("label", ""),
                "payload":   attrs.get("payload", {}),
            }
            if include_embeddings:
                n["embedding"] = attrs.get("embedding")
            nodes.append(n)

        edges = [
            {
                "src":       src,
                "dst":       dst,
                "edge_type": attrs.get("edge_type"),
                "reason":    attrs.get("reason", ""),
            }
            for src, dst, attrs in self._g.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict) -> "ThenaKnowledgeGraph":
        kg = cls()
        for n in data.get("nodes", []):
            kg.add_node(KGNode(
                node_id=n["node_id"],
                node_type=n["node_type"],
                label=n.get("label", ""),
                payload=n.get("payload", {}),
                embedding=n.get("embedding"),
            ))
        for e in data.get("edges", []):
            try:
                kg.add_edge(
                    e["src"], e["dst"], e["edge_type"],
                    reason=e.get("reason", ""),
                )
            except ValueError:
                pass  # skip edges whose nodes weren't in the serialized data
        return kg

    def to_networkx(self) -> nx.DiGraph:
        return self._g.copy()
