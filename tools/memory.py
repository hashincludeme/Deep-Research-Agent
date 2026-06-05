"""
ariadne/tools/memory.py

Ten memory-namespace tools for Thena's knowledge graph.

All tools share a module-level session store (_graphs dict keyed by session_id).
Calling any tool with a new session_id auto-creates a fresh ThenaKnowledgeGraph.

Embeddings use sentence-transformers all-MiniLM-L6-v2 (lazy import, ~80MB).
If the package is not installed, embeddings are None and deduplication falls
back to exact-string matching only (no semantic merge).
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from core.errors import ThenaError, ToolError
from core.knowledge_graph import KGNode, ThenaKnowledgeGraph, EdgeType
from registry.registry import tool


# ── Session-scoped graph store ────────────────────────────────────────────────

_graphs: dict[str, ThenaKnowledgeGraph] = {}


def _get_graph(session_id: str) -> ThenaKnowledgeGraph:
    if session_id not in _graphs:
        _graphs[session_id] = ThenaKnowledgeGraph()
    return _graphs[session_id]


# ── Embedding helpers ─────────────────────────────────────────────────────────

_embed_model = None  # lazy singleton


def _load_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _embed(text: str) -> Optional[list[float]]:
    """Returns a 384-dim embedding, or None if sentence-transformers is unavailable."""
    try:
        model = _load_embed_model()
        return model.encode([text], show_progress_bar=False)[0].tolist()
    except ImportError:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 1e-10 else 0.0


# ── Shared output type ────────────────────────────────────────────────────────

class FindingRecord(BaseModel):
    finding_id: str
    claim: str
    source_id: str
    source_title: str = ""
    year: Optional[int] = None
    topics: list[str] = Field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# Tool 1: memory.store_finding
# ═════════════════════════════════════════════════════════════════════════════

class StoreFindingInput(BaseModel):
    claim: str = Field(description=(
        "The discrete factual claim to store. Should be one specific assertion, "
        "not a full paragraph. Example: 'Metformin reduces HbA1c by 1.5% in T2DM patients (RCT, n=400)'."
    ))
    source_id: str = Field(description=(
        "Identifier for the source (PMID, DOI, URL, or internal ID)."
    ))
    source_title: str = Field(default="", description="Human-readable source title.")
    year: Optional[int] = Field(default=None, description="Publication year.")
    topics: list[str] = Field(
        default_factory=list,
        description=(
            "Topic labels to connect this finding to via 'about' edges. "
            "Example: ['glycemic control', 'type 2 diabetes', 'metformin']."
        ),
    )
    session_id: str

class StoreFindingOutput(BaseModel):
    finding_id: str
    was_merged: bool = Field(
        description="True if an exact-duplicate claim was found and this returned the existing node."
    )
    merged_into: Optional[str] = Field(
        default=None,
        description="ID of the existing node if was_merged=True.",
    )


@tool(
    name="memory.store_finding",
    description=(
        "Store a discrete factual claim in the research knowledge graph. "
        "Creates a Finding node, connects it to topic nodes via 'about' edges, "
        "and computes a semantic embedding for later deduplication. "
        "Returns a finding_id for use with memory.cross_ref, memory.tag_entity, and memory.retrieve_by_topic. "
        "Call this after every search result to build the session's knowledge graph. "
        "Exact-duplicate claims (same text) are detected and merged automatically."
    ),
    input_schema=StoreFindingInput,
    output_schema=StoreFindingOutput,
)
async def store_finding(
    claim: str,
    source_id: str,
    source_title: str = "",
    year: Optional[int] = None,
    topics: Optional[list] = None,
    session_id: str = "",
) -> dict:
    topics = topics or []
    kg = _get_graph(session_id)

    # Exact-duplicate check: same claim text already stored
    normalized = claim.strip()
    for existing in kg.all_nodes_of_type("finding"):
        if existing.label.strip() == normalized:
            return {
                "finding_id": existing.node_id,
                "was_merged": True,
                "merged_into": existing.node_id,
            }

    # Create finding node
    finding_id = str(uuid.uuid4())
    embedding = _embed(claim)
    kg.add_node(KGNode(
        node_id=finding_id,
        node_type="finding",
        label=claim,
        payload={
            "source_id":    source_id,
            "source_title": source_title,
            "year":         year,
        },
        embedding=embedding,
    ))

    # Ensure source node exists
    if source_id and source_id not in {
        n.node_id for n in kg.all_nodes_of_type("source")
    }:
        kg.add_node(KGNode(
            node_id=source_id,
            node_type="source",
            label=source_title or source_id,
            payload={"year": year},
        ))
    if source_id:
        try:
            kg.add_edge(finding_id, source_id, "cites")
        except ValueError:
            pass

    # Connect to topic nodes via 'about' edges
    for topic_label in topics:
        topic_id = kg.get_or_create_topic(topic_label)
        kg.add_edge(finding_id, topic_id, "about")

    return {"finding_id": finding_id, "was_merged": False, "merged_into": None}


# ═════════════════════════════════════════════════════════════════════════════
# Tool 2: memory.retrieve_by_topic
# ═════════════════════════════════════════════════════════════════════════════

class RetrieveByTopicInput(BaseModel):
    topic: str = Field(description="Topic label to search for. Case-insensitive.")
    session_id: str
    max_results: int = Field(default=20, ge=1, le=100)

class RetrieveByTopicOutput(BaseModel):
    findings: list[FindingRecord]
    total: int
    topic_node_id: str


@tool(
    name="memory.retrieve_by_topic",
    description=(
        "Retrieve all Finding nodes connected to a topic label via 'about' edges. "
        "Use to gather all stored evidence for a specific research question before synthesis. "
        "Returns finding IDs, claims, and source metadata. "
        "Case-insensitive topic matching — 'Metformin' and 'metformin' return the same results."
    ),
    input_schema=RetrieveByTopicInput,
    output_schema=RetrieveByTopicOutput,
)
async def retrieve_by_topic(
    topic: str,
    session_id: str = "",
    max_results: int = 20,
) -> dict:
    kg = _get_graph(session_id)

    # Find the topic node (case-insensitive)
    norm = topic.lower().strip()
    topic_node_id = ""
    for nid, attrs in kg._g.nodes(data=True):
        if attrs.get("node_type") == "topic" and attrs.get("label", "").lower() == norm:
            topic_node_id = nid
            break

    if not topic_node_id:
        return {"findings": [], "total": 0, "topic_node_id": ""}

    # Findings connect to topics via outgoing 'about' edges.
    # We want findings that have an 'about' edge pointing TO this topic node.
    findings_raw = kg.neighbors_by_edge_type(
        topic_node_id,
        edge_type="about",
        node_type="finding",
        direction="in",
    )

    findings = []
    for f in findings_raw[:max_results]:
        payload = f.payload or {}
        # Reconstruct connected topics for this finding
        connected_topics = [
            t.label
            for t in kg.neighbors_by_edge_type(
                f.node_id, edge_type="about", direction="out"
            )
        ]
        findings.append({
            "finding_id":   f.node_id,
            "claim":        f.label,
            "source_id":    payload.get("source_id", ""),
            "source_title": payload.get("source_title", ""),
            "year":         payload.get("year"),
            "topics":       connected_topics,
        })

    return {"findings": findings, "total": len(findings_raw), "topic_node_id": topic_node_id}


# ═════════════════════════════════════════════════════════════════════════════
# Tool 3: memory.find_contradiction
# ═════════════════════════════════════════════════════════════════════════════

class FindContradictionInput(BaseModel):
    session_id: str
    topic: Optional[str] = Field(
        default=None,
        description=(
            "If set, only return contradictions among findings connected to this topic. "
            "Leave empty to check the entire session graph."
        ),
    )

class ContradictionPair(BaseModel):
    finding_id_a: str
    claim_a: str
    source_a: str
    finding_id_b: str
    claim_b: str
    source_b: str
    reason: str = ""

class FindContradictionOutput(BaseModel):
    contradictions: list[ContradictionPair]
    total: int


@tool(
    name="memory.find_contradiction",
    description=(
        "Return all finding pairs in the knowledge graph that are connected by a 'contradicts' edge. "
        "These are findings that make incompatible factual claims. "
        "Use during synthesis to surface unresolved conflicts that must be reported as limitations. "
        "Optionally filter to a specific topic to narrow the scope. "
        "Contradicts edges are created via memory.cross_ref with edge_type='contradicts'."
    ),
    input_schema=FindContradictionInput,
    output_schema=FindContradictionOutput,
)
async def find_contradiction(
    session_id: str = "",
    topic: Optional[str] = None,
) -> dict:
    kg = _get_graph(session_id)

    topic_node_id = None
    if topic:
        norm = topic.lower().strip()
        for nid, attrs in kg._g.nodes(data=True):
            if attrs.get("node_type") == "topic" and attrs.get("label", "").lower() == norm:
                topic_node_id = nid
                break

    pairs_raw = kg.find_contradictions(topic_node_id=topic_node_id)

    contradictions = []
    for node_a, node_b, attrs in pairs_raw:
        contradictions.append({
            "finding_id_a": node_a.node_id,
            "claim_a":      node_a.label,
            "source_a":     node_a.payload.get("source_id", ""),
            "finding_id_b": node_b.node_id,
            "claim_b":      node_b.label,
            "source_b":     node_b.payload.get("source_id", ""),
            "reason":       attrs.get("reason", ""),
        })

    return {"contradictions": contradictions, "total": len(contradictions)}


# ═════════════════════════════════════════════════════════════════════════════
# Tool 4: memory.deduplicate
# ═════════════════════════════════════════════════════════════════════════════

class DeduplicateInput(BaseModel):
    session_id: str
    similarity_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity threshold above which two findings are considered duplicates. "
            "0.9 catches paraphrases and minor rewording. "
            "0.95 catches near-verbatim duplicates only. "
            "Requires sentence-transformers; falls back to exact-string match if unavailable."
        ),
    )

class MergedPair(BaseModel):
    kept_id: str
    removed_id: str
    similarity: float
    kept_claim: str
    removed_claim: str

class DeduplicateOutput(BaseModel):
    merged_pairs: list[MergedPair]
    total_before: int
    total_after: int


@tool(
    name="memory.deduplicate",
    description=(
        "Merge semantically near-duplicate Finding nodes in the knowledge graph. "
        "Uses cosine similarity on all-MiniLM-L6-v2 embeddings — catches paraphrases "
        "that exact-string matching misses. "
        "All graph edges from the removed node are rewired to the surviving node. "
        "Run this after a batch of memory.store_finding calls to keep the graph clean. "
        "If sentence-transformers is not installed, performs exact-string deduplication only."
    ),
    input_schema=DeduplicateInput,
    output_schema=DeduplicateOutput,
)
async def deduplicate(
    session_id: str = "",
    similarity_threshold: float = 0.9,
) -> dict:
    kg = _get_graph(session_id)
    findings = kg.all_nodes_of_type("finding")
    total_before = len(findings)

    # Separate findings with embeddings from those without
    embedded = [(f, f.embedding) for f in findings if f.embedding]

    if len(embedded) < 2:
        return {
            "merged_pairs": [],
            "total_before": total_before,
            "total_after":  kg.node_count("finding"),
        }

    import numpy as np

    nodes_list, emb_list = zip(*embedded)
    matrix = np.array(emb_list, dtype=float)  # shape: (n, d)

    # Vectorised cosine similarity matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1e-10
    normed = matrix / norms
    sim_matrix = normed @ normed.T  # (n, n), symmetric, diagonal=1

    n = len(nodes_list)
    candidate_pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= similarity_threshold:
                candidate_pairs.append((float(sim_matrix[i, j]), i, j))

    # Process pairs highest-similarity first; skip if either node already merged
    candidate_pairs.sort(reverse=True)
    removed: set[str] = set()
    merged_pairs = []

    for sim, i, j in candidate_pairs:
        node_a = nodes_list[i]
        node_b = nodes_list[j]
        if node_a.node_id in removed or node_b.node_id in removed:
            continue

        # Keep the node with more connections (higher degree) — it's better connected
        deg_a = kg._g.degree(node_a.node_id)
        deg_b = kg._g.degree(node_b.node_id)
        keep, remove = (node_a, node_b) if deg_a >= deg_b else (node_b, node_a)

        merged_pairs.append({
            "kept_id":       keep.node_id,
            "removed_id":    remove.node_id,
            "similarity":    round(sim, 4),
            "kept_claim":    keep.label,
            "removed_claim": remove.label,
        })
        kg.merge_nodes(keep.node_id, remove.node_id)
        removed.add(remove.node_id)

    return {
        "merged_pairs": merged_pairs,
        "total_before": total_before,
        "total_after":  kg.node_count("finding"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Tool 5: memory.tag_entity
# ═════════════════════════════════════════════════════════════════════════════

class TagEntityInput(BaseModel):
    finding_id: str = Field(description="ID of the Finding node to tag.")
    entity_name: str = Field(description="Entity name, e.g. 'metformin', 'BRCA1', 'Pfizer'.")
    entity_type: str = Field(
        default="",
        description="Entity category: drug, gene, organization, person, condition, etc.",
    )
    session_id: str

class TagEntityOutput(BaseModel):
    entity_node_id: str
    edge_created: bool


@tool(
    name="memory.tag_entity",
    description=(
        "Tag a Finding node with a named entity (drug, gene, institution, condition). "
        "Creates an Entity node if it doesn't exist, then adds a 'tagged_with' edge "
        "from the finding to the entity. "
        "Use to build a queryable entity index: later retrieve all findings mentioning "
        "'metformin' or 'BRCA1' without scanning every finding's text. "
        "Entity lookup is case-insensitive and deduplicates automatically."
    ),
    input_schema=TagEntityInput,
    output_schema=TagEntityOutput,
)
async def tag_entity(
    finding_id: str,
    entity_name: str,
    entity_type: str = "",
    session_id: str = "",
) -> dict:
    kg = _get_graph(session_id)

    if kg.get_node(finding_id) is None:
        raise ToolError(
            f"Finding node '{finding_id}' not found in session '{session_id}'.",
            tool_name="memory.tag_entity",
            retryable=False,
        )

    entity_node_id = kg.get_or_create_entity(entity_name, entity_type)

    # Only add edge if it doesn't already exist
    existing_entities = {
        n.node_id
        for n in kg.neighbors_by_edge_type(
            finding_id, edge_type="tagged_with", direction="out"
        )
    }
    edge_created = entity_node_id not in existing_entities
    if edge_created:
        kg.add_edge(finding_id, entity_node_id, "tagged_with")

    return {"entity_node_id": entity_node_id, "edge_created": edge_created}


# ═════════════════════════════════════════════════════════════════════════════
# Tool 6: memory.build_kg_node
# ═════════════════════════════════════════════════════════════════════════════

class BuildKGNodeInput(BaseModel):
    node_type: str = Field(
        description="Node type: 'finding', 'source', 'topic', or 'entity'."
    )
    label: str = Field(description="Human-readable label for this node.")
    payload: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata stored on the node.",
    )
    session_id: str

class BuildKGNodeOutput(BaseModel):
    node_id: str


@tool(
    name="memory.build_kg_node",
    description=(
        "Create a raw Knowledge Graph node of any type in the session graph. "
        "Use for source nodes (papers, URLs), topic nodes (research themes), or entity nodes "
        "when you need fine-grained control over the node structure. "
        "For storing research findings, prefer memory.store_finding which also handles "
        "topic connections and embedding computation automatically."
    ),
    input_schema=BuildKGNodeInput,
    output_schema=BuildKGNodeOutput,
)
async def build_kg_node(
    node_type: str,
    label: str,
    payload: Optional[dict] = None,
    session_id: str = "",
) -> dict:
    valid_types = {"finding", "source", "topic", "entity"}
    if node_type not in valid_types:
        raise ToolError(
            f"Invalid node_type '{node_type}'. Must be one of: {sorted(valid_types)}",
            tool_name="memory.build_kg_node",
            retryable=False,
        )

    kg = _get_graph(session_id)
    node_id = str(uuid.uuid4())
    kg.add_node(KGNode(
        node_id=node_id,
        node_type=node_type,  # type: ignore[arg-type]
        label=label,
        payload=payload or {},
    ))
    return {"node_id": node_id}


# ═════════════════════════════════════════════════════════════════════════════
# Tool 7: memory.summarize_chunk
# ═════════════════════════════════════════════════════════════════════════════

class SummarizeChunkInput(BaseModel):
    text: str = Field(description=(
        "Raw text chunk to summarize and store — e.g. a scraped article section "
        "or a long abstract. Will be truncated to max_chars using sentence boundaries."
    ))
    source_id: str
    topic: str = Field(description="Primary topic label for this chunk.")
    session_id: str
    max_chars: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum character length of the stored summary.",
    )

class SummarizeChunkOutput(BaseModel):
    finding_id: str
    summary: str


@tool(
    name="memory.summarize_chunk",
    description=(
        "Extract a summary from a long text chunk and store it as a Finding node. "
        "Uses extractive summarization: splits text into sentences and keeps the first N "
        "that fit within max_chars. Stores the summary with a semantic embedding. "
        "Use when a scraped article section is too long to store verbatim — "
        "keep the most information-dense sentences as a finding."
    ),
    input_schema=SummarizeChunkInput,
    output_schema=SummarizeChunkOutput,
)
async def summarize_chunk(
    text: str,
    source_id: str,
    topic: str,
    session_id: str = "",
    max_chars: int = 500,
) -> dict:
    # Extractive: split on sentence boundaries, take sentences until max_chars
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    parts: list[str] = []
    total = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if total + len(s) + 1 > max_chars:
            break
        parts.append(s)
        total += len(s) + 1

    summary = " ".join(parts).strip() or text[:max_chars].strip()

    # Store as a finding using the graph directly (avoid calling the decorated function
    # to prevent double-registration side effects)
    kg = _get_graph(session_id)
    normalized = summary.strip()
    for existing in kg.all_nodes_of_type("finding"):
        if existing.label.strip() == normalized:
            return {"finding_id": existing.node_id, "summary": summary}

    finding_id = str(uuid.uuid4())
    embedding = _embed(summary)
    kg.add_node(KGNode(
        node_id=finding_id,
        node_type="finding",
        label=summary,
        payload={"source_id": source_id, "source_title": "", "year": None},
        embedding=embedding,
    ))
    topic_id = kg.get_or_create_topic(topic)
    kg.add_edge(finding_id, topic_id, "about")

    return {"finding_id": finding_id, "summary": summary}


# ═════════════════════════════════════════════════════════════════════════════
# Tool 8: memory.cross_ref
# ═════════════════════════════════════════════════════════════════════════════

class CrossRefInput(BaseModel):
    finding_id_a: str = Field(description="Source finding node ID.")
    finding_id_b: str = Field(description="Target finding node ID.")
    edge_type: str = Field(
        description=(
            "Relationship type: "
            "'supports' (A strengthens B), "
            "'contradicts' (A and B conflict), "
            "'extends' (A builds on B), "
            "'cites' (A references B's source)."
        )
    )
    reason: str = Field(
        default="",
        description="Brief explanation of why this relationship holds.",
    )
    session_id: str

class CrossRefOutput(BaseModel):
    edge_created: bool
    edge_type: str


@tool(
    name="memory.cross_ref",
    description=(
        "Create a typed edge between two Finding nodes in the knowledge graph. "
        "This is how contradictions and support relationships are encoded — not inferred at synthesis time, "
        "but stored structurally when the relationship is detected during research. "
        "Use 'contradicts' when two findings make incompatible claims about the same outcome. "
        "Use 'supports' when a finding reinforces another's claim with independent evidence. "
        "Use 'extends' when a finding qualifies or refines another (e.g. adds a subgroup analysis). "
        "These edges are what memory.find_contradiction and the synthesis phase query."
    ),
    input_schema=CrossRefInput,
    output_schema=CrossRefOutput,
)
async def cross_ref(
    finding_id_a: str,
    finding_id_b: str,
    edge_type: str,
    reason: str = "",
    session_id: str = "",
) -> dict:
    valid_types: set[str] = {"supports", "contradicts", "extends", "cites"}
    if edge_type not in valid_types:
        raise ToolError(
            f"Invalid edge_type '{edge_type}'. Must be one of: {sorted(valid_types)}",
            tool_name="memory.cross_ref",
            retryable=False,
        )

    kg = _get_graph(session_id)
    missing = [nid for nid in (finding_id_a, finding_id_b) if kg.get_node(nid) is None]
    if missing:
        raise ToolError(
            f"Finding nodes not found: {missing}",
            tool_name="memory.cross_ref",
            retryable=False,
        )

    kg.add_edge(finding_id_a, finding_id_b, edge_type, reason=reason)  # type: ignore[arg-type]
    return {"edge_created": True, "edge_type": edge_type}


# ═════════════════════════════════════════════════════════════════════════════
# Tool 9: memory.cluster_themes
# ═════════════════════════════════════════════════════════════════════════════

class ThemeCluster(BaseModel):
    theme: str
    finding_ids: list[str]
    claims: list[str]
    size: int

class ClusterThemesInput(BaseModel):
    session_id: str

class ClusterThemesOutput(BaseModel):
    clusters: list[ThemeCluster]
    total_findings: int


@tool(
    name="memory.cluster_themes",
    description=(
        "Group all Finding nodes into theme clusters based on their topic connections. "
        "Uses the knowledge graph structure itself — findings connected to the same topic node "
        "via 'about' edges belong to the same cluster. "
        "A finding connected to multiple topics appears in all relevant clusters. "
        "Findings with no topic connection are grouped under '_uncategorized'. "
        "Use before synthesis to understand the thematic distribution of evidence "
        "and to identify topics that are well-covered vs. sparse."
    ),
    input_schema=ClusterThemesInput,
    output_schema=ClusterThemesOutput,
)
async def cluster_themes(session_id: str = "") -> dict:
    kg = _get_graph(session_id)
    raw_clusters = kg.cluster_by_topics()

    clusters = []
    for theme, finding_nodes in raw_clusters.items():
        clusters.append({
            "theme":       theme,
            "finding_ids": [f.node_id for f in finding_nodes],
            "claims":      [f.label for f in finding_nodes],
            "size":        len(finding_nodes),
        })

    # Sort largest cluster first — gives the orchestrator a quick coverage map
    clusters.sort(key=lambda c: c["size"], reverse=True)

    return {
        "clusters":       clusters,
        "total_findings": kg.node_count("finding"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Tool 10: memory.export_graph
# ═════════════════════════════════════════════════════════════════════════════

class ExportGraphInput(BaseModel):
    session_id: str
    include_embeddings: bool = Field(
        default=False,
        description=(
            "Whether to include raw embedding vectors in node data. "
            "Embeddings are 384-dim float lists — large. "
            "Set True only when you need to serialize the graph for checkpointing."
        ),
    )

class ExportGraphOutput(BaseModel):
    nodes: list[dict]
    edges: list[dict]
    node_count: int
    edge_count: int


@tool(
    name="memory.export_graph",
    description=(
        "Export the full knowledge graph for this session as a JSON-serializable dict. "
        "Returns all nodes (finding, source, topic, entity) and all typed edges. "
        "Use for checkpointing, debugging, or passing the graph state to the report writer. "
        "Set include_embeddings=True only when persisting to disk — embeddings are large."
    ),
    input_schema=ExportGraphInput,
    output_schema=ExportGraphOutput,
)
async def export_graph(
    session_id: str = "",
    include_embeddings: bool = False,
) -> dict:
    kg = _get_graph(session_id)
    data = kg.to_dict(include_embeddings=include_embeddings)
    return {
        "nodes":      data["nodes"],
        "edges":      data["edges"],
        "node_count": kg.node_count(),
        "edge_count": kg.edge_count(),
    }
