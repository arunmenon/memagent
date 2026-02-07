"""
Data models for the Context Retriever Graph.

The CRG represents memories as nodes in a multi-relation graph where edges
encode typed relationships (semantic, causal, temporal, hierarchical,
contradicts, supports). Retrieval is performed via intent-driven traversal
with cognitive-science-inspired activation spreading.
"""

from __future__ import annotations

import uuid
import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class EdgeType(Enum):
    """Types of relationships between context nodes."""

    SEMANTIC = "semantic"        # Topically similar content
    CAUSAL = "causal"            # A causes/leads-to B
    TEMPORAL = "temporal"        # A happened before/after B
    HIERARCHICAL = "hierarchical"  # A is parent/child of B
    CONTRADICTS = "contradicts"  # A conflicts with B
    SUPPORTS = "supports"        # A reinforces/corroborates B


# Default traversal weights per intent category.
# Keys are intent labels, values map EdgeType -> weight multiplier.
INTENT_EDGE_WEIGHTS: Dict[str, Dict[EdgeType, float]] = {
    "explore": {
        EdgeType.SEMANTIC: 1.0,
        EdgeType.CAUSAL: 0.6,
        EdgeType.TEMPORAL: 0.4,
        EdgeType.HIERARCHICAL: 0.8,
        EdgeType.CONTRADICTS: 0.3,
        EdgeType.SUPPORTS: 0.5,
    },
    "debug": {
        EdgeType.SEMANTIC: 0.5,
        EdgeType.CAUSAL: 1.0,
        EdgeType.TEMPORAL: 0.8,
        EdgeType.HIERARCHICAL: 0.4,
        EdgeType.CONTRADICTS: 0.9,
        EdgeType.SUPPORTS: 0.7,
    },
    "summarize": {
        EdgeType.SEMANTIC: 0.7,
        EdgeType.CAUSAL: 0.3,
        EdgeType.TEMPORAL: 0.2,
        EdgeType.HIERARCHICAL: 1.0,
        EdgeType.CONTRADICTS: 0.1,
        EdgeType.SUPPORTS: 0.6,
    },
    "verify": {
        EdgeType.SEMANTIC: 0.4,
        EdgeType.CAUSAL: 0.6,
        EdgeType.TEMPORAL: 0.3,
        EdgeType.HIERARCHICAL: 0.2,
        EdgeType.CONTRADICTS: 1.0,
        EdgeType.SUPPORTS: 1.0,
    },
    "plan": {
        EdgeType.SEMANTIC: 0.6,
        EdgeType.CAUSAL: 1.0,
        EdgeType.TEMPORAL: 0.9,
        EdgeType.HIERARCHICAL: 0.7,
        EdgeType.CONTRADICTS: 0.5,
        EdgeType.SUPPORTS: 0.8,
    },
    "recall": {
        EdgeType.SEMANTIC: 1.0,
        EdgeType.CAUSAL: 0.5,
        EdgeType.TEMPORAL: 0.7,
        EdgeType.HIERARCHICAL: 0.6,
        EdgeType.CONTRADICTS: 0.2,
        EdgeType.SUPPORTS: 0.4,
    },
}


@dataclass
class Edge:
    """A typed, weighted, directed edge between two context nodes."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    @property
    def key(self) -> str:
        return f"{self.source_id}|{self.target_id}|{self.edge_type.value}"


@dataclass
class ContextNode:
    """
    A node in the context graph representing a discrete unit of knowledge.

    Each node holds content, an embedding vector, structural metadata,
    and an activation level that decays over time and spreads to neighbours.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    summary: str = ""
    embedding: List[float] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    layer: int = 0  # 0 = leaf (raw fact), 1 = cluster summary, 2 = theme …
    activation: float = 0.0
    importance: float = 0.5
    access_count: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    last_accessed: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def touch(self) -> None:
        """Record an access, bumping count and timestamp."""
        self.access_count += 1
        self.last_accessed = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def strength(self) -> float:
        """Compute memory strength from importance, access frequency, and recency."""
        hours_since = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.datetime.fromisoformat(self.last_accessed)
        ).total_seconds() / 3600.0
        return self.importance * (1 + self.access_count) * (1 / (1 + hours_since))


@dataclass
class ActivationResult:
    """Result of an activation-spreading retrieval pass."""

    node_id: str
    activation: float
    depth: int
    path: List[str]  # IDs of nodes on the traversal path from seed to this node


@dataclass
class RetrievalPlan:
    """
    An intent-driven retrieval plan produced by the planner.

    Contains the classified intent, edge weights to use during traversal,
    seed node IDs, and budget constraints.
    """

    intent: str
    edge_weights: Dict[EdgeType, float]
    seed_ids: List[str]
    max_depth: int = 3
    max_nodes: int = 20
    token_budget: int = 4000
    filters: Dict[str, str] = field(default_factory=dict)
