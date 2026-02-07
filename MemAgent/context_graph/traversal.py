"""
Intent-Driven Traversal with Activation Spreading.

The traversal engine is the heart of the CRG retrieval system.  Instead of
plain nearest-neighbour search, it:

1. **Plans** — classifies the agent's current intent and selects edge-weight
   multipliers so that different relationship types are prioritised depending
   on what the agent is trying to do (debugging favours causal edges,
   summarising favours hierarchical edges, etc.).

2. **Seeds** — uses the existing ChromaDB embeddings to find the top-k
   semantically closest nodes as starting points.

3. **Spreads** — performs a bounded breadth-first activation spread from the
   seeds through the graph, where each edge's effective weight is the product
   of its base weight and the intent-specific multiplier for its type.

4. **Collects** — returns the top-N nodes ranked by final activation, along
   with the traversal paths that reached them (for explainability).

The algorithm is inspired by *spreading activation* models from cognitive
science (Collins & Loftus, 1975) and adapts the idea to a typed graph
with a planning layer on top.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    ActivationResult,
    ContextNode,
    Edge,
    EdgeType,
    INTENT_EDGE_WEIGHTS,
    RetrievalPlan,
)
from .graph import ContextGraph


class IntentDrivenTraversal:
    """
    Performs activation-spreading traversal over a ContextGraph guided
    by a RetrievalPlan.
    """

    # Activation below this threshold is not propagated.
    MIN_ACTIVATION: float = 0.01

    # Decay factor per hop — activation is multiplied by this at each step.
    DEFAULT_HOP_DECAY: float = 0.6

    def __init__(
        self,
        graph: ContextGraph,
        hop_decay: float = DEFAULT_HOP_DECAY,
    ) -> None:
        self.graph = graph
        self.hop_decay = hop_decay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def traverse(self, plan: RetrievalPlan) -> List[ActivationResult]:
        """
        Execute activation spreading according to *plan* and return nodes
        ranked by descending activation.

        Parameters
        ----------
        plan : RetrievalPlan
            Contains seed IDs, edge weights, depth/node budgets, etc.

        Returns
        -------
        list[ActivationResult]
            Nodes with their final activation levels, depth, and the path
            from the nearest seed that contributed the most activation.
        """
        # Accumulate best activation per node.
        best: Dict[str, ActivationResult] = {}

        # BFS frontier: (-activation, node_id, depth, path)
        # We use a max-heap via negative activation.
        frontier: list[Tuple[float, str, int, List[str]]] = []

        # Initialise seeds with activation = 1.0
        for seed_id in plan.seed_ids:
            seed_node = self.graph.get_node(seed_id)
            if seed_node is None:
                continue
            initial_activation = 1.0 * seed_node.importance
            entry = (-initial_activation, seed_id, 0, [seed_id])
            heapq.heappush(frontier, entry)

        visited_at_depth: Dict[str, int] = {}  # node_id -> min depth seen

        while frontier:
            neg_act, node_id, depth, path = heapq.heappop(frontier)
            activation = -neg_act

            # Skip if we've already seen this node at a shallower or equal depth
            # with higher activation.
            if node_id in best and best[node_id].activation >= activation:
                continue

            best[node_id] = ActivationResult(
                node_id=node_id,
                activation=activation,
                depth=depth,
                path=list(path),
            )

            # Stop expanding if we've hit depth or node budget.
            if depth >= plan.max_depth:
                continue
            if len(best) >= plan.max_nodes * 2:
                # Allow some over-expansion then trim at the end.
                continue

            # Spread activation to neighbours.
            for neighbour, edge in self.graph.neighbours(node_id):
                if neighbour.id in path:
                    # Avoid cycles within a single path.
                    continue

                effective_weight = self._effective_weight(edge, plan.edge_weights)
                child_activation = activation * self.hop_decay * effective_weight

                if child_activation < self.MIN_ACTIVATION:
                    continue

                # Boost by the neighbour's own importance.
                child_activation *= (0.5 + 0.5 * neighbour.importance)

                new_path = path + [neighbour.id]
                heapq.heappush(
                    frontier,
                    (-child_activation, neighbour.id, depth + 1, new_path),
                )

        # Rank by activation descending and trim to budget.
        ranked = sorted(best.values(), key=lambda r: r.activation, reverse=True)
        return ranked[: plan.max_nodes]

    def multi_seed_traverse(
        self,
        plans: List[RetrievalPlan],
    ) -> List[ActivationResult]:
        """
        Run multiple traversal plans and merge the results, keeping the
        highest activation per node across all plans.
        """
        merged: Dict[str, ActivationResult] = {}
        for plan in plans:
            results = self.traverse(plan)
            for r in results:
                if r.node_id not in merged or r.activation > merged[r.node_id].activation:
                    merged[r.node_id] = r

        ranked = sorted(merged.values(), key=lambda r: r.activation, reverse=True)
        max_nodes = max((p.max_nodes for p in plans), default=20)
        return ranked[:max_nodes]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _effective_weight(
        edge: Edge, intent_weights: Dict[EdgeType, float]
    ) -> float:
        """
        Compute the effective traversal weight for an edge given the
        intent-driven weight multipliers.

        effective = edge.weight * intent_multiplier_for_edge_type

        If the edge type has no multiplier in the plan, use 0.3 as a
        conservative default so that unknown edge types are not
        completely ignored.
        """
        multiplier = intent_weights.get(edge.edge_type, 0.3)
        return edge.weight * multiplier

    # ------------------------------------------------------------------
    # Convenience: build plans from intent strings
    # ------------------------------------------------------------------

    @staticmethod
    def plan_from_intent(
        intent: str,
        seed_ids: List[str],
        max_depth: int = 3,
        max_nodes: int = 20,
        token_budget: int = 4000,
        filters: Optional[Dict[str, str]] = None,
    ) -> RetrievalPlan:
        """
        Create a RetrievalPlan from a human-readable intent string.

        If the intent matches a known category in INTENT_EDGE_WEIGHTS,
        use those weights. Otherwise fall back to uniform weights.
        """
        intent_lower = intent.lower().strip()

        # Try exact match first.
        if intent_lower in INTENT_EDGE_WEIGHTS:
            edge_weights = INTENT_EDGE_WEIGHTS[intent_lower]
        else:
            # Try substring match.
            matched = None
            for key in INTENT_EDGE_WEIGHTS:
                if key in intent_lower:
                    matched = key
                    break
            if matched:
                edge_weights = INTENT_EDGE_WEIGHTS[matched]
            else:
                # Uniform fallback.
                edge_weights = {et: 0.5 for et in EdgeType}

        return RetrievalPlan(
            intent=intent_lower,
            edge_weights=edge_weights,
            seed_ids=seed_ids,
            max_depth=max_depth,
            max_nodes=max_nodes,
            token_budget=token_budget,
            filters=filters or {},
        )
