"""
Context Budget Manager — token-aware context window packing.

Given a ranked list of activated nodes, the budget manager decides which
nodes to include in the final context, how to compress them, and how to
order them for maximum LLM comprehension within a fixed token budget.

Key ideas:
- **Token estimation** without a tokeniser dependency (4 chars ≈ 1 token).
- **Tiered compression**: high-activation nodes get full content,
  medium-activation nodes get summaries, low-activation nodes get
  one-line keyword stubs.
- **Structural ordering**: nodes are arranged so that hierarchical parents
  appear before children and temporal ordering is preserved where possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .models import ActivationResult, ContextNode
from .graph import ContextGraph


@dataclass
class BudgetSlot:
    """A single slot in the packed context."""

    node_id: str
    content: str            # The text that will go into the context window
    tier: str               # "full", "summary", or "stub"
    activation: float
    estimated_tokens: int
    depth: int


@dataclass
class PackedContext:
    """The final output of the budget manager."""

    slots: List[BudgetSlot]
    total_tokens: int
    token_budget: int
    nodes_included: int
    nodes_dropped: int
    tiers: Dict[str, int]   # tier name -> count


class ContextBudgetManager:
    """
    Packs activated nodes into a token-limited context window using
    tiered compression and structural ordering.
    """

    # Activation thresholds for tier assignment.
    FULL_THRESHOLD: float = 0.4
    SUMMARY_THRESHOLD: float = 0.15

    # Approximate characters per token (GPT-family heuristic).
    CHARS_PER_TOKEN: float = 4.0

    def __init__(
        self,
        graph: ContextGraph,
        token_budget: int = 4000,
        full_threshold: float = FULL_THRESHOLD,
        summary_threshold: float = SUMMARY_THRESHOLD,
    ) -> None:
        self.graph = graph
        self.token_budget = token_budget
        self.full_threshold = full_threshold
        self.summary_threshold = summary_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pack(
        self,
        activated: List[ActivationResult],
        token_budget: Optional[int] = None,
    ) -> PackedContext:
        """
        Pack activated nodes into a context window respecting *token_budget*.

        Nodes are assigned to tiers based on their activation level:
        - **full**: include the node's complete content.
        - **summary**: include node.summary (or truncated content if no summary).
        - **stub**: a one-liner with keywords and a content snippet.

        Nodes are then greedily added in activation order until the budget
        is exhausted.
        """
        budget = token_budget or self.token_budget

        # Assign tiers and prepare candidate slots.
        candidates: List[BudgetSlot] = []
        for ar in activated:
            node = self.graph.get_node(ar.node_id)
            if node is None:
                continue

            tier, text = self._tiered_content(node, ar.activation)
            tokens = self._estimate_tokens(text)
            candidates.append(
                BudgetSlot(
                    node_id=node.id,
                    content=text,
                    tier=tier,
                    activation=ar.activation,
                    estimated_tokens=tokens,
                    depth=ar.depth,
                )
            )

        # Greedy packing: highest activation first.
        candidates.sort(key=lambda s: s.activation, reverse=True)

        slots: List[BudgetSlot] = []
        used_tokens = 0
        dropped = 0

        for slot in candidates:
            if used_tokens + slot.estimated_tokens <= budget:
                slots.append(slot)
                used_tokens += slot.estimated_tokens
            else:
                # Try downgrading tier to fit.
                downgraded = self._downgrade_slot(slot, budget - used_tokens)
                if downgraded is not None:
                    slots.append(downgraded)
                    used_tokens += downgraded.estimated_tokens
                else:
                    dropped += 1

        # Structural reordering: sort by (layer ASC, depth ASC) so parents
        # appear before children and shallower context comes first.
        slots.sort(key=lambda s: (self._node_layer(s.node_id), s.depth))

        tier_counts: Dict[str, int] = {}
        for s in slots:
            tier_counts[s.tier] = tier_counts.get(s.tier, 0) + 1

        return PackedContext(
            slots=slots,
            total_tokens=used_tokens,
            token_budget=budget,
            nodes_included=len(slots),
            nodes_dropped=dropped,
            tiers=tier_counts,
        )

    def render(self, packed: PackedContext) -> str:
        """Render a PackedContext into a single string suitable for an LLM prompt."""
        lines: List[str] = []
        for i, slot in enumerate(packed.slots, 1):
            prefix = f"[{slot.tier.upper()}|act={slot.activation:.2f}]"
            lines.append(f"{prefix} {slot.content}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tiered_content(
        self, node: ContextNode, activation: float
    ) -> Tuple[str, str]:
        """Return (tier_name, text) for a node given its activation."""
        if activation >= self.full_threshold:
            return ("full", node.content)
        elif activation >= self.summary_threshold:
            text = node.summary if node.summary else self._truncate(node.content, 200)
            return ("summary", text)
        else:
            keywords = ", ".join(node.keywords[:5]) if node.keywords else ""
            snippet = self._truncate(node.content, 80)
            stub = f"[{keywords}] {snippet}" if keywords else snippet
            return ("stub", stub)

    def _downgrade_slot(
        self, slot: BudgetSlot, remaining_tokens: int
    ) -> Optional[BudgetSlot]:
        """Try to produce a smaller version of a slot that fits."""
        node = self.graph.get_node(slot.node_id)
        if node is None:
            return None

        # Try summary tier.
        if slot.tier == "full":
            text = node.summary if node.summary else self._truncate(node.content, 200)
            tokens = self._estimate_tokens(text)
            if tokens <= remaining_tokens:
                return BudgetSlot(
                    node_id=slot.node_id,
                    content=text,
                    tier="summary",
                    activation=slot.activation,
                    estimated_tokens=tokens,
                    depth=slot.depth,
                )

        # Try stub tier.
        keywords = ", ".join((node.keywords or [])[:5])
        snippet = self._truncate(node.content, 80)
        stub = f"[{keywords}] {snippet}" if keywords else snippet
        tokens = self._estimate_tokens(stub)
        if tokens <= remaining_tokens:
            return BudgetSlot(
                node_id=slot.node_id,
                content=stub,
                tier="stub",
                activation=slot.activation,
                estimated_tokens=tokens,
                depth=slot.depth,
            )

        return None

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count using the 4-chars heuristic."""
        return max(1, math.ceil(len(text) / self.CHARS_PER_TOKEN))

    def _node_layer(self, node_id: str) -> int:
        node = self.graph.get_node(node_id)
        return node.layer if node else 0

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."
