"""
ContextRetriever — the main orchestrator for the Context Retriever Graph.

This is the primary public API.  It:

1. **Ingests** new information, automatically extracting entities/facts,
   creating nodes, and discovering edges to existing nodes via LLM-powered
   relationship classification.

2. **Retrieves** context for a given query + intent by:
   a) Finding seed nodes via embedding similarity (ChromaDB).
   b) Building a RetrievalPlan from the intent.
   c) Running activation-spreading traversal.
   d) Packing the results into a token-budgeted context string.

3. **Self-organises** the graph over time by:
   - Merging near-duplicate nodes.
   - Promoting frequently co-activated leaf nodes into cluster summaries.
   - Pruning weak nodes that have decayed below threshold.
   - Detecting and marking contradictions between nodes.

The design philosophy: *retrieval is not search — it is planning-aware
graph navigation with active memory management.*
"""

from __future__ import annotations

import json
import re
import uuid
import math
import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import openai
import chromadb

from .models import (
    ActivationResult,
    ContextNode,
    Edge,
    EdgeType,
    RetrievalPlan,
)
from .graph import ContextGraph
from .traversal import IntentDrivenTraversal
from .budget import ContextBudgetManager, PackedContext


# The prompt templates used by the retriever.
_RELATIONSHIP_PROMPT = """Analyze the relationship between these two pieces of information.

NEW: "{new_text}"
EXISTING: "{existing_text}"

Classify their relationship as one or more of:
- SEMANTIC: They discuss similar topics
- CAUSAL: One causes or leads to the other
- TEMPORAL: They have a time-based ordering
- HIERARCHICAL: One is a more specific/general version of the other
- CONTRADICTS: They conflict or disagree
- SUPPORTS: One reinforces or corroborates the other
- NONE: No meaningful relationship

For each relationship found, rate its strength from 0.0 to 1.0.

Respond ONLY with a JSON array like:
[{{"type": "SEMANTIC", "weight": 0.8, "direction": "bidirectional"}}, {{"type": "CAUSAL", "weight": 0.6, "direction": "new_to_existing"}}]

Direction must be "bidirectional", "new_to_existing", or "existing_to_new".
If NONE, respond with: []"""

_EXTRACT_PROMPT = """Extract structured knowledge from this text.
Return a JSON object with:
- "facts": list of distinct factual claims (strings)
- "keywords": list of key terms/concepts (strings)
- "summary": a one-sentence summary

Text: "{text}"

Respond ONLY with valid JSON."""

_INTENT_PROMPT = """Classify the intent behind this query into exactly one of these categories:
- explore: Broad information gathering, understanding a topic
- debug: Finding causes of problems, tracing errors
- summarize: Getting an overview or condensed understanding
- verify: Checking claims, finding supporting/contradicting evidence
- plan: Preparing for future actions, understanding dependencies
- recall: Retrieving specific previously stored information

Query: "{query}"
Context hint: "{hint}"

Respond with ONLY the category name (one word)."""


class ContextRetriever:
    """
    Main API for the Context Retriever Graph system.

    Parameters
    ----------
    db_path : str
        SQLite path for graph persistence (default in-memory).
    collection_name : str
        ChromaDB collection name for embedding search.
    token_budget : int
        Default token budget for context packing.
    auto_organise : bool
        Whether to trigger self-organisation after ingestion.
    llm_model : str
        OpenAI model for relationship classification and extraction.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        collection_name: str = "context_graph",
        token_budget: int = 4000,
        auto_organise: bool = True,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.graph = ContextGraph(db_path=db_path)
        self.traversal = IntentDrivenTraversal(self.graph)
        self.budget_manager = ContextBudgetManager(self.graph, token_budget=token_budget)
        self.token_budget = token_budget
        self.auto_organise = auto_organise
        self.llm_model = llm_model
        self.collection_name = collection_name

        # ChromaDB for embedding-based seed finding.
        self._chroma = chromadb.Client()
        self._openai = openai.OpenAI()

    # ==================================================================
    # INGEST — add new information to the graph
    # ==================================================================

    def ingest(
        self,
        text: str,
        metadata: Optional[Dict[str, str]] = None,
        discover_edges: bool = True,
    ) -> ContextNode:
        """
        Ingest a piece of text into the context graph.

        Steps:
        1. Extract facts, keywords, summary via LLM.
        2. Create a ContextNode with an embedding.
        3. Find related existing nodes via embedding similarity.
        4. Classify relationships and add typed edges.
        5. Optionally trigger self-organisation.

        Returns the created node.
        """
        metadata = metadata or {}

        # Step 1: Extract structured knowledge.
        extracted = self._extract_knowledge(text)
        facts = extracted.get("facts", [text])
        keywords = extracted.get("keywords", [])
        summary = extracted.get("summary", "")

        # Step 2: Create embedding and node.
        embedding = self._get_embedding(text)
        node = ContextNode(
            content=text,
            summary=summary,
            embedding=embedding,
            keywords=keywords,
            layer=0,
            metadata=metadata,
        )
        self.graph.add_node(node)

        # Store embedding in ChromaDB for seed-finding.
        collection = self._chroma.get_or_create_collection(self.collection_name)
        collection.add(
            ids=[node.id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata] if metadata else None,
        )

        # Step 3-4: Discover edges to existing nodes.
        if discover_edges and self.graph.node_count() > 1:
            self._discover_edges(node)

        # Step 5: Self-organise.
        if self.auto_organise:
            self._maybe_organise()

        return node

    def ingest_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, str]]] = None,
    ) -> List[ContextNode]:
        """Ingest multiple texts. Edge discovery runs after all nodes are added."""
        metadata_list = metadata_list or [{}] * len(texts)
        nodes: List[ContextNode] = []

        for text, meta in zip(texts, metadata_list):
            node = self.ingest(text, metadata=meta, discover_edges=False)
            nodes.append(node)

        # Now discover edges for all new nodes.
        for node in nodes:
            self._discover_edges(node)

        if self.auto_organise:
            self._maybe_organise()

        return nodes

    # ==================================================================
    # RETRIEVE — get context for a query + intent
    # ==================================================================

    def retrieve(
        self,
        query: str,
        intent: Optional[str] = None,
        intent_hint: str = "",
        max_depth: int = 3,
        max_nodes: int = 20,
        token_budget: Optional[int] = None,
        n_seeds: int = 5,
    ) -> PackedContext:
        """
        Retrieve context for a query using intent-driven graph traversal.

        Parameters
        ----------
        query : str
            The retrieval query.
        intent : str, optional
            Explicit intent label.  If None, the LLM classifies it.
        intent_hint : str
            Additional context to help intent classification.
        max_depth : int
            Max traversal hops from seeds.
        max_nodes : int
            Max nodes to return.
        token_budget : int, optional
            Override the default token budget.
        n_seeds : int
            Number of seed nodes from embedding search.

        Returns
        -------
        PackedContext
            Token-budgeted context with tiered compression.
        """
        budget = token_budget or self.token_budget

        # Step 1: Classify intent.
        if intent is None:
            intent = self._classify_intent(query, intent_hint)

        # Step 2: Find seed nodes via embedding similarity.
        seed_ids = self._find_seeds(query, n=n_seeds)

        if not seed_ids:
            # No seeds found — return empty context.
            return PackedContext(
                slots=[], total_tokens=0, token_budget=budget,
                nodes_included=0, nodes_dropped=0, tiers={},
            )

        # Step 3: Build retrieval plan.
        plan = IntentDrivenTraversal.plan_from_intent(
            intent=intent,
            seed_ids=seed_ids,
            max_depth=max_depth,
            max_nodes=max_nodes,
            token_budget=budget,
        )

        # Step 4: Traverse.
        activated = self.traversal.traverse(plan)

        # Touch accessed nodes (update access stats).
        for ar in activated:
            node = self.graph.get_node(ar.node_id)
            if node:
                node.touch()

        # Step 5: Pack into budget.
        packed = self.budget_manager.pack(activated, token_budget=budget)
        return packed

    def retrieve_text(self, query: str, **kwargs: Any) -> str:
        """Convenience: retrieve and render to a string."""
        packed = self.retrieve(query, **kwargs)
        return self.budget_manager.render(packed)

    # ==================================================================
    # SELF-ORGANISATION
    # ==================================================================

    def organise(self) -> Dict[str, int]:
        """
        Run all self-organisation passes and return a summary of changes.
        """
        results: Dict[str, int] = {
            "pruned": 0,
            "clusters_created": 0,
        }

        results["pruned"] = self._prune_weak_nodes()
        results["clusters_created"] = self._create_cluster_summaries()
        return results

    def _maybe_organise(self) -> None:
        """Trigger organisation periodically (every 10 ingestions)."""
        if self.graph.node_count() > 0 and self.graph.node_count() % 10 == 0:
            self.organise()

    def _prune_weak_nodes(self, threshold: float = 0.05) -> int:
        """Remove leaf nodes with very low strength."""
        pruned = 0
        for node in list(self.graph.nodes_by_layer(0)):
            if node.strength() < threshold and node.access_count == 0:
                self.graph.remove_node(node.id)
                self._remove_from_chroma(node.id)
                pruned += 1
        return pruned

    def _create_cluster_summaries(self, min_cluster_size: int = 3) -> int:
        """
        Find groups of tightly-connected leaf nodes and create summary
        nodes (layer=1) above them with HIERARCHICAL edges.
        """
        created = 0
        visited: set[str] = set()

        for node in self.graph.nodes_by_layer(0):
            if node.id in visited:
                continue

            # Find all semantically-connected neighbours.
            cluster_ids = {node.id}
            frontier = [node.id]
            while frontier:
                current_id = frontier.pop()
                for neighbour, edge in self.graph.neighbours(current_id, EdgeType.SEMANTIC):
                    if neighbour.id not in cluster_ids and neighbour.layer == 0:
                        cluster_ids.add(neighbour.id)
                        frontier.append(neighbour.id)

            visited |= cluster_ids

            if len(cluster_ids) < min_cluster_size:
                continue

            # Build a summary node for the cluster.
            cluster_nodes = [
                self.graph.get_node(nid) for nid in cluster_ids
                if self.graph.get_node(nid) is not None
            ]
            combined_keywords: list[str] = []
            for cn in cluster_nodes:
                combined_keywords.extend(cn.keywords)
            # Deduplicate while preserving order.
            seen_kw: set[str] = set()
            unique_keywords: list[str] = []
            for kw in combined_keywords:
                if kw.lower() not in seen_kw:
                    seen_kw.add(kw.lower())
                    unique_keywords.append(kw)

            combined_content = " | ".join(cn.content[:100] for cn in cluster_nodes)
            summary_text = f"Cluster of {len(cluster_nodes)} related memories: {combined_content[:300]}"

            summary_node = ContextNode(
                content=summary_text,
                summary=summary_text,
                keywords=unique_keywords[:10],
                layer=1,
                importance=max(cn.importance for cn in cluster_nodes),
            )

            # Average the embeddings.
            if all(cn.embedding for cn in cluster_nodes):
                dim = len(cluster_nodes[0].embedding)
                avg_emb = [0.0] * dim
                for cn in cluster_nodes:
                    for j in range(dim):
                        avg_emb[j] += cn.embedding[j]
                for j in range(dim):
                    avg_emb[j] /= len(cluster_nodes)
                summary_node.embedding = avg_emb

            self.graph.add_node(summary_node)

            # Add hierarchical edges from summary to each member.
            for cn in cluster_nodes:
                self.graph.add_edge(Edge(
                    source_id=summary_node.id,
                    target_id=cn.id,
                    edge_type=EdgeType.HIERARCHICAL,
                    weight=0.9,
                ))

            created += 1

        return created

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _get_embedding(self, text: str) -> List[float]:
        response = self._openai.embeddings.create(
            input=text, model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def _get_llm_response(self, prompt: str) -> str:
        response = self._openai.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    def _extract_knowledge(self, text: str) -> Dict[str, Any]:
        prompt = _EXTRACT_PROMPT.format(text=text[:2000])
        raw = self._get_llm_response(prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from the response.
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"facts": [text], "keywords": [], "summary": ""}

    def _classify_intent(self, query: str, hint: str = "") -> str:
        prompt = _INTENT_PROMPT.format(query=query, hint=hint)
        raw = self._get_llm_response(prompt).strip().lower()
        valid = {"explore", "debug", "summarize", "verify", "plan", "recall"}
        if raw in valid:
            return raw
        # Fuzzy match.
        for v in valid:
            if v in raw:
                return v
        return "explore"

    def _find_seeds(self, query: str, n: int = 5) -> List[str]:
        """Find the n most similar existing nodes via ChromaDB."""
        collection = self._chroma.get_or_create_collection(self.collection_name)
        if collection.count() == 0:
            return []

        query_embedding = self._get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, collection.count()),
        )
        if results and results["ids"]:
            return results["ids"][0]
        return []

    def _discover_edges(self, new_node: ContextNode) -> None:
        """
        Find existing nodes related to *new_node* and classify the
        relationships using an LLM, then add typed edges.
        """
        # Find top-k similar nodes via ChromaDB.
        collection = self._chroma.get_or_create_collection(self.collection_name)
        if collection.count() <= 1:
            return

        results = collection.query(
            query_embeddings=[new_node.embedding],
            n_results=min(6, collection.count()),
        )
        if not results or not results["ids"]:
            return

        candidate_ids = [
            cid for cid in results["ids"][0] if cid != new_node.id
        ][:5]

        for cid in candidate_ids:
            existing_node = self.graph.get_node(cid)
            if existing_node is None:
                continue

            relationships = self._classify_relationship(
                new_node.content, existing_node.content
            )
            for rel in relationships:
                edge_type_str = rel.get("type", "").upper()
                try:
                    edge_type = EdgeType(edge_type_str.lower())
                except ValueError:
                    continue

                weight = float(rel.get("weight", 0.5))
                direction = rel.get("direction", "bidirectional")

                if direction in ("new_to_existing", "bidirectional"):
                    self.graph.add_edge(Edge(
                        source_id=new_node.id,
                        target_id=existing_node.id,
                        edge_type=edge_type,
                        weight=weight,
                    ))
                if direction in ("existing_to_new", "bidirectional"):
                    self.graph.add_edge(Edge(
                        source_id=existing_node.id,
                        target_id=new_node.id,
                        edge_type=edge_type,
                        weight=weight,
                    ))

    def _classify_relationship(
        self, new_text: str, existing_text: str
    ) -> List[Dict[str, Any]]:
        prompt = _RELATIONSHIP_PROMPT.format(
            new_text=new_text[:500], existing_text=existing_text[:500]
        )
        raw = self._get_llm_response(prompt)
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    pass
        return []

    def _remove_from_chroma(self, node_id: str) -> None:
        try:
            collection = self._chroma.get_or_create_collection(self.collection_name)
            collection.delete(ids=[node_id])
        except Exception:
            pass

    # ==================================================================
    # Inspection / debugging
    # ==================================================================

    def stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        return self.graph.stats()

    def explain_retrieval(
        self, query: str, intent: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Retrieve context and return a detailed explanation of how
        each node was reached, useful for debugging retrieval quality.
        """
        budget = kwargs.get("token_budget", self.token_budget)

        if intent is None:
            intent = self._classify_intent(query)

        seed_ids = self._find_seeds(query, n=kwargs.get("n_seeds", 5))

        plan = IntentDrivenTraversal.plan_from_intent(
            intent=intent,
            seed_ids=seed_ids,
            max_depth=kwargs.get("max_depth", 3),
            max_nodes=kwargs.get("max_nodes", 20),
            token_budget=budget,
        )

        activated = self.traversal.traverse(plan)
        packed = self.budget_manager.pack(activated, token_budget=budget)

        return {
            "intent": intent,
            "seed_ids": seed_ids,
            "plan": {
                "edge_weights": {k.value: v for k, v in plan.edge_weights.items()},
                "max_depth": plan.max_depth,
                "max_nodes": plan.max_nodes,
            },
            "activated_nodes": [
                {
                    "node_id": ar.node_id,
                    "activation": ar.activation,
                    "depth": ar.depth,
                    "path": ar.path,
                    "content_preview": (
                        self.graph.get_node(ar.node_id).content[:100]
                        if self.graph.get_node(ar.node_id)
                        else ""
                    ),
                }
                for ar in activated
            ],
            "packed": {
                "total_tokens": packed.total_tokens,
                "nodes_included": packed.nodes_included,
                "nodes_dropped": packed.nodes_dropped,
                "tiers": packed.tiers,
            },
        }
