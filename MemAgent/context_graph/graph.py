"""
ContextGraph — the core multi-relation graph with SQLite persistence.

Stores ContextNodes and typed Edges in an adjacency-list structure backed
by SQLite so the graph survives restarts without requiring an external
graph database.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Dict, List, Optional, Set, Tuple

from .models import ContextNode, Edge, EdgeType


class ContextGraph:
    """
    An in-memory multi-relation directed graph persisted to SQLite.

    Nodes are ContextNode instances keyed by their string ID.
    Edges are stored in a forward adjacency list (source -> [Edge])
    and a reverse adjacency list (target -> [Edge]) for bidirectional traversal.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._nodes: Dict[str, ContextNode] = {}
        self._forward: Dict[str, List[Edge]] = {}   # source_id -> edges
        self._reverse: Dict[str, List[Edge]] = {}   # target_id -> edges
        self._edge_index: Dict[str, Edge] = {}       # edge.key -> Edge

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        self._load_from_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cg_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                embedding TEXT NOT NULL DEFAULT '[]',
                keywords TEXT NOT NULL DEFAULT '[]',
                layer INTEGER NOT NULL DEFAULT 0,
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cg_edges (
                key TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES cg_nodes(id),
                FOREIGN KEY (target_id) REFERENCES cg_nodes(id)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_cg_edges_source ON cg_edges(source_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_cg_edges_target ON cg_edges(target_id)
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_from_db(self) -> None:
        """Hydrate the in-memory graph from SQLite."""
        cur = self._conn.cursor()

        for row in cur.execute("SELECT * FROM cg_nodes"):
            node = ContextNode(
                id=row[0],
                content=row[1],
                summary=row[2],
                embedding=json.loads(row[3]),
                keywords=json.loads(row[4]),
                layer=row[5],
                importance=row[6],
                access_count=row[7],
                metadata=json.loads(row[8]),
                created_at=row[9],
                last_accessed=row[10],
            )
            self._nodes[node.id] = node
            self._forward.setdefault(node.id, [])
            self._reverse.setdefault(node.id, [])

        for row in cur.execute("SELECT * FROM cg_edges"):
            edge = Edge(
                source_id=row[1],
                target_id=row[2],
                edge_type=EdgeType(row[3]),
                weight=row[4],
                metadata=json.loads(row[5]),
                created_at=row[6],
            )
            self._edge_index[edge.key] = edge
            self._forward.setdefault(edge.source_id, []).append(edge)
            self._reverse.setdefault(edge.target_id, []).append(edge)

    def _persist_node(self, node: ContextNode) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO cg_nodes
               (id, content, summary, embedding, keywords, layer,
                importance, access_count, metadata, created_at, last_accessed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.id,
                node.content,
                node.summary,
                json.dumps(node.embedding),
                json.dumps(node.keywords),
                node.layer,
                node.importance,
                node.access_count,
                json.dumps(node.metadata),
                node.created_at,
                node.last_accessed,
            ),
        )
        self._conn.commit()

    def _persist_edge(self, edge: Edge) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO cg_edges
               (key, source_id, target_id, edge_type, weight, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                edge.key,
                edge.source_id,
                edge.target_id,
                edge.edge_type.value,
                edge.weight,
                json.dumps(edge.metadata),
                edge.created_at,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: ContextNode) -> ContextNode:
        """Add a node to the graph and persist it."""
        self._nodes[node.id] = node
        self._forward.setdefault(node.id, [])
        self._reverse.setdefault(node.id, [])
        self._persist_node(node)
        return node

    def get_node(self, node_id: str) -> Optional[ContextNode]:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all edges touching it."""
        if node_id not in self._nodes:
            return

        # Remove forward edges from this node
        for edge in list(self._forward.get(node_id, [])):
            self._reverse[edge.target_id] = [
                e for e in self._reverse.get(edge.target_id, []) if e.key != edge.key
            ]
            self._edge_index.pop(edge.key, None)
        self._forward.pop(node_id, None)

        # Remove reverse edges pointing to this node
        for edge in list(self._reverse.get(node_id, [])):
            self._forward[edge.source_id] = [
                e for e in self._forward.get(edge.source_id, []) if e.key != edge.key
            ]
            self._edge_index.pop(edge.key, None)
        self._reverse.pop(node_id, None)

        del self._nodes[node_id]

        cur = self._conn.cursor()
        cur.execute("DELETE FROM cg_edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
        cur.execute("DELETE FROM cg_nodes WHERE id = ?", (node_id,))
        self._conn.commit()

    def all_nodes(self) -> List[ContextNode]:
        return list(self._nodes.values())

    def node_count(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: Edge) -> Edge:
        """Add or update an edge."""
        if edge.key in self._edge_index:
            old = self._edge_index[edge.key]
            self._forward[old.source_id] = [
                e for e in self._forward.get(old.source_id, []) if e.key != old.key
            ]
            self._reverse[old.target_id] = [
                e for e in self._reverse.get(old.target_id, []) if e.key != old.key
            ]

        self._edge_index[edge.key] = edge
        self._forward.setdefault(edge.source_id, []).append(edge)
        self._reverse.setdefault(edge.target_id, []).append(edge)
        self._persist_edge(edge)
        return edge

    def get_edges_from(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[Edge]:
        edges = self._forward.get(node_id, [])
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_edges_to(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[Edge]:
        edges = self._reverse.get(node_id, [])
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def neighbours(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[Tuple[ContextNode, Edge]]:
        """Return (neighbour_node, edge) pairs reachable from node_id."""
        result: List[Tuple[ContextNode, Edge]] = []
        for edge in self.get_edges_from(node_id, edge_type):
            target = self._nodes.get(edge.target_id)
            if target is not None:
                result.append((target, edge))
        # Also follow reverse edges (undirected traversal for semantic/supports)
        for edge in self.get_edges_to(node_id, edge_type):
            source = self._nodes.get(edge.source_id)
            if source is not None:
                result.append((source, edge))
        return result

    def edge_count(self) -> int:
        return len(self._edge_index)

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

    def nodes_by_layer(self, layer: int) -> List[ContextNode]:
        return [n for n in self._nodes.values() if n.layer == layer]

    def nodes_by_keyword(self, keyword: str) -> List[ContextNode]:
        kw_lower = keyword.lower()
        return [
            n for n in self._nodes.values()
            if any(kw_lower in k.lower() for k in n.keywords)
        ]

    def subgraph(self, node_ids: Set[str]) -> "ContextGraph":
        """Return a new in-memory ContextGraph containing only the given nodes and edges between them."""
        sub = ContextGraph.__new__(ContextGraph)
        sub._nodes = {nid: self._nodes[nid] for nid in node_ids if nid in self._nodes}
        sub._forward = {}
        sub._reverse = {}
        sub._edge_index = {}
        sub._conn = sqlite3.connect(":memory:")

        for nid in node_ids:
            sub._forward[nid] = []
            sub._reverse[nid] = []
            for edge in self._forward.get(nid, []):
                if edge.target_id in node_ids:
                    sub._edge_index[edge.key] = edge
                    sub._forward[nid].append(edge)
                    sub._reverse.setdefault(edge.target_id, []).append(edge)

        return sub

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all nodes and edges."""
        self._nodes.clear()
        self._forward.clear()
        self._reverse.clear()
        self._edge_index.clear()
        cur = self._conn.cursor()
        cur.execute("DELETE FROM cg_edges")
        cur.execute("DELETE FROM cg_nodes")
        self._conn.commit()

    def stats(self) -> Dict[str, int]:
        edge_type_counts = {}
        for edge in self._edge_index.values():
            edge_type_counts[edge.edge_type.value] = (
                edge_type_counts.get(edge.edge_type.value, 0) + 1
            )
        return {
            "nodes": self.node_count(),
            "edges": self.edge_count(),
            "layers": len({n.layer for n in self._nodes.values()}),
            **{f"edges_{k}": v for k, v in edge_type_counts.items()},
        }
