"""
graph_store.py
Neo4j interface for storing and traversing the code dependency graph.
Nodes: Function, Class, Module
Edges: CALLS, IMPORTS, INHERITS, REFERENCES
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CodeNode:
    """Represents a code entity (function, class, module)."""
    id: str                        # unique: "module::ClassName::method_name"
    name: str
    kind: str                      # "function" | "class" | "module"
    file_path: str
    start_line: int
    end_line: int
    source_code: str
    docstring: str = ""
    language: str = "python"
    score: float = 1.0             # relevance score (decays with hops)
    hop: int = 0                   # how many hops from anchor


@dataclass
class CodeEdge:
    """Represents a dependency between two code nodes."""
    source_id: str
    target_id: str
    kind: str                      # "CALLS" | "IMPORTS" | "INHERITS" | "REFERENCES"
    weight: float = 1.0


class GraphStore:
    """
    Neo4j-backed code dependency graph.

    Usage:
        store = GraphStore()
        store.add_node(node)
        store.add_edge(edge)
        neighbors = store.get_neighbors("module::MyClass::my_method", hops=2)
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._ensure_constraints()
            print("Connected to Neo4j.")
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j ({e}). Graph features will be disabled. Only Vector RAG will run.")

    # ------------------------------------------------------------------ #
    #  Schema                                                              #
    # ------------------------------------------------------------------ #

    def _ensure_constraints(self):
        if not self.driver: return
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT code_node_id IF NOT EXISTS "
                "FOR (n:CodeNode) REQUIRE n.id IS UNIQUE"
            )

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def add_node(self, node: CodeNode) -> None:
        if not self.driver: return
        query = """
        MERGE (n:CodeNode {id: $id})
        SET n.name       = $name,
            n.kind       = $kind,
            n.file_path  = $file_path,
            n.start_line = $start_line,
            n.end_line   = $end_line,
            n.source_code= $source_code,
            n.docstring  = $docstring,
            n.language   = $language
        """
        with self.driver.session() as session:
            session.run(query, **node.__dict__)

    def add_edge(self, edge: CodeEdge) -> None:
        if not self.driver: return
        query = f"""
        MATCH (a:CodeNode {{id: $source_id}})
        MATCH (b:CodeNode {{id: $target_id}})
        MERGE (a)-[r:{edge.kind}]->(b)
        SET r.weight = $weight
        """
        with self.driver.session() as session:
            session.run(
                query,
                source_id=edge.source_id,
                target_id=edge.target_id,
                weight=edge.weight,
            )

    def add_nodes_batch(self, nodes: list[CodeNode]) -> None:
        for node in nodes:
            self.add_node(node)

    def add_edges_batch(self, edges: list[CodeEdge]) -> None:
        for edge in edges:
            self.add_edge(edge)

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def get_node(self, node_id: str) -> Optional[CodeNode]:
        if not self.driver: return None
        query = "MATCH (n:CodeNode {id: $id}) RETURN n"
        with self.driver.session() as session:
            result = session.run(query, id=node_id).single()
            if result:
                return self._record_to_node(result["n"])
        return None

    def get_neighbors(
        self,
        node_id: str,
        hops: int = 2,
        decay: float = 0.8,
        edge_types: Optional[list[str]] = None,
    ) -> list[CodeNode]:
        """
        BFS traversal up to `hops` levels.
        Returns neighbors with decayed relevance scores.
        """
        if not self.driver: return []
        edge_filter = ""
        if edge_types:
            types = "|".join(edge_types)
            edge_filter = f":{types}"

        query = f"""
        MATCH path = (start:CodeNode {{id: $node_id}})-[r{edge_filter}*1..{hops}]-(neighbor:CodeNode)
        WHERE neighbor.id <> $node_id
        WITH neighbor, min(length(path)) AS hop_distance
        RETURN neighbor, hop_distance
        ORDER BY hop_distance
        """
        results: list[CodeNode] = []
        seen: set[str] = set()

        with self.driver.session() as session:
            for record in session.run(query, node_id=node_id):
                node = self._record_to_node(record["neighbor"])
                hop = record["hop_distance"]
                if node.id not in seen:
                    node.hop = hop
                    node.score = decay ** hop
                    results.append(node)
                    seen.add(node.id)

        return results

    def get_callers(self, node_id: str) -> list[CodeNode]:
        """Who calls this node?"""
        if not self.driver: return []
        query = """
        MATCH (caller:CodeNode)-[:CALLS]->(n:CodeNode {id: $id})
        RETURN caller
        """
        with self.driver.session() as session:
            return [
                self._record_to_node(r["caller"])
                for r in session.run(query, id=node_id)
            ]

    def get_callees(self, node_id: str) -> list[CodeNode]:
        """What does this node call?"""
        if not self.driver: return []
        query = """
        MATCH (n:CodeNode {id: $id})-[:CALLS]->(callee:CodeNode)
        RETURN callee
        """
        with self.driver.session() as session:
            return [
                self._record_to_node(r["callee"])
                for r in session.run(query, id=node_id)
            ]

    def clear(self) -> None:
        """Delete all nodes and edges — useful for re-indexing."""
        if not self.driver: return
        with self.driver.session() as session:
            session.run("MATCH (n:CodeNode) DETACH DELETE n")

    def node_count(self) -> int:
        if not self.driver: return 0
        with self.driver.session() as session:
            result = session.run("MATCH (n:CodeNode) RETURN count(n) AS c").single()
            return result["c"] if result else 0

    def edge_count(self) -> int:
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
            return result["c"] if result else 0

    def get_health_stats(self) -> dict:
        """Analyze graph structure and return health metrics."""
        stats = {}
        with self.driver.session() as session:
            # Orphan nodes (no edges)
            orphan_q = "MATCH (n:CodeNode) WHERE NOT (n)--() RETURN count(n) AS c"
            stats["orphan_nodes"] = session.run(orphan_q).single()["c"]
            
            # Nodes by kind
            kind_q = "MATCH (n:CodeNode) RETURN n.kind AS kind, count(n) AS count"
            stats["by_kind"] = {r["kind"]: r["count"] for r in session.run(kind_q)}
            
            # Avg degree
            degree_q = "MATCH (n:CodeNode) WITH n, size((n)--()) AS degree RETURN avg(degree) AS avg_deg"
            stats["avg_degree"] = session.run(degree_q).single()["avg_deg"] or 0
            
        return stats

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _record_to_node(record) -> CodeNode:
        return CodeNode(
            id=record["id"],
            name=record["name"],
            kind=record["kind"],
            file_path=record["file_path"],
            start_line=record["start_line"],
            end_line=record["end_line"],
            source_code=record["source_code"],
            docstring=record.get("docstring", ""),
            language=record.get("language", "python"),
        )

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()