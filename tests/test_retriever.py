"""
Tests for the Retriever — 3-layer retrieval pipeline.
Uses mock stores to avoid requiring live Neo4j/ChromaDB.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.graph_store import CodeNode, CodeEdge
from src.retriever import Retriever


def make_node(name: str, kind: str = "function", hop: int = 0, score: float = 1.0) -> CodeNode:
    return CodeNode(
        id=f"module::{name}",
        name=name,
        kind=kind,
        file_path=f"service.py",
        start_line=1,
        end_line=10,
        source_code=f"def {name}(): pass",
        score=score,
        hop=hop,
    )


class TestRetriever:

    def setup_method(self):
        self.graph = MagicMock()
        self.vector = MagicMock()
        self.retriever = Retriever(
            graph_store=self.graph,
            vector_store=self.vector,
            max_hops=2,
            top_k_anchor=3,
            relevance_decay=0.8,
            max_context_nodes=10,
        )

    def test_retrieve_returns_nodes(self):
        anchor = make_node("process")
        neighbor = make_node("get_connection", hop=1, score=0.8)

        self.vector.search.return_value = [("module::process", 0.95)]
        self.graph.get_node.return_value = anchor
        self.graph.get_neighbors.return_value = [neighbor]

        results = self.retriever.retrieve("payment failure")

        assert len(results) > 0
        assert any(n.name == "process" for n in results)

    def test_retrieve_empty_when_no_vectors(self):
        self.vector.search.return_value = []
        results = self.retriever.retrieve("anything")
        assert results == []

    def test_scores_decay_with_hops(self):
        anchor = make_node("root", hop=0, score=1.0)
        hop1 = make_node("child", hop=1, score=0.8)
        hop2 = make_node("grandchild", hop=2, score=0.64)

        self.vector.search.return_value = [("module::root", 1.0)]
        self.graph.get_node.return_value = anchor
        self.graph.get_neighbors.return_value = [hop1, hop2]

        results = self.retriever.retrieve("test")
        scores = {n.name: n.score for n in results}

        if "root" in scores and "child" in scores:
            assert scores["root"] >= scores["child"]
        if "child" in scores and "grandchild" in scores:
            assert scores["child"] >= scores["grandchild"]

    def test_max_context_nodes_limit(self):
        # Vector returns 3 anchors, each expands to 5 neighbors
        self.vector.search.return_value = [
            (f"module::fn{i}", 0.9 - i * 0.1) for i in range(3)
        ]
        self.graph.get_node.return_value = make_node("fn0")
        self.graph.get_neighbors.return_value = [
            make_node(f"dep{i}", hop=1, score=0.7) for i in range(5)
        ]

        results = self.retriever.retrieve("query")
        assert len(results) <= self.retriever.max_context_nodes

    def test_deduplication(self):
        # Same node returned from two anchors
        shared_node = make_node("shared_dep", hop=1, score=0.8)
        anchor1 = make_node("service1")
        anchor2 = make_node("service2")

        self.vector.search.return_value = [
            ("module::service1", 0.9),
            ("module::service2", 0.85),
        ]
        self.graph.get_node.side_effect = [anchor1, anchor2]
        self.graph.get_neighbors.return_value = [shared_node]

        results = self.retriever.retrieve("query")
        ids = [n.id for n in results]
        assert len(ids) == len(set(ids)), "Duplicate nodes found in results!"

    def test_format_context(self):
        nodes = [make_node(f"fn{i}") for i in range(3)]
        context = Retriever.format_context(nodes)
        assert "fn0" in context
        assert "fn1" in context
        assert isinstance(context, str)

    def test_format_context_respects_max_chars(self):
        nodes = [make_node(f"fn{i}") for i in range(100)]
        context = Retriever.format_context(nodes, max_chars=500)
        assert len(context) <= 600  # some buffer for last node

    def test_retrieve_with_decomposition(self):
        anchor = make_node("payment_process")
        self.vector.search.return_value = [("module::payment_process", 0.9)]
        self.graph.get_node.return_value = anchor
        self.graph.get_neighbors.return_value = []

        results = self.retriever.retrieve_with_decomposition(
            query="main query",
            sub_queries=["sub1", "sub2"],
        )
        assert isinstance(results, list)
