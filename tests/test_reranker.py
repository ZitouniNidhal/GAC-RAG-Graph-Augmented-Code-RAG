"""
Tests for the Reranker — LLM-based context pruning.
Mocks the Anthropic API to avoid real calls during testing.
"""
import pytest
import json
from unittest.mock import MagicMock, patch
from src.graph_store import CodeNode
from src.reranker import Reranker


def make_node(name: str, kind: str = "function") -> CodeNode:
    return CodeNode(
        id=f"module::{name}",
        name=name,
        kind=kind,
        file_path="service.py",
        start_line=1,
        end_line=10,
        source_code=f"def {name}(): pass",
    )


MOCK_RERANKER_RESPONSE = {
    "relevant_ids": ["module::process", "module::get_connection"],
    "reasoning": "These nodes are directly involved in the payment failure chain."
}


class TestReranker:

    def _make_reranker_with_mock(self, response_json: dict) -> Reranker:
        reranker = Reranker(api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(response_json))]
        reranker.client = MagicMock()
        reranker.client.messages.create.return_value = mock_response
        return reranker

    def test_reranker_filters_nodes(self):
        nodes = [make_node("process"), make_node("get_connection"), make_node("unrelated_fn")]
        reranker = self._make_reranker_with_mock(MOCK_RERANKER_RESPONSE)

        filtered, reasoning = reranker.rerank("payment failure query", nodes)

        filtered_names = {n.name for n in filtered}
        assert "process" in filtered_names
        assert "get_connection" in filtered_names
        assert "unrelated_fn" not in filtered_names

    def test_reranker_returns_reasoning(self):
        nodes = [make_node("process")]
        reranker = self._make_reranker_with_mock(MOCK_RERANKER_RESPONSE)

        _, reasoning = reranker.rerank("query", nodes)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_reranker_fallback_on_empty_result(self):
        """If LLM returns no relevant IDs, fallback to top-5 by score."""
        nodes = [make_node(f"fn{i}") for i in range(8)]
        reranker = self._make_reranker_with_mock({"relevant_ids": [], "reasoning": "none"})

        filtered, _ = reranker.rerank("query", nodes)
        assert len(filtered) > 0  # fallback should return something

    def test_reranker_handles_json_parse_error(self):
        """On JSON error, fallback gracefully."""
        nodes = [make_node("fn1"), make_node("fn2"), make_node("fn3")]
        reranker = Reranker(api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not valid json {{")]
        reranker.client = MagicMock()
        reranker.client.messages.create.return_value = mock_response

        filtered, reasoning = reranker.rerank("query", nodes)
        assert len(filtered) > 0
        assert "error" in reasoning.lower() or "fallback" in reasoning.lower()

    def test_reranker_handles_empty_input(self):
        reranker = self._make_reranker_with_mock(MOCK_RERANKER_RESPONSE)
        filtered, reasoning = reranker.rerank("query", [])
        assert filtered == []

    def test_reranker_strips_markdown_fences(self):
        """LLM sometimes wraps JSON in ```json``` — should still parse."""
        nodes = [make_node("process")]
        wrapped = f"```json\n{json.dumps(MOCK_RERANKER_RESPONSE)}\n```"
        reranker = Reranker(api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=wrapped)]
        reranker.client = MagicMock()
        reranker.client.messages.create.return_value = mock_response

        filtered, _ = reranker.rerank("query", nodes)
        assert isinstance(filtered, list)

    def test_node_summary_format(self):
        nodes = [make_node("MyClass", kind="class"), make_node("my_func")]
        summary = Reranker._build_node_summaries(nodes)
        assert "module::MyClass" in summary
        assert "module::my_func" in summary
        assert "class" in summary
        assert "function" in summary
