"""
retriever.py
The heart of GAC-RAG: 3-layer retrieval pipeline.

Layer 1 — Semantic Anchor    : ChromaDB vector search → top-k entry points
Layer 2 — Graph Expansion    : Neo4j N-hop traversal → dependency subgraph
Layer 3 — LLM Reranking      : Claude prunes irrelevant nodes
"""

from __future__ import annotations
import os
from typing import Optional
from dotenv import load_dotenv

from src.graph_store import GraphStore, CodeNode
from src.vector_store import VectorStore

load_dotenv()


class Retriever:
    """
    GAC-RAG 3-layer retrieval pipeline.

    Usage:
        retriever = Retriever(graph_store, vector_store)
        context_nodes = retriever.retrieve("Why does payment fail silently?")
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        max_hops: int = 2,
        top_k_anchor: int = 5,
        relevance_decay: float = 0.8,
        max_context_nodes: int = 20,
    ):
        self.graph = graph_store
        self.vector = vector_store
        self.max_hops = int(os.getenv("MAX_HOPS", max_hops))
        self.top_k_anchor = int(os.getenv("TOP_K_ANCHOR", top_k_anchor))
        self.relevance_decay = float(os.getenv("RELEVANCE_DECAY", relevance_decay))
        self.max_context_nodes = int(os.getenv("MAX_CONTEXT_NODES", max_context_nodes))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        verbose: bool = False,
    ) -> list[CodeNode]:
        """
        Full 3-layer retrieval.

        Args:
            query:   Natural language question about the codebase.
            verbose: Print layer-by-layer diagnostics.

        Returns:
            Ordered list of relevant CodeNode objects (best first).
        """
        # ── Layer 1: Semantic anchor ──────────────────────────────────
        anchor_hits = self._layer1_semantic_anchor(query)
        if verbose:
            print(f"\n[Layer 1] Semantic anchors ({len(anchor_hits)}):")
            for node_id, score in anchor_hits:
                print(f"  {score:.3f}  {node_id}")

        if not anchor_hits:
            return []

        anchor_ids = [node_id for node_id, _ in anchor_hits]
        anchor_scores = {node_id: score for node_id, score in anchor_hits}

        # ── Layer 2: Graph expansion ──────────────────────────────────
        expanded = self._layer2_graph_expand(anchor_ids, anchor_scores)
        if verbose:
            print(f"\n[Layer 2] After graph expansion: {len(expanded)} nodes")
            for node in expanded[:10]:
                print(f"  hop={node.hop} score={node.score:.3f}  {node.id}")

        # ── Dedup + sort by score ─────────────────────────────────────
        seen: dict[str, CodeNode] = {}
        for node in expanded:
            if node.id not in seen or node.score > seen[node.id].score:
                seen[node.id] = node

        ranked = sorted(seen.values(), key=lambda n: n.score, reverse=True)
        ranked = ranked[: self.max_context_nodes]

        if verbose:
            print(f"\n[Pre-rerank] Top {len(ranked)} nodes ready for LLM reranker")

        return ranked

    def retrieve_with_decomposition(
        self,
        query: str,
        sub_queries: list[str],
        verbose: bool = False,
    ) -> list[CodeNode]:
        """
        Advanced: retrieve for each sub-query independently, then merge.
        Useful for complex multi-hop questions.

        Args:
            query:       Original user question (for dedup context).
            sub_queries: Decomposed sub-questions.
            verbose:     Print diagnostics.

        Returns:
            Merged, deduplicated, ranked CodeNode list.
        """
        all_nodes: dict[str, CodeNode] = {}

        for sq in sub_queries:
            if verbose:
                print(f"\n── Sub-query: {sq}")
            nodes = self.retrieve(sq, verbose=verbose)
            for node in nodes:
                if node.id not in all_nodes or node.score > all_nodes[node.id].score:
                    all_nodes[node.id] = node

        merged = sorted(all_nodes.values(), key=lambda n: n.score, reverse=True)
        return merged[: self.max_context_nodes]

    # ------------------------------------------------------------------ #
    #  Layer implementations                                              #
    # ------------------------------------------------------------------ #

    def _layer1_semantic_anchor(self, query: str) -> list[tuple[str, float]]:
        """ChromaDB similarity search → top-k (node_id, score) pairs."""
        return self.vector.search(query, top_k=self.top_k_anchor)

    def _layer2_graph_expand(
        self,
        anchor_ids: list[str],
        anchor_scores: dict[str, float],
    ) -> list[CodeNode]:
        """
        For each anchor node, traverse the dependency graph up to max_hops.
        Scores decay by relevance_decay per hop from the anchor.
        """
        result: list[CodeNode] = []

        for anchor_id in anchor_ids:
            anchor_base_score = anchor_scores.get(anchor_id, 1.0)

            # Add the anchor itself
            anchor_node = self.graph.get_node(anchor_id)
            if anchor_node:
                anchor_node.score = anchor_base_score
                anchor_node.hop = 0
                result.append(anchor_node)

            # Expand neighbors
            neighbors = self.graph.get_neighbors(
                anchor_id,
                hops=self.max_hops,
                decay=self.relevance_decay,
            )
            for neighbor in neighbors:
                # Combine anchor score with hop decay
                neighbor.score = anchor_base_score * (self.relevance_decay ** neighbor.hop)
                result.append(neighbor)

        return result

    # ------------------------------------------------------------------ #
    #  Context formatting                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_context(nodes: list[CodeNode], max_chars: int = 8000) -> str:
        """
        Format retrieved nodes into a single context string for the LLM.
        Respects max_chars budget.
        """
        parts: list[str] = []
        total = 0

        for node in nodes:
            header = (
                f"### [{node.kind.upper()}] {node.name}\n"
                f"# File: {node.file_path} | Lines {node.start_line}-{node.end_line} "
                f"| Score: {node.score:.3f}\n"
            )
            body = node.source_code.strip()
            if node.docstring:
                body = f'"""{node.docstring}"""\n' + body

            block = f"{header}{body}\n\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)

        return "".join(parts)