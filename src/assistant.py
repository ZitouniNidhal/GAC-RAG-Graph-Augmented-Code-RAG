"""
assistant.py
Main orchestrator for the GAC-RAG Code Assistant.

Ties together: Indexer → GraphStore + VectorStore → Retriever → Reranker → LLM
"""

from __future__ import annotations
import os
import json
from typing import Optional
import anthropic
from dotenv import load_dotenv

from src.indexer import Indexer
from src.graph_store import GraphStore
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.reranker import Reranker

load_dotenv()

ANSWER_SYSTEM_PROMPT = """You are an expert code assistant with deep knowledge of software architecture.





class CodeAssistant:
    """
    End-to-end GAC-RAG Code Assistant.

    Usage:
        assistant = CodeAssistant(repo_path="./my_project")
        assistant.index()
        answer = assistant.ask("How does authentication work in this codebase?")
        print(answer)
    """

    def __init__(
        self,
        repo_path: str,
        api_key: Optional[str] = None,
        graph_store: Optional[GraphStore] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.repo_path = repo_path
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        self.graph_store = graph_store or GraphStore()
        self.vector_store = vector_store or VectorStore()
        self.indexer = Indexer()
        self.retriever = Retriever(self.graph_store, self.vector_store)
        self.reranker = Reranker(api_key=self.api_key)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    # ------------------------------------------------------------------ #
    #  Indexing                                                            #
    # ------------------------------------------------------------------ #

    def index(self, clear_existing: bool = False) -> dict:
        """
        Parse the repository and build the graph + vector index.

        Args:
            clear_existing: Wipe existing index before rebuilding.

        Returns:
            Stats dict: {"nodes": int, "edges": int}
        """
        if clear_existing:
            print("🗑️  Clearing existing index...")
            self.graph_store.clear()
            self.vector_store.clear()

        print(f"🔍 Indexing repository: {self.repo_path}")
        nodes, edges = self.indexer.index_repo(self.repo_path)

        print(f"📊 Storing {len(nodes)} nodes in Neo4j + ChromaDB...")
        self.graph_store.add_nodes_batch(nodes)
        self.graph_store.add_edges_batch(edges)
        self.vector_store.add_nodes_batch(nodes)

        stats = {
            "nodes": self.graph_store.node_count(),
            "edges": self.graph_store.edge_count(),
            "vectors": self.vector_store.count(),
        }
        print(f"✅ Index complete: {stats}")
        return stats

    # ------------------------------------------------------------------ #
    #  Querying                                                            #
    # ------------------------------------------------------------------ #

    def ask(
        self,
        query: str,
        use_reranker: bool = True,
        verbose: bool = False,
    ) -> str:
        """
        Ask a question about the codebase.

        Args:
            query:        Natural language question.
            use_reranker: Whether to apply LLM reranking (Layer 3).
            verbose:      Print retrieval diagnostics.

        Returns:
            Answer string from Claude.
        """
        # ── Retrieve ─────────────────────────────────────────────────
        print(f"\n🔎 Retrieving context for: {query!r}")
        candidates = self.retriever.retrieve(query, verbose=verbose)
        print(f"   Retrieved {len(candidates)} candidate nodes")

        # ── Rerank ───────────────────────────────────────────────────
        if use_reranker and candidates:
            print("   Running LLM reranker...")
            final_nodes, reasoning = self.reranker.rerank(query, candidates, verbose=verbose)
            print(f"   Reranker kept {len(final_nodes)} nodes")
        else:
            final_nodes = candidates
            reasoning = ""

        if not final_nodes:
            return "I couldn't find relevant code context for that question. Try re-indexing or rephrasing."

        # ── Format context ───────────────────────────────────────────
        context = self.retriever.format_context(final_nodes)

        # ── Generate answer ──────────────────────────────────────────
        print("   Generating answer...")
        user_message = f"""Question: {query}

Relevant code context:
{context}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=ANSWER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

    def ask_with_decomposition(
        self,
        query: str,
        verbose: bool = False,
    ) -> str:
        """
        Advanced: automatically decompose the query into sub-questions,
        retrieve for each, then answer holistically. Best for complex
        multi-hop questions.
        """
        sub_queries = self._decompose_query(query)
        if verbose:
            print(f"\n🔀 Decomposed into {len(sub_queries)} sub-queries:")
            for sq in sub_queries:
                print(f"   • {sq}")

        candidates = self.retriever.retrieve_with_decomposition(
            query, sub_queries, verbose=verbose
        )
        final_nodes, _ = self.reranker.rerank(query, candidates, verbose=verbose)
        context = self.retriever.format_context(final_nodes)

        user_message = f"""Question: {query}

Relevant code context:
{context}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=ANSWER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _decompose_query(self, query: str) -> list[str]:
        """Use Claude to decompose a complex question into sub-questions."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    f"Decompose this code question into 2-4 specific sub-questions "
                    f"that together answer the main question. "
                    f"Return ONLY a JSON array of strings, no other text.\n\n"
                    f"Question: {query}"
                ),
            }],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            sub_queries = json.loads(raw.strip())
            return sub_queries if isinstance(sub_queries, list) else [query]
        except Exception:
            return [query]

    # ------------------------------------------------------------------ #
    #  Comparison helper (for the demo notebook)                          #
    # ------------------------------------------------------------------ #

    def compare_with_naive_rag(self, query: str) -> dict:
        """
        Run both naive RAG (vector only) and GAC-RAG (full pipeline),
        return both answers + context stats for comparison.
        """
        # Naive RAG: vector search only, no graph expansion, no reranker
        anchor_hits = self.retriever._layer1_semantic_anchor(query)
        naive_node_ids = [node_id for node_id, _ in anchor_hits]
        naive_nodes = [self.graph_store.get_node(nid) for nid in naive_node_ids]
        naive_nodes = [n for n in naive_nodes if n]
        naive_context = self.retriever.format_context(naive_nodes)

        naive_response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=ANSWER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Question: {query}\n\nContext:\n{naive_context}"}],
        )

        # GAC-RAG: full pipeline
        gac_answer = self.ask(query, use_reranker=True)

        return {
            "query": query,
            "naive_rag": {
                "answer": naive_response.content[0].text,
                "nodes_retrieved": len(naive_nodes),
                "context_preview": naive_context[:500],
            },
            "gac_rag": {
                "answer": gac_answer,
                "nodes_retrieved": len(anchor_hits),
            },
        }
