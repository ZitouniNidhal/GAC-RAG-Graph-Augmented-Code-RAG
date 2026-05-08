"""
assistant.py
Main orchestrator for the GAC-RAG Code Assistant.

Ties together: Indexer → GraphStore + VectorStore → Retriever → Reranker → LLM
"""

from __future__ import annotations
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Iterator, Literal
import anthropic
from openai import OpenAI
from dotenv import load_dotenv

from src.indexer import Indexer
from src.graph_store import GraphStore, CodeNode
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.reranker import Reranker

load_dotenv()

ANSWER_SYSTEM_PROMPT = """You are an expert code assistant with deep knowledge of software architecture.

You will be given:
1. A user question about a codebase
2. Relevant code context (functions, classes, modules) retrieved from the codebase

Your task: Answer the question thoroughly using ONLY the provided code context.

Guidelines:
- Reference specific function/class names and file paths when relevant
- Explain the call chain or data flow when tracing bugs or understanding behavior
- If the context is insufficient, say so clearly rather than guessing
- Format code references as `ClassName.method_name()`
"""


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
        llm_provider: Literal["anthropic", "openai"] = "anthropic",
        model: Optional[str] = None,
        graph_store: Optional[GraphStore] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.repo_path = repo_path
        self.llm_provider = llm_provider
        
        if llm_provider == "anthropic":
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.model = model or "claude-3-5-sonnet-20240620"
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or "gpt-4-turbo"
            self.client = OpenAI(api_key=self.api_key)

        self.graph_store = graph_store or GraphStore()
        self.vector_store = vector_store or VectorStore()
        self.indexer = Indexer()
        self.retriever = Retriever(self.graph_store, self.vector_store)
        self.reranker = Reranker(api_key=self.api_key, provider=llm_provider)

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

        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=ANSWER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=2000,
            )
            return response.choices[0].message.content

    def stream_ask(
        self,
        query: str,
        use_reranker: bool = True,
    ) -> Iterator[str]:
        """Stream the answer for real-time interaction."""
        candidates = self.retriever.retrieve(query)
        if use_reranker and candidates:
            final_nodes, _ = self.reranker.rerank(query, candidates)
        else:
            final_nodes = candidates

        context = self.retriever.format_context(final_nodes)
        user_message = f"Question: {query}\n\nContext:\n{context}"

        if self.llm_provider == "anthropic":
            with self.client.messages.stream(
                model=self.model,
                max_tokens=2000,
                system=ANSWER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        else:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

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

    # ------------------------------------------------------------------ #
    #  Visualization                                                       #
    # ------------------------------------------------------------------ #

    def visualize_context(self, nodes: list[CodeNode], title: str = "Retrieved Context Graph"):
        """Generate a plot of the retrieved nodes and their relationships."""
        G = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node.id, label=node.name, kind=node.kind)
        
        # Add edges (only between retrieved nodes)
        node_ids = {n.id for n in nodes}
        for node in nodes:
            # We'd need a way to get edges for these nodes. 
            # For simplicity, let's query the graph store for edges between these nodes.
            pass # Simplified for now, or we could fetch them
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        colors = {"function": "lightblue", "class": "orange", "module": "lightgreen"}
        node_colors = [colors.get(G.nodes[n]["kind"], "gray") for n in G.nodes]
        
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
                node_color=node_colors, node_size=2000, font_size=10, arrowsize=20)
        
        plt.title(title)
        plt.show()
