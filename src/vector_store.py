"""
vector_store.py
ChromaDB interface for semantic search over code embeddings.
"""

from __future__ import annotations
import os
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from src.graph_store import CodeNode

load_dotenv()


class VectorStore:
    """
    ChromaDB-backed semantic search for code nodes.

    Embeds: name + docstring + source_code (truncated)
    Returns: top-k CodeNode IDs ordered by similarity
    """

    COLLECTION_NAME = "gac_rag_code_nodes"

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def add_node(self, node: CodeNode) -> None:
        """Embed and store a single code node."""
        text = self._node_to_text(node)
        self.collection.upsert(
            ids=[node.id],
            documents=[text],
            metadatas=[
                {
                    "name": node.name,
                    "kind": node.kind,
                    "file_path": node.file_path,
                    "start_line": node.start_line,
                    "end_line": node.end_line,
                    "language": node.language,
                }
            ],
        )

    def add_nodes_batch(self, nodes: list[CodeNode]) -> None:
        """Batch embed and store nodes (faster than one-by-one)."""
        if not nodes:
            return
        ids = [n.id for n in nodes]
        docs = [self._node_to_text(n) for n in nodes]
        metas = [
            {
                "name": n.name,
                "kind": n.kind,
                "file_path": n.file_path,
                "start_line": n.start_line,
                "end_line": n.end_line,
                "language": n.language,
            }
            for n in nodes
        ]
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Semantic search over all code nodes.

        Returns:
            List of (node_id, similarity_score) tuples, best first.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["distances", "metadatas"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for node_id, distance in zip(results["ids"][0], results["distances"][0]):
            similarity = 1.0 - distance          # cosine distance → similarity
            output.append((node_id, round(similarity, 4)))

        return output

    def clear(self) -> None:
        """Delete all stored embeddings."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _node_to_text(node: CodeNode) -> str:
        """
        Combine metadata + source code into a single embedding document.
        Truncate source to 512 chars to keep embeddings stable.
        """
        source_preview = node.source_code[:512].replace("\n", " ")
        parts = [
            f"[{node.kind}] {node.name}",
            f"file: {node.file_path}",
        ]
        if node.docstring:
            parts.append(f"doc: {node.docstring[:200]}")
        parts.append(f"code: {source_preview}")
        return " | ".join(parts)