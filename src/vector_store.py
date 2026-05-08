
from __future__ import annotations
import os
from typing import Optional, Any
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from src.graph_store import CodeNode

load_dotenv()


class VectorStore:
    """
    ChromaDB-backed semantic search for code nodes.
    Supports both local (SentenceTransformer) and remote (OpenAI) embeddings.
    """

    COLLECTION_NAME = "gac_rag_code_nodes"

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        provider: str = "local",
        embedding_model: Optional[str] = None,
    ):
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            model_name = embedding_model or "text-embedding-3-small"
            self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=model_name
            )
        else:
            model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

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
                    "language": node.language,
                }
            ],
        )

    def add_nodes_batch(self, nodes: list[CodeNode]) -> None:
        """Batch embed and store nodes."""
        if not nodes:
            return
        ids = [n.id for n in nodes]
        docs = [self._node_to_text(n) for n in nodes]
        metas = [
            {
                "name": n.name,
                "kind": n.kind,
                "file_path": n.file_path,
                "language": n.language,
            }
            for n in nodes
        ]
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def search(
        self, 
        query: str, 
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None
    ) -> list[tuple[str, float]]:
        """
        Semantic search with optional metadata filtering.
        
        Example where: {"language": "python"} or {"kind": "class"}
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["distances"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for node_id, distance in zip(results["ids"][0], results["distances"][0]):
            similarity = 1.0 - distance
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

    @staticmethod
    def _node_to_text(node: CodeNode) -> str:
        source_preview = node.source_code[:512].replace("\n", " ")
        parts = [
            f"[{node.kind}] {node.name} in {node.language}",
            f"file: {node.file_path}",
        ]
        if node.docstring:
            parts.append(f"doc: {node.docstring[:200]}")
        parts.append(f"code: {source_preview}")
        return " | ".join(parts)
