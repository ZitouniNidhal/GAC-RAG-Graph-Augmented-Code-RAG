# GAC-RAG Architecture

## Overview

GAC-RAG introduces a **3-layer retrieval pipeline** that replaces pure vector similarity search with dependency-graph-aware retrieval for codebases.

## Why Code Needs Graph-Aware RAG

Standard RAG treats code as a collection of text chunks. But code has **structural properties** that text doesn't:

- Functions call other functions → **call graph**
- Classes inherit from other classes → **inheritance graph**
- Modules import other modules → **dependency graph**

When answering questions like *"Why does X fail?"* or *"How does data flow from A to B?"*, the relevant context spans **multiple files and call chains** — not just the single most similar chunk.

## The 3-Layer Pipeline

### Layer 1: Semantic Anchor

**Backend**: ChromaDB + SentenceTransformer embeddings

Each code node (function, class, module) is embedded as:
```
[kind] name | file: path | doc: docstring | code: source_preview
```

At query time, cosine similarity retrieves the top-k most semantically similar nodes. These are the **entry points** into the dependency graph.

### Layer 2: Graph Expansion ★

**Backend**: Neo4j

From each anchor node, we traverse the code dependency graph up to N hops. Each hop applies a **relevance decay factor** (default: 0.8):

```
score(node at hop h) = anchor_score × decay^h
```

**Edge types traversed:**
| Edge | Meaning |
|---|---|
| CALLS | fn A calls fn B |
| IMPORTS | module A imports module B |
| INHERITS | class A extends class B |
| CONTAINS | module/class contains fn/class |

### Layer 3: LLM Reranker

**Backend**: Claude (claude-3-5-sonnet) or GPT-4 (via OpenAI)

The expanded node list (potentially 20–50 nodes) is passed to the selected LLM with the original query. The LLM returns only the node IDs that are **necessary** to answer the question, with reasoning.

This prevents context window bloat and removes noise introduced by graph traversal.

## Data Model

### CodeNode
```python
@dataclass
class CodeNode:
    id: str          # "module::ClassName::method_name"
    name: str
    kind: str        # "function" | "class" | "module"
    file_path: str
    start_line: int
    end_line: int
    source_code: str
    docstring: str
    language: str
    score: float     # relevance score
    hop: int         # distance from anchor
```

### CodeEdge
```python
@dataclass
class CodeEdge:
    source_id: str
    target_id: str
    kind: str        # "CALLS" | "IMPORTS" | "INHERITS" | "CONTAINS"
    weight: float
```

## Indexing Pipeline

```
Repository
    │
    ▼
Indexer.index_repo()
    │
    ├── PythonParser (ast-based)
    │   ├── Extract functions, classes, modules
    │   ├── Extract call edges (ast.Call nodes)
    │   ├── Extract inheritance edges (ast.ClassDef.bases)
    │   └── Extract import edges (ast.Import / ast.ImportFrom)
    │
    ├── TreeSitterParser (tree-sitter based) ★
    │   ├── Multi-language support (JS/TS, Java, Go, etc.)
    │   ├── Robust syntax tree traversal
    │   └── Precise node and edge extraction
    │
    └── GenericParser (regex-based, fallback)
        ├── Extract function/class names
        └── Minimal edge extraction
    │
    ▼
CodeNode list + CodeEdge list
    │
    ├── → GraphStore.add_nodes_batch()   → Neo4j
    └── → VectorStore.add_nodes_batch()  → ChromaDB
```

## Query Pipeline

```
User Query (CLI or API)
    │
    ▼
VectorStore.search(query, top_k=5)
→ [(node_id, similarity_score), ...]
    │
    ▼
GraphStore.get_neighbors(node_id, hops=2, decay=0.8)
→ [CodeNode, ...] with decayed scores
    │
    ▼
Reranker.rerank(query, expanded_nodes)
→ [CodeNode, ...] pruned to relevant only
    │
    ▼
Retriever.format_context(final_nodes)
→ context string
    │
    ▼
LLM (Anthropic/OpenAI).messages.create(query + context) ★
→ Final answer (Streaming support included)
```

## Features Summary

- 🕸️ **Graph-Augmented Retrieval**: Uses dependency graphs to find relevant context that pure vector search misses.
- 🌳 **Tree-Sitter Indexing**: Powerful, multi-language parsing for accurate code structure extraction.
- 🤖 **Multi-LLM Support**: Seamlessly switch between Anthropic (Claude) and OpenAI (GPT-4).
- 📺 **Interactive CLI**: Rich terminal interface for indexing and querying.
- 🌊 **Streaming Responses**: Real-time feedback for long-form answers.
- 📊 **Graph Visualization**: Tools to visualize the retrieved code subgraph.

## Configuration

All parameters are configurable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `MAX_HOPS` | 2 | Graph traversal depth |
| `TOP_K_ANCHOR` | 5 | Semantic search top-k |
| `RELEVANCE_DECAY` | 0.8 | Score decay per hop |
| `MAX_CONTEXT_NODES` | 20 | Max nodes sent to LLM |
| `LLM_PROVIDER` | anthropic | `anthropic` or `openai` |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | SentenceTransformer model |
