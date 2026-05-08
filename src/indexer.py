"""
indexer.py
Parses source code repositories into CodeNode + CodeEdge objects
using tree-sitter for multi-language support.

Supported: Python, JavaScript/TypeScript, Java, Go, C/C++
Falls back to a simple regex parser for unknown languages.
"""

from __future__ import annotations
import os
import re
import ast
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from src.graph_store import CodeNode, CodeEdge
from src.summarizer import Summarizer


# ------------------------------------------------------------------ #
#  Language detection                                                  #
# ------------------------------------------------------------------ #

EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".rb": "ruby",
    ".rs": "rust",
    ".php": "php",
}

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".idea", ".vscode",
}


# ------------------------------------------------------------------ #
#  Python-specific parser (AST-based, most accurate)                 #
# ------------------------------------------------------------------ #

class PythonParser:
    """Extracts nodes and edges from Python source using the `ast` module."""

    def parse_file(self, file_path: str, source: str) -> tuple[list[CodeNode], list[CodeEdge]]:
        nodes: list[CodeNode] = []
        edges: list[CodeEdge] = []
        module_id = self._module_id(file_path)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return nodes, edges

        # Module node
        module_node = CodeNode(
            id=module_id,
            name=Path(file_path).stem,
            kind="module",
            file_path=file_path,
            start_line=1,
            end_line=source.count("\n") + 1,
            source_code=source[:300],
            language="python",
        )
        nodes.append(module_node)

        lines = source.splitlines()

        for item in ast.walk(tree):
            # Functions
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_id = f"{module_id}::{item.name}"
                docstring = ast.get_docstring(item) or ""
                fn_source = self._extract_lines(lines, item.lineno, item.end_lineno)
                fn_node = CodeNode(
                    id=fn_id,
                    name=item.name,
                    kind="function",
                    file_path=file_path,
                    start_line=item.lineno,
                    end_line=item.end_lineno,
                    source_code=fn_source,
                    docstring=docstring,
                    language="python",
                )
                nodes.append(fn_node)

                # Module CONTAINS function
                edges.append(CodeEdge(source_id=module_id, target_id=fn_id, kind="CONTAINS"))

                # Call edges
                for call in self._extract_calls(item):
                    call_target = f"{module_id}::{call}"
                    edges.append(CodeEdge(source_id=fn_id, target_id=call_target, kind="CALLS"))

            # Classes
            elif isinstance(item, ast.ClassDef):
                cls_id = f"{module_id}::{item.name}"
                docstring = ast.get_docstring(item) or ""
                cls_source = self._extract_lines(lines, item.lineno, item.end_lineno)
                cls_node = CodeNode(
                    id=cls_id,
                    name=item.name,
                    kind="class",
                    file_path=file_path,
                    start_line=item.lineno,
                    end_line=item.end_lineno,
                    source_code=cls_source[:500],
                    docstring=docstring,
                    language="python",
                )
                nodes.append(cls_node)
                edges.append(CodeEdge(source_id=module_id, target_id=cls_id, kind="CONTAINS"))

                # Inheritance edges
                for base in item.bases:
                    base_name = ast.unparse(base) if hasattr(ast, "unparse") else getattr(base, "id", None)
                    if base_name:
                        base_id = f"{module_id}::{base_name}"
                        edges.append(CodeEdge(source_id=cls_id, target_id=base_id, kind="INHERITS"))

                # Methods
                for method in [n for n in ast.walk(item) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                    m_id = f"{cls_id}::{method.name}"
                    m_doc = ast.get_docstring(method) or ""
                    m_source = self._extract_lines(lines, method.lineno, method.end_lineno)
                    m_node = CodeNode(
                        id=m_id,
                        name=method.name,
                        kind="function",
                        file_path=file_path,
                        start_line=method.lineno,
                        end_line=method.end_lineno,
                        source_code=m_source,
                        docstring=m_doc,
                        language="python",
                    )
                    nodes.append(m_node)
                    edges.append(CodeEdge(source_id=cls_id, target_id=m_id, kind="CONTAINS"))

                    for call in self._extract_calls(method):
                        call_target = f"{module_id}::{call}"
                        edges.append(CodeEdge(source_id=m_id, target_id=call_target, kind="CALLS"))

        # Import edges
        for item in ast.walk(tree):
            if isinstance(item, (ast.Import, ast.ImportFrom)):
                imported = self._extract_import_name(item)
                if imported:
                    imp_id = f"module::{imported}"
                    edges.append(CodeEdge(source_id=module_id, target_id=imp_id, kind="IMPORTS"))

        return nodes, edges

    def _extract_calls(self, node: ast.AST) -> list[str]:
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return calls

    def _extract_import_name(self, node) -> Optional[str]:
        if isinstance(node, ast.Import):
            return node.names[0].name if node.names else None
        elif isinstance(node, ast.ImportFrom):
            return node.module
        return None

    @staticmethod
    def _module_id(file_path: str) -> str:
        stem = Path(file_path).stem
        return f"module::{stem}"

    @staticmethod
    def _extract_lines(lines: list[str], start: int, end: int) -> str:
        return "\n".join(lines[start - 1 : end])


# ------------------------------------------------------------------ #
#  Tree-Sitter based parser (Multi-language)                         #
# ------------------------------------------------------------------ #

from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
import tree_sitter_java as tsjava

class TreeSitterParser:
    """
    Robust multi-language parser using tree-sitter.
    Extracts nodes (functions, classes) and edges (calls, imports).
    """

    def __init__(self):
        self.languages = {
            "python": Language(tspython.language()),
            "javascript": Language(tsjs.language()),
            "java": Language(tsjava.language()),
        }
        self.parser = Parser()

    def parse_file(self, file_path: str, source: str, language: str) -> tuple[list[CodeNode], list[CodeEdge]]:
        if language not in self.languages:
            return [], []

        self.parser.set_language(self.languages[language])
        tree = self.parser.parse(bytes(source, "utf8"))
        
        nodes: list[CodeNode] = []
        edges: list[CodeEdge] = []
        module_id = f"module::{Path(file_path).stem}"

        # Create module node
        module_node = CodeNode(
            id=module_id,
            name=Path(file_path).stem,
            kind="module",
            file_path=file_path,
            start_line=1,
            end_line=source.count("\n") + 1,
            source_code=source[:500],
            language=language,
        )
        nodes.append(module_node)

        # Basic queries for different languages
        queries = {
            "python": """
                (function_definition name: (identifier) @func.name) @func.def
                (class_definition name: (identifier) @class.name) @class.def
                (call function: (identifier) @call.name) @call
                (import_from_statement module_name: (dotted_name) @import.name)
            """,
            "javascript": """
                (function_declaration name: (identifier) @func.name) @func.def
                (class_declaration name: (identifier) @class.name) @class.def
                (call_expression function: (identifier) @call.name) @call
            """,
            "java": """
                (method_declaration name: (identifier) @func.name) @func.def
                (class_declaration name: (identifier) @class.name) @class.def
                (method_invocation name: (identifier) @call.name) @call
            """
        }

        if language in queries:
            query = self.languages[language].query(queries[language])
            captures = query.captures(tree.root_node)

            for node, tag in captures:
                if tag == "func.def":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        name = source[name_node.start_byte : name_node.end_byte]
                        fn_id = f"{module_id}::{name}"
                        nodes.append(CodeNode(
                            id=fn_id, name=name, kind="function",
                            file_path=file_path, 
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            source_code=source[node.start_byte : node.end_byte],
                            language=language,
                        ))
                        edges.append(CodeEdge(source_id=module_id, target_id=fn_id, kind="CONTAINS"))
                
                elif tag == "class.def":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        name = source[name_node.start_byte : name_node.end_byte]
                        cls_id = f"{module_id}::{name}"
                        nodes.append(CodeNode(
                            id=cls_id, name=name, kind="class",
                            file_path=file_path,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            source_code=source[node.start_byte : node.end_byte][:500],
                            language=language,
                        ))
                        edges.append(CodeEdge(source_id=module_id, target_id=cls_id, kind="CONTAINS"))

        return nodes, edges


# ------------------------------------------------------------------ #
#  Generic fallback parser (regex-based)                             #
# ------------------------------------------------------------------ #

class GenericParser:
    """
    Minimal regex-based parser for languages not yet covered by
    a dedicated tree-sitter parser. Extracts function/class names only.
    """

    FUNC_RE = re.compile(
        r"(?:def|function|func|fn|void|int|string|public|private|protected)?\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        re.MULTILINE,
    )
    CLASS_RE = re.compile(r"(?:class|struct|interface)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

    def parse_file(self, file_path: str, source: str, language: str) -> tuple[list[CodeNode], list[CodeEdge]]:
        nodes: list[CodeNode] = []
        edges: list[CodeEdge] = []
        lines = source.splitlines()
        module_id = f"module::{Path(file_path).stem}"

        module_node = CodeNode(
            id=module_id,
            name=Path(file_path).stem,
            kind="module",
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            source_code=source[:300],
            language=language,
        )
        nodes.append(module_node)

        for match in self.FUNC_RE.finditer(source):
            name = match.group(1)
            line_no = source[: match.start()].count("\n") + 1
            fn_id = f"{module_id}::{name}"
            nodes.append(CodeNode(
                id=fn_id, name=name, kind="function",
                file_path=file_path, start_line=line_no, end_line=line_no,
                source_code="", language=language,
            ))
            edges.append(CodeEdge(source_id=module_id, target_id=fn_id, kind="CONTAINS"))

        for match in self.CLASS_RE.finditer(source):
            name = match.group(1)
            line_no = source[: match.start()].count("\n") + 1
            cls_id = f"{module_id}::{name}"
            nodes.append(CodeNode(
                id=cls_id, name=name, kind="class",
                file_path=file_path, start_line=line_no, end_line=line_no,
                source_code="", language=language,
            ))
            edges.append(CodeEdge(source_id=module_id, target_id=cls_id, kind="CONTAINS"))

        return nodes, edges


# ------------------------------------------------------------------ #
#  Main Indexer                                                        #
# ------------------------------------------------------------------ #

class Indexer:
    """
    Walks a repository directory, parses every source file,
    and returns lists of CodeNode + CodeEdge objects ready to be
    stored in GraphStore + VectorStore.
    """

    def __init__(self, use_summarizer: bool = False):
        self.python_parser = PythonParser()
        self.ts_parser = TreeSitterParser()
        self.generic_parser = GenericParser()
        self.summarizer = Summarizer() if use_summarizer else None

    def index_repo(self, repo_path: str) -> tuple[list[CodeNode], list[CodeEdge]]:
        """
        Walk repo_path and parse all source files.

        Returns:
            (nodes, edges) — deduplicated by node ID
        """
        all_nodes: dict[str, CodeNode] = {}
        all_edges: list[CodeEdge] = []

        files = self._collect_files(repo_path)
        print(f"📂 Found {len(files)} source files to index...")

        for file_path in tqdm(files, desc="Indexing"):
            language = EXTENSION_MAP.get(Path(file_path).suffix.lower(), "unknown")
            try:
                source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if language == "python":
                nodes, edges = self.python_parser.parse_file(file_path, source)
            elif language in ["javascript", "typescript", "java"]:
                nodes, edges = self.ts_parser.parse_file(file_path, source, language)
            else:
                nodes, edges = self.generic_parser.parse_file(file_path, source, language)

            # Enrich with summaries if enabled
            if self.summarizer:
                for n in nodes:
                    if n.kind in ["function", "class"] and not n.docstring:
                        n.docstring = self.summarizer.summarize(n.name, n.kind, n.source_code)

            for n in nodes:
                all_nodes[n.id] = n
            all_edges.extend(edges)

        # Filter edges: only keep edges where both endpoints exist
        node_ids = set(all_nodes.keys())
        valid_edges = [
            e for e in all_edges
            if e.source_id in node_ids and e.target_id in node_ids
        ]

        print(f"✅ Indexed {len(all_nodes)} nodes, {len(valid_edges)} edges")
        return list(all_nodes.values()), valid_edges

    @staticmethod
    def _collect_files(repo_path: str) -> list[str]:
        result = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext in EXTENSION_MAP:
                    result.append(os.path.join(root, fname))
        return result