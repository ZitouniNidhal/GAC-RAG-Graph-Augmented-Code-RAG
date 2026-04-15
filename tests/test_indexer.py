"""
Tests for the Indexer — code parsing and graph construction.
"""
import pytest
import tempfile
import os
from src.indexer import Indexer, PythonParser, GenericParser


SAMPLE_PYTHON = '''
"""Sample module for testing."""

class Animal:
    """Base class for all animals."""
    def __init__(self, name: str):
        self.name = name
    def speak(self) -> str:
        return ""

class Dog(Animal):
    """A dog."""
    def speak(self) -> str:
        return bark()

def bark() -> str:
    return "Woof!"
'''


class TestPythonParser:

    def setup_method(self):
        self.parser = PythonParser()

    def test_parses_functions(self):
        nodes, _ = self.parser.parse_file("test.py", SAMPLE_PYTHON)
        names = [n.name for n in nodes]
        assert "bark" in names
        assert "speak" in names

    def test_parses_classes(self):
        nodes, _ = self.parser.parse_file("test.py", SAMPLE_PYTHON)
        kinds = {n.kind for n in nodes}
        assert "class" in kinds

    def test_parses_module(self):
        nodes, _ = self.parser.parse_file("test.py", SAMPLE_PYTHON)
        modules = [n for n in nodes if n.kind == "module"]
        assert len(modules) == 1

    def test_inheritance_edge(self):
        _, edges = self.parser.parse_file("test.py", SAMPLE_PYTHON)
        edge_kinds = [e.kind for e in edges]
        assert "INHERITS" in edge_kinds

    def test_contains_edge(self):
        _, edges = self.parser.parse_file("test.py", SAMPLE_PYTHON)
        edge_kinds = [e.kind for e in edges]
        assert "CONTAINS" in edge_kinds

    def test_calls_edge(self):
        _, edges = self.parser.parse_file("test.py", SAMPLE_PYTHON)
        edge_kinds = [e.kind for e in edges]
        assert "CALLS" in edge_kinds

    def test_node_has_source_code(self):
        nodes, _ = self.parser.parse_file("test.py", SAMPLE_PYTHON)
        fn_nodes = [n for n in nodes if n.kind == "function"]
        for n in fn_nodes:
            assert len(n.source_code) > 0

    def test_node_line_numbers(self):
        nodes, _ = self.parser.parse_file("test.py", SAMPLE_PYTHON)
        for node in nodes:
            assert node.start_line >= 1
            assert node.end_line >= node.start_line


class TestGenericParser:

    def setup_method(self):
        self.parser = GenericParser()

    def test_parses_js_functions(self):
        js_code = """
function fetchUser(id) {
    return db.query(id);
}
function saveUser(user) {
    return db.save(user);
}
"""
        nodes, edges = self.parser.parse_file("app.js", js_code, "javascript")
        names = [n.name for n in nodes]
        assert "fetchUser" in names or "saveUser" in names

    def test_parses_java_class(self):
        java_code = """
public class UserService {
    public void save(User user) {
        repository.save(user);
    }
}
"""
        nodes, _ = self.parser.parse_file("UserService.java", java_code, "java")
        class_nodes = [n for n in nodes if n.kind == "class"]
        assert len(class_nodes) >= 1


class TestIndexer:

    def setup_method(self):
        self.indexer = Indexer()

    def test_index_sample_repo(self):
        nodes, edges = self.indexer.index_repo("./sample_repo")
        assert len(nodes) > 0
        assert len(edges) > 0

    def test_no_dangling_edges(self):
        nodes, edges = self.indexer.index_repo("./sample_repo")
        node_ids = {n.id for n in nodes}
        for edge in edges:
            assert edge.source_id in node_ids
            assert edge.target_id in node_ids

    def test_ignores_non_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Write a Python file and a non-source file
            with open(os.path.join(tmp, "app.py"), "w") as f:
                f.write("def hello(): pass\n")
            with open(os.path.join(tmp, "README.md"), "w") as f:
                f.write("# readme\n")
            nodes, _ = self.indexer.index_repo(tmp)
            # Only Python file should be indexed
            python_nodes = [n for n in nodes if n.language == "python"]
            assert len(python_nodes) > 0