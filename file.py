import os

# Structure interne seulement
structure = {
    "src": [
        "indexer.py",
        "graph_store.py",
        "vector_store.py",
        "retriever.py",
        "reranker.py",
        "assistant.py",
    ],
    "notebooks": ["demo.ipynb"],
    "tests": [
        "test_indexer.py",
        "test_retriever.py",
        "test_reranker.py",
    ],
    "sample_repo": {
        "services": [],
        "models": [],
        "utils": [],
    },
    "docs": ["architecture.md"],
    "files": [
        "requirements.txt",
        ".env.example",
        "README.md",
    ],
}


def create_structure(base_path, tree):
    for name, content in tree.items():
        current_path = os.path.join(base_path, name)

        if isinstance(content, dict):
            os.makedirs(current_path, exist_ok=True)
            create_structure(current_path, content)

        elif isinstance(content, list):
            os.makedirs(current_path, exist_ok=True)
            for file in content:
                file_path = os.path.join(current_path, file)
                if not os.path.exists(file_path):
                    with open(file_path, "w") as f:
                        f.write("")


def create_files(base_path, files):
    for file in files:
        file_path = os.path.join(base_path, file)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("")


if __name__ == "__main__":
    base = os.getcwd()

    create_structure(base, structure)
    create_files(base, structure["files"])

    print("✅ Structure ajoutée avec succès sans recréer le dossier racine!")