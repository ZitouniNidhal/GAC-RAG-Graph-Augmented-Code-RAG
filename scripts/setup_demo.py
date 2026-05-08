
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.assistant import CodeAssistant

def setup_demo():
    print("Starting GAC-RAG Demo Setup...")
    
    # Path to the sample repository
    repo_path = "./sample_repo"
    
    # Check if sample repo exists, if not create a dummy one
    if not os.path.exists(repo_path):
        print(f"Creating sample repository at {repo_path}...")
        os.makedirs(os.path.join(repo_path, "src"), exist_ok=True)
        with open(os.path.join(repo_path, "src", "main.py"), "w") as f:
            f.write("def hello_world():\n    print('Hello GAC-RAG!')\n\ndef run():\n    hello_world()")
        with open(os.path.join(repo_path, "src", "utils.py"), "w") as f:
            f.write("def helper():\n    return 'I help main.py'")
            
    # Initialize assistant
    print("Initializing Assistant...")
    assistant = CodeAssistant(repo_path=repo_path)
    
    # Start Indexing
    print("Indexing repository (this may take a minute)...")
    try:
        stats = assistant.index(clear_existing=True)
        print(f"Indexing complete! Stats: {stats}")
        
        print("\nRunning a test query...")
        response = assistant.ask("What functions are in main.py?")
        print(f"\nAssistant Response:\n{response}")
        
    except Exception as e:
        print(f"Error during indexing: {e}")
        print("\nNote: Make sure Neo4j is running at bolt://localhost:7687")

if __name__ == "__main__":
    setup_demo()
