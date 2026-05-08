
import os
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

from src.assistant import CodeAssistant

load_dotenv()

console = Console()

def main():
    parser = argparse.ArgumentParser(description="GAC-RAG: Graph-Augmented Code Assistant")
    parser.add_argument("--repo", type=str, default=".", help="Path to the repository to index")
    parser.add_argument("--provider", type=str, choices=["anthropic", "openai"], default="anthropic", help="LLM provider")
    parser.add_argument("--index", action="store_true", help="Re-index the repository")
    parser.add_argument("--query", type=str, help="Initial query to ask")
    
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY") and args.provider == "anthropic":
        console.print("[red]Error: ANTHROPIC_API_KEY not found in .env[/red]")
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY") and args.provider == "openai":
        console.print("[red]Error: OPENAI_API_KEY not found in .env[/red]")
        sys.exit(1)

    assistant = CodeAssistant(repo_path=args.repo, llm_provider=args.provider)

    if args.index:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Indexing repository...", total=None)
            assistant.index(clear_existing=True)
        console.print("[green]✅ Repository indexed successfully![/green]")

    if args.query:
        run_query(assistant, args.query)
    else:
        interactive_mode(assistant)

def run_query(assistant: CodeAssistant, query: str):
    console.print(Panel(f"[bold blue]Query:[/bold blue] {query}"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="GAC-RAG is thinking...", total=None)
        
        # We use the stream_ask to show progress
        response_text = ""
        console.print("\n[bold green]Assistant:[/bold green]")
        for chunk in assistant.stream_ask(query):
            response_text += chunk
            console.print(chunk, end="")
        console.print("\n")

def interactive_mode(assistant: CodeAssistant):
    console.print(Panel.fit(
        "[bold cyan]Welcome to GAC-RAG CLI![/bold cyan]\n"
        "Ask questions about your codebase. Type 'exit' to quit.",
        title="GAC-RAG"
    ))
    
    while True:
        try:
            query = console.input("[bold yellow]>>> [/bold yellow]")
            if query.lower() in ["exit", "quit", "q"]:
                break
            if not query.strip():
                continue
                
            run_query(assistant, query)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
