import os
import json
from typing import Optional, Literal, List, Tuple
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
from src.graph_store import CodeNode

load_dotenv()

RERANKER_SYSTEM_PROMPT = """You are a code relevance judge for a RAG system.

You will receive:
1. A user question about a codebase
2. A list of code nodes (functions, classes, modules) retrieved from the codebase

Your task: Identify which nodes are NECESSARY to answer the question.
A node is necessary if:
- It directly answers the question
- It is a dependency of a relevant node (called, imported, inherited)
- Understanding it is required for the answer to make sense

Respond ONLY with a JSON object in this exact format:
{
  "relevant_ids": ["node_id_1", "node_id_2", ...],
  "reasoning": "Brief explanation of why these nodes matter"
}

Do not include any text outside the JSON object.
"""


class Reranker:
    """
    LLM-powered reranker that filters the expanded node list
    down to only the most relevant nodes for the query.

    This is Layer 3 of the GAC-RAG pipeline.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        provider: Literal["anthropic", "openai"] = "anthropic",
        model: Optional[str] = None
    ):
        self.provider = provider
        if provider == "anthropic":
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.model = model or "claude-3-5-sonnet-20240620"
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or "gpt-4-turbo"
            self.client = OpenAI(api_key=self.api_key)

    def rerank(
        self,
        query: str,
        nodes: list[CodeNode],
        verbose: bool = False,
    ) -> tuple[list[CodeNode], str]:
        """
        Filter nodes to only the ones relevant for the query.
        """
        if not nodes:
            return [], "No nodes to rerank."

        node_summaries = self._build_node_summaries(nodes)

        prompt = f"""Question: {query}

Code nodes to evaluate:
{node_summaries}

Which of these nodes are necessary to answer the question?"""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    system=RERANKER_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": RERANKER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                )
                raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            relevant_ids = set(parsed.get("relevant_ids", []))
            reasoning = parsed.get("reasoning", "")

            filtered = [n for n in nodes if n.id in relevant_ids]

            if verbose:
                print(f"\n[Layer 3 - Reranker] Kept {len(filtered)}/{len(nodes)} nodes")
                print(f"  Reasoning: {reasoning}")

            if not filtered:
                filtered = sorted(nodes, key=lambda n: n.score, reverse=True)[:5]
                reasoning = "Fallback: reranker returned empty — using top-5 by score."

            return filtered, reasoning

        except (json.JSONDecodeError, Exception) as e:
            fallback = sorted(nodes, key=lambda n: n.score, reverse=True)[:8]
            return fallback, f"Reranker error ({e}), using top-8 by score."

    @staticmethod
    def _build_node_summaries(nodes: list[CodeNode]) -> str:
        lines = []
        for node in nodes:
            doc = f" — {node.docstring[:80]}" if node.docstring else ""
            lines.append(
                f"- id: {node.id}\n"
                f"  type: {node.kind} | file: {node.file_path} | hop: {node.hop}{doc}"
            )
        return "\n".join(lines)
