
import os
from typing import Optional, Literal
import anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Summarizer:
    """
    Uses a fast LLM to generate high-level summaries for code snippets.
    This improves retrieval by providing more descriptive text for embeddings.
    """

    def __init__(
        self,
        provider: Literal["anthropic", "openai"] = "openai",
        model: Optional[str] = None
    ):
        self.provider = provider
        if provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            self.model = model or "claude-3-haiku-20240307"
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.model = model or "gpt-3.5-turbo"
            self.client = OpenAI(api_key=self.api_key)

    def summarize(self, name: str, kind: str, code: str) -> str:
        """Generate a one-sentence summary of what this code does."""
        prompt = f"Summarize what this {kind} '{name}' does in one concise sentence. Focus on its purpose in the codebase.\n\nCode:\n{code[:1000]}"
        
        try:
            if self.provider == "anthropic":
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.content[0].text.strip()
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100
                )
                return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Code entity: {name}"
