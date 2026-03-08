"""OpenAI API loader — same generate/prompt interface as HFLoader."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from .base import BaseModelLoader


class OpenAILoader(BaseModelLoader):
    """
    Call OpenAI chat API with the same interface as HFLoader.
    No local model; uses OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Args:
            model: OpenAI model id (e.g. "gpt-4o-mini", "gpt-4o").
            api_key: Override OPENAI_API_KEY env var.
            base_url: Optional base URL (e.g. for Azure or a proxy endpoint).
        """
        self.model = model
        self._client = OpenAI(api_key=api_key or None, base_url=base_url or None)

    def load(self) -> None:
        """No-op for API backends; exists so all loaders share the same interface."""

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """Call OpenAI chat completions. Ignores top_p (not forwarded by default)."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()

    def prompt(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Single prompt → API response."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.generate(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )
