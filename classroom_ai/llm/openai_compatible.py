from __future__ import annotations

import json
import urllib.request

from classroom_ai.llm.base import BaseLLM, LLMResponse


class OpenAICompatibleLLM(BaseLLM):
    """Minimal OpenAI-compatible chat completions adapter for later API mode."""

    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return LLMResponse(text=payload["choices"][0]["message"]["content"], raw=payload)
