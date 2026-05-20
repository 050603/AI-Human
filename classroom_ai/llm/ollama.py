from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request

from classroom_ai.llm.base import BaseLLM, LLMResponse


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3",
        timeout_seconds: int = 300,
        max_retries: int = 6,
        retry_backoff_base_seconds: float = 1.5,
        retry_jitter_seconds: float = 0.5,
        keep_alive: str = "30m",
        options: dict | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_base_seconds = retry_backoff_base_seconds
        self.retry_jitter_seconds = retry_jitter_seconds
        self.keep_alive = keep_alive
        self.options = options or {}

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": self.options,
            }
        ).encode("utf-8")

        url = f"{self.host}/api/chat"
        last_error = None
        retryable_http_codes = {408, 429, 500, 502, 503, 504}

        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return LLMResponse(text=payload["message"]["content"], raw=payload)
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 404:
                    detail = e.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Ollama model not found or endpoint missing (HTTP 404): model={self.model}, host={self.host}, detail={detail[:240]}") from e
                if e.code in retryable_http_codes:
                    wait_s = self.retry_backoff_base_seconds * (2 ** attempt) + random.uniform(0, self.retry_jitter_seconds)
                    time.sleep(wait_s)
                    continue
                raise
            except (TimeoutError, socket.timeout, ConnectionError, urllib.error.URLError) as e:
                last_error = e
                wait_s = self.retry_backoff_base_seconds * (2 ** attempt) + random.uniform(0, self.retry_jitter_seconds)
                time.sleep(wait_s)

        raise RuntimeError(f"Ollama request failed after {self.max_retries} retries: {last_error}")
