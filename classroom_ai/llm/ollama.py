from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from classroom_ai.llm.base import BaseLLM, LLMResponse


class OllamaLLM(BaseLLM):
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3") -> None:
        self.host = host.rstrip("/")
        self.model = model

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
                "keep_alive": "10m",
            }
        ).encode("utf-8")

        url = f"{self.host}/api/chat"
        last_error = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=300) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return LLMResponse(text=payload["message"]["content"], raw=payload)
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 404:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
            except Exception:
                last_error = None
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Ollama request failed after 3 retries: {last_error}")