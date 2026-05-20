from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request

from classroom_ai.embedding.base import BaseEmbedder, Vector


_MAX_CHARS_PER_TEXT = 5000


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text") -> None:
        self.host = host.rstrip("/")
        self.model = model

    def encode(self, texts: list[str], normalize: bool = True) -> list[Vector]:
        truncated = [t[:_MAX_CHARS_PER_TEXT] if len(t) > _MAX_CHARS_PER_TEXT else t for t in texts]
        body = json.dumps(
            {
                "model": self.model,
                "input": truncated,
                "keep_alive": "10m",
            }
        ).encode("utf-8")

        url = f"{self.host}/api/embed"
        last_error = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 400 and "context length" in str(e.read().decode("utf-8", errors="replace")).lower():
                    truncated = [t[: _MAX_CHARS_PER_TEXT // 2] if len(t) > _MAX_CHARS_PER_TEXT // 2 else t for t in texts]
                    body = json.dumps({"model": self.model, "input": truncated, "keep_alive": "10m"}).encode("utf-8")
                    time.sleep(1 * (attempt + 1))
                    continue
                if e.code == 404:
                    detail = e.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Ollama embedding model not found or endpoint missing (HTTP 404): model={self.model}, host={self.host}, detail={detail[:240]}") from e
                raise
            except Exception:
                last_error = None
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Ollama embed request failed after 3 retries: {last_error}")

        vectors = [[float(value) for value in embedding] for embedding in payload["embeddings"]]
        if normalize:
            normalized = []
            for vector in vectors:
                norm = math.sqrt(sum(value * value for value in vector))
                normalized.append([value / norm for value in vector] if norm > 0 else vector)
            vectors = normalized
        return vectors