from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request

from classroom_ai.embedding.base import BaseEmbedder, Vector


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text") -> None:
        self.host = host.rstrip("/")
        self.model = model

    def encode(self, texts: list[str], normalize: bool = True) -> list[Vector]:
        body = json.dumps(
            {
                "model": self.model,
                "input": texts,
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
                if e.code == 404:
                    time.sleep(2 * (attempt + 1))
                    continue
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