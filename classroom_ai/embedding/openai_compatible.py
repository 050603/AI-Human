from __future__ import annotations

import json
import math
import urllib.request

from classroom_ai.embedding.base import BaseEmbedder, Vector


class OpenAICompatibleEmbedder(BaseEmbedder):
    """Minimal OpenAI-compatible embeddings adapter for later API mode."""

    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def encode(self, texts: list[str], normalize: bool = True) -> list[Vector]:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        vectors = [[float(value) for value in item["embedding"]] for item in payload["data"]]
        if normalize:
            normalized = []
            for vector in vectors:
                norm = math.sqrt(sum(value * value for value in vector))
                normalized.append([value / norm for value in vector] if norm > 0 else vector)
            vectors = normalized
        return vectors
