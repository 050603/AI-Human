from __future__ import annotations

import hashlib
import math
import re

from classroom_ai.embedding.base import BaseEmbedder, Vector


class HashingEmbedder(BaseEmbedder):
    """Dependency-light local embedder for offline pipeline validation.

    It hashes character n-grams into a fixed-size vector. Production deployments
    should swap this provider for a local sentence-transformer/bge/e5 model while
    preserving the same encode interface.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def encode(self, texts: list[str], normalize: bool = True) -> list[Vector]:
        matrix: list[Vector] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in self._tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                index = value % self.dimensions
                sign = 1.0 if value & 1 else -1.0
                vector[index] += sign
            if normalize:
                norm = math.sqrt(sum(value * value for value in vector))
                if norm > 0:
                    vector = [value / norm for value in vector]
            matrix.append(vector)
        return matrix

    def _tokens(self, text: str) -> list[str]:
        compact = re.sub(r"\s+", "", text.lower())
        grams = []
        for n in (2, 3, 4):
            grams.extend(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return grams + words or [compact]
