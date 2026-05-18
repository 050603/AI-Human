from __future__ import annotations

from abc import ABC, abstractmethod

Vector = list[float]


class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str], normalize: bool = True) -> list[Vector]:
        raise NotImplementedError
