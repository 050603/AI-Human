from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    raw: dict | None = None


class BaseLLM(ABC):
    """Provider-neutral LLM interface used by the evaluation pipeline."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        raise NotImplementedError
