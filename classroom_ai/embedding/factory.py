from __future__ import annotations

import os
from typing import Any

from classroom_ai.embedding.base import BaseEmbedder
from classroom_ai.embedding.hashing import HashingEmbedder
from classroom_ai.embedding.ollama import OllamaEmbedder
from classroom_ai.embedding.openai_compatible import OpenAICompatibleEmbedder


def build_embedder(embedding_config: dict[str, Any], env: dict[str, str] | None = None) -> BaseEmbedder:
    env_map = env or os.environ
    provider = embedding_config["provider"]

    if provider == "hashing":
        return HashingEmbedder(dimensions=int(embedding_config.get("dimensions", 256)))
    if provider in {"ollama", "local_ollama"}:
        return OllamaEmbedder(host=embedding_config.get("host", "http://localhost:11434"), model=embedding_config["model"])
    if provider in {"openai_compatible", "litellm"}:
        api_key_env = embedding_config.get("api_key_env")
        api_key = embedding_config.get("api_key") or (env_map.get(api_key_env, "") if api_key_env else "")
        if not api_key:
            raise ValueError(f"Missing API key for embedding provider={provider}; set embedding.api_key or env {api_key_env}")
        return OpenAICompatibleEmbedder(api_base=embedding_config["api_base"], api_key=api_key, model=embedding_config["model"])
    raise ValueError(f"Unsupported embedding provider: {provider}")
