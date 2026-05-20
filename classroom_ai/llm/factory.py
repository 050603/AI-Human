from __future__ import annotations

import os
from typing import Any

from classroom_ai.llm.base import BaseLLM
from classroom_ai.llm.mock_local import MockLocalLLM
from classroom_ai.llm.ollama import OllamaLLM
from classroom_ai.llm.openai_compatible import OpenAICompatibleLLM


def build_llm(llm_config: dict[str, Any], env: dict[str, str] | None = None) -> BaseLLM:
    env_map = env or os.environ
    provider = llm_config["provider"]

    if provider == "mock_local":
        return MockLocalLLM()
    if provider in {"ollama", "local_ollama"}:
        return OllamaLLM(
            host=llm_config.get("host", "http://localhost:11434"),
            model=llm_config["model"],
            timeout_seconds=int(llm_config.get("timeout_seconds", 300)),
            max_retries=int(llm_config.get("max_retries", 6)),
            retry_backoff_base_seconds=float(llm_config.get("retry_backoff_base_seconds", 1.5)),
            retry_jitter_seconds=float(llm_config.get("retry_jitter_seconds", 0.5)),
            keep_alive=str(llm_config.get("keep_alive", "30m")),
            options=llm_config.get("options", {}),
        )
    if provider in {"openai_compatible", "litellm"}:
        api_key_env = llm_config.get("api_key_env")
        api_key = llm_config.get("api_key") or (env_map.get(api_key_env, "") if api_key_env else "")
        if not api_key:
            raise ValueError(f"Missing API key for provider={provider}; set llm.api_key or env {api_key_env}")
        return OpenAICompatibleLLM(api_base=llm_config["api_base"], api_key=api_key, model=llm_config["model"])
    raise ValueError(f"Unsupported LLM provider: {provider}")
