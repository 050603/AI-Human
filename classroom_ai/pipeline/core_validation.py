from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


from classroom_ai.embedding.hashing import HashingEmbedder
from classroom_ai.embedding.openai_compatible import OpenAICompatibleEmbedder
from classroom_ai.evaluation.parser import parse_evaluation_response
from classroom_ai.evaluation.prompts import build_class_prompt
from classroom_ai.llm.mock_local import MockLocalLLM
from classroom_ai.llm.openai_compatible import OpenAICompatibleLLM
from classroom_ai.schemas.transcript import Transcript
from classroom_ai.slicing.time_slicer import slice_transcript_by_time
from classroom_ai.uncertainty.decision import majority_vote, route_decision
from classroom_ai.uncertainty.entropy import shannon_entropy
from classroom_ai.uncertainty.semantic_entropy import compute_semantic_entropy


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a small YAML/JSON config without mandatory third-party dependencies."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the limited nested key/value YAML used by this prototype config."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, separator, value = line.strip().partition(":")
        if not separator:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"\'')


def load_transcript(path: str | Path) -> Transcript:
    with Path(path).open("r", encoding="utf-8") as handle:
        return Transcript.from_dict(json.load(handle))


def build_llm(config: dict[str, Any]):
    llm_config = config["llm"]
    provider = llm_config["provider"]
    if provider == "mock_local":
        return MockLocalLLM()
    if provider == "openai_compatible":
        import os

        return OpenAICompatibleLLM(
            api_base=llm_config["api_base"],
            api_key=os.environ[llm_config["api_key_env"]],
            model=llm_config["model"],
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def build_embedder(config: dict[str, Any]):
    embedding_config = config["embedding"]
    provider = embedding_config["provider"]
    if provider == "hashing":
        return HashingEmbedder(dimensions=int(embedding_config.get("dimensions", 256)))
    if provider == "openai_compatible":
        import os

        return OpenAICompatibleEmbedder(
            api_base=embedding_config["api_base"],
            api_key=os.environ[embedding_config["api_key_env"]],
            model=embedding_config["model"],
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def run_core_validation(transcript_path: str | Path, config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    transcript = load_transcript(transcript_path)
    llm = build_llm(config)
    embedder = build_embedder(config)

    slicing_config = config.get("slicing", {})
    slices = slice_transcript_by_time(
        transcript,
        window_seconds=int(slicing_config.get("window_seconds", 600)),
        overlap_seconds=int(slicing_config.get("overlap_seconds", 120)),
    )

    llm_config = config["llm"]
    uncertainty_config = config["uncertainty"]
    sample_count = int(llm_config.get("monte_carlo_samples", 20))

    results = []
    for transcript_slice in slices:
        messages = build_class_prompt(transcript_slice)
        parsed_samples = []
        for _ in range(sample_count):
            response = llm.generate(
                messages=messages,
                temperature=float(llm_config.get("temperature", 0.7)),
                max_tokens=int(llm_config.get("max_tokens", 2048)),
            )
            parsed_samples.append(parse_evaluation_response(response.text))

        scores = [sample["score"] for sample in parsed_samples if sample["score"]]
        reasons = [sample["reason"] for sample in parsed_samples]
        score_entropy = shannon_entropy(scores)
        semantic_entropy, labels, clusters = compute_semantic_entropy(
            reasons,
            embedder=embedder,
            similarity_threshold=float(uncertainty_config.get("similarity_threshold", 0.82)),
        )
        decision = route_decision(
            semantic_entropy=semantic_entropy,
            score_entropy=score_entropy,
            semantic_entropy_threshold=float(uncertainty_config["semantic_entropy_threshold"]),
            score_entropy_threshold=float(uncertainty_config["score_entropy_threshold"]),
        )

        results.append(
            {
                "slice_id": transcript_slice.slice_id,
                "lesson_id": transcript.lesson_id,
                "start": transcript_slice.start,
                "end": transcript_slice.end,
                "monte_carlo_samples": sample_count,
                "score_distribution": dict(sorted(Counter(scores).items())),
                "majority_score": majority_vote(scores),
                "score_entropy": score_entropy,
                "semantic_entropy": semantic_entropy,
                "semantic_cluster_labels": labels,
                "semantic_clusters": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "count": cluster.count,
                        "representative_reason": cluster.representative_text,
                        "member_indices": cluster.member_indices,
                    }
                    for cluster in clusters
                ],
                "decision": decision,
                "samples": parsed_samples,
            }
        )

    return {
        "lesson_id": transcript.lesson_id,
        "config": config,
        "slice_count": len(results),
        "results": results,
    }
