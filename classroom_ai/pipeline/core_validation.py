from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


from classroom_ai.embedding.factory import build_embedder as build_embedder_from_factory
from classroom_ai.evaluation.parser import parse_evaluation_response
from classroom_ai.evaluation.prompts import build_class_prompt
from classroom_ai.llm.factory import build_llm as build_llm_from_factory
from classroom_ai.schemas.transcript import Transcript
from classroom_ai.slicing.time_slicer import slice_transcript_by_time
from classroom_ai.slicing.phase_slicer import slice_transcript_by_phase
from classroom_ai.uncertainty.decision import majority_vote, route_decision
from classroom_ai.uncertainty.entropy import shannon_entropy
from classroom_ai.uncertainty.semantic_entropy import compute_semantic_entropy, EmbeddingEntailmentJudge


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
    return build_llm_from_factory(config["llm"] if "llm" in config else config)


def build_embedder(config: dict[str, Any]):
    return build_embedder_from_factory(config["embedding"] if "embedding" in config else config)


def run_core_validation(transcript_path: str | Path, config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    transcript = load_transcript(transcript_path)
    llm_config = config["llm"]
    llm = build_llm_from_factory(llm_config)

    uncertainty_config = config["uncertainty"]
    embedding_config = config.get("embedding", {})
    embedder = build_embedder_from_factory(embedding_config) if embedding_config else None
    sim_threshold = float(uncertainty_config.get("embedding_similarity_threshold", 0.75))
    semantic_judge = EmbeddingEntailmentJudge(embedder, similarity_threshold=sim_threshold) if embedder is not None else None

    slicing_config = config.get("slicing", {})
    strategy = slicing_config.get("strategy", "time")
    window_s = int(slicing_config.get("window_seconds", 600))
    overlap_s = int(slicing_config.get("overlap_seconds", 120))

    if strategy == "phase":
        slices = slice_transcript_by_phase(transcript, llm, window_s, overlap_s)
    else:
        slices = slice_transcript_by_time(transcript, window_s, overlap_s)

    sample_count = int(llm_config.get("monte_carlo_samples", 20))

    results = []
    total_slices = len(slices)
    for slice_idx, transcript_slice in enumerate(slices):
        print(f"[Slice {slice_idx + 1}/{total_slices}] {transcript_slice.slice_id} "
              f"({transcript_slice.start:.0f}s-{transcript_slice.end:.0f}s) - sampling ...", flush=True)
        messages = build_class_prompt(transcript_slice)
        parsed_samples = []
        ensemble_models = llm_config.get("ensemble_models", [])
        if isinstance(ensemble_models, str):
            ensemble_models = [m.strip() for m in ensemble_models.split(",") if m.strip()]
        for sample_i in range(sample_count):
            active_llm = llm
            if ensemble_models:
                model_name = ensemble_models[sample_i % len(ensemble_models)]
                active_llm = build_llm_from_factory({**llm_config, "model": model_name})
            response = active_llm.generate(
                messages=messages,
                temperature=float(llm_config.get("temperature", 0.7)),
                max_tokens=int(llm_config.get("max_tokens", 2048)),
            )
            parsed_samples.append(parse_evaluation_response(response.text))
            print(f"  sample {sample_i + 1}/{sample_count} done (score={parsed_samples[-1]['score']})", flush=True)

        scores = [sample["score"] for sample in parsed_samples if sample["score"]]
        reasons = [sample["reason"] for sample in parsed_samples]
        score_entropy = shannon_entropy(scores)
        semantic_entropy, labels, clusters = compute_semantic_entropy(reasons, judge=semantic_judge)
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
                "phase_label": transcript_slice.phase_label,
                "segment_count": len(transcript_slice.segments),
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
        print(f"  -> majority={majority_vote(scores)}, "
              f"score_entropy={score_entropy:.3f}, "
              f"semantic_entropy={semantic_entropy:.3f}, "
              f"decision={decision}", flush=True)

    return {
        "lesson_id": transcript.lesson_id,
        "config": config,
        "slice_count": len(results),
        "results": results,
    }
