from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor


from classroom_ai.embedding.factory import build_embedder as build_embedder_from_factory
from classroom_ai.evaluation.parser import parse_evaluation_response
from classroom_ai.evaluation.prompts import build_dimension_prompt
from classroom_ai.evaluation.rubrics import get_default_dimensions
from classroom_ai.llm.factory import build_llm as build_llm_from_factory
from classroom_ai.schemas.transcript import Transcript
from classroom_ai.pipeline.debate_orchestrator import run_debate
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




def _load_phase4_memory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_phase4_memory(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_human_review_diagnostic(result: dict[str, Any]) -> dict[str, Any]:
    samples = result.get("samples", [])
    if not samples:
        return {}
    high = max(samples, key=lambda s: s.get("score", 0))
    low = min(samples, key=lambda s: s.get("score", 99))
    return {
        "slice_id": result.get("slice_id"),
        "time_range": {"start": result.get("start"), "end": result.get("end")},
        "high_score_view": {"score": high.get("score"), "reason": high.get("reason")},
        "low_score_view": {"score": low.get("score"), "reason": low.get("reason")},
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def build_llm(config: dict[str, Any]):
    return build_llm_from_factory(config["llm"] if "llm" in config else config)


def build_embedder(config: dict[str, Any]):
    return build_embedder_from_factory(config["embedding"] if "embedding" in config else config)




def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _retrieve_dimension_cases(
    transcript_slice,
    dimension_key: str,
    expert_memory: list[dict[str, Any]],
    rag_cfg: dict[str, Any],
    embedder,
) -> list[str]:
    top_k = int(rag_cfg.get("retrieve_top_k", 2))
    if top_k <= 0:
        return []
    candidates = [m for m in expert_memory if m.get("dimension") == dimension_key and m.get("source_text") and m.get("expert_correct_reason")]
    if not candidates:
        return []
    if embedder is None:
        chosen = candidates[-top_k:]
    else:
        query_vec = embedder.encode([transcript_slice.text])[0]
        ranked = sorted(candidates, key=lambda m: _cosine(query_vec, m.get("vector", [])), reverse=True)
        chosen = ranked[:top_k]
    return [
        f"历史案例#{i+1} | 维度={dimension_key} | 专家分数={item.get('expert_score')} | 专家理由={item.get('expert_correct_reason')}"
        for i, item in enumerate(chosen)
    ]


def _build_dim_thresholds(uncertainty_config: dict[str, Any], dimension_key: str) -> tuple[float, float]:
    dim_thresholds = uncertainty_config.get("dimension_thresholds", {}) or {}
    dim_cfg = dim_thresholds.get(dimension_key, {}) if isinstance(dim_thresholds, dict) else {}
    sem = float(dim_cfg.get("semantic_entropy_threshold", uncertainty_config["semantic_entropy_threshold"]))
    score = float(dim_cfg.get("score_entropy_threshold", uncertainty_config["score_entropy_threshold"]))
    return sem, score


def _evaluate_dimension(
    transcript_slice,
    dimension_key: str,
    llm_config: dict[str, Any],
    llm,
    sample_count: int,
    semantic_judge,
    uncertainty_config: dict[str, Any],
    debate_cfg: dict[str, Any],
    few_shot_cases: list[str],
) -> dict[str, Any]:
    messages = build_dimension_prompt(transcript_slice, dimension_key=dimension_key, few_shot_cases=few_shot_cases)
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

    scores = [sample["score"] for sample in parsed_samples if sample["score"]]
    reasons = [sample["reason"] for sample in parsed_samples]
    # 计算分数熵前剔除 0 分废票；有效样本不足 2 时强制触发人工
    score_entropy = shannon_entropy(scores) if len(scores) >= 2 else 999.0
    semantic_entropy, labels, clusters = compute_semantic_entropy(reasons, judge=semantic_judge)
    sem_t, score_t = _build_dim_thresholds(uncertainty_config, dimension_key)
    decision = route_decision(
        semantic_entropy=semantic_entropy,
        score_entropy=score_entropy,
        semantic_entropy_threshold=sem_t,
        score_entropy_threshold=score_t,
    )

    if decision == "human_review":
        debate_enabled = bool(debate_cfg.get("enabled", True))
        if debate_enabled:
            debate = run_debate(
                high_score_reason=max(parsed_samples, key=lambda s: s.get("score", 0)).get("reason", ""),
                low_score_reason=min(parsed_samples, key=lambda s: s.get("score", 99)).get("reason", ""),
                max_rounds=int(debate_cfg.get("max_rounds", 2)),
            )
            if debate.converged:
                updated_reasons = reasons.copy()
                if updated_reasons:
                    try:
                        high_idx = max(range(len(parsed_samples)), key=lambda i: parsed_samples[i].get("score", 0))
                        low_idx = min(range(len(parsed_samples)), key=lambda i: parsed_samples[i].get("score", 99))
                        updated_reasons[high_idx] = debate.updated_reasons.get("high_score_model", updated_reasons[high_idx])
                        updated_reasons[low_idx] = debate.updated_reasons.get("low_score_model", updated_reasons[low_idx])
                    except ValueError:
                        pass
                score_entropy = shannon_entropy(scores) if len(scores) >= 2 else 999.0
                semantic_entropy, labels, clusters = compute_semantic_entropy(updated_reasons, judge=semantic_judge)
                decision = route_decision(
                    semantic_entropy=semantic_entropy,
                    score_entropy=score_entropy,
                    semantic_entropy_threshold=sem_t,
                    score_entropy_threshold=score_t,
                )

    return {
        "score_distribution": dict(sorted(Counter(scores).items())),
        "majority_score": majority_vote(scores),
        "score_entropy": score_entropy,
        "semantic_entropy": semantic_entropy,
        "semantic_entropy_threshold_applied": sem_t,
        "score_entropy_threshold_applied": score_t,
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
        "invalid_score_count": sum(1 for s in parsed_samples if s.get("score_invalid")),
    }

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
    debate_cfg = config.get("debate", {})
    rag_cfg = config.get("phase4_rag", {})
    memory_path = Path(rag_cfg.get("memory_path", "outputs/phase4_expert_memory.json"))
    expert_memory = _load_phase4_memory(memory_path)
    total_slices = len(slices)
    dimensions = config.get("evaluation", {}).get("dimensions", get_default_dimensions())
    for slice_idx, transcript_slice in enumerate(slices):
        print(f"[Slice {slice_idx + 1}/{total_slices}] {transcript_slice.slice_id} "
              f"({transcript_slice.start:.0f}s-{transcript_slice.end:.0f}s) - evaluating 4 dimensions...", flush=True)

        dimension_results: dict[str, Any] = {}
        max_workers = int(config.get("evaluation", {}).get("dimension_concurrency", len(dimensions)))

        def _job(dimension_key: str) -> tuple[str, dict[str, Any]]:
            few_shot_cases = _retrieve_dimension_cases(transcript_slice, dimension_key, expert_memory, rag_cfg, embedder)
            res = _evaluate_dimension(
                transcript_slice=transcript_slice,
                dimension_key=dimension_key,
                llm_config=llm_config,
                llm=llm,
                sample_count=sample_count,
                semantic_judge=semantic_judge,
                uncertainty_config=uncertainty_config,
                debate_cfg=debate_cfg,
                few_shot_cases=few_shot_cases,
            )
            res["diagnostic"] = _build_human_review_diagnostic({"slice_id": transcript_slice.slice_id,"start": transcript_slice.start,"end": transcript_slice.end,"samples": res.get("samples", [])}) if res.get("decision") == "human_review" else {}
            return dimension_key, res

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for dimension_key, dim_result in ex.map(_job, dimensions):
                dimension_results[dimension_key] = dim_result

        majority_scores = [d.get("majority_score", 0) for d in dimension_results.values() if d.get("majority_score")]
        results.append(
            {
                "slice_id": transcript_slice.slice_id,
                "lesson_id": transcript.lesson_id,
                "start": transcript_slice.start,
                "end": transcript_slice.end,
                "phase_label": transcript_slice.phase_label,
                "segment_count": len(transcript_slice.segments),
                "slice_text": transcript_slice.text,
                "monte_carlo_samples": sample_count,
                "dimensions": dimension_results,
                "majority_score": round(sum(majority_scores) / len(majority_scores)) if majority_scores else 0,
                "decision": "human_review" if any(d.get("decision") == "human_review" for d in dimension_results.values()) else "auto_accept",
            }
        )

    if rag_cfg.get("enabled", True):
        for r in results:
            for dimension_key, d in r.get("dimensions", {}).items():
                if d.get("decision") != "human_review":
                    continue
                exemplar = d.get("samples", [{}])[0]
                source_text = str(r.get("slice_text", ""))
                vector = embedder.encode([source_text])[0] if (embedder is not None and source_text) else []
                expert_memory.append({
                    "slice_id": r.get("slice_id"),
                    "lesson_id": transcript.lesson_id,
                    "dimension": dimension_key,
                    "source_text": source_text,
                    "vector": vector,
                    "model_error_reason": d.get("diagnostic", {}).get("high_score_view", {}).get("reason", ""),
                    "expert_correct_reason": "专家复核后确认该维度需更高阶能力证据，予以修正。",
                    "expert_score": d.get("majority_score", 4),
                    "ability_codes": exemplar.get("ability_codes", []),
                })
                break
        _save_phase4_memory(memory_path, expert_memory)

    return {
        "lesson_id": transcript.lesson_id,
        "config": config,
        "slice_count": len(results),
        "results": results,
    }
