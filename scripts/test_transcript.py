#!/usr/bin/env python3
"""Comprehensive test runner for classroom transcript evaluation."""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from classroom_ai.pipeline.core_validation import load_config, load_transcript, build_llm, build_embedder
from classroom_ai.slicing.time_slicer import slice_transcript_by_time
from classroom_ai.slicing.phase_slicer import slice_transcript_by_phase
from classroom_ai.evaluation.prompts import build_class_prompt
from classroom_ai.evaluation.parser import parse_evaluation_response
from classroom_ai.uncertainty.decision import majority_vote, route_decision
from classroom_ai.uncertainty.entropy import shannon_entropy
from classroom_ai.uncertainty.semantic_entropy import compute_semantic_entropy

TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S+08:00")


def run_test(transcript_path: str, config_path: str, output_path: str) -> None:
    test_start = time.time()
    report: dict[str, Any] = {
        "report_meta": {
            "generated_at": now_iso(),
            "pipeline_version": "classroom-ai-core v0.1.0 + Ollama",
            "llm_model": "Qwen2.5:7b",
            "embed_model": "nomic-embed-text",
        },
        "errors": [],
        "timing": {},
    }

    config = load_config(config_path)
    report["config"] = config

    transcript = load_transcript(transcript_path)
    report["transcript_info"] = {
        "lesson_id": transcript.lesson_id,
        "segment_count": len(transcript.segments),
        "total_chars": sum(len(s.text) for s in transcript.segments),
        "total_duration_seconds": transcript.segments[-1].end - transcript.segments[0].start if transcript.segments else 0,
        "speakers": list({s.speaker for s in transcript.segments}),
        "segments": [
            {"segment_id": s.segment_id, "start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text, "text_length": len(s.text)}
            for s in transcript.segments
        ],
    }

    llm = build_llm(config)
    embedder = build_embedder(config)
    report["providers"] = {
        "llm": {"type": type(llm).__name__, "model": llm.model, "host": llm.host},
        "embedder": {"type": type(embedder).__name__, "model": embedder.model, "host": embedder.host},
    }

    slicing_config = config.get("slicing", {})
    strategy = slicing_config.get("strategy", "time")
    window_s = int(slicing_config.get("window_seconds", 600))
    overlap_s = int(slicing_config.get("overlap_seconds", 120))

    if strategy == "phase":
        slices = slice_transcript_by_phase(transcript, llm, window_s, overlap_s)
    else:
        slices = slice_transcript_by_time(transcript, window_seconds=window_s, overlap_seconds=overlap_s)

    report["slicing_info"] = {
        "strategy": strategy,
        "window_seconds": window_s,
        "overlap_seconds": overlap_s,
        "step_seconds": window_s - overlap_s if strategy != "phase" else 0,
        "slice_count": len(slices),
        "slices_meta": [
            {"slice_id": sl.slice_id, "start": sl.start, "end": sl.end,
             "phase_label": sl.phase_label, "segment_count": len(sl.segments),
             "text_total_chars": len(sl.text)}
            for sl in slices
        ],
    }

    llm_config = config["llm"]
    uncertainty_config = config["uncertainty"]
    sample_count = int(llm_config.get("monte_carlo_samples", 20))
    temperature = float(llm_config.get("temperature", 0.7))
    max_tokens = int(llm_config.get("max_tokens", 2048))

    total_llm_calls = 0
    total_llm_time = 0.0
    total_embed_time = 0.0
    all_decisions: list[str] = []
    all_scores: list[int] = []
    slice_results: list[dict[str, Any]] = []

    for slice_idx, transcript_slice in enumerate(slices):
        slice_start = time.time()
        print(
            f"\n{'=' * 60}\n"
            f"[Slice {slice_idx + 1}/{len(slices)}] {transcript_slice.slice_id}\n"
            f"Time: {transcript_slice.start:.0f}s – {transcript_slice.end:.0f}s | "
            f"Segments: {len(transcript_slice.segments)}\n"
            f"{'=' * 60}",
            flush=True,
        )

        messages = build_class_prompt(transcript_slice)
        print(f"Prompt: {len(messages[1]['content']) if len(messages) > 1 else 0} chars", flush=True)

        parsed_samples: list[dict[str, Any]] = []
        llm_call_log: list[dict[str, Any]] = []
        for sample_i in range(sample_count):
            call_start = time.time()
            try:
                response = llm.generate(messages=messages, temperature=temperature, max_tokens=max_tokens)
                elapsed = time.time() - call_start
                parsed = parse_evaluation_response(response.text)
                parsed_samples.append(parsed)
                llm_call_log.append({"sample": sample_i, "score": parsed.get("score"), "reason": parsed.get("reason", "")[:120], "time_s": round(elapsed, 2), "status": "ok"})
                total_llm_time += elapsed
                total_llm_calls += 1
                print(f"  [{sample_i + 1:2d}/{sample_count}] score={parsed.get('score')} | {elapsed:.1f}s", flush=True)
            except Exception as e:
                elapsed = time.time() - call_start
                llm_call_log.append({"sample": sample_i, "score": None, "time_s": round(elapsed, 2), "status": "error", "error": str(e)})
                total_llm_time += elapsed
                total_llm_calls += 1
                report["errors"].append({"slice": transcript_slice.slice_id, "sample": sample_i, "error": str(e)})
                print(f"  [{sample_i + 1:2d}/{sample_count}] ERROR: {e}", flush=True)

        scores = [s["score"] for s in parsed_samples if s.get("score") is not None]
        reasons = [s["reason"] for s in parsed_samples if s.get("reason")]
        all_scores.extend(scores)

        score_entropy = shannon_entropy(scores)

        embed_start = time.time()
        semantic_entropy, labels, clusters = compute_semantic_entropy(reasons, embedder=embedder, similarity_threshold=float(uncertainty_config.get("similarity_threshold", 0.82)))
        embed_time = time.time() - embed_start
        total_embed_time += embed_time

        majority = majority_vote(scores)
        decision = route_decision(
            semantic_entropy=semantic_entropy, score_entropy=score_entropy,
            semantic_entropy_threshold=float(uncertainty_config["semantic_entropy_threshold"]),
            score_entropy_threshold=float(uncertainty_config["score_entropy_threshold"]),
        )
        all_decisions.append(decision)

        score_dist = dict(sorted(Counter(scores).items()))
        slice_result = {
            "slice_id": transcript_slice.slice_id,
            "lesson_id": transcript.lesson_id,
            "phase_label": transcript_slice.phase_label,
            "time_start": transcript_slice.start,
            "time_end": transcript_slice.end,
            "segment_count_in_slice": len(transcript_slice.segments),
            "monte_carlo_samples": sample_count,
            "temperature": temperature,
            "score_distribution": score_dist,
            "majority_score": majority,
            "score_entropy": round(score_entropy, 6),
            "semantic_entropy": round(semantic_entropy, 6),
            "semantic_cluster_count": len(clusters),
            "semantic_cluster_labels": labels,
            "semantic_clusters": [{"cluster_id": c.cluster_id, "count": c.count, "representative_reason": c.representative_text, "member_indices": c.member_indices} for c in clusters],
            "decision": decision,
            "slice_processing_time_s": round(time.time() - slice_start, 1),
            "embedding_time_s": round(embed_time, 3),
            "llm_call_log": llm_call_log,
            "samples": parsed_samples,
        }
        slice_results.append(slice_result)

        print(f"  -> score_dist={score_dist} majority={majority}\n"
              f"  -> score_entropy={score_entropy:.4f} semantic_entropy={semantic_entropy:.4f} clusters={len(clusters)}\n"
              f"  -> decision={decision} | {time.time() - slice_start:.1f}s", flush=True)

    total_elapsed = time.time() - test_start
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    decision_counts = dict(Counter(all_decisions))

    report["timing"] = {
        "total_elapsed_seconds": round(total_elapsed, 1),
        "total_elapsed_minutes": round(total_elapsed / 60, 1),
        "total_llm_calls": total_llm_calls,
        "total_llm_time_seconds": round(total_llm_time, 1),
        "avg_llm_call_seconds": round(total_llm_time / total_llm_calls, 2) if total_llm_calls else 0,
        "total_embed_time_seconds": round(total_embed_time, 3),
    }
    report["slice_results"] = slice_results
    report["summary"] = {
        "lesson_id": transcript.lesson_id,
        "total_slices": len(slices),
        "total_samples": total_llm_calls,
        "overall_avg_score": round(avg_score, 2),
        "score_range": [min(all_scores), max(all_scores)] if all_scores else [0, 0],
        "decisions": decision_counts,
        "auto_accept_rate": round(decision_counts.get("auto_accept", 0) / len(all_decisions), 3) if all_decisions else 0,
        "human_review_rate": round(decision_counts.get("human_review", 0) / len(all_decisions), 3) if all_decisions else 0,
        "error_count": len(report["errors"]),
        "conclusion": (
            f"课程「{transcript.lesson_id}」{len(slices)} 个切片，{total_llm_calls} 次采样。"
            f"自动通过 {decision_counts.get('auto_accept', 0)}，人工审核 {decision_counts.get('human_review', 0)}。"
            f"均值 {avg_score:.1f}/7，耗时 {total_elapsed:.0f}s（{total_elapsed / 60:.1f}min）。"
        ),
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'=' * 60}\nReport: {out_path} ({out_path.stat().st_size / 1024:.0f}KB)\n{'=' * 60}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_transcript.py <lesson_id> [samples]")
        sys.exit(1)

    lesson_id = sys.argv[1]
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    transcript_path = f"data/sample/{lesson_id}.json"
    output_path = f"outputs/{lesson_id}_full_test_report.json"

    run_test(transcript_path, "configs/local_ollama.yaml", output_path)