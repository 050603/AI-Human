#!/usr/bin/env python3
"""Comprehensive test runner for S3-4 classroom transcript evaluation."""
from __future__ import annotations

import json
import time
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from classroom_ai.pipeline.core_validation import (
    load_config,
    load_transcript,
    build_llm,
    build_embedder,
)
from classroom_ai.slicing.time_slicer import slice_transcript_by_time
from classroom_ai.evaluation.prompts import build_class_prompt
from classroom_ai.evaluation.parser import parse_evaluation_response
from classroom_ai.uncertainty.decision import majority_vote, route_decision
from classroom_ai.uncertainty.entropy import shannon_entropy
from classroom_ai.uncertainty.semantic_entropy import compute_semantic_entropy

TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S+08:00")


def main() -> None:
    test_start = time.time()
    report: dict[str, Any] = {
        "report_meta": {
            "title": "S3-4 人工智能通识课 CLASS 教学支持评估报告",
            "generated_at": now_iso(),
            "pipeline_version": "classroom-ai-core v0.1.0 + Ollama",
            "notes": "使用 Qwen2.5:7b 作为评估 LLM，nomic-embed-text 作为语义 embedding",
        },
        "transcript_info": {},
        "config": {},
        "slicing_info": {},
        "slice_results": [],
        "summary": {},
        "errors": [],
        "timing": {},
    }

    transcript_path = "data/sample/S3-4.json"
    config_path = "configs/local_ollama.yaml"

    # ── Step 1: Load config ──
    step_start = time.time()
    config = load_config(config_path)
    report["config"] = config
    report["timing"]["load_config_seconds"] = round(time.time() - step_start, 3)

    # ── Step 2: Load transcript ──
    step_start = time.time()
    transcript = load_transcript(transcript_path)
    report["transcript_info"] = {
        "lesson_id": transcript.lesson_id,
        "segment_count": len(transcript.segments),
        "total_chars": sum(len(s.text) for s in transcript.segments),
        "total_duration_seconds": (
            transcript.segments[-1].end - transcript.segments[0].start
            if transcript.segments
            else 0
        ),
        "speakers": list({s.speaker for s in transcript.segments}),
        "segments": [
            {
                "segment_id": s.segment_id,
                "start": s.start,
                "end": s.end,
                "speaker": s.speaker,
                "text": s.text,
                "text_length": len(s.text),
            }
            for s in transcript.segments
        ],
    }
    report["timing"]["load_transcript_seconds"] = round(time.time() - step_start, 3)

    # ── Step 3: Build LLM & Embedder ──
    step_start = time.time()
    llm = build_llm(config)
    embedder = build_embedder(config)
    report["providers"] = {
        "llm": {"type": type(llm).__name__, "model": llm.model, "host": llm.host},
        "embedder": {
            "type": type(embedder).__name__,
            "model": embedder.model,
            "host": embedder.host,
        },
    }
    report["timing"]["build_providers_seconds"] = round(time.time() - step_start, 3)

    # ── Step 4: Slice ──
    step_start = time.time()
    slicing_config = config.get("slicing", {})
    window_s = int(slicing_config.get("window_seconds", 600))
    overlap_s = int(slicing_config.get("overlap_seconds", 120))
    slices = slice_transcript_by_time(transcript, window_seconds=window_s, overlap_seconds=overlap_s)
    report["slicing_info"] = {
        "window_seconds": window_s,
        "overlap_seconds": overlap_s,
        "step_seconds": window_s - overlap_s,
        "slice_count": len(slices),
        "slices_meta": [
            {
                "slice_id": sl.slice_id,
                "start": sl.start,
                "end": sl.end,
                "segment_count": len(sl.segments),
                "text_preview": sl.text[:300] + ("..." if len(sl.text) > 300 else ""),
                "text_total_chars": len(sl.text),
            }
            for sl in slices
        ],
    }
    report["timing"]["slicing_seconds"] = round(time.time() - step_start, 3)

    # ── Step 5: Per-slice evaluation ──
    llm_config = config["llm"]
    uncertainty_config = config["uncertainty"]
    sample_count = int(llm_config.get("monte_carlo_samples", 20))
    temperature = float(llm_config.get("temperature", 0.7))
    max_tokens = int(llm_config.get("max_tokens", 2048))

    total_llm_calls = 0
    total_llm_time = 0.0
    total_embed_calls = 0
    total_embed_time = 0.0
    all_decisions: list[str] = []
    all_scores: list[int] = []

    for slice_idx, transcript_slice in enumerate(slices):
        slice_start = time.time()
        print(
            f"\n{'=' * 60}\n"
            f"[Slice {slice_idx + 1}/{len(slices)}] {transcript_slice.slice_id}\n"
            f"Time range: {transcript_slice.start:.0f}s – {transcript_slice.end:.0f}s\n"
            f"Segments: {len(transcript_slice.segments)}\n"
            f"{'=' * 60}",
            flush=True,
        )

        # Build prompt
        messages = build_class_prompt(transcript_slice)
        prompt_text = messages[1]["content"] if len(messages) > 1 else ""
        print(f"Prompt length: {len(prompt_text)} chars", flush=True)

        # Monte Carlo sampling
        parsed_samples: list[dict[str, Any]] = []
        llm_call_details: list[dict[str, Any]] = []
        for sample_i in range(sample_count):
            call_start = time.time()
            try:
                response = llm.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                call_elapsed = time.time() - call_start
                parsed = parse_evaluation_response(response.text)
                parsed_samples.append(parsed)
                llm_call_details.append(
                    {
                        "sample_index": sample_i,
                        "score": parsed.get("score"),
                        "reason_preview": parsed.get("reason", "")[:100],
                        "llm_time_seconds": round(call_elapsed, 3),
                        "response_length": len(response.text),
                        "status": "success",
                    }
                )
                total_llm_time += call_elapsed
                total_llm_calls += 1
                print(
                    f"  sample {sample_i + 1:2d}/{sample_count} | "
                    f"score={parsed.get('score'):>2} | "
                    f"{call_elapsed:.1f}s",
                    flush=True,
                )
            except Exception as e:
                call_elapsed = time.time() - call_start
                llm_call_details.append(
                    {
                        "sample_index": sample_i,
                        "score": None,
                        "reason_preview": "",
                        "llm_time_seconds": round(call_elapsed, 3),
                        "status": "error",
                        "error": str(e),
                    }
                )
                total_llm_time += call_elapsed
                total_llm_calls += 1
                report["errors"].append(
                    {
                        "slice": transcript_slice.slice_id,
                        "sample": sample_i,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(
                    f"  sample {sample_i + 1:2d}/{sample_count} | ERROR: {e}",
                    flush=True,
                )

        # Extract scores and reasons
        scores = [s["score"] for s in parsed_samples if s.get("score") is not None]
        reasons = [s["reason"] for s in parsed_samples if s.get("reason")]
        all_scores.extend(scores)

        # Score entropy
        score_entropy = shannon_entropy(scores)

        # Semantic entropy
        embed_start = time.time()
        semantic_entropy, labels, clusters = compute_semantic_entropy(
            reasons,
            embedder=embedder,
            similarity_threshold=float(uncertainty_config.get("similarity_threshold", 0.82)),
        )
        embed_time = time.time() - embed_start
        total_embed_time += embed_time
        total_embed_calls += 1

        # Routing decision
        majority = majority_vote(scores)
        decision = route_decision(
            semantic_entropy=semantic_entropy,
            score_entropy=score_entropy,
            semantic_entropy_threshold=float(uncertainty_config["semantic_entropy_threshold"]),
            score_entropy_threshold=float(uncertainty_config["score_entropy_threshold"]),
        )
        all_decisions.append(decision)

        slice_elapsed = time.time() - slice_start

        # Build detailed slice result
        score_dist = dict(sorted(Counter(scores).items()))
        slice_result: dict[str, Any] = {
            "slice_id": transcript_slice.slice_id,
            "lesson_id": transcript.lesson_id,
            "time_start": transcript_slice.start,
            "time_end": transcript_slice.end,
            "segment_count_in_slice": len(transcript_slice.segments),
            "monte_carlo_samples": sample_count,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "score_distribution": score_dist,
            "majority_score": majority,
            "score_entropy": round(score_entropy, 6),
            "semantic_entropy": round(semantic_entropy, 6),
            "semantic_entropy_threshold_applied": float(uncertainty_config["semantic_entropy_threshold"]),
            "score_entropy_threshold_applied": float(uncertainty_config["score_entropy_threshold"]),
            "similarity_threshold_applied": float(uncertainty_config.get("similarity_threshold", 0.82)),
            "semantic_cluster_count": len(clusters),
            "semantic_cluster_labels": labels,
            "semantic_clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "count": c.count,
                    "representative_reason": c.representative_text,
                    "member_indices": c.member_indices,
                }
                for c in clusters
            ],
            "decision": decision,
            "decision_rationale": (
                f"score_entropy={score_entropy:.4f} (threshold={uncertainty_config['score_entropy_threshold']}), "
                f"semantic_entropy={semantic_entropy:.4f} (threshold={uncertainty_config['semantic_entropy_threshold']})"
                f" → {decision}"
            ),
            "slice_processing_time_seconds": round(slice_elapsed, 3),
            "embedding_time_seconds": round(embed_time, 3),
            "llm_call_details": llm_call_details,
            "raw_samples": [s for s in parsed_samples],
        }
        report["slice_results"].append(slice_result)

        print(
            f"  → score_dist={score_dist}, majority={majority}\n"
            f"  → score_entropy={score_entropy:.4f}, semantic_entropy={semantic_entropy:.4f}\n"
            f"  → decision={decision} | slice time={slice_elapsed:.1f}s",
            flush=True,
        )

    # ── Step 6: Summary ──
    total_elapsed = time.time() - test_start
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    report["timing"].update(
        {
            "total_elapsed_seconds": round(total_elapsed, 3),
            "total_elapsed_minutes": round(total_elapsed / 60, 1),
            "total_llm_calls": total_llm_calls,
            "total_llm_time_seconds": round(total_llm_time, 3),
            "avg_llm_call_seconds": round(total_llm_time / total_llm_calls, 3) if total_llm_calls else 0,
            "total_embed_calls": total_embed_calls,
            "total_embed_time_seconds": round(total_embed_time, 3),
        }
    )

    decision_counts = dict(Counter(all_decisions))
    report["summary"] = {
        "lesson_id": transcript.lesson_id,
        "total_slices": len(slices),
        "total_samples": total_llm_calls,
        "overall_avg_score": round(avg_score, 2),
        "score_range": [min(all_scores), max(all_scores)] if all_scores else [0, 0],
        "decisions": decision_counts,
        "auto_accept_rate": (
            round(decision_counts.get("auto_accept", 0) / len(all_decisions), 3)
            if all_decisions
            else 0
        ),
        "human_review_rate": (
            round(decision_counts.get("human_review", 0) / len(all_decisions), 3)
            if all_decisions
            else 0
        ),
        "conclusion": (
            f"课程「{transcript.lesson_id}」共切分为 {len(slices)} 个切片，"
            f"执行 {total_llm_calls} 次 LLM 评估采样。"
            f"自动通过 {decision_counts.get('auto_accept', 0)} 个切片，"
            f"需人工审核 {decision_counts.get('human_review', 0)} 个切片。"
            f"总体评分均值 {avg_score:.1f}（CLASS 1-7 量表），"
            f"耗时 {total_elapsed:.1f}s（{total_elapsed / 60:.1f}min）。"
        ),
        "error_count": len(report["errors"]),
    }

    # Write report
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "S3-4_full_test_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"\n{'=' * 60}\n"
        f"Report complete!\n"
        f"  Slices: {len(slices)}\n"
        f"  LLM calls: {total_llm_calls}\n"
        f"  Auto-accept: {decision_counts.get('auto_accept', 0)}\n"
        f"  Human-review: {decision_counts.get('human_review', 0)}\n"
        f"  Avg score: {avg_score:.1f}\n"
        f"  Total time: {total_elapsed:.1f}s\n"
        f"  Output: {report_path}\n"
        f"{'=' * 60}",
        flush=True,
    )


if __name__ == "__main__":
    main()