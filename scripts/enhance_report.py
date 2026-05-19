#!/usr/bin/env python3
"""Enhance the core validation report with comprehensive metadata and statistics."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S+08:00")


def main():
    if len(sys.argv) < 3:
        print("Usage: python enhance_report.py <input_report.json> <output_report.json> [transcript.json]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    transcript_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    with input_path.open("r", encoding="utf-8") as f:
        core = json.load(f)

    report: dict[str, Any] = {}

    report["report_meta"] = {
        "generated_at": now_iso(),
        "pipeline_version": "classroom-ai-core v0.1.0 + Ollama Ensemble",
        "evaluation_mode": "Ollama Heterogeneous Model Pool Ensemble",
        "llm_provider": "ollama",
        "llm_main_model": core["config"]["llm"]["model"],
        "llm_ensemble_models": core["config"]["llm"]["ensemble_models"],
        "embed_provider": core["config"]["embedding"]["provider"],
        "embed_model": core["config"]["embedding"]["model"],
        "slicing_strategy": core["config"]["slicing"]["strategy"],
        "debate_enabled": core["config"].get("debate", {}).get("enabled", False),
    }

    report["config"] = core["config"]

    if transcript_path and transcript_path.exists():
        with transcript_path.open("r", encoding="utf-8") as f:
            t = json.load(f)
        report["transcript_info"] = {
            "lesson_id": t["lesson_id"],
            "total_segments": len(t["segments"]),
            "total_chars": sum(len(s["text"]) for s in t["segments"]),
            "total_duration_seconds": t["duration_seconds"],
            "speakers": list({s["speaker"] for s in t["segments"]}),
            "segments": [
                {
                    "segment_id": s["segment_id"],
                    "start": s["start"],
                    "end": s["end"],
                    "speaker": s["speaker"],
                    "text_length": len(s["text"]),
                    "text_preview": s["text"][:80] + ("..." if len(s["text"]) > 80 else ""),
                }
                for s in t["segments"]
            ],
        }

    report["slice_count"] = core["slice_count"]

    all_dimension_decisions: list[str] = []
    all_dimension_scores: list[int] = []
    dimension_stats: dict[str, dict[str, Any]] = {}

    for dim_key in ["human_centered", "ai_ethics", "ai_tech_and_app", "ai_system_design"]:
        dimension_stats[dim_key] = {
            "auto_accept": 0,
            "human_review": 0,
            "scores": [],
            "semantic_entropies": [],
            "score_entropies": [],
        }

    slice_summaries: list[dict[str, Any]] = []

    for res in core["results"]:
        dims = res["dimensions"]
        slice_summary = {
            "slice_id": res["slice_id"],
            "phase_label": res["phase_label"],
            "time_range": {"start": res["start"], "end": res["end"]},
            "segment_count": res["segment_count"],
            "dimensions": {},
            "overall_decision": res.get("decision", "N/A"),
        }

        for dk, dv in dims.items():
            all_dimension_decisions.append(dv["decision"])
            all_dimension_scores.append(dv["majority_score"])
            if dk in dimension_stats:
                dimension_stats[dk]["scores"].append(dv["majority_score"])
                dimension_stats[dk]["semantic_entropies"].append(dv["semantic_entropy"])
                dimension_stats[dk]["score_entropies"].append(dv["score_entropy"])
                if dv["decision"] == "auto_accept":
                    dimension_stats[dk]["auto_accept"] += 1
                else:
                    dimension_stats[dk]["human_review"] += 1

            slice_summary["dimensions"][dk] = {
                "majority_score": dv["majority_score"],
                "score_distribution": dv["score_distribution"],
                "score_entropy": dv["score_entropy"],
                "semantic_entropy": dv["semantic_entropy"],
                "semantic_cluster_count": len(dv.get("semantic_clusters", [])),
                "decision": dv["decision"],
            }

        slice_summaries.append(slice_summary)

    report["slice_summaries"] = slice_summaries

    decision_counts = dict(Counter(all_dimension_decisions))
    total_dim_evals = len(all_dimension_decisions)

    for dk in dimension_stats:
        ds = dimension_stats[dk]
        ds["avg_score"] = round(sum(ds["scores"]) / len(ds["scores"]), 2) if ds["scores"] else 0
        ds["avg_semantic_entropy"] = round(sum(ds["semantic_entropies"]) / len(ds["semantic_entropies"]), 4) if ds["semantic_entropies"] else 0
        ds["avg_score_entropy"] = round(sum(ds["score_entropies"]) / len(ds["score_entropies"]), 4) if ds["score_entropies"] else 0
        ds["auto_accept_rate"] = round(ds["auto_accept"] / (ds["auto_accept"] + ds["human_review"]), 3) if (ds["auto_accept"] + ds["human_review"]) > 0 else 0

    report["dimension_statistics"] = dimension_stats

    avg_score = sum(all_dimension_scores) / len(all_dimension_scores) if all_dimension_scores else 0

    report["summary"] = {
        "lesson_id": core["lesson_id"],
        "total_slices": core["slice_count"],
        "total_dimension_evaluations": total_dim_evals,
        "total_monte_carlo_samples": total_dim_evals * core["results"][0]["monte_carlo_samples"] if core["results"] else 0,
        "overall_avg_score": round(avg_score, 2),
        "score_range": [min(all_dimension_scores), max(all_dimension_scores)] if all_dimension_scores else [0, 0],
        "dimension_decisions": decision_counts,
        "auto_accept_rate": round(decision_counts.get("auto_accept", 0) / total_dim_evals, 3) if total_dim_evals else 0,
        "human_review_rate": round(decision_counts.get("human_review", 0) / total_dim_evals, 3) if total_dim_evals else 0,
        "slice_level_decisions": dict(Counter(r.get("decision", "N/A") for r in core["results"])),
        "conclusion": (
            f"课程「{core['lesson_id']}」共 {core['slice_count']} 个教学阶段切片，"
            f"{total_dim_evals} 个维度评估（{total_dim_evals * core['results'][0]['monte_carlo_samples']} 次 Monte Carlo 采样）。"
            f"维度级自动通过 {decision_counts.get('auto_accept', 0)}/{total_dim_evals}"
            f"（{round(decision_counts.get('auto_accept', 0) / total_dim_evals * 100, 1)}%），"
            f"人工审核 {decision_counts.get('human_review', 0)}/{total_dim_evals}"
            f"（{round(decision_counts.get('human_review', 0) / total_dim_evals * 100, 1)}%）。"
            f"整体均值 {avg_score:.1f}/7。"
        ),
    }

    report["detailed_results"] = core["results"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Enhanced report written to {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()