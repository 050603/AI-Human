#!/usr/bin/env python3
"""Auto-generate Stage 2 human expert feedback for human_review items.

This script simulates human expert intervention by:
1. Reading stage1 results
2. For each human_review dimension, generating expert question + auto-selecting the best option
3. Saving stage2_human_feedback.json into the same run_dir as stage1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classroom_ai.pipeline.core_validation import build_llm, load_config
from classroom_ai.pipeline.micro_intervention import generate_expert_question


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-generate Stage 2 human feedback")
    ap.add_argument("--stage1", required=True, help="Path to stage1_result.json")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--run-dir", default=None, help="Output directory (default: same dir as stage1)")
    args = ap.parse_args()

    config = load_config(args.config)
    llm = build_llm(config)
    stage1_path = Path(args.stage1)
    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))

    run_dir = Path(args.run_dir) if args.run_dir else stage1_path.parent

    feedback = []
    for result in stage1.get("results", []):
        for dim, detail in result.get("dimensions", {}).items():
            if detail.get("decision") != "human_review":
                continue

            diagnostic = detail.get("diagnostic", {})
            high_view = diagnostic.get("high_score_view", {})
            low_view = diagnostic.get("low_score_view", {})

            high_reason = high_view.get("reason", "该维度表现优秀")
            low_reason = low_view.get("reason", "该维度表现不足")

            print(f"  Generating question for {result['slice_id']}/{dim}...")
            question = generate_expert_question(llm, high_reason, low_reason)

            options = question.get("options", [])
            if options:
                selected = options[0]["id"]
            else:
                selected = "A"

            feedback.append({
                "slice_id": result["slice_id"],
                "dimension": dim,
                "needs_human_intervention": True,
                "expert_choice_id": selected,
                "expert_custom_text": "",
                "core_conflict": f"{high_reason} VS {low_reason}",
                "question_payload": question,
            })

    payload = {"feedback": feedback}
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "stage2_human_feedback.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nStage 2 feedback saved: {out}")
    print(f"Total human_review items: {len(feedback)}")


if __name__ == "__main__":
    main()