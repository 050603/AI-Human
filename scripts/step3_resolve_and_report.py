#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classroom_ai.pipeline.core_validation import build_llm, load_config
from classroom_ai.pipeline.micro_intervention import resolve_with_expert_feedback


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage3: 局部重评与终报生成")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--output-root", default="outputs")
    args = ap.parse_args()

    config = load_config(args.config)
    llm = build_llm(config)
    stage1 = json.loads(Path(args.stage1).read_text(encoding="utf-8"))
    stage2 = json.loads(Path(args.stage2).read_text(encoding="utf-8"))

    index = {r["slice_id"]: r for r in stage1.get("results", [])}
    for item in stage2.get("feedback", []):
        if not item.get("needs_human_intervention"):
            continue
        slice_id = item.get("slice_id")
        dimension = item.get("dimension")
        row = index.get(slice_id)
        if not row or dimension not in row.get("dimensions", {}):
            continue
        resolved = resolve_with_expert_feedback(
            llm,
            dimension=dimension,
            original_conflict=item.get("core_conflict", ""),
            vqa_question=item.get("question_payload", {}),
            expert_choice_id=item.get("expert_choice_id", ""),
            expert_custom_text=item.get("expert_custom_text", ""),
        )
        row["dimensions"][dimension]["majority_score"] = resolved.get("score", 0)
        row["dimensions"][dimension]["resolved_reason"] = resolved.get("reason", "")
        row["dimensions"][dimension]["resolved_ability_codes"] = resolved.get("ability_codes", [])
        row["dimensions"][dimension]["decision"] = "resolved_by_human_ai_collab"

    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "final_report.json"
    out.write_text(json.dumps(stage1, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Final report written: {out}")


if __name__ == "__main__":
    main()
