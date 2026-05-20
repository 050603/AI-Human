#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classroom_ai.pipeline.core_validation import run_core_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: AI 全量初评")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result = run_core_validation(args.transcript, args.config)
    out = run_dir / "stage1_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stage1 done: {out}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
