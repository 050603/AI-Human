#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classroom_ai.pipeline.core_validation import run_core_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run notebook-level classroom uncertainty validation.")
    parser.add_argument("--transcript", required=True, help="Path to structured transcript JSON.")
    parser.add_argument("--config", required=True, help="Path to YAML runtime config.")
    parser.add_argument("--output", required=True, help="Path to write validation result JSON.")
    args = parser.parse_args()

    result = run_core_validation(args.transcript, args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {result['slice_count']} slice result(s) to {output_path}")


if __name__ == "__main__":
    main()
