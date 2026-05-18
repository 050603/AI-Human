#!/usr/bin/env python3
"""Convert S3-4 raw transcript to standard JSON format."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAW_PATH = Path("/home/lkj/AIED/AI-Human/S3-4_原文.txt")
OUT_PATH = Path("/home/lkj/AIED/AI-Human/data/sample/S3-4.json")


def parse_timestamp(ts_str: str) -> float:
    parts = ts_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0


def main() -> None:
    text = RAW_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    segments_raw = []
    current_ts = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        speaker_match = re.match(r"^发言人(\d+)\s+([\d:.]+)$", stripped)
        if speaker_match:
            if current_ts is not None and current_lines:
                segments_raw.append((current_ts, current_lines))
            current_ts = parse_timestamp(speaker_match.group(2))
            current_lines = []
            continue
        if stripped == "S3-4_原文":
            continue
        if re.match(r"^\d{4}年\d{2}月\d{2}日", stripped):
            continue
        if stripped:
            current_lines.append(stripped)

    if current_ts is not None and current_lines:
        segments_raw.append((current_ts, current_lines))

    timestamps = [ts for ts, _ in segments_raw]

    segments = []
    for i, (start_ts, text_lines) in enumerate(segments_raw):
        segment_text = "".join(text_lines).strip().replace("\u3000", "")
        if i < len(timestamps) - 1:
            end_ts = timestamps[i + 1]
        else:
            end_ts = start_ts + 120.0

        segments.append(
            {
                "segment_id": f"seg_{i + 1:04d}",
                "start": start_ts,
                "end": end_ts,
                "speaker": "SPEAKER_00",
                "text": segment_text,
            }
        )

    result = {
        "lesson_id": "S3-4",
        "lesson_name": "人工智能通识课-探秘AI基础原理",
        "topic": "人工智能基础原理入门",
        "grade": "初中/小学高段",
        "duration_seconds": timestamps[-1] - timestamps[0] if timestamps else 0,
        "segments": segments,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    total_text = sum(len(s["text"]) for s in segments)
    print(f"Converted: {len(segments)} segments, {total_text} chars")
    print(f"Duration: {timestamps[0]}s – {timestamps[-1]}s ({timestamps[-1] - timestamps[0]:.0f}s)")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()