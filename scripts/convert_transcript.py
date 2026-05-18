#!/usr/bin/env python3
"""Convert raw classroom transcript TXT to standard JSON format."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_timestamp(ts_str: str) -> float:
    parts = ts_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0


def convert(raw_path: Path, out_path: Path, lesson_id: str) -> None:
    text = raw_path.read_text(encoding="utf-8")
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
        if stripped.startswith("S") and "_原文" in stripped:
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
        end_ts = timestamps[i + 1] if i < len(timestamps) - 1 else start_ts + 120.0
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
        "lesson_id": lesson_id,
        "duration_seconds": timestamps[-1] - timestamps[0] if timestamps else 0,
        "segments": segments,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    total_text = sum(len(s["text"]) for s in segments)
    print(f"Converted {raw_path.name}: {len(segments)} segments, {total_text} chars")
    print(f"Duration: {timestamps[0]}s – {timestamps[-1]}s ({timestamps[-1] - timestamps[0]:.0f}s)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_transcript.py <raw.txt> <lesson_id>")
        sys.exit(1)
    raw_path = Path(sys.argv[1])
    lesson_id = sys.argv[2]
    out_path = Path(f"data/sample/{lesson_id}.json")
    convert(raw_path, out_path, lesson_id)