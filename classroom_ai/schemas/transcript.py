from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TranscriptSegment:
    """A speaker-attributed transcript segment with second-based timestamps."""

    segment_id: str
    start: float
    end: float
    speaker: str
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            segment_id=str(data.get("segment_id", "")),
            start=float(data["start"]),
            end=float(data["end"]),
            speaker=str(data.get("speaker", "UNKNOWN")),
            text=str(data.get("text", "")).strip(),
            words=list(data.get("words", [])),
        )


@dataclass(frozen=True)
class Transcript:
    lesson_id: str
    segments: list[TranscriptSegment]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        return cls(
            lesson_id=str(data["lesson_id"]),
            segments=[TranscriptSegment.from_dict(item) for item in data.get("segments", [])],
        )
