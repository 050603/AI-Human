from __future__ import annotations

from dataclasses import dataclass

from classroom_ai.schemas.transcript import TranscriptSegment


@dataclass(frozen=True)
class TranscriptSlice:
    slice_id: str
    lesson_id: str
    start: float
    end: float
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        lines = []
        for seg in self.segments:
            lines.append(f"[{seg.start:.3f}-{seg.end:.3f}] {seg.speaker}: {seg.text}")
        return "\n".join(lines)
