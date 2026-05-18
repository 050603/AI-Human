from __future__ import annotations

from classroom_ai.schemas.slice import TranscriptSlice
from classroom_ai.schemas.transcript import Transcript


def slice_transcript_by_time(
    transcript: Transcript,
    window_seconds: int = 600,
    overlap_seconds: int = 120,
) -> list[TranscriptSlice]:
    """Create fixed-size transcript slices with overlap.

    Segments are included when they overlap the slice time range. This is the
    minimal text-only slicer for notebook validation; activity-aware slicing can
    be added later without changing downstream interfaces.
    """

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= window_seconds:
        raise ValueError("overlap_seconds must be non-negative and smaller than window_seconds")
    if not transcript.segments:
        return []

    lesson_start = min(seg.start for seg in transcript.segments)
    lesson_end = max(seg.end for seg in transcript.segments)
    step = window_seconds - overlap_seconds

    slices: list[TranscriptSlice] = []
    start = lesson_start
    index = 1
    while start < lesson_end:
        end = min(start + window_seconds, lesson_end)
        selected = [seg for seg in transcript.segments if seg.end > start and seg.start < end]
        if selected:
            slices.append(
                TranscriptSlice(
                    slice_id=f"{transcript.lesson_id}_slice_{index:04d}",
                    lesson_id=transcript.lesson_id,
                    start=start,
                    end=end,
                    segments=selected,
                )
            )
            index += 1
        if end >= lesson_end:
            break
        start += step

    return slices
