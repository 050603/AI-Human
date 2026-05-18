from __future__ import annotations

import json
import re

from classroom_ai.llm.base import BaseLLM
from classroom_ai.schemas.slice import TranscriptSlice
from classroom_ai.schemas.transcript import Transcript

PHASE_DETECTION_PROMPT = """你是一位课堂分析专家。请根据以下课堂转录稿，识别其中自然的教学阶段划分。

每个阶段应代表一个完整的教学意图或活动类型。根据教学内容的变化点判断阶段切换，而非固定时间间隔。

常见阶段类型参考：
- 导入/引入：课程开场、激发兴趣、联系旧知
- 讲授/讲解：教师系统讲解新知识、新概念
- 活动/实践：学生动手操作、小组讨论、互动体验
- 讨论/分享：学生分享成果、师生互动讨论
- 总结/收尾：课程总结、回顾要点、结束语
- 过渡/衔接：不同活动间的过渡衔接

要求：
1. 根据内容语义识别阶段边界，而非固定时长
2. 每个阶段建议持续90-600秒
3. 阶段之间无缝衔接，覆盖完整课堂
4. 阶段数量建议3-8个

请严格输出JSON（不要用markdown代码块包裹，直接输出纯JSON）：
{"phases": [{"name": "导入", "start": 起始秒数, "end": 结束秒数}, ...]}"""


def _format_segments_for_llm(segments) -> str:
    lines = []
    for seg in segments:
        ts = _format_time(seg.start)
        lines.append(f"[{ts}] {seg.speaker}: {seg.text}")
    return "\n".join(lines)


def _format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def parse_phases_from_response(raw_text: str) -> list[dict] | None:
    """从LLM返回的文本中提取阶段JSON。"""
    try:
        return json.loads(raw_text.strip()).get("phases")
    except (json.JSONDecodeError, AttributeError):
        pass
    match = re.search(r"\{.*\"phases\".*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()).get("phases")
        except (json.JSONDecodeError, AttributeError):
            pass
    return None


def slice_transcript_by_phase(
    transcript: Transcript,
    llm: BaseLLM,
    fallback_window_seconds: int = 600,
    fallback_overlap_seconds: int = 120,
) -> list[TranscriptSlice]:
    """按课程阶段切片，LLM检测失败时回退到时间切片。"""

    segment_text = _format_segments_for_llm(transcript.segments)
    messages = [
        {"role": "system", "content": "你是一名专业的课堂教学分析师，输出必须是严格的JSON格式。"},
        {"role": "user", "content": f"{PHASE_DETECTION_PROMPT}\n\n课堂转录稿：\n{segment_text}"},
    ]

    try:
        response = llm.generate(messages, temperature=0.2, max_tokens=2048)
        phases = parse_phases_from_response(response.text)
    except Exception:
        phases = None

    if not phases:
        from classroom_ai.slicing.time_slicer import slice_transcript_by_time
        return slice_transcript_by_time(transcript, fallback_window_seconds, fallback_overlap_seconds)

    slices: list[TranscriptSlice] = []
    for i, phase in enumerate(phases):
        name = phase.get("name", f"阶段{i+1}")
        start = float(phase.get("start", 0))
        end = float(phase.get("end", start + 120))
        selected = [seg for seg in transcript.segments if seg.end > start and seg.start < end]
        if selected:
            slices.append(
                TranscriptSlice(
                    slice_id=f"{transcript.lesson_id}_phase_{i + 1:02d}_{name}",
                    lesson_id=transcript.lesson_id,
                    start=start,
                    end=end,
                    segments=selected,
                    phase_label=name,
                )
            )

    return slices