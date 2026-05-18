from __future__ import annotations

from classroom_ai.schemas.slice import TranscriptSlice


SYSTEM_PROMPT = "你是一名课堂观察专家，严格依据转录文本进行教学评价。"


def build_class_prompt(transcript_slice: TranscriptSlice, few_shot_cases: list[str] | None = None) -> list[dict[str, str]]:
    cases_block = ""
    if few_shot_cases:
        cases_block = "\n\n历史专家裁定边界案例：\n" + "\n\n".join(few_shot_cases)

    user_prompt = f"""
请根据以下课堂转录片段，对 CLASS 量表中的“教学支持 instructional_support”维度给出初步评分。

要求：
- 只依据给定文本，不要编造视频中未出现的信息。
- 分数范围为 1-7。
- 每个分数必须给出文本证据。
- 如果证据不足，请在 uncertainty 中说明。
- 输出严格 JSON，字段为 dimension、score、reason、evidence、uncertainty。
{cases_block}

课堂片段：
{transcript_slice.text}
""".strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
