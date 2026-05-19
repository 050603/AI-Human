from __future__ import annotations

from classroom_ai.evaluation.rubrics import UNESCO_RUBRIC
from classroom_ai.schemas.slice import TranscriptSlice


SYSTEM_PROMPT = "你是一名课堂观察专家，严格依据转录文本进行教学评价。"


def build_dimension_prompt(
    transcript_slice: TranscriptSlice,
    dimension_key: str,
    few_shot_cases: list[str] | None = None,
) -> list[dict[str, str]]:
    rubric = UNESCO_RUBRIC[dimension_key]
    levels = "\n".join([f"- {k}: {v}" for k, v in rubric["levels"].items()])
    cases_block = ""
    if few_shot_cases:
        cases_block = "\n\n历史专家裁定边界案例：\n" + "\n\n".join(few_shot_cases)

    user_prompt = f"""
请根据以下课堂转录片段，对 UNESCO 学生AI能力框架中的“{rubric['name']}”维度给出评分。

该维度评分量规（1-7分）：
{levels}

要求：
- 只依据给定文本，不要编造视频中未出现的信息。
- 分数范围为 1-7，且**必须输出纯整数**（禁止小数、区间、文字分级）。
- 输出语言必须统一为中文。
- 必须给出命中的能力编码列表，字段名为 ability_codes，例如 ["HAL", "DAMA"]。
- 每个分数必须给出文本证据。
- 如果证据不足，请在 uncertainty 中说明。
- 输出严格 JSON，字段为 dimension、score、reason、ability_codes、evidence、uncertainty。
- evidence 必须是对象数组，格式为[{{"line_start": 整数, "line_end": 整数, "text": "证据原文"}}]。
{cases_block}

课堂阶段：{transcript_slice.phase_label}
课堂片段：
{transcript_slice.text}
""".strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]



def build_class_prompt(transcript_slice: TranscriptSlice, few_shot_cases: list[str] | None = None) -> list[dict[str, str]]:
    """Backward-compatible prompt builder. Defaults to human_centered dimension."""
    return build_dimension_prompt(transcript_slice, dimension_key="human_centered", few_shot_cases=few_shot_cases)
