from __future__ import annotations

import json
from typing import Any

from classroom_ai.evaluation.parser import parse_evaluation_response

CUSTOM_OPTION = {"id": "CUSTOM", "text": "以上选项都不准确，我有自己的看法。"}


def generate_expert_question(llm, high_score_reason: str, low_score_reason: str) -> dict[str, Any]:
    prompt = (
        "你是课堂评价冲突分析助手。基于高分派与低分派理由，生成一个单项选择题帮助专家快速裁决。"
        "只输出 JSON，格式为："
        '{"core_conflict":"...","question":"...","options":[{"id":"A","text":"..."}]}.\n'
        f"高分派理由：{high_score_reason}\n低分派理由：{low_score_reason}"
    )
    resp = llm.generate(messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=800)
    try:
        data = json.loads(resp.text)
    except Exception:
        data = {}
    options = data.get("options", []) if isinstance(data, dict) else []
    if not isinstance(options, list):
        options = []
    options.append(CUSTOM_OPTION)
    return {
        "core_conflict": str(data.get("core_conflict", "高低分理由存在评价依据冲突。")) if isinstance(data, dict) else "高低分理由存在评价依据冲突。",
        "question": str(data.get("question", "以下哪种判断更符合该课堂片段？")) if isinstance(data, dict) else "以下哪种判断更符合该课堂片段？",
        "options": options,
    }


def resolve_with_expert_feedback(
    llm,
    *,
    dimension: str,
    original_conflict: str,
    vqa_question: dict[str, Any],
    expert_choice_id: str,
    expert_custom_text: str | None = None,
) -> dict[str, Any]:
    expert_final = (
        expert_custom_text.strip()
        if expert_choice_id == "CUSTOM" and expert_custom_text and expert_custom_text.strip()
        else f"专家选择了选项 {expert_choice_id}"
    )
    prompt = (
        "你是课堂评价终审模型。请综合原始冲突、选择题与专家意见，输出最终裁决。\n"
        "仅输出 JSON，字段必须包含：score(1-7整数), reason(字符串), ability_codes(数组)。\n"
        f"维度: {dimension}\n"
        f"原始冲突: {original_conflict}\n"
        f"题目: {json.dumps(vqa_question, ensure_ascii=False)}\n"
        f"专家最终意见: {expert_final}"
    )
    resp = llm.generate(messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=1200)
    parsed = parse_evaluation_response(resp.text)
    return {
        "score": parsed.get("score", 0),
        "reason": parsed.get("reason", ""),
        "ability_codes": parsed.get("ability_codes", []),
        "raw_text": resp.text,
    }
