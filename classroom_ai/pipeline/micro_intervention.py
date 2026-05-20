from __future__ import annotations

import json
import re
from typing import Any

from classroom_ai.evaluation.parser import parse_evaluation_response

CUSTOM_OPTION = {"id": "CUSTOM", "text": "以上选项都不准确，我有自己的看法。"}


def _extract_options_from_question(question: str) -> tuple[str, list[dict[str, str]]]:
    if not question:
        return question, []
    pattern = r"([A-D])\s*[\.、:：]\s*(.*?)(?=(?:\s+[A-D]\s*[\.、:：])|$)"
    matches = re.findall(pattern, question, flags=re.S)
    if len(matches) < 2:
        return question, []
    options = [{"id": key, "text": value.strip()} for key, value in matches if value.strip()]
    cleaned = re.sub(pattern, "", question, flags=re.S).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned or "以下哪种判断更符合该课堂片段？", options


def generate_expert_question(llm, high_score_reason: str, low_score_reason: str) -> dict[str, Any]:
    prompt = (
        "你是课堂评价冲突分析助手。基于高分派与低分派理由，生成一个单项选择题帮助专家快速裁决。\n"
        "【致命约束】严禁在 question 字段中包含选项内容（如 A/B/C 等标识）！你必须将具体的选项内容作为对象结构化地放入 options 数组中！\n"
        "只输出 JSON，格式为："
        '{"core_conflict":"...","question":"...","options":[{"id":"A","text":"..."}]}.\n'
        "正确示例：\n"
        '{"core_conflict":"教师是否有效引导了伦理探讨","question":"教师此处提及数据隐私问题，属于什么认知层级？","options":[{"id":"A","text":"仅停留在理解层，因为没有让学生实际操作。"},{"id":"B","text":"达到了应用层，因为引发了学生的具体案例探讨。"}]}\n'
        f"高分派理由：{high_score_reason}\n低分派理由：{low_score_reason}"
    )

    def _invoke() -> dict[str, Any]:
        resp = llm.generate(messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=800)
        try:
            return json.loads(resp.text)
        except Exception:
            return {}

    data = _invoke()
    options = data.get("options", []) if isinstance(data, dict) else []
    if not isinstance(options, list):
        options = []

    question = str(data.get("question", "以下哪种判断更符合该课堂片段？")) if isinstance(data, dict) else "以下哪种判断更符合该课堂片段？"

    if len(options) == 0:
        data = _invoke()  # retry once for option collapse
        options = data.get("options", []) if isinstance(data, dict) else []
        if not isinstance(options, list):
            options = []
        question = str(data.get("question", question)) if isinstance(data, dict) else question

    if len(options) == 0:
        question, extracted_options = _extract_options_from_question(question)
        options = extracted_options

    options.append(CUSTOM_OPTION)
    return {
        "core_conflict": str(data.get("core_conflict", "高低分理由存在评价依据冲突。")) if isinstance(data, dict) else "高低分理由存在评价依据冲突。",
        "question": question,
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
