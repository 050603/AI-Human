from __future__ import annotations

import json
import re
from typing import Any

from classroom_ai.schemas.evidence import normalize_evidence


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_INVALID_ESCAPE_RE = re.compile(r"\\(?![\"\\/bfnrtu])")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_MISSING_COMMA_RE = re.compile(r"([}\]])\s*([{\[])")
_CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)```", flags=re.DOTALL)
_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")


_SMART_QUOTE_TABLE = str.maketrans("\u201c\u201d\u2018\u2019", "\"\"''")


def _sanitize_json(text: str) -> str:
    sanitized = _CONTROL_CHAR_RE.sub(" ", text)
    sanitized = sanitized.translate(_SMART_QUOTE_TABLE)
    sanitized = _INVALID_ESCAPE_RE.sub(r"\\\\", sanitized)
    sanitized = _TRAILING_COMMA_RE.sub(r"\1", sanitized)
    sanitized = _MISSING_COMMA_RE.sub(r"\1,\2", sanitized)
    return sanitized


def _extract_json_block(text: str) -> str:
    stripped = text.strip()
    # 剥离任意 markdown 代码块标记（```json / ```python / ``` ...）
    stripped = _CODE_FENCE_RE.sub(lambda m: m.group(1).strip(), stripped).strip()

    brace_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return stripped


def _robust_json_parse(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_sanitize_json(text))
    except json.JSONDecodeError:
        pass
    return {}


def _extract_score_value(score_raw: Any) -> tuple[int, bool]:
    """解析 1-7 分；失败或越界均记 0（废票）。"""

    if isinstance(score_raw, (int, float)):
        score = int(round(float(score_raw)))
        return (score if 1 <= score <= 7 else 0), not (1 <= score <= 7)

    if isinstance(score_raw, str):
        match = _SCORE_RE.search(score_raw)
        if match:
            score = int(round(float(match.group(0))))
            return (score if 1 <= score <= 7 else 0), not (1 <= score <= 7)

    return 0, True


def parse_evaluation_response(text: str) -> dict[str, Any]:
    """Parse JSON LLM output with a conservative fallback for notebook use."""

    clean = _extract_json_block(text)
    data = _robust_json_parse(clean)

    if isinstance(data, list):
        data = data[0] if data else {}

    score_raw = data.get("score", 0)
    score, score_invalid = _extract_score_value(score_raw)

    reason = str(data.get("reason") or data.get("summary") or text).strip()
    evidence = normalize_evidence(data.get("evidence", []))

    ability_codes = data.get("ability_codes", data.get("target_codes", []))
    if isinstance(ability_codes, str):
        ability_codes = [ability_codes]
    if not isinstance(ability_codes, list):
        ability_codes = []

    return {
        "dimension": str(data.get("dimension", "instructional_support")),
        "score": score,
        "reason": reason,
        "ability_codes": [str(code).strip() for code in ability_codes if str(code).strip()],
        "evidence": evidence,
        "uncertainty": str(data.get("uncertainty", "")),
        "raw_text": text,
        "score_invalid": score_invalid,
        "score_raw": score_raw,
    }
