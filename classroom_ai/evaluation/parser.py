from __future__ import annotations

import json
import re
from typing import Any

from classroom_ai.schemas.evidence import normalize_evidence


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_INVALID_ESCAPE_RE = re.compile(r"\\(?![\"\\/bfnrtu])")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_MISSING_COMMA_RE = re.compile(r"([}\]])\s*([{\[])")


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

    fence_match = re.match(r"```(?:json)?\s*\n(.*?)\n```", stripped, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

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


def parse_evaluation_response(text: str) -> dict[str, Any]:
    """Parse JSON LLM output with a conservative fallback for notebook use."""

    clean = _extract_json_block(text)
    data = _robust_json_parse(clean)

    score_raw = data.get("score", 0)
    score_invalid = False
    try:
        score = int(round(float(score_raw)))
    except (TypeError, ValueError):
        score = 0
        score_invalid = True
    if score and not (1 <= score <= 7):
        score_invalid = True
    score = max(1, min(7, score)) if score else 0

    reason = str(data.get("reason") or data.get("summary") or text).strip()
    evidence = normalize_evidence(data.get("evidence", []))

    return {
        "dimension": str(data.get("dimension", "instructional_support")),
        "score": score,
        "reason": reason,
        "evidence": evidence,
        "uncertainty": str(data.get("uncertainty", "")),
        "raw_text": text,
        "score_invalid": score_invalid,
        "score_raw": score_raw,
    }
