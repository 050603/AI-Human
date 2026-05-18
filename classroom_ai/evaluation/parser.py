from __future__ import annotations

import json
import re
from typing import Any


def parse_evaluation_response(text: str) -> dict[str, Any]:
    """Parse JSON LLM output with a conservative fallback for notebook use."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            data = {}
        else:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = {}

    score = data.get("score", 0)
    try:
        score = int(round(float(score)))
    except (TypeError, ValueError):
        score = 0
    score = max(1, min(7, score)) if score else 0

    reason = str(data.get("reason") or data.get("summary") or text).strip()
    evidence = data.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []

    return {
        "dimension": str(data.get("dimension", "instructional_support")),
        "score": score,
        "reason": reason,
        "evidence": [str(item) for item in evidence],
        "uncertainty": str(data.get("uncertainty", "")),
        "raw_text": text,
    }
