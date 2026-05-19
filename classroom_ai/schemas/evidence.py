from __future__ import annotations

import json
from typing import Any


def _coerce_item(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        try:
            parsed = json.loads(item.replace("'", '"'))
            if isinstance(parsed, dict):
                item = parsed
            else:
                return {"line_start": None, "line_end": None, "text": item}
        except Exception:
            return {"line_start": None, "line_end": None, "text": item}
    if isinstance(item, dict):
        return {
            "line_start": item.get("line_start"),
            "line_end": item.get("line_end"),
            "text": str(item.get("text", "")).strip(),
        }
    return None


def normalize_evidence(evidence: Any) -> list[dict[str, Any]]:
    if isinstance(evidence, dict):
        evidence = [evidence]
    elif isinstance(evidence, str):
        evidence = [evidence]
    elif not isinstance(evidence, list):
        evidence = []

    result = []
    for item in evidence:
        normalized = _coerce_item(item)
        if normalized and normalized.get("text"):
            result.append(normalized)
    return result
