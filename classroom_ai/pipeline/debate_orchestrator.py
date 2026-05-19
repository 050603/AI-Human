from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DebateRoundResult:
    converged: bool
    rounds: int
    updated_reasons: dict[str, str]


def run_debate(
    high_score_reason: str,
    low_score_reason: str,
    max_rounds: int = 2,
) -> DebateRoundResult:
    """Stub debate protocol for heterogeneous model reasoning exchange."""

    reasons = {
        "high_score_model": high_score_reason,
        "low_score_model": low_score_reason,
    }
    converged = _normalize(high_score_reason) == _normalize(low_score_reason)
    rounds = 0

    while not converged and rounds < max_rounds:
        rounds += 1
        reasons["high_score_model"] = (
            f"参考另一位AI专家的质疑后复核：{high_score_reason}；同时关注反方观点：{low_score_reason}"
        )
        reasons["low_score_model"] = (
            f"参考另一位AI专家的支持后复核：{low_score_reason}；同时吸收正方证据：{high_score_reason}"
        )
        converged = rounds >= max_rounds

    return DebateRoundResult(converged=converged, rounds=rounds, updated_reasons=reasons)


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower().strip() if ch.isalnum())
