from __future__ import annotations

from collections import Counter


def majority_vote(scores: list[int]) -> int:
    if not scores:
        return 0
    return Counter(scores).most_common(1)[0][0]


def route_decision(
    semantic_entropy: float,
    score_entropy: float,
    semantic_entropy_threshold: float,
    score_entropy_threshold: float,
) -> str:
    if semantic_entropy <= semantic_entropy_threshold and score_entropy <= score_entropy_threshold:
        return "auto_accept"
    return "human_review"
