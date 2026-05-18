from __future__ import annotations

import hashlib
import json
import random
import re

from classroom_ai.llm.base import BaseLLM, LLMResponse


class MockLocalLLM(BaseLLM):
    """Deterministic offline LLM stand-in for intranet smoke tests.

    The mock reads the transcript prompt, applies simple teaching-behavior
    heuristics, and injects temperature-controlled variation. It is not a real
    evaluator; it exists so the complete Monte Carlo and entropy pipeline can run
    before local model weights are installed.
    """

    positive_terms = ("追问", "为什么", "解释", "分析", "证据", "鼓励", "很好", "想一想")
    negative_terms = ("太简单", "不对", "错了", "安静", "不要说话", "批评")

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
        prompt = "\n".join(message.get("content", "") for message in messages)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16) + len(prompt)
        rng = random.Random(seed + random.randint(0, 1_000_000))

        pos = sum(prompt.count(term) for term in self.positive_terms)
        neg = sum(prompt.count(term) for term in self.negative_terms)
        questions = len(re.findall(r"[？?]", prompt))
        base_score = 4 + min(2, pos // 2) + min(1, questions // 3) - min(2, neg // 2)
        jitter = rng.choices([-1, 0, 1], weights=[temperature, 1.8, temperature], k=1)[0]
        score = max(1, min(7, base_score + jitter))

        if score >= 5:
            reason = "教师通过提问、追问或鼓励性反馈支持学生表达和概念理解。"
        elif score <= 3:
            reason = "教师反馈偏控制或否定，文本证据不足以说明形成了高质量教学支持。"
        else:
            reason = "片段中存在一定互动，但追问深度和支架质量仍不稳定。"

        payload = {
            "dimension": "instructional_support",
            "score": score,
            "reason": reason,
            "evidence": self._extract_evidence(prompt),
            "uncertainty": "mock_local provider; replace with local LLM or API provider for formal evaluation.",
        }
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False), raw=payload)

    def _extract_evidence(self, prompt: str) -> list[str]:
        evidence = []
        for line in prompt.splitlines():
            if any(term in line for term in self.positive_terms + self.negative_terms):
                evidence.append(line.strip())
            if len(evidence) >= 3:
                break
        return evidence
