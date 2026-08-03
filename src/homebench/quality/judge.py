"""Optional LLM-as-judge.

Uses one of the local models (or any provider model you point it at) to score
a response against a reference on a 1-5 scale. Kept deliberately simple and
deterministic (temperature 0); it's a signal, not an oracle.
"""

from __future__ import annotations

import re

from ..providers.base import Provider, ProviderError
from .graders import Grade

_JUDGE_TEMPLATE = """You are a strict grader. Rate how well the RESPONSE answers the TASK.

TASK:
{task}

REFERENCE (an example of a good answer):
{reference}

RESPONSE:
{response}

Grade on a 1-5 scale where 1 = wrong/irrelevant and 5 = fully correct and \
well-formed. Reply with exactly one line: "SCORE: <n>" followed by a short \
reason on the same line."""


class LLMJudge:
    def __init__(self, provider: Provider, model: str, timeout: float = 120.0):
        self.provider = provider
        self.model = model
        self.timeout = timeout

    def score(self, task_prompt: str, response: str, reference: str = "") -> Grade:
        prompt = _JUDGE_TEMPLATE.format(
            task=task_prompt.strip(),
            reference=(reference or "(none provided)").strip(),
            response=(response or "(empty response)").strip(),
        )
        try:
            result = self.provider.generate(
                self.model, prompt, max_tokens=120, temperature=0.0,
                timeout=self.timeout,
            )
        except ProviderError as exc:
            return Grade(0.0, False, f"judge error: {exc}")

        text = result.text
        m = re.search(r"SCORE\s*[:=]?\s*([1-5])", text, re.IGNORECASE)
        if not m:
            m = re.search(r"\b([1-5])\b", text)
        if not m:
            return Grade(0.0, False, f"judge gave no score: {text[:80]!r}")
        raw = int(m.group(1))
        norm = (raw - 1) / 4.0  # 1..5 -> 0..1
        reason = text.strip().splitlines()[0][:120] if text.strip() else ""
        return Grade(norm, raw >= 3, f"judge {raw}/5 — {reason}")
