"""Deterministic graders.

Each grader is a callable ``response -> Grade``. They are intentionally
forgiving about surrounding prose (models love to explain) but strict about
the actual answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, List, Sequence


@dataclass
class Grade:
    score: float          # 0.0 .. 1.0
    passed: bool
    detail: str = ""


Grader = Callable[[str], Grade]


_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _numbers(text: str) -> List[float]:
    out: List[float] = []
    for m in _NUMBER_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def exact_number(expected: float, tol: float = 1e-6) -> Grader:
    """Pass if the last number in the response equals ``expected``."""

    def grade(response: str) -> Grade:
        nums = _numbers(response)
        if not nums:
            return Grade(0.0, False, "no number found in response")
        got = nums[-1]
        ok = abs(got - expected) <= tol
        return Grade(1.0 if ok else 0.0, ok, f"got {got}, expected {expected}")

    return grade


def multiple_choice(expected: str) -> Grader:
    """Pass if the first standalone A-E letter matches ``expected``."""
    expected = expected.strip().upper()

    def grade(response: str) -> Grade:
        # Prefer an explicit "answer: X" if present, else first bare letter.
        m = re.search(r"answer\s*[:\-]?\s*\(?([A-E])\b", response, re.IGNORECASE)
        if not m:
            m = re.search(r"\b([A-E])\b", response)
        if not m:
            return Grade(0.0, False, "no A-E choice found")
        got = m.group(1).upper()
        ok = got == expected
        return Grade(1.0 if ok else 0.0, ok, f"chose {got}, expected {expected}")

    return grade


def contains_any(answers: Sequence[str]) -> Grader:
    """Pass if the response contains any accepted answer (case-insensitive)."""
    lowered = [a.lower() for a in answers]

    def grade(response: str) -> Grade:
        r = response.lower()
        for a in lowered:
            if a in r:
                return Grade(1.0, True, f"matched {a!r}")
        return Grade(0.0, False, f"none of {list(answers)} present")

    return grade


def regex_match(pattern: str, flags: int = re.IGNORECASE) -> Grader:
    rx = re.compile(pattern, flags)

    def grade(response: str) -> Grade:
        ok = rx.search(response) is not None
        return Grade(1.0 if ok else 0.0, ok, f"pattern {pattern!r}")

    return grade


def _extract_json(text: str, open_ch: str = "{", close_ch: str = "}") -> str:
    """Pull the first JSON value out of a response, tolerating code fences."""
    fenced = re.search(
        r"```(?:json)?\s*(" + re.escape(open_ch) + r".*?" + re.escape(close_ch) + r")\s*```",
        text, re.DOTALL,
    )
    if fenced:
        return fenced.group(1)
    span = re.search(re.escape(open_ch) + r".*" + re.escape(close_ch), text, re.DOTALL)
    return span.group(0) if span else text


def valid_json(required_keys: Sequence[str] = ()) -> Grader:
    """Pass if the response contains a JSON object with all required keys."""

    def grade(response: str) -> Grade:
        raw = _extract_json(response)
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            return Grade(0.0, False, f"invalid JSON: {exc}")
        if not isinstance(obj, dict):
            return Grade(0.0, False, "JSON is not an object")
        missing = [k for k in required_keys if k not in obj]
        if missing:
            return Grade(0.0, False, f"missing keys: {missing}")
        return Grade(1.0, True, "valid JSON with required keys")

    return grade


def valid_json_array(length: int = None) -> Grader:
    """Pass if the response contains a JSON array (optionally of exact length)."""

    def grade(response: str) -> Grade:
        raw = _extract_json(response, "[", "]")
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            return Grade(0.0, False, f"invalid JSON: {exc}")
        if not isinstance(obj, list):
            return Grade(0.0, False, "JSON is not an array")
        if length is not None and len(obj) != length:
            return Grade(0.0, False, f"array length {len(obj)}, expected {length}")
        return Grade(1.0, True, f"valid JSON array (len {len(obj)})")

    return grade


def all_of(*graders: Grader) -> Grader:
    """Average the sub-grader scores; pass only if every sub-grader passes."""

    def grade(response: str) -> Grade:
        grades = [g(response) for g in graders]
        score = sum(g.score for g in grades) / len(grades)
        passed = all(g.passed for g in grades)
        detail = "; ".join(g.detail for g in grades)
        return Grade(score, passed, detail)

    return grade
