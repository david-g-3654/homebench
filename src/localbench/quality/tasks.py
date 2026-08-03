"""The curated quality suite.

Small on purpose: enough tasks across categories to separate models, few
enough to run every model in a couple of minutes on a laptop. Every task
is deterministically graded except the ``open`` category, which is only
included when an LLM judge is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .graders import (
    Grader,
    contains_any,
    exact_number,
    multiple_choice,
    regex_match,
    valid_json,
)

# Nudge small models toward parseable output without over-constraining them.
_MATH_SUFFIX = " Think briefly, then end with the final answer on its own line."
_MC_SUFFIX = " Respond with only the letter of the correct answer."


@dataclass
class Task:
    id: str
    category: str
    prompt: str
    grader: Optional[Grader] = None   # None => open-ended, judge-only
    max_tokens: int = 256
    reference: str = ""               # ideal answer, shown to the judge


_TASKS: List[Task] = [
    # -- arithmetic / math word problems -------------------------------
    Task(
        "math_mult", "math",
        "What is 17 multiplied by 23?" + _MATH_SUFFIX,
        exact_number(391), max_tokens=200, reference="391",
    ),
    Task(
        "math_word", "math",
        "A store had 48 apples. It sold three quarters of them, then received "
        "15 more. How many apples does the store have now?" + _MATH_SUFFIX,
        exact_number(27), max_tokens=300, reference="27",
    ),
    Task(
        "math_rate", "math",
        "A train travels 60 miles in 1.5 hours. What is its average speed in "
        "miles per hour?" + _MATH_SUFFIX,
        exact_number(40), max_tokens=200, reference="40 mph",
    ),
    # -- reasoning (multiple choice) -----------------------------------
    Task(
        "reason_seq", "reasoning",
        "Which number continues the sequence 2, 6, 12, 20, 30, ?\n"
        "A) 36  B) 42  C) 40  D) 44" + _MC_SUFFIX,
        multiple_choice("B"), max_tokens=100, reference="B (42)",
    ),
    Task(
        "reason_logic", "reasoning",
        "All roses are flowers. Some flowers fade quickly. Which statement must "
        "be true?\nA) All roses fade quickly\nB) Some roses fade quickly\n"
        "C) It cannot be concluded whether any roses fade quickly\n"
        "D) No roses fade quickly" + _MC_SUFFIX,
        multiple_choice("C"), max_tokens=100, reference="C",
    ),
    # -- closed-book factual -------------------------------------------
    Task(
        "fact_capital", "factual",
        "What is the capital city of Australia? Answer with just the city name.",
        contains_any(["Canberra"]), max_tokens=60, reference="Canberra",
    ),
    Task(
        "fact_element", "factual",
        "Which chemical element has the symbol 'Fe'? Answer with just the name.",
        contains_any(["Iron"]), max_tokens=60, reference="Iron",
    ),
    # -- instruction following / structured output ----------------------
    Task(
        "format_json", "format",
        'Output ONLY a JSON object (no prose, no code fence) with exactly the '
        'keys "name" and "age" describing a person named Alice who is 30 years '
        "old.",
        valid_json(["name", "age"]), max_tokens=120,
        reference='{"name": "Alice", "age": 30}',
    ),
    # -- extraction -----------------------------------------------------
    Task(
        "extract_email", "extraction",
        "Extract the email address from the following text and output only the "
        "address:\n\"Please contact us at support@example.com for assistance.\"",
        regex_match(r"support@example\.com"), max_tokens=60,
        reference="support@example.com",
    ),
    # -- code understanding --------------------------------------------
    Task(
        "code_eval", "code",
        "What does this Python expression evaluate to? len('hello') + 2\n"
        "End with the final answer on its own line.",
        exact_number(7), max_tokens=150, reference="7",
    ),
    # -- open-ended (judge only) ---------------------------------------
    Task(
        "open_summary", "open",
        "Summarize in one sentence why the sky appears blue during the day.",
        grader=None, max_tokens=200,
        reference="Sunlight scatters off air molecules, and shorter (blue) "
        "wavelengths scatter much more than longer ones (Rayleigh scattering), "
        "so we see blue.",
    ),
    Task(
        "open_email", "open",
        "Write a polite two-sentence email declining a meeting invitation "
        "because of a scheduling conflict.",
        grader=None, max_tokens=200,
        reference="A courteous, well-formed two-sentence email that declines "
        "and cites a scheduling conflict.",
    ),
]


def default_suite(include_open: bool = False) -> List[Task]:
    """Return the suite. Open-ended tasks are excluded unless a judge is on."""
    if include_open:
        return list(_TASKS)
    return [t for t in _TASKS if t.category != "open"]


def suite_categories(include_open: bool = False) -> List[str]:
    seen: List[str] = []
    for t in default_suite(include_open):
        if t.category not in seen:
            seen.append(t.category)
    return seen
