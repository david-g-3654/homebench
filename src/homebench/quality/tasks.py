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
    valid_json_array,
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
    Task(
        "math_percent", "math",
        "What is 15% of 240?" + _MATH_SUFFIX,
        exact_number(36), max_tokens=200, reference="36",
    ),
    Task(
        "math_algebra", "math",
        "If 3x + 7 = 22, what is the value of x?" + _MATH_SUFFIX,
        exact_number(5), max_tokens=200, reference="x = 5",
    ),
    Task(
        "math_sum", "math",
        "What is the sum of all integers from 1 to 10 inclusive?" + _MATH_SUFFIX,
        exact_number(55), max_tokens=200, reference="55",
    ),
    Task(
        "math_scale", "math",
        "A recipe needs 2.5 cups of flour per loaf. How many cups are needed "
        "for 4 loaves?" + _MATH_SUFFIX,
        exact_number(10), max_tokens=200, reference="10 cups",
    ),
    # -- reasoning (multiple choice + short answer) --------------------
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
    Task(
        "reason_syllogism", "reasoning",
        "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops "
        "necessarily Lazzies?\nA) Yes  B) No  C) Cannot be determined" + _MC_SUFFIX,
        multiple_choice("A"), max_tokens=100, reference="A (yes)",
    ),
    Task(
        "reason_odd", "reasoning",
        "Which word does not belong with the others?\n"
        "A) Dog  B) Cat  C) Car  D) Horse" + _MC_SUFFIX,
        multiple_choice("C"), max_tokens=100, reference="C (Car)",
    ),
    Task(
        "reason_order", "reasoning",
        "Tom is taller than Sam. Sam is taller than Bob. Who is the shortest? "
        "Answer with just the name.",
        contains_any(["Bob"]), max_tokens=60, reference="Bob",
    ),
    Task(
        "reason_time", "reasoning",
        "A meeting starts at 2:45 PM and lasts 90 minutes. What time does it "
        "end? Answer in H:MM format.",
        contains_any(["4:15"]), max_tokens=100, reference="4:15 PM",
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
    Task(
        "fact_planets", "factual",
        "How many planets are officially recognised in our solar system by the "
        "IAU? Answer with just the number.",
        contains_any(["8", "eight"]), max_tokens=60, reference="8",
    ),
    Task(
        "fact_water", "factual",
        "What is the chemical formula for water? Answer with just the formula.",
        contains_any(["H2O", "H₂O"]), max_tokens=60, reference="H2O",
    ),
    Task(
        "fact_author", "factual",
        "Who wrote the play 'Romeo and Juliet'? Answer with just the name.",
        contains_any(["Shakespeare"]), max_tokens=60, reference="William Shakespeare",
    ),
    Task(
        "fact_ocean", "factual",
        "What is the largest ocean on Earth? Answer with just the name.",
        contains_any(["Pacific"]), max_tokens=60, reference="Pacific Ocean",
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
    Task(
        "format_colors", "format",
        'Output ONLY a JSON object with a single key "colors" whose value is a '
        "list of the three primary colors.",
        valid_json(["colors"]), max_tokens=120,
        reference='{"colors": ["red", "yellow", "blue"]}',
    ),
    Task(
        "format_array", "format",
        "Output ONLY a JSON array containing the numbers 1, 2, and 3, in order.",
        valid_json_array(3), max_tokens=80, reference="[1, 2, 3]",
    ),
    Task(
        "format_upper", "format",
        "Respond with only the word 'hello' written in all uppercase letters, "
        "and nothing else.",
        regex_match(r"HELLO", flags=0), max_tokens=40, reference="HELLO",
    ),
    # -- extraction -----------------------------------------------------
    Task(
        "extract_email", "extraction",
        "Extract the email address from the following text and output only the "
        "address:\n\"Please contact us at support@example.com for assistance.\"",
        regex_match(r"support@example\.com"), max_tokens=60,
        reference="support@example.com",
    ),
    Task(
        "extract_year", "extraction",
        "Extract the year from the following text and output only the year:\n"
        "\"The company was founded in 1998 by two university students.\"",
        exact_number(1998), max_tokens=60, reference="1998",
    ),
    Task(
        "extract_name", "extraction",
        "Extract the person's full name from the following text and output only "
        "the name:\n\"The report was submitted by Dr. Maria Chen last Friday.\"",
        contains_any(["Maria Chen"]), max_tokens=60, reference="Maria Chen",
    ),
    # -- code understanding --------------------------------------------
    Task(
        "code_eval", "code",
        "What does this Python expression evaluate to? len('hello') + 2\n"
        "End with the final answer on its own line.",
        exact_number(7), max_tokens=150, reference="7",
    ),
    Task(
        "code_pow", "code",
        "What is the output of this Python code?\nprint(2 ** 5)\n"
        "End with the answer on its own line.",
        exact_number(32), max_tokens=150, reference="32",
    ),
    Task(
        "code_len", "code",
        "In Python, what does len([1, [2, 3], 4]) return?\n"
        "End with the answer on its own line.",
        exact_number(3), max_tokens=150, reference="3",
    ),
    Task(
        "code_listcomp", "code",
        "What does this Python expression evaluate to? [x * 2 for x in range(4)]\n"
        "End with the answer on its own line.",
        regex_match(r"\[\s*0\s*,\s*2\s*,\s*4\s*,\s*6\s*\]"), max_tokens=150,
        reference="[0, 2, 4, 6]",
    ),
    Task(
        "code_str", "code",
        "In Python, what is the result of 'ab' * 3? Output only the resulting "
        "string.",
        contains_any(["ababab"]), max_tokens=60, reference="ababab",
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
    Task(
        "open_hash", "open",
        "Explain in two sentences what a hash function is.",
        grader=None, max_tokens=200,
        reference="A hash function maps input data of any size to a fixed-size "
        "value; it is deterministic and hard to invert, which makes it useful "
        "for lookups and integrity checks.",
    ),
    Task(
        "open_haiku", "open",
        "Write a haiku about autumn (three short lines).",
        grader=None, max_tokens=120,
        reference="A three-line poem evoking autumn imagery.",
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
