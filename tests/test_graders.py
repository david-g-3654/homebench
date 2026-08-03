from homebench.quality.graders import (
    all_of,
    contains_any,
    exact_number,
    multiple_choice,
    regex_match,
    valid_json,
)


def test_exact_number_takes_last_number():
    g = exact_number(391)
    assert g("17 * 20 = 340, plus 51 = 391").passed
    assert g("The answer is 391.").passed
    assert not g("I think it is 343").passed


def test_exact_number_handles_commas_and_none():
    assert exact_number(1234)("that's 1,234 total").passed
    r = exact_number(5)("no digits here")
    assert not r.passed and "no number" in r.detail


def test_multiple_choice_prefers_answer_line():
    g = multiple_choice("C")
    assert g("Answer: C because it cannot be concluded").passed
    assert g("C").passed
    assert not g("B").passed
    # "A" appearing inside a word should not count as bare letter
    assert g("The answer is C").passed


def test_contains_any_case_insensitive():
    g = contains_any(["Canberra"])
    assert g("The capital is canberra.").passed
    assert not g("Sydney").passed


def test_valid_json_requires_keys():
    g = valid_json(["name", "age"])
    assert g('{"name": "Alice", "age": 30}').passed
    assert g('```json\n{"name":"A","age":1}\n```').passed          # fenced
    assert not g('{"name": "Alice"}').passed                        # missing key
    assert not g("not json at all").passed


def test_regex_match():
    g = regex_match(r"support@example\.com")
    assert g("mail: support@example.com now").passed
    assert not g("support@other.com").passed


def test_all_of_averages_and_requires_all():
    g = all_of(contains_any(["a"]), contains_any(["b"]))
    full = g("a and b")
    assert full.passed and full.score == 1.0
    half = g("only a")
    assert not half.passed and half.score == 0.5
