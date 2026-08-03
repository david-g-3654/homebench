"""Guards on the quality suite itself."""

import pytest

from localbench.quality import default_suite, suite_categories


DETERMINISTIC = [t for t in default_suite(include_open=True) if t.grader is not None]


def test_task_ids_are_unique():
    ids = [t.id for t in default_suite(include_open=True)]
    assert len(ids) == len(set(ids))


def test_suite_has_variety():
    cats = suite_categories()
    for expected in ("math", "reasoning", "factual", "format", "extraction", "code"):
        assert expected in cats


def test_open_tasks_excluded_by_default():
    assert all(t.category != "open" for t in default_suite())
    assert any(t.category == "open" for t in default_suite(include_open=True))


@pytest.mark.parametrize("task", DETERMINISTIC, ids=[t.id for t in DETERMINISTIC])
def test_reference_answer_passes_its_grader(task):
    """Every deterministic task's reference must satisfy its own grader."""
    grade = task.grader(task.reference)
    assert grade.passed, f"{task.id}: reference {task.reference!r} -> {grade.detail}"
