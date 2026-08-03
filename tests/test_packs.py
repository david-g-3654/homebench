import json
import os

import pytest

from localbench.quality import default_suite, load_pack, load_packs
from localbench.quality.packs import PackError

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data) if name.endswith(".json") else data)
    return str(p)


def test_load_json_pack(tmp_path):
    path = _write(tmp_path, "p.json", {
        "name": "t",
        "tasks": [
            {"id": "a", "category": "math", "prompt": "2+2?",
             "grader": {"type": "exact_number", "value": 4}, "reference": "4"},
            {"id": "b", "category": "factual", "prompt": "cap of France?",
             "grader": {"type": "contains_any", "values": ["Paris"]},
             "reference": "Paris"},
        ],
    })
    tasks = load_pack(path)
    assert [t.id for t in tasks] == ["a", "b"]
    assert tasks[0].grader("the answer is 4").passed
    assert tasks[1].grader("Paris").passed


def test_bare_list_pack(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "x", "prompt": "hi", "grader": {"type": "contains_any", "values": ["y"]}},
    ])
    tasks = load_pack(path)
    assert len(tasks) == 1 and tasks[0].category == "custom"


def test_open_task_has_no_grader(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "o", "prompt": "explain X", "reference": "because"},
    ])
    t = load_pack(path)[0]
    assert t.grader is None
    assert t.category == "open"  # normalized


def test_all_grader_types(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "n", "prompt": "?", "grader": {"type": "exact_number", "value": 7},
         "reference": "7"},
        {"id": "mc", "prompt": "?", "grader": {"type": "multiple_choice", "value": "B"},
         "reference": "B"},
        {"id": "c", "prompt": "?", "grader": {"type": "contains_any", "values": ["a", "b"]},
         "reference": "a"},
        {"id": "r", "prompt": "?", "grader": {"type": "regex", "pattern": "foo\\d+"},
         "reference": "foo12"},
        {"id": "j", "prompt": "?", "grader": {"type": "valid_json", "keys": ["k"]},
         "reference": "{\"k\": 1}"},
        {"id": "arr", "prompt": "?", "grader": {"type": "valid_json_array", "length": 2},
         "reference": "[1, 2]"},
    ])
    for t in load_pack(path):
        assert t.grader(t.reference).passed, t.id


def test_regex_case_sensitive_flag(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "r", "prompt": "?",
         "grader": {"type": "regex", "pattern": "HELLO", "ignorecase": False}},
    ])
    g = load_pack(path)[0].grader
    assert g("HELLO").passed
    assert not g("hello").passed


def test_unknown_grader_type_errors(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "x", "prompt": "?", "grader": {"type": "nonsense"}},
    ])
    with pytest.raises(PackError):
        load_pack(path)


def test_missing_required_fields(tmp_path):
    p1 = _write(tmp_path, "p1.json", [{"category": "math", "prompt": "?"}])
    with pytest.raises(PackError):
        load_pack(p1)
    p2 = _write(tmp_path, "p2.json", [{"id": "a"}])
    with pytest.raises(PackError):
        load_pack(p2)


def test_duplicate_ids(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "a", "prompt": "?", "grader": {"type": "contains_any", "values": ["x"]}},
        {"id": "a", "prompt": "?", "grader": {"type": "contains_any", "values": ["y"]}},
    ])
    with pytest.raises(PackError):
        load_pack(path)


def test_empty_pack_errors(tmp_path):
    path = _write(tmp_path, "p.json", {"tasks": []})
    with pytest.raises(PackError):
        load_pack(path)


def test_load_packs_dedup_across_files(tmp_path):
    a = _write(tmp_path, "a.json", [
        {"id": "dup", "prompt": "?", "grader": {"type": "contains_any", "values": ["x"]}}])
    b = _write(tmp_path, "b.json", [
        {"id": "dup", "prompt": "?", "grader": {"type": "contains_any", "values": ["y"]}}])
    with pytest.raises(PackError):
        load_packs([a, b])


def test_shipped_example_pack_json():
    path = os.path.join(EXAMPLES, "sample_pack.json")
    tasks = load_pack(path)
    assert len(tasks) == 4
    for t in tasks:
        if t.grader is not None:
            assert t.grader(t.reference).passed, t.id


def test_yaml_pack():
    yaml = pytest.importorskip("yaml")  # noqa: F841
    path = os.path.join(EXAMPLES, "sample_pack.yaml")
    tasks = load_pack(path)
    assert any(t.category == "open" for t in tasks)
    for t in tasks:
        if t.grader is not None:
            assert t.grader(t.reference).passed, t.id
