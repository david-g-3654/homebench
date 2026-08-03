"""User-supplied task packs.

A task pack is a JSON or YAML file describing tasks with declarative graders,
so anyone can add domain-specific evals without writing Python. Example
(YAML)::

    name: my-pack
    tasks:
      - id: capital_fr
        category: factual
        prompt: "What is the capital of France? Answer with just the city."
        grader: {type: contains_any, values: ["Paris"]}
        reference: Paris
      - id: add
        category: math
        prompt: "What is 2 + 2? End with the answer on its own line."
        grader: {type: exact_number, value: 4}
      - id: explain            # no grader -> open-ended, scored by --judge
        category: open
        prompt: "Explain gravity in one sentence."
        reference: "Mass attracts mass; near Earth it accelerates objects downward."

Supported grader ``type`` values map to the built-in graders:
``exact_number``, ``multiple_choice``, ``contains_any``, ``regex``,
``valid_json``, ``valid_json_array``. Omit ``grader`` (or set it null / type
``judge``) for an open-ended task.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .graders import (
    Grader,
    contains_any,
    exact_number,
    multiple_choice,
    regex_match,
    valid_json,
    valid_json_array,
)
from .tasks import Task


class PackError(ValueError):
    """Raised when a task pack is malformed."""


# ---- grader specs ----------------------------------------------------------
def _grader_from_spec(spec: Optional[Dict[str, Any]], where: str) -> Optional[Grader]:
    if spec is None:
        return None  # open-ended (judge only)
    if not isinstance(spec, dict):
        raise PackError(f"{where}: 'grader' must be a mapping or null")
    gtype = spec.get("type")
    try:
        if gtype in (None, "judge", "open"):
            return None
        if gtype == "exact_number":
            return exact_number(float(spec["value"]), float(spec.get("tol", 1e-6)))
        if gtype == "multiple_choice":
            return multiple_choice(str(spec["value"]))
        if gtype == "contains_any":
            values = spec.get("values") or spec.get("value")
            if isinstance(values, str):
                values = [values]
            if not values:
                raise KeyError("values")
            return contains_any([str(v) for v in values])
        if gtype in ("regex", "regex_match"):
            flags = re.IGNORECASE if spec.get("ignorecase", True) else 0
            return regex_match(str(spec["pattern"]), flags=flags)
        if gtype == "valid_json":
            return valid_json(list(spec.get("keys", [])))
        if gtype == "valid_json_array":
            return valid_json_array(spec.get("length"))
    except KeyError as exc:
        raise PackError(f"{where}: grader type {gtype!r} is missing field {exc}")
    raise PackError(f"{where}: unknown grader type {gtype!r}")


def _task_from_spec(spec: Dict[str, Any], where: str) -> Task:
    if not isinstance(spec, dict):
        raise PackError(f"{where}: each task must be a mapping")
    if "id" not in spec:
        raise PackError(f"{where}: task is missing required field 'id'")
    tid = str(spec["id"])
    if "prompt" not in spec:
        raise PackError(f"task {tid!r}: missing required field 'prompt'")
    grader = _grader_from_spec(spec.get("grader"), f"task {tid!r}")
    category = str(spec.get("category", "custom"))
    if grader is None and category != "open":
        category = "open"  # normalize: no grader means judge-only
    return Task(
        id=tid,
        category=category,
        prompt=str(spec["prompt"]),
        grader=grader,
        max_tokens=int(spec.get("max_tokens", 256)),
        reference=str(spec.get("reference", "")),
    )


# ---- file loading ----------------------------------------------------------
def _parse_text(text: str, path: str) -> Any:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return json.loads(text)
    if ext in (".yaml", ".yml"):
        return _parse_yaml(text, path)
    # unknown extension: try JSON, then YAML
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _parse_yaml(text, path)


def _parse_yaml(text: str, path: str) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError:
        raise PackError(
            f"{path}: YAML task packs need PyYAML. Install it "
            "(`pip install pyyaml`) or use a .json pack."
        )
    return yaml.safe_load(text)


def load_pack(path: str) -> List[Task]:
    """Load a task pack file and return its tasks."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        raise PackError(f"could not read task pack {path!r}: {exc}")

    try:
        data = _parse_text(text, path)
    except (json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, PackError):
            raise
        raise PackError(f"{path}: could not parse pack: {exc}")

    if isinstance(data, dict):
        raw_tasks = data.get("tasks")
    elif isinstance(data, list):
        raw_tasks = data
    else:
        raw_tasks = None
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise PackError(f"{path}: pack must contain a non-empty list of tasks")

    tasks: List[Task] = []
    seen = set()
    for i, spec in enumerate(raw_tasks):
        task = _task_from_spec(spec, f"{path}[task {i}]")
        if task.id in seen:
            raise PackError(f"{path}: duplicate task id {task.id!r}")
        seen.add(task.id)
        tasks.append(task)
    return tasks


def load_packs(paths: List[str]) -> List[Task]:
    """Load and concatenate several packs, guarding against duplicate ids."""
    tasks: List[Task] = []
    seen = set()
    for p in paths:
        for task in load_pack(p):
            if task.id in seen:
                raise PackError(f"duplicate task id {task.id!r} across packs")
            seen.add(task.id)
            tasks.append(task)
    return tasks
