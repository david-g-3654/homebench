"""Deterministic response cache for the quality suite.

Quality runs use temperature 0 and a fixed seed, so a given (model version,
task, generation params) always produces the same output. We cache the raw
response text keyed on those, so re-running only pays for *new* models/tasks
and re-grading (which is cheap) — instead of regenerating everything.

Only deterministic generations (temperature 0) are cached, and only the raw
response is stored, so changing a grader still takes effect on the next run.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

from .history import default_home


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


class ResponseCache:
    def __init__(self, home: Optional[str] = None, enabled: bool = True,
                 refresh: bool = False):
        self.enabled = enabled
        self.path = os.path.join(home or default_home(), "response-cache.json")
        self._data: Dict[str, Any] = {} if refresh else self._load()
        self._dirty = False
        self.hits = 0
        self.misses = 0

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            return blob.get("responses", {}) if isinstance(blob, dict) else {}
        except (OSError, json.JSONDecodeError, ValueError):
            return {}

    @staticmethod
    def key(model_id: str, task_id: str, prompt: str, max_tokens: int,
            temperature: float, seed: Optional[int]) -> str:
        return "|".join([
            model_id, task_id, _hash(prompt),
            f"mt={max_tokens}", f"t={temperature}", f"s={seed}",
        ])

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        hit = self._data.get(key)
        if hit is not None:
            self.hits += 1
        else:
            self.misses += 1
        return hit

    def set(self, key: str, response: str, output_tokens: int) -> None:
        if not self.enabled:
            return
        self._data[key] = {"response": response, "output_tokens": output_tokens}
        self._dirty = True

    def save(self) -> None:
        if not self.enabled or not self._dirty:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"responses": self._data}, f)
            self._dirty = False
        except OSError:
            pass  # caching is best-effort
