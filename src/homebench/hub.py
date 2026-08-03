"""Live model list from the HuggingFace Hub.

Fetches the most popular text-generation models and reads their parameter
counts (from safetensors metadata) so `homebench fit` can size them against
your hardware. Results are cached under ``$HOMEBENCH_HOME`` so repeat runs are
fast and work offline; if the Hub is unreachable we fall back to any cache.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Tuple

import httpx

from .catalog import ModelSpec
from .history import default_home

HF_API = "https://huggingface.co/api/models"
_SORT_MAP = {"downloads": "downloads", "trending": "trendingScore", "likes": "likes"}
_DEFAULT_TTL = 24 * 3600  # 1 day


class HubError(RuntimeError):
    pass


def _parse(data, limit: int) -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for m in data:
        mid = m.get("id") or ""
        if not mid or "testing" in mid.lower():
            continue
        st = m.get("safetensors") or {}
        total = st.get("total")
        if not total:
            continue  # can't size it without a param count
        params_b = total / 1_000_000_000
        if params_b < 0.1:  # drop micro/test repos
            continue
        out.append(ModelSpec(
            name=mid,
            params_b=round(params_b, 2),
            family=(mid.split("/")[0] if "/" in mid else ""),
            ollama=None,
            hf=mid,
        ))
        if len(out) >= limit:
            break
    return out


def fetch_top_models(limit: int = 50, sort: str = "downloads",
                     timeout: float = 15.0) -> List[ModelSpec]:
    sort_key = _SORT_MAP.get(sort, sort)
    # Over-fetch, since some entries get filtered (no param count / test repos).
    params = [
        ("pipeline_tag", "text-generation"),
        ("sort", sort_key),
        ("direction", "-1"),
        ("limit", str(max(limit * 3, limit + 20))),
        ("expand[]", "safetensors"),
    ]
    try:
        r = httpx.get(HF_API, params=params, timeout=timeout,
                      headers={"User-Agent": "homebench"})
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HubError(f"could not reach HuggingFace Hub: {exc}")
    if not isinstance(data, list):
        raise HubError("unexpected response from HuggingFace Hub")
    return _parse(data, limit)


# ---- caching ---------------------------------------------------------------
def _cache_path(sort: str, home: Optional[str] = None) -> str:
    return os.path.join(home or default_home(), f"hf-top-{sort}.json")


def load_cache(sort: str, home: Optional[str] = None) -> Optional[Tuple[float, List[ModelSpec]]]:
    path = _cache_path(sort, home)
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    ts = float(blob.get("fetched_at", 0) or 0)
    models = [
        ModelSpec(name=m["name"], params_b=m["params_b"], family=m.get("family", ""),
                  ollama=m.get("ollama"), hf=m.get("hf"))
        for m in blob.get("models", [])
    ]
    return ts, models


def save_cache(sort: str, models: List[ModelSpec], home: Optional[str] = None) -> None:
    d = home or default_home()
    try:
        os.makedirs(d, exist_ok=True)
        with open(_cache_path(sort, home), "w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(),
                       "models": [m.to_dict() for m in models]}, f, indent=2)
    except OSError:
        pass  # caching is best-effort


def top_models(limit: int = 50, sort: str = "downloads", refresh: bool = False,
               ttl: float = _DEFAULT_TTL, home: Optional[str] = None
               ) -> Tuple[List[ModelSpec], str]:
    """Return (models, source-note). Uses fresh cache, else fetches, else stale cache."""
    if sort not in _SORT_MAP:
        raise HubError(f"unknown sort {sort!r}. Choices: {', '.join(_SORT_MAP)}")

    cached = load_cache(sort, home)
    if not refresh and cached is not None:
        ts, models = cached
        if models and (time.time() - ts) < ttl:
            return models[:limit], f"HuggingFace top by {sort} (cached)"

    try:
        models = fetch_top_models(limit=limit, sort=sort)
        save_cache(sort, models, home)
        return models, f"HuggingFace top {len(models)} by {sort} (live)"
    except HubError:
        if cached is not None and cached[1]:
            return cached[1][:limit], f"HuggingFace top by {sort} (stale cache — offline)"
        raise
