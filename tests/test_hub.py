import json

import pytest

from homebench import hub
from homebench.hub import (
    HubError,
    _parse,
    fetch_top_models,
    load_cache,
    save_cache,
    top_models,
)

SAMPLE = [
    {"id": "Qwen/Qwen3-8B", "safetensors": {"total": 8_190_735_360}},
    {"id": "meta-llama/Llama-3.2-1B-Instruct", "safetensors": {"total": 1_240_000_000}},
    {"id": "facebook/opt-125m", "safetensors": {}},               # no total -> skip
    {"id": "some/no-safetensors"},                                # missing -> skip
    {"id": "trl-internal-testing/tiny-model", "safetensors": {"total": 2_000_000}},  # testing -> skip
    {"id": "micro/nano", "safetensors": {"total": 5_000_000}},    # < 0.1B -> skip
]


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_parse_filters_and_sizes():
    models = _parse(SAMPLE, limit=50)
    names = [m.name for m in models]
    assert names == ["Qwen/Qwen3-8B", "meta-llama/Llama-3.2-1B-Instruct"]
    m = models[0]
    assert round(m.params_b, 2) == 8.19
    assert m.hf == "Qwen/Qwen3-8B"
    assert m.ollama is None          # unknown for arbitrary HF ids
    assert m.family == "Qwen"


def test_parse_respects_limit():
    assert len(_parse(SAMPLE, limit=1)) == 1


def test_fetch_top_models(monkeypatch):
    monkeypatch.setattr(hub.httpx, "get", lambda *a, **k: _FakeResp(SAMPLE))
    models = fetch_top_models(limit=10, sort="downloads")
    assert [m.name for m in models] == ["Qwen/Qwen3-8B", "meta-llama/Llama-3.2-1B-Instruct"]


def test_fetch_bad_sort_still_maps(monkeypatch):
    seen = {}

    def fake_get(url, params=None, **k):
        seen["params"] = dict(params)
        return _FakeResp(SAMPLE)

    monkeypatch.setattr(hub.httpx, "get", fake_get)
    fetch_top_models(sort="trending")
    assert seen["params"]["sort"] == "trendingScore"


def test_fetch_network_error(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(hub.httpx, "get", boom)
    with pytest.raises(HubError):
        fetch_top_models()


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(hub.httpx, "get", lambda *a, **k: _FakeResp(SAMPLE))
    models = fetch_top_models()
    save_cache("downloads", models, home=str(tmp_path))
    ts, loaded = load_cache("downloads", home=str(tmp_path))
    assert ts > 0
    assert [m.name for m in loaded] == [m.name for m in models]


def test_top_models_uses_fresh_cache(tmp_path, monkeypatch):
    # Pre-populate cache; network must NOT be called.
    save_cache("downloads", _parse(SAMPLE, 50), home=str(tmp_path))

    def fail(*a, **k):
        raise AssertionError("should not fetch when cache is fresh")

    monkeypatch.setattr(hub.httpx, "get", fail)
    models, note = top_models(sort="downloads", home=str(tmp_path))
    assert models and "cached" in note


def test_top_models_refresh_bypasses_cache(tmp_path, monkeypatch):
    save_cache("downloads", _parse(SAMPLE, 50), home=str(tmp_path))
    monkeypatch.setattr(hub.httpx, "get", lambda *a, **k: _FakeResp(SAMPLE))
    models, note = top_models(sort="downloads", refresh=True, home=str(tmp_path))
    assert "live" in note


def test_top_models_stale_cache_on_offline(tmp_path, monkeypatch):
    save_cache("downloads", _parse(SAMPLE, 50), home=str(tmp_path))
    # force cache "stale" and network down -> should still return stale cache
    monkeypatch.setattr(hub, "_DEFAULT_TTL", -1)

    def boom(*a, **k):
        import httpx
        raise httpx.ConnectError("down")

    monkeypatch.setattr(hub.httpx, "get", boom)
    models, note = top_models(sort="downloads", ttl=-1, home=str(tmp_path))
    assert models and "stale" in note


def test_top_models_unknown_sort(tmp_path):
    with pytest.raises(HubError):
        top_models(sort="bogus", home=str(tmp_path))
