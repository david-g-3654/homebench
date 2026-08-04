"""Tests for the laptop-speed defaults: quick subset + response cache."""

from homebench.cache import ResponseCache
from homebench.quality import default_suite, suite_categories
from homebench.runner import RunConfig, Runner, resolve_suite
from tests.fakes import FakeProvider


# ---- quick subset ----------------------------------------------------------
def test_quick_subset_is_smaller_and_covers_categories():
    full = default_suite()
    quick = default_suite(quick=True)
    assert 0 < len(quick) < len(full)
    # every quick task is flagged and deterministic
    assert all(t.quick and t.grader is not None for t in quick)
    # covers the same deterministic categories as the full suite
    assert set(suite_categories(quick=True)) == set(suite_categories())


def test_runner_defaults_to_quick():
    provider = FakeProvider()
    runner = Runner(provider, RunConfig(sample_rss=False, use_cache=False))
    result = runner.run([provider.list_models()[0]])
    assert len(result.reports[0].task_results) == len(default_suite(quick=True))


# ---- response cache --------------------------------------------------------
def test_cache_roundtrip(tmp_path):
    c = ResponseCache(home=str(tmp_path), enabled=True)
    k = c.key("model@digest", "task1", "the prompt", 128, 0.0, 42)
    assert c.get(k) is None and c.misses == 1
    c.set(k, "hello", 3)
    c.save()

    c2 = ResponseCache(home=str(tmp_path), enabled=True)
    hit = c2.get(k)
    assert hit == {"response": "hello", "output_tokens": 3}
    assert c2.hits == 1


def test_cache_disabled_is_noop(tmp_path):
    c = ResponseCache(home=str(tmp_path), enabled=False)
    k = c.key("m", "t", "p", 1, 0.0, 1)
    c.set(k, "x", 1)
    assert c.get(k) is None  # disabled -> never stores/returns


def test_refresh_ignores_existing(tmp_path):
    c = ResponseCache(home=str(tmp_path))
    k = c.key("m", "t", "p", 1, 0.0, 1)
    c.set(k, "x", 1)
    c.save()
    fresh = ResponseCache(home=str(tmp_path), refresh=True)
    assert fresh.get(k) is None


def test_runner_uses_cache_on_second_run(monkeypatch):
    provider = FakeProvider()
    calls = {"n": 0}
    real = provider.generate

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(provider, "generate", counting)

    # warmup also calls generate; disable it so we count only task generations.
    cfg = lambda: RunConfig(sample_rss=False, run_speed=False, warmup=False, use_cache=True)
    models = [provider.list_models()[0]]

    Runner(provider, cfg()).run(models)          # cold: generates every task
    first = calls["n"]
    assert first > 0

    Runner(provider, cfg()).run(models)          # warm: served from cache
    assert calls["n"] == first                    # no new generate calls


def test_refresh_cache_regenerates(monkeypatch):
    provider = FakeProvider()
    calls = {"n": 0}
    real = provider.generate
    monkeypatch.setattr(provider, "generate",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1])
    models = [provider.list_models()[0]]

    Runner(provider, RunConfig(sample_rss=False, run_speed=False, warmup=False)).run(models)
    first = calls["n"]
    Runner(provider, RunConfig(sample_rss=False, run_speed=False, warmup=False,
                               refresh_cache=True)).run(models)
    assert calls["n"] == first * 2                # refresh -> regenerate


def test_digest_cache_id():
    from homebench.models import ModelInfo

    assert ModelInfo("m", "ollama", digest="abc123").cache_id == "abc123"
    assert ModelInfo("m", "ollama", size_bytes=42).cache_id == "m:42"
