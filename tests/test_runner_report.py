import json

from homebench.quality import default_suite
from homebench.report import rank_reports, to_json, to_markdown
from homebench.runner import RunConfig, Runner
from tests.fakes import FakeProvider

SUITE_SIZE = len(default_suite(include_open=False))


def _run(**cfg_kwargs):
    provider = FakeProvider()
    # These tests exercise the full suite; the quick subset has its own test.
    cfg_kwargs.setdefault("quick", False)
    runner = Runner(provider, RunConfig(sample_rss=False, **cfg_kwargs))
    result = runner.run(provider.list_models())
    return provider, result


def test_runner_produces_report_per_model():
    provider, result = _run()
    assert len(result.reports) == 2
    names = {r.model.name for r in result.reports}
    assert names == {"fast:1b", "smart:8b"}
    for r in result.reports:
        assert r.error is None
        assert r.speed.tokens_per_sec > 0
        assert r.quality_score is not None
        assert len(r.task_results) == SUITE_SIZE


def test_all_canned_answers_pass():
    _, result = _run()
    for r in result.reports:
        # FakeProvider answers every deterministic task with its reference
        assert r.quality_score == 100.0
        assert r.tasks_passed == SUITE_SIZE


def test_unload_called_between_models():
    provider, _ = _run()
    assert set(provider.unloaded) == {"fast:1b", "smart:8b"}


def test_events_emitted():
    provider = FakeProvider()
    runner = Runner(provider, RunConfig(sample_rss=False, quick=False))
    seen = []
    runner.run(provider.list_models(), observer=lambda ev, **d: seen.append(ev))
    assert "run_start" in seen
    assert "model_done" in seen
    assert "run_done" in seen
    assert seen.count("task_done") == SUITE_SIZE * 2


def test_ranking_quality_then_speed():
    # equal quality -> faster model ranks first
    _, result = _run()
    ranked = rank_reports(result.reports)
    assert ranked[0].model.name == "fast:1b"  # 120 tps beats 40 tps


def test_markdown_and_json_export():
    _, result = _run()
    md = to_markdown(result)
    assert "# homebench results" in md
    assert "## Leaderboard" in md
    assert "fast:1b" in md and "smart:8b" in md
    assert "Quality by category" in md

    data = json.loads(to_json(result))
    assert data["provider"] == "ollama"
    assert len(data["reports"]) == 2
    assert data["reports"][0]["tasks_total"] == SUITE_SIZE


def test_speed_only_skips_quality():
    _, result = _run(run_quality=False)
    for r in result.reports:
        assert r.task_results == []
        assert r.quality_score is None
        assert r.speed.tokens_per_sec > 0
