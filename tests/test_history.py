import pytest

from homebench import history
from homebench.history import (
    HistoryError,
    diff_table,
    list_runs,
    resolve_ref,
    save_run,
)
from homebench.models import (
    BenchmarkResult,
    MemoryMetrics,
    ModelInfo,
    ModelReport,
    SpeedMetrics,
    TaskResult,
)


def _make_result(started, models):
    """models: list of (name, tps, passed, total)."""
    reports = []
    for name, tps, passed, total in models:
        r = ModelReport(model=ModelInfo(name, "ollama", size_bytes=1_000_000_000,
                                        parameter_size="1B"))
        r.speed = SpeedMetrics(tokens_per_sec=tps, ttft_s=0.1)
        r.memory = MemoryMetrics(size_bytes=2_000_000_000)
        r.task_results = (
            [TaskResult(f"p{i}", "math", 1.0, True) for i in range(passed)]
            + [TaskResult(f"f{i}", "math", 0.0, False) for i in range(total - passed)]
        )
        reports.append(r)
    return BenchmarkResult(reports=reports, provider="ollama",
                           started_at=started, finished_at=started + 1)


def test_save_and_read_back(tmp_path):
    home = str(tmp_path)
    res = _make_result(1000.0, [("llama3.2", 50.0, 8, 10)])
    path = save_run(res, home=home, label="baseline")
    assert path.endswith(".json")
    runs = list_runs(home=home)
    assert len(runs) == 1
    rec = runs[0]
    assert rec.provider == "ollama"
    assert rec.label == "baseline"
    assert rec.model_names == ["llama3.2"]
    assert rec.best()["model"]["name"] == "llama3.2"


def test_runs_sorted_newest_first(tmp_path):
    home = str(tmp_path)
    save_run(_make_result(1000.0, [("a", 10.0, 5, 10)]), home=home)
    save_run(_make_result(2000.0, [("b", 20.0, 6, 10)]), home=home)
    runs = list_runs(home=home)
    assert [r.started_at for r in runs] == [2000.0, 1000.0]


def test_resolve_ref_variants(tmp_path):
    home = str(tmp_path)
    save_run(_make_result(1000.0, [("a", 10.0, 5, 10)]), home=home)
    save_run(_make_result(2000.0, [("b", 20.0, 6, 10)]), home=home)

    assert resolve_ref("latest", home=home).started_at == 2000.0
    assert resolve_ref("prev", home=home).started_at == 1000.0
    assert resolve_ref("1", home=home).started_at == 2000.0
    assert resolve_ref("2", home=home).started_at == 1000.0
    # by filename
    fname = list_runs(home=home)[0].filename
    assert resolve_ref(fname, home=home).started_at == 2000.0


def test_resolve_ref_insufficient(tmp_path):
    home = str(tmp_path)
    with pytest.raises(HistoryError):
        resolve_ref("latest", home=home)
    save_run(_make_result(1000.0, [("a", 10.0, 5, 10)]), home=home)
    with pytest.raises(HistoryError):
        resolve_ref("prev", home=home)  # only one run
    with pytest.raises(HistoryError):
        resolve_ref("nope.json", home=home)


def test_diff_metrics_and_table(tmp_path):
    home = str(tmp_path)
    save_run(_make_result(1000.0, [("shared", 10.0, 5, 10), ("gone", 30.0, 9, 10)]),
             home=home)
    save_run(_make_result(2000.0, [("shared", 20.0, 8, 10), ("fresh", 40.0, 7, 10)]),
             home=home)
    runs = list_runs(home=home)
    new, base = runs[0], runs[1]

    # metric extraction from the newer 'shared' report
    shared_b = next(r for r in new.reports if r["model"]["name"] == "shared")
    m = history._metrics(shared_b)
    assert m["tps"] == 20.0
    assert m["quality"] == 80.0

    table = diff_table(base, new)
    # union of models: shared, fresh (new), gone (removed) == 3 rows
    assert table.row_count == 3


def test_home_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HOMEBENCH_HOME", str(tmp_path))
    save_run(_make_result(1000.0, [("a", 10.0, 5, 10)]))  # no home arg -> uses env
    assert len(list_runs()) == 1


# ---- regression detection --------------------------------------------------
def _rec(started, models):
    return history.RunRecord(path="x.json", data=_make_result(started, models).to_dict())


def test_quality_regression_detected():
    base = _rec(1000.0, [("m", 20.0, 8, 10)])   # 80%
    new = _rec(2000.0, [("m", 20.0, 6, 10)])    # 60% -> -20 pts
    regs = history.regressions(base, new)
    assert len(regs) == 1
    assert regs[0].metric == "quality" and regs[0].drop == 20.0


def test_speed_regression_detected():
    base = _rec(1000.0, [("m", 20.0, 8, 10)])
    new = _rec(2000.0, [("m", 10.0, 8, 10)])    # -50% tok/s
    regs = history.regressions(base, new)
    assert len(regs) == 1 and regs[0].metric == "speed" and regs[0].drop == 50.0


def test_within_thresholds_is_clean():
    base = _rec(1000.0, [("m", 20.0, 8, 10)])
    new = _rec(2000.0, [("m", 19.0, 8, 10)])    # -5% speed, same quality
    assert history.regressions(base, new) == []


def test_thresholds_are_tunable():
    base = _rec(1000.0, [("m", 20.0, 8, 10)])
    new = _rec(2000.0, [("m", 20.0, 7, 10)])    # -10 pts quality
    assert history.regressions(base, new, quality_threshold=15.0) == []
    assert len(history.regressions(base, new, quality_threshold=5.0)) == 1


def test_non_shared_models_ignored():
    base = _rec(1000.0, [("gone", 20.0, 9, 10)])
    new = _rec(2000.0, [("fresh", 5.0, 1, 10)])
    assert history.regressions(base, new) == []
