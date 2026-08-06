"""Value score + `homebench doctor` diagnostics."""

from homebench.doctor import Check, run_checks, summarize
from homebench.models import (
    MemoryMetrics,
    ModelInfo,
    ModelReport,
    SpeedMetrics,
    TaskResult,
)
from homebench.score import best_value, value_scores, value_verdict

GB = 1_000_000_000


def _report(name, quality_pct, tps, mem_gb, err=None):
    r = ModelReport(model=ModelInfo(name, "ollama", size_bytes=int(mem_gb * GB)))
    r.speed = SpeedMetrics(tokens_per_sec=tps)
    r.memory = MemoryMetrics(size_bytes=int(mem_gb * GB))
    if quality_pct is not None:
        n = 10
        passed = round(quality_pct / 100 * n)
        r.task_results = ([TaskResult(f"p{i}", "x", 1.0, True) for i in range(passed)]
                          + [TaskResult(f"f{i}", "x", 0.0, False) for i in range(n - passed)])
    r.error = err
    return r


# ---- value score -----------------------------------------------------------
def test_value_scores_reward_quality_speed_and_low_memory():
    reports = [
        _report("great", 90, 100, 2),   # high q, high speed, small -> best
        _report("big", 90, 100, 20),    # same but 10x memory -> lower value
        _report("slow", 90, 5, 2),      # same but slow -> lower value
    ]
    scores = value_scores(reports)
    assert scores["great"] == 100.0
    assert scores["great"] > scores["big"]
    assert scores["great"] > scores["slow"]


def test_best_value_picks_top():
    reports = [_report("a", 60, 20, 4), _report("b", 90, 30, 3)]
    rep, score = best_value(reports)
    assert rep.model.name == "b"
    assert 0 < score <= 100


def test_verdict_needs_two_models():
    assert value_verdict([_report("solo", 80, 20, 3)]) is None
    v = value_verdict([_report("a", 60, 10, 5), _report("b", 90, 30, 2)])
    assert v and v.startswith("b — ") and "value" in v


def test_errored_models_excluded():
    reports = [_report("ok", 80, 20, 3), _report("bad", None, 0, 0, err="boom")]
    scores = value_scores(reports)
    assert "bad" not in scores and "ok" in scores


def test_value_without_quality_uses_speed_and_memory():
    # --no-quality run: quality_score is None for all
    reports = [_report("fast_small", None, 100, 2), _report("slow_big", None, 10, 20)]
    scores = value_scores(reports)
    assert scores["fast_small"] > scores["slow_big"]


# ---- doctor ----------------------------------------------------------------
def test_doctor_runs_and_reports_python_ok():
    checks = run_checks(check_pypi=False)   # no network
    assert all(isinstance(c, Check) for c in checks)
    assert all(c.status in {"ok", "warn", "fail", "info"} for c in checks)
    names = {c.name for c in checks}
    assert "Python" in names and "Hardware" in names and "Home directory" in names
    py = next(c for c in checks if c.name == "Python")
    assert py.status == "ok"                # tests run on 3.9+


def test_doctor_home_writable_and_summary(tmp_path):
    checks = run_checks(home=str(tmp_path / "hb"), check_pypi=False)
    home = next(c for c in checks if c.name == "Home directory")
    assert home.status == "ok" and "writable" in home.detail
    fails, warns = summarize(checks)
    assert fails >= 0 and warns >= 0


def test_doctor_pypi_check_is_optional(monkeypatch):
    import homebench.doctor as mod
    monkeypatch.setattr(mod, "_latest_pypi", lambda timeout=3.0: "9.9.9")
    checks = run_checks(check_pypi=True)
    ver = next(c for c in checks if c.name == "Version")
    assert "9.9.9" in ver.detail and ver.status == "info"
