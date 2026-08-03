from localbench.quality import LLMJudge
from tests.fakes import FakeProvider


def test_judge_parses_score():
    judge = LLMJudge(FakeProvider(), "smart:8b")
    grade = judge.score("summarize why sky is blue", "Rayleigh scattering.", "ref")
    # FakeProvider returns "SCORE: 5 ..." for the judge prompt
    assert grade.passed
    assert grade.score == 1.0
    assert "5/5" in grade.detail


def test_runner_includes_open_tasks_with_judge():
    from localbench.runner import RunConfig, Runner

    provider = FakeProvider()
    judge = LLMJudge(provider, "smart:8b")
    runner = Runner(provider, RunConfig(sample_rss=False, include_open=True), judge=judge)
    result = runner.run([provider.list_models()[0]])
    r = result.reports[0]
    from localbench.quality import default_suite
    assert len(r.task_results) == len(default_suite(include_open=True))
    assert any(t.category == "open" for t in r.task_results)
