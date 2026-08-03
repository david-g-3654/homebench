from localbench.quality import Task
from localbench.quality.graders import contains_any
from localbench.runner import RunConfig, Runner, resolve_suite
from tests.fakes import FakeProvider


def _suite():
    return [
        Task("c1", "custom", "prompt one", contains_any(["a"])),
        Task("o1", "open", "prompt two", grader=None),
    ]


def test_resolve_suite_drops_open_without_judge():
    cfg = RunConfig(suite=_suite())
    assert [t.id for t in resolve_suite(cfg, judge_present=False)] == ["c1"]


def test_resolve_suite_keeps_open_with_judge():
    cfg = RunConfig(suite=_suite(), include_open=True)
    assert [t.id for t in resolve_suite(cfg, judge_present=True)] == ["c1", "o1"]


def test_resolve_suite_defaults_to_builtin():
    from localbench.quality import default_suite

    cfg = RunConfig()
    assert len(resolve_suite(cfg, judge_present=False)) == len(default_suite())


def test_runner_honors_custom_suite():
    provider = FakeProvider()
    runner = Runner(provider, RunConfig(sample_rss=False, suite=_suite()))
    result = runner.run([provider.list_models()[0]])
    # only the one deterministic task runs (open dropped, no judge)
    assert [t.task_id for t in result.reports[0].task_results] == ["c1"]
