import pytest

from localbench.metrics.throughput import (
    ThroughputResult,
    measure_throughput,
    parse_levels,
)
from localbench.providers.base import ProviderError
from localbench.report import throughput_table
from tests.fakes import FakeProvider


def test_parse_levels():
    assert parse_levels("1,2,4,8") == [1, 2, 4, 8]
    assert parse_levels("8, 1, 4, 1") == [1, 4, 8]  # sorted + deduped
    assert parse_levels(" 2 ") == [2]
    with pytest.raises(ValueError):
        parse_levels("")
    with pytest.raises(ValueError):
        parse_levels("abc")


def test_measure_throughput_basic():
    provider = FakeProvider()
    res = measure_throughput(
        provider, "fast:1b", concurrency_levels=[1, 2, 4], requests=6,
    )
    assert isinstance(res, ThroughputResult)
    assert res.error is None
    assert [p.concurrency for p in res.points] == [1, 2, 4]
    for p in res.points:
        assert p.requests == 6
        assert p.completed == 6
        assert p.errors == 0
        assert p.total_output_tokens > 0
        assert p.aggregate_tps > 0
        assert p.mean_latency_s >= 0
    # speedup is relative to the first (lowest-concurrency) point
    assert res.speedup(res.points[0]) == pytest.approx(1.0)


def test_auto_requests_scale_with_concurrency():
    provider = FakeProvider()
    res = measure_throughput(provider, "fast:1b", concurrency_levels=[1, 4])
    reqs = {p.concurrency: p.requests for p in res.points}
    assert reqs[1] == 4      # max(4, 3*1)
    assert reqs[4] == 12     # 3*4


def test_errors_are_counted(monkeypatch):
    provider = FakeProvider()
    calls = {"n": 0}
    real = provider.generate

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise ProviderError("boom")
        return real(*a, **k)

    monkeypatch.setattr(provider, "generate", flaky)
    res = measure_throughput(provider, "fast:1b", concurrency_levels=[1], requests=4)
    p = res.points[0]
    assert p.errors == 2
    assert p.completed == 2


def test_warmup_failure_sets_error(monkeypatch):
    provider = FakeProvider()

    def boom(*a, **k):
        raise ProviderError("cannot load")

    monkeypatch.setattr(provider, "warmup", boom)
    res = measure_throughput(provider, "fast:1b", concurrency_levels=[1, 2])
    assert res.error == "cannot load"
    assert res.points == []


def test_throughput_table_renders():
    provider = FakeProvider()
    res = measure_throughput(provider, "fast:1b", concurrency_levels=[1, 2], requests=4)
    table = throughput_table(res)
    assert table.row_count == 2

    err = ThroughputResult(model="m", provider="p", error="nope")
    assert throughput_table(err).row_count == 1
