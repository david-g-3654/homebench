import asyncio

from textual.widgets import DataTable

from localbench.runner import RunConfig, Runner
from localbench.tui.app import LocalbenchApp
from tests.fakes import FakeProvider


def test_tui_runs_and_fills_leaderboard():
    provider = FakeProvider()
    runner = Runner(provider, RunConfig(sample_rss=False))
    models = provider.list_models()

    async def scenario():
        app = LocalbenchApp(runner, models)
        async with app.run_test() as pilot:
            for _ in range(200):
                if app.result is not None:
                    break
                await pilot.pause(0.05)
            assert app.result is not None, "benchmark did not finish"
            table = app.query_one("#board", DataTable)
            assert table.row_count == 2
            assert len(app.result.reports) == 2

    asyncio.run(scenario())
