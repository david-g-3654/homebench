"""Plain (non-full-screen) renderer.

Used when stdout isn't an interactive terminal, or when ``--no-tui`` is passed.
Renders a live-updating leaderboard with Rich, so piping output or running in
CI still gives a clean, readable comparison.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .models import BenchmarkResult, ModelInfo, ModelReport
from .report import (
    fmt_quality,
    fmt_tps,
    fmt_ttft,
    rank_reports,
    _memory_display,
)
from .runner import (
    EV_MODEL_DONE,
    EV_MODEL_START,
    EV_PHASE,
    EV_TASK_DONE,
    Runner,
    resolve_suite,
)


class PlainReporter:
    def __init__(self, models: List[ModelInfo], total_tasks: int):
        self.order = [m.name for m in models]
        self.total_tasks = total_tasks
        self.status: Dict[str, str] = {n: "queued" for n in self.order}
        self.progress: Dict[str, int] = {n: 0 for n in self.order}
        self.reports: Dict[str, ModelReport] = {}
        self.current: Optional[str] = None

    # observer callback -------------------------------------------------
    def __call__(self, event: str, **data) -> None:
        if event == EV_MODEL_START:
            self.current = data["model"]
            self.status[data["model"]] = "starting"
        elif event == EV_PHASE:
            phase = data["phase"]
            if phase == "error":
                self.status[data["model"]] = "error"
            else:
                self.status[data["model"]] = phase
        elif event == EV_TASK_DONE:
            self.progress[data["model"]] = self.progress.get(data["model"], 0) + 1
        elif event == EV_MODEL_DONE:
            report = data["report"]
            self.reports[report.model.name] = report
            self.status[report.model.name] = "error" if report.error else "done"

    # rendering ---------------------------------------------------------
    def _status_text(self, name: str) -> Text:
        s = self.status.get(name, "queued")
        if s == "quality":
            done = self.progress.get(name, 0)
            return Text(f"quality {done}/{self.total_tasks}", style="yellow")
        colors = {
            "done": "green", "error": "red", "queued": "dim",
            "warmup": "cyan", "speed": "cyan", "starting": "cyan",
        }
        return Text(s, style=colors.get(s, "yellow"))

    def render(self) -> Group:
        table = Table(header_style="bold cyan", expand=False)
        table.add_column("Model", style="bold")
        table.add_column("Status")
        table.add_column("Quality", justify="right")
        table.add_column("Pass", justify="right")
        table.add_column("tok/s", justify="right", style="green")
        table.add_column("TTFT", justify="right")
        table.add_column("Memory", justify="right")

        done = [self.reports[n] for n in self.order if n in self.reports]
        pending = [n for n in self.order if n not in self.reports]
        for r in rank_reports(done):
            if r.error:
                table.add_row(r.model.name, Text("error", style="red"),
                              "–", "–", "–", "–", "–")
                continue
            passed = f"{r.tasks_passed}/{len(r.task_results)}" if r.task_results else "–"
            table.add_row(
                r.model.name, Text("done", style="green"),
                fmt_quality(r.quality_score), passed,
                fmt_tps(r.speed.tokens_per_sec), fmt_ttft(r.speed.ttft_s),
                _memory_display(r),
            )
        for n in pending:
            table.add_row(n, self._status_text(n), "–", "–", "–", "–", "–")

        header = Text("localbench — benchmarking local models", style="bold magenta")
        return Group(header, table)


def run_plain(runner: Runner, models: List[ModelInfo], console: Console) -> BenchmarkResult:
    total_tasks = (
        len(resolve_suite(runner.config, runner.judge is not None))
        if runner.config.run_quality else 0
    )
    reporter = PlainReporter(models, total_tasks)

    console.print(
        f"[bold]localbench[/bold] · provider [cyan]{runner.provider.name}[/cyan] · "
        f"{len(models)} model(s)\n"
    )
    with Live(reporter.render(), console=console, refresh_per_second=8) as live:
        def observer(event: str, **data):
            reporter(event, **data)
            live.update(reporter.render())

        result = runner.run(models, observer=observer)
        live.update(reporter.render())

    console.print()
    from .report import leaderboard_table

    console.print(leaderboard_table(result, title="Final leaderboard"))
    return result
