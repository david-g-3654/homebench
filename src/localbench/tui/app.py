"""The full-screen leaderboard TUI (Textual).

The benchmark runs in a background thread; observer events are marshalled onto
the UI thread with ``call_from_thread`` and the leaderboard is re-rendered on
each update.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static
from textual import work

from ..models import BenchmarkResult, ModelInfo, ModelReport
from ..quality import default_suite
from ..report import (
    _memory_display,
    fmt_quality,
    fmt_tps,
    fmt_ttft,
    rank_reports,
)
from ..runner import (
    EV_MODEL_DONE,
    EV_MODEL_START,
    EV_PHASE,
    EV_RUN_DONE,
    EV_TASK_DONE,
    Runner,
)


class LocalbenchApp(App):
    TITLE = "localbench"
    SUB_TITLE = "local LLM leaderboard"

    CSS = """
    Screen { background: $surface; }
    #status { height: 3; padding: 1 2; color: $text-muted; }
    DataTable { height: 1fr; margin: 0 1; }
    #hint { height: 1; padding: 0 2; color: $text-muted; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "save", "Save report"),
    ]

    def __init__(self, runner: Runner, models: List[ModelInfo]):
        super().__init__()
        self.runner = runner
        self.models = models
        self.order = [m.name for m in models]
        include_open = runner.config.include_open and runner.judge is not None
        self.total_tasks = (
            len(default_suite(include_open=include_open))
            if runner.config.run_quality else 0
        )
        self.status: Dict[str, str] = {n: "queued" for n in self.order}
        self.progress: Dict[str, int] = {n: 0 for n in self.order}
        self.reports: Dict[str, ModelReport] = {}
        self.result: Optional[BenchmarkResult] = None

    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static("Starting…", id="status")
            yield DataTable(id="board", zebra_stripes=True, cursor_type="row")
            yield Static("Press [b]q[/b] to quit · [b]s[/b] to save a Markdown report",
                         id="hint")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#board", DataTable)
        table.add_columns("Model", "Status", "Quality", "Pass", "tok/s", "TTFT", "Memory")
        self._rebuild_table()
        self.run_benchmark()

    # ------------------------------------------------------------------
    @work(thread=True, exclusive=True)
    def run_benchmark(self) -> None:
        self.runner.run(self.models, observer=self._observer)

    def _observer(self, event: str, **data) -> None:
        # Called from the worker thread; hop onto the UI thread to touch widgets.
        self.call_from_thread(self._handle_event, event, data)

    def _handle_event(self, event: str, data: dict) -> None:
        if event == EV_MODEL_START:
            self.status[data["model"]] = "starting"
        elif event == EV_PHASE:
            phase = data["phase"]
            self.status[data["model"]] = phase
            note = data.get("note")
            self._set_status(data["model"], phase, note)
        elif event == EV_TASK_DONE:
            self.progress[data["model"]] = self.progress.get(data["model"], 0) + 1
        elif event == EV_MODEL_DONE:
            report = data["report"]
            self.reports[report.model.name] = report
            self.status[report.model.name] = "error" if report.error else "done"
        elif event == EV_RUN_DONE:
            self.result = data["result"]
            self._set_done()
        self._rebuild_table()

    # ------------------------------------------------------------------
    def _status_label(self, name: str) -> str:
        s = self.status.get(name, "queued")
        if s == "quality":
            return f"quality {self.progress.get(name, 0)}/{self.total_tasks}"
        return s

    def _rebuild_table(self) -> None:
        table = self.query_one("#board", DataTable)
        table.clear()
        done = [self.reports[n] for n in self.order if n in self.reports]
        pending = [n for n in self.order if n not in self.reports]
        for r in rank_reports(done):
            if r.error:
                table.add_row(r.model.name, "[red]error[/red]", "–", "–", "–", "–", "–")
                continue
            passed = f"{r.tasks_passed}/{len(r.task_results)}" if r.task_results else "–"
            table.add_row(
                r.model.name, "[green]done[/green]",
                fmt_quality(r.quality_score), passed,
                fmt_tps(r.speed.tokens_per_sec), fmt_ttft(r.speed.ttft_s),
                _memory_display(r),
            )
        for n in pending:
            label = self._status_label(n)
            style = "yellow" if label != "queued" else "dim"
            table.add_row(n, f"[{style}]{label}[/{style}]", "–", "–", "–", "–", "–")

    def _set_status(self, model: str, phase: str, note: Optional[str]) -> None:
        widget = self.query_one("#status", Static)
        if phase == "error":
            widget.update(f"[red]{model}: {note or 'error'}[/red]")
        else:
            done = len(self.reports)
            widget.update(
                f"Running [b]{model}[/b] — {phase}  "
                f"([green]{done}[/green]/{len(self.order)} models done)"
            )

    def _set_done(self) -> None:
        widget = self.query_one("#status", Static)
        widget.update("[green]✓ Benchmark complete.[/green] "
                      "Press [b]s[/b] to save a report, [b]q[/b] to quit.")

    # ------------------------------------------------------------------
    def action_save(self) -> None:
        if self.result is None:
            self.notify("Run still in progress…", severity="warning")
            return
        from ..report import to_markdown

        path = "localbench-report.md"
        with open(path, "w") as f:
            f.write(to_markdown(self.result))
        self.notify(f"Saved {path}")


def run_tui(runner: Runner, models: List[ModelInfo]) -> Optional[BenchmarkResult]:
    app = LocalbenchApp(runner, models)
    app.run()
    return app.result
