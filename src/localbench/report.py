"""Rendering & export: a Rich leaderboard table, plus Markdown / JSON output."""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from rich.table import Table

from .models import BenchmarkResult, ModelReport


# ---- formatting helpers ----------------------------------------------------
def fmt_bytes(n: int) -> str:
    if not n:
        return "–"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit in ("B", "KB") else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def fmt_tps(v: float) -> str:
    return f"{v:.1f}" if v else "–"


def fmt_ttft(v: float) -> str:
    if not v:
        return "–"
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s"


def fmt_quality(v: Optional[float]) -> str:
    return f"{v:.0f}%" if v is not None else "–"


def _memory_display(r: ModelReport) -> str:
    # Prefer the provider's resident-model size; fall back to on-disk size.
    n = r.memory.size_bytes or r.model.size_bytes
    return fmt_bytes(n)


# ---- ranking ---------------------------------------------------------------
def rank_reports(reports: List[ModelReport]) -> List[ModelReport]:
    """Rank by quality first, then throughput. Errored models sink to the end."""

    def key(r: ModelReport):
        q = r.quality_score if r.quality_score is not None else -1
        return (r.error is not None, -q, -r.speed.tokens_per_sec)

    return sorted(reports, key=key)


# ---- Rich table (shared by CLI + TUI) --------------------------------------
def leaderboard_table(result: BenchmarkResult, title: str = "localbench") -> Table:
    table = Table(title=title, expand=False, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Model", style="bold")
    table.add_column("Params", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Pass", justify="right")
    table.add_column("tok/s", justify="right", style="green")
    table.add_column("TTFT", justify="right")
    table.add_column("Memory", justify="right")

    ranked = rank_reports(result.reports)
    for i, r in enumerate(ranked, start=1):
        if r.error:
            table.add_row(
                str(i), r.model.name, r.model.parameter_size or "–",
                "[red]error[/red]", "–", "–", "–", "–",
            )
            continue
        passed = f"{r.tasks_passed}/{len(r.task_results)}" if r.task_results else "–"
        table.add_row(
            str(i),
            r.model.name,
            r.model.parameter_size or "–",
            fmt_quality(r.quality_score),
            passed,
            fmt_tps(r.speed.tokens_per_sec),
            fmt_ttft(r.speed.ttft_s),
            _memory_display(r),
        )
    return table


# ---- Markdown --------------------------------------------------------------
def to_markdown(result: BenchmarkResult) -> str:
    started = datetime.fromtimestamp(result.started_at).strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# localbench results")
    lines.append("")
    lines.append(f"- **Provider:** {result.provider}")
    lines.append(f"- **Run at:** {started}")
    lines.append(f"- **Models:** {len(result.reports)}")
    cfg = result.config or {}
    if cfg.get("judge_model"):
        lines.append(f"- **Judge:** {cfg['judge_model']}")
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| # | Model | Params | Quality | Pass | tok/s | TTFT | Memory |")
    lines.append("|---|-------|-------:|--------:|-----:|------:|-----:|-------:|")
    for i, r in enumerate(rank_reports(result.reports), start=1):
        if r.error:
            lines.append(
                f"| {i} | {r.model.name} | {r.model.parameter_size or '–'} "
                f"| error | – | – | – | – |"
            )
            continue
        passed = f"{r.tasks_passed}/{len(r.task_results)}" if r.task_results else "–"
        lines.append(
            f"| {i} | {r.model.name} | {r.model.parameter_size or '–'} "
            f"| {fmt_quality(r.quality_score)} | {passed} "
            f"| {fmt_tps(r.speed.tokens_per_sec)} | {fmt_ttft(r.speed.ttft_s)} "
            f"| {_memory_display(r)} |"
        )
    lines.append("")

    # Per-category quality breakdown
    cats = _category_set(result)
    if cats:
        lines.append("## Quality by category")
        lines.append("")
        header = "| Model | " + " | ".join(cats) + " |"
        sep = "|-------|" + "|".join(["------:"] * len(cats)) + "|"
        lines.append(header)
        lines.append(sep)
        for r in rank_reports(result.reports):
            if r.error:
                continue
            cells = []
            for c in cats:
                scores = [t.score for t in r.task_results if t.category == c]
                cells.append(f"{100 * sum(scores) / len(scores):.0f}%" if scores else "–")
            lines.append(f"| {r.model.name} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Generated by [localbench](https://github.com/localbench/localbench)._")
    return "\n".join(lines)


def _category_set(result: BenchmarkResult) -> List[str]:
    cats: List[str] = []
    for r in result.reports:
        for t in r.task_results:
            if t.category not in cats:
                cats.append(t.category)
    return cats


def to_json(result: BenchmarkResult, indent: int = 2) -> str:
    return json.dumps(result.to_dict(), indent=indent)
