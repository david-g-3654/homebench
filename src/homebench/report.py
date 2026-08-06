"""Rendering & export: a Rich leaderboard table, plus Markdown / JSON output."""

from __future__ import annotations

import html as _html
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


# ---- environment -----------------------------------------------------------
def _hw(env: dict) -> dict:
    return (env or {}).get("hardware", {}) or {}


def env_summary(env: dict) -> str:
    """One-line host summary from a captured environment dict."""
    hw = _hw(env)
    parts: List[str] = []
    if hw.get("os"):
        os_part = f"{hw['os']} {hw.get('os_version', '')}".strip()
        if hw.get("arch"):
            os_part += f" ({hw['arch']})"
        parts.append(os_part)
    if hw.get("cpu"):
        cpu = hw["cpu"]
        if hw.get("cpu_cores"):
            cpu += f" ×{hw['cpu_cores']}"
        parts.append(cpu)
    if hw.get("ram_total_bytes"):
        parts.append(f"{fmt_bytes(hw['ram_total_bytes'])} RAM")
    gpu = hw.get("gpu") or {}
    if gpu.get("kind") == "nvidia" and gpu.get("vram_bytes"):
        parts.append(f"{gpu.get('name', 'GPU')} {fmt_bytes(gpu['vram_bytes'])}")
    elif gpu.get("kind") == "apple":
        parts.append("Apple GPU (unified)")
    if (env or {}).get("homebench_version"):
        parts.append(f"homebench {env['homebench_version']}")
    if hw.get("python_version"):
        parts.append(f"Python {hw['python_version']}")
    return " · ".join(parts)


# ---- ranking ---------------------------------------------------------------
def rank_reports(reports: List[ModelReport]) -> List[ModelReport]:
    """Rank by quality first, then throughput. Errored models sink to the end."""

    def key(r: ModelReport):
        q = r.quality_score if r.quality_score is not None else -1
        return (r.error is not None, -q, -r.speed.tokens_per_sec)

    return sorted(reports, key=key)


# ---- Rich table (shared by CLI + TUI) --------------------------------------
def leaderboard_table(result: BenchmarkResult, title: str = "homebench") -> Table:
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
    lines.append("# homebench results")
    lines.append("")
    lines.append(f"- **Provider:** {result.provider}")
    lines.append(f"- **Run at:** {started}")
    lines.append(f"- **Models:** {len(result.reports)}")
    env = env_summary(result.environment)
    if env:
        lines.append(f"- **Environment:** {env}")
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
    lines.append("_Generated by [homebench](https://github.com/david-g-3654/homebench)._")
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


# ---- HTML (self-contained, shareable) --------------------------------------
_HTML_CSS = """
:root { --bg:#fff; --fg:#1c2024; --muted:#6b7280; --line:#e5e7eb; --card:#f9fafb;
        --q:#16a34a; --s:#2563eb; --bar-bg:#eef1f4; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1115; --fg:#e6e8eb; --muted:#9aa4b2; --line:#232833; --card:#161a21;
          --q:#22c55e; --s:#3b82f6; --bar-bg:#1c222c; } }
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1rem; background:var(--bg); color:var(--fg);
       font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:960px; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 .25rem; }
h2 { font-size:1.05rem; margin:2rem 0 .75rem; }
.sub { color:var(--muted); font-size:.9rem; margin:0 0 1rem; }
.env { background:var(--card); border:1px solid var(--line); border-radius:8px;
       padding:.6rem .9rem; color:var(--muted); font-size:.85rem; margin:0 0 1.5rem; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
th,td { text-align:left; padding:.55rem .6rem; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; font-size:.78rem; text-transform:uppercase;
     letter-spacing:.03em; }
td.num,th.num { text-align:right; }
.rank { color:var(--muted); }
.model { font-weight:600; }
.bar { position:relative; min-width:120px; }
.bar .track { height:1.05rem; background:var(--bar-bg); border-radius:4px; overflow:hidden; }
.bar .fill { height:100%; border-radius:4px; }
.bar .fill.q { background:var(--q); } .bar .fill.s { background:var(--s); }
.bar .val { position:absolute; right:.4rem; top:0; font-size:.8rem; line-height:1.05rem; }
.err { color:#dc2626; font-weight:600; }
footer { margin-top:2.5rem; color:var(--muted); font-size:.82rem;
         border-top:1px solid var(--line); padding-top:1rem; }
a { color:var(--s); }
"""


def _bar(value: Optional[float], vmax: float, cls: str, label: str) -> str:
    pct = 0.0 if not value or vmax <= 0 else max(0.0, min(100.0, 100.0 * value / vmax))
    return (f'<div class="bar"><div class="track">'
            f'<div class="fill {cls}" style="width:{pct:.0f}%"></div></div>'
            f'<span class="val">{_html.escape(label)}</span></div>')


def to_html(result: BenchmarkResult, title: str = "homebench report") -> str:
    ranked = rank_reports(result.reports)
    ok = [r for r in ranked if not r.error]
    max_q = max((r.quality_score or 0 for r in ok), default=100) or 100
    max_tps = max((r.speed.tokens_per_sec for r in ok), default=1) or 1
    when = datetime.fromtimestamp(result.started_at).strftime("%Y-%m-%d %H:%M:%S")
    env = env_summary(result.environment)

    rows = []
    for i, r in enumerate(ranked, start=1):
        name = _html.escape(r.model.name)
        params = _html.escape(r.model.parameter_size or "–")
        if r.error:
            rows.append(
                f'<tr><td class="rank">{i}</td><td class="model">{name}</td>'
                f'<td class="num">{params}</td>'
                f'<td colspan="5" class="err">error: {_html.escape(r.error)}</td></tr>')
            continue
        passed = f"{r.tasks_passed}/{len(r.task_results)}" if r.task_results else "–"
        q = r.quality_score
        rows.append(
            f'<tr><td class="rank">{i}</td><td class="model">{name}</td>'
            f'<td class="num">{params}</td>'
            f'<td>{_bar(q, max_q, "q", fmt_quality(q))}</td>'
            f'<td class="num">{passed}</td>'
            f'<td>{_bar(r.speed.tokens_per_sec, max_tps, "s", fmt_tps(r.speed.tokens_per_sec))}</td>'
            f'<td class="num">{fmt_ttft(r.speed.ttft_s)}</td>'
            f'<td class="num">{_memory_display(r)}</td></tr>')

    # per-category quality
    cats = _category_set(result)
    cat_html = ""
    if cats:
        head = "".join(f"<th class='num'>{_html.escape(c)}</th>" for c in cats)
        body = []
        for r in ranked:
            if r.error:
                continue
            cells = []
            for c in cats:
                scores = [t.score for t in r.task_results if t.category == c]
                cells.append(f"<td class='num'>{100*sum(scores)/len(scores):.0f}%</td>"
                             if scores else "<td class='num'>–</td>")
            body.append(f"<tr><td class='model'>{_html.escape(r.model.name)}</td>"
                        + "".join(cells) + "</tr>")
        cat_html = (f"<h2>Quality by category</h2><table><thead><tr>"
                    f"<th>Model</th>{head}</tr></thead><tbody>"
                    + "".join(body) + "</tbody></table>")

    env_html = f'<div class="env">{_html.escape(env)}</div>' if env else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title><style>{_HTML_CSS}</style></head>
<body><div class="wrap">
<h1>{_html.escape(title)}</h1>
<p class="sub">{_html.escape(result.provider)} · {len(result.reports)} model(s) · {when}</p>
{env_html}
<h2>Leaderboard</h2>
<table><thead><tr><th class="num">#</th><th>Model</th><th class="num">Params</th>
<th>Quality</th><th class="num">Pass</th><th>tok/s</th><th class="num">TTFT</th>
<th class="num">Memory</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{cat_html}
<footer>Generated by <a href="https://github.com/david-g-3654/homebench">homebench</a>
— a local-first LLM benchmark. Numbers reflect this machine at run time; see the
project's Limitations.</footer>
</div></body></html>
"""


# ---- hardware / model-fit --------------------------------------------------
def hardware_table(hw) -> Table:
    """Render captured hardware as a key/value summary."""
    budget, label = hw.memory_budget()
    t = Table(title="This machine", header_style="bold cyan", show_header=False)
    t.add_column("k", style="bold", justify="right")
    t.add_column("v")
    t.add_row("OS", f"{hw.os} {hw.os_version} ({hw.arch})")
    t.add_row("CPU", f"{hw.cpu}  ·  {hw.cpu_cores} cores")
    t.add_row("RAM", f"{fmt_bytes(hw.ram_total_bytes)} total  "
                     f"· {fmt_bytes(hw.ram_available_bytes)} available")
    if hw.gpu.kind == "nvidia":
        t.add_row("GPU", f"{hw.gpu.name}  · {fmt_bytes(hw.gpu.vram_bytes)} VRAM")
    elif hw.gpu.kind == "apple":
        t.add_row("GPU", f"{hw.gpu.name} (unified memory)")
    else:
        t.add_row("GPU", "none detected (CPU inference)")
    t.add_row("Model budget", f"[green]{fmt_bytes(budget)}[/green]  ({label})")
    return t


_FIT_MARK = {
    "fits": "[green]✓ fits[/green]",
    "tight": "[yellow]⚠ tight[/yellow]",
    "no": "[red]✗ too big[/red]",
}


def fit_table(results, show_all: bool = False) -> Table:
    """Render model-fit results (from catalog.evaluate_catalog)."""
    t = Table(
        title="Model fit for your hardware", header_style="bold cyan",
        caption="Install: [b]ollama pull <tag>[/b]  ·  HuggingFace repos also "
                "work in LM Studio & vLLM",
        caption_justify="left",
    )
    t.add_column("Model", style="bold", no_wrap=True)
    t.add_column("Params", justify="right")
    t.add_column("Quant", justify="center")
    t.add_column("~Needs", justify="right")
    t.add_column("Fit", justify="left", no_wrap=True)
    t.add_column("Ollama tag", overflow="fold")
    t.add_column("HuggingFace repo", overflow="fold", style="dim")

    for r in results:
        if not show_all and r.status == "no":
            continue
        m = r.model
        t.add_row(
            m.name,
            f"{m.params_b:g}B",
            r.quant or "–",
            fmt_bytes(r.required_bytes),
            _FIT_MARK.get(r.status, r.status),
            m.ollama or "–",
            m.hf or "–",
        )
    return t


# ---- throughput ------------------------------------------------------------
def throughput_table(result) -> Table:
    """Render a ThroughputResult as a concurrency-sweep table."""
    title = f"Batch throughput — {result.model} ({result.provider})"
    table = Table(title=title, header_style="bold cyan")
    table.add_column("Conc", justify="right")
    table.add_column("Reqs", justify="right")
    table.add_column("Agg tok/s", justify="right", style="green")
    table.add_column("Speedup", justify="right")
    table.add_column("Req tok/s", justify="right")
    table.add_column("Mean lat", justify="right")
    table.add_column("p95 lat", justify="right")
    table.add_column("Errors", justify="right")

    if result.error:
        table.add_row("–", "–", "[red]error[/red]", "–", "–", "–", "–", "–")
        return table

    for p in result.points:
        speedup = result.speedup(p)
        speedup_txt = f"{speedup:.2f}×" if speedup is not None else "–"
        errors_txt = f"[red]{p.errors}[/red]" if p.errors else "0"
        table.add_row(
            str(p.concurrency),
            str(p.requests),
            f"{p.aggregate_tps:.1f}",
            speedup_txt,
            f"{p.mean_req_tps:.1f}",
            fmt_ttft(p.mean_latency_s),
            fmt_ttft(p.p95_latency_s),
            errors_txt,
        )
    return table
