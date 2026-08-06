"""Command-line entry point for homebench."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from rich.console import Console

from . import __version__
from .models import ModelInfo
from .providers import ProviderError, detect_provider, get_provider
from .quality import LLMJudge
from .runner import RunConfig, Runner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="homebench",
        description="Benchmark the local LLMs you already have — speed, memory, "
        "and quality — as a live terminal leaderboard.",
    )
    p.add_argument("--version", action="version", version=f"homebench {__version__}")
    sub = p.add_subparsers(dest="command")

    # ---- shared run options (also used when no subcommand is given) ----
    def add_run_options(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-m", "--models", default=None,
                        help="comma-separated model names to benchmark")
        sp.add_argument("--all", action="store_true",
                        help="benchmark every discovered model "
                             "(default: the 3 smallest)")
        sp.add_argument("--limit", type=int, default=None,
                        help="benchmark at most N models")
        sp.add_argument("--provider", default=None,
                        help="ollama|lmstudio|llamacpp|vllm|mlx|openai "
                             "(default: auto-detect)")
        sp.add_argument("--host", default=None,
                        help="provider host URL (default: the provider's env var "
                             "or its standard localhost port)")
        sp.add_argument("--full", action="store_true",
                        help="run the full quality suite (default: a fast subset)")
        sp.add_argument("--max-tokens", type=int, default=256,
                        help="max tokens per quality task (default: 256)")
        sp.add_argument("--speed-tokens", type=int, default=100,
                        help="tokens to generate for the speed probe (default: 100)")
        sp.add_argument("--repeat", type=int, default=1,
                        help="speed-probe repetitions, best kept (default: 1)")
        sp.add_argument("--timeout", type=float, default=300.0,
                        help="per-request timeout in seconds (default: 300)")
        sp.add_argument("--seed", type=int, default=42,
                        help="sampling seed for reproducibility (default: 42)")
        sp.add_argument("--no-quality", action="store_true",
                        help="skip the quality suite (speed/memory only)")
        sp.add_argument("--no-speed", action="store_true",
                        help="skip the speed probe (quality only)")
        sp.add_argument("--no-warmup", action="store_true",
                        help="don't pre-load models before timing")
        sp.add_argument("--no-unload", action="store_true",
                        help="keep models loaded between runs")
        sp.add_argument("--no-rss", action="store_true",
                        help="disable psutil RSS sampling")
        sp.add_argument("--judge", default=None, metavar="MODEL",
                        help="enable LLM-as-judge using MODEL (adds open-ended tasks)")
        sp.add_argument("--no-tui", action="store_true",
                        help="use the plain renderer instead of the full-screen TUI")
        sp.add_argument("--tui", action="store_true",
                        help="force the full-screen TUI even when piped")
        sp.add_argument("--md", "--out", dest="md", default=None, metavar="FILE",
                        help="write a Markdown report to FILE")
        sp.add_argument("--json", dest="json_out", default=None, metavar="FILE",
                        help="write raw JSON results to FILE")
        sp.add_argument("--html", dest="html_out", default=None, metavar="FILE",
                        help="write a self-contained shareable HTML report to FILE")
        sp.add_argument("--tasks", action="append", default=None, metavar="PACK",
                        help="use this task pack instead of the built-in suite "
                             "(JSON/YAML; repeatable)")
        sp.add_argument("--add-tasks", dest="add_tasks", action="append",
                        default=None, metavar="PACK",
                        help="append this task pack to the current suite (repeatable)")
        sp.add_argument("--label", default=None,
                        help="label to tag this run in history (for diffing)")
        sp.add_argument("--no-save", action="store_true",
                        help="don't save this run to the history store")
        sp.add_argument("--no-cache", action="store_true",
                        help="don't reuse cached quality responses")
        sp.add_argument("--refresh-cache", dest="refresh_cache", action="store_true",
                        help="ignore and overwrite the response cache")

    run_p = sub.add_parser("run", help="run the benchmark (default)")
    add_run_options(run_p)

    list_p = sub.add_parser("list", help="list discovered models and exit")
    list_p.add_argument("--provider", default=None)
    list_p.add_argument("--host", default=None)

    tasks_p = sub.add_parser("tasks", help="list the quality tasks and exit")
    tasks_p.add_argument("--tasks", action="append", default=None, metavar="PACK",
                         help="preview a task pack instead of the built-in suite")
    tasks_p.add_argument("--add-tasks", dest="add_tasks", action="append",
                         default=None, metavar="PACK")

    hist_p = sub.add_parser("history", help="list saved runs")
    hist_p.add_argument("--limit", type=int, default=20)

    sub.add_parser("doctor", help="diagnose provider / models / setup")

    diff_p = sub.add_parser("diff", help="diff two saved runs (base -> new)")
    diff_p.add_argument("a", nargs="?", default=None,
                        help="base run: 'latest'/'prev', a 1-based index, or a path")
    diff_p.add_argument("b", nargs="?", default=None,
                        help="new run (default: latest)")
    diff_p.add_argument("--fail-on-regression", dest="fail_on_regression",
                        action="store_true",
                        help="exit non-zero if a shared model regressed (for CI)")
    diff_p.add_argument("--quality-threshold", dest="quality_threshold",
                        type=float, default=5.0, metavar="PTS",
                        help="quality-drop tolerance in points (default: 5)")
    diff_p.add_argument("--speed-threshold", dest="speed_threshold",
                        type=float, default=10.0, metavar="PCT",
                        help="tok/s-drop tolerance in percent (default: 10)")

    report_p = sub.add_parser(
        "report", help="render a saved run as an HTML / Markdown report")
    report_p.add_argument("ref", nargs="?", default="latest",
                          help="run to render: 'latest'/'prev', index, or path")
    report_p.add_argument("--html", dest="html_out", default=None, metavar="FILE",
                          help="write a self-contained HTML report")
    report_p.add_argument("--md", dest="md", default=None, metavar="FILE",
                          help="write a Markdown report")

    tp = sub.add_parser(
        "throughput",
        help="measure batch/concurrent throughput (scales on server backends)")
    tp.add_argument("-m", "--models", default=None,
                    help="comma-separated model names (default: all discovered)")
    tp.add_argument("--limit", type=int, default=None)
    tp.add_argument("--provider", default=None)
    tp.add_argument("--host", default=None)
    tp.add_argument("--concurrency", default="1,2,4,8",
                    help="comma-separated concurrency levels (default: 1,2,4,8)")
    tp.add_argument("--requests", type=int, default=None,
                    help="requests per level (default: auto = 3×concurrency)")
    tp.add_argument("--max-tokens", type=int, default=128,
                    help="tokens to generate per request (default: 128)")
    tp.add_argument("--timeout", type=float, default=300.0)
    tp.add_argument("--seed", type=int, default=42)
    tp.add_argument("--no-warmup", action="store_true")
    tp.add_argument("--json", dest="json_out", default=None, metavar="FILE",
                    help="write raw throughput results to FILE")

    fit = sub.add_parser(
        "fit", help="capture your hardware and show which popular models fit")
    fit.add_argument("--all", action="store_true",
                     help="include models that don't fit")
    fit.add_argument("--context", type=int, default=4096,
                     help="context length to budget KV cache for (default: 4096)")
    fit.add_argument("--quant", default=None,
                     help="evaluate a specific quant (e.g. Q4_K_M) instead of the best")
    fit.add_argument("--catalog", action="append", default=None, metavar="FILE",
                     help="add models from a JSON catalog file (repeatable)")
    fit.add_argument("--online", action="store_true",
                     help="source models from the HuggingFace Hub (top by popularity) "
                          "instead of the built-in catalog")
    fit.add_argument("--top", type=int, default=50, metavar="N",
                     help="with --online, how many models to fetch (default: 50)")
    fit.add_argument("--sort", default="downloads",
                     choices=["downloads", "trending", "likes"],
                     help="with --online, ranking to pull the top models by")
    fit.add_argument("--refresh", action="store_true",
                     help="with --online, bypass the cache and refetch")
    fit.add_argument("--ram", type=float, default=None, metavar="GB",
                     help="override detected system RAM (GB) for what-if planning")
    fit.add_argument("--vram", type=float, default=None, metavar="GB",
                     help="override/assume GPU VRAM (GB) for what-if planning")
    fit.add_argument("--json", dest="json_out", default=None, metavar="FILE",
                     help="write hardware + fit results to FILE")
    return p


_COMMANDS = {"run", "list", "tasks", "history", "diff", "throughput", "fit",
             "report", "doctor"}


def _inject_default_command(argv: List[str]) -> List[str]:
    """Make ``run`` the default: `homebench --no-tui` == `homebench run --no-tui`.

    Global help/version are left for the top-level parser to handle.
    """
    if not argv:
        return ["run"]
    if argv[0] in _COMMANDS or argv[0] in ("-h", "--help", "--version"):
        return argv
    return ["run"] + argv


# ---------------------------------------------------------------------------
def _resolve_provider(args, console: Console):
    kwargs = {}
    if getattr(args, "host", None):
        kwargs["host"] = args.host
    if getattr(args, "provider", None):
        provider = get_provider(args.provider, **kwargs)
        if not provider.is_available():
            raise ProviderError(
                f"Provider {args.provider!r} is not reachable. Is it running?"
            )
        return provider
    return detect_provider(**kwargs)


def _select_models(provider, args, console: Console) -> List[ModelInfo]:
    available = provider.list_models()
    if not available:
        raise ProviderError(
            f"No models found for provider {provider.name!r}. "
            "Pull one first, e.g. `ollama pull llama3.2`."
        )
    # Smallest first, so the fast default and early results favor quick models.
    by_size = sorted(available, key=lambda m: m.size_bytes or 0)

    if getattr(args, "models", None):
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        by_name = {m.name: m for m in available}
        chosen: List[ModelInfo] = []
        for w in wanted:
            if w in by_name:
                chosen.append(by_name[w])
                continue
            matches = [m for m in available if m.name.startswith(w)]
            if matches:
                chosen.append(matches[0])
            else:
                console.print(f"[yellow]warning:[/yellow] no model matching {w!r}")
        selected = chosen or by_size
    elif getattr(args, "all", False):
        selected = by_size
    else:
        # Fast default: the smallest few models. Everything via --all.
        default_n = 3
        selected = by_size[:default_n]
        if len(by_size) > len(selected):
            console.print(
                f"[dim]Benchmarking the {len(selected)} smallest of "
                f"{len(by_size)} models — use [b]--all[/b] for everything, "
                f"or [b]-m name,…[/b] to choose.[/dim]"
            )
    if getattr(args, "limit", None):
        selected = selected[: args.limit]
    return selected


def _resolve_suite(args):
    """Build a custom suite from --tasks / --add-tasks, or None for built-in."""
    from .quality import default_suite, load_packs

    tasks = getattr(args, "tasks", None)
    add_tasks = getattr(args, "add_tasks", None)
    if not tasks and not add_tasks:
        return None
    suite = load_packs(tasks) if tasks else list(default_suite(include_open=True))
    if add_tasks:
        suite = suite + load_packs(add_tasks)
    return suite


def _build_runner(provider, args) -> Runner:
    judge = None
    include_open = False
    if getattr(args, "judge", None):
        judge = LLMJudge(provider, args.judge)
        include_open = True
    suite = _resolve_suite(args)
    if suite is not None:
        include_open = include_open or any(t.grader is None for t in suite)
    config = RunConfig(
        max_tokens=args.max_tokens,
        speed_max_tokens=args.speed_tokens,
        repeat=max(1, args.repeat),
        timeout=args.timeout,
        seed=args.seed,
        warmup=not args.no_warmup,
        unload_between=not args.no_unload,
        sample_rss=not args.no_rss,
        include_open=include_open,
        judge_model=args.judge,
        run_quality=not args.no_quality,
        run_speed=not args.no_speed,
        suite=suite,
        quick=not getattr(args, "full", False),
        use_cache=not getattr(args, "no_cache", False),
        refresh_cache=getattr(args, "refresh_cache", False),
    )
    return Runner(provider, config, judge=judge)


def _export(result, args, console: Console) -> None:
    from .report import to_html, to_json, to_markdown

    if getattr(args, "md", None):
        with open(args.md, "w") as f:
            f.write(to_markdown(result))
        console.print(f"[green]✓[/green] wrote Markdown report to [bold]{args.md}[/bold]")
    if getattr(args, "json_out", None):
        with open(args.json_out, "w") as f:
            f.write(to_json(result))
        console.print(f"[green]✓[/green] wrote JSON results to [bold]{args.json_out}[/bold]")
    if getattr(args, "html_out", None):
        with open(args.html_out, "w") as f:
            f.write(to_html(result))
        console.print(f"[green]✓[/green] wrote HTML report to [bold]{args.html_out}[/bold]")


# ---------------------------------------------------------------------------
def cmd_list(args, console: Console) -> int:
    try:
        provider = _resolve_provider(args, console)
        models = provider.list_models()
    except ProviderError as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1
    from rich.table import Table

    t = Table(title=f"Discovered models ({provider.name})", header_style="bold cyan")
    t.add_column("Model", style="bold")
    t.add_column("Params", justify="right")
    t.add_column("Quant")
    t.add_column("Family")
    t.add_column("Disk size", justify="right")
    from .report import fmt_bytes

    for m in models:
        t.add_row(m.name, m.parameter_size or "–", m.quantization or "–",
                  m.family or "–", fmt_bytes(m.size_bytes))
    console.print(t)
    return 0


def cmd_tasks(args, console: Console) -> int:
    from rich.table import Table

    from .quality import PackError, default_suite

    try:
        suite = _resolve_suite(args)
    except PackError as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1
    if suite is None:
        suite = default_suite(include_open=True)

    t = Table(title="Quality suite", header_style="bold cyan")
    t.add_column("ID", style="bold")
    t.add_column("Category")
    t.add_column("Grading")
    t.add_column("Quick", justify="center")
    quick_n = 0
    for task in suite:
        grading = "deterministic" if task.grader is not None else "LLM judge"
        is_quick = getattr(task, "quick", False)
        quick_n += 1 if is_quick else 0
        t.add_row(task.id, task.category, grading,
                  "[green]✓[/green]" if is_quick else "")
    console.print(t)
    console.print(f"[dim]{len(suite)} tasks · {quick_n} in the fast default "
                  "subset (the rest run with --full)[/dim]")
    return 0


def cmd_history(args, console: Console) -> int:
    from .history import history_table, list_runs

    runs = list_runs()
    if not runs:
        console.print("No saved runs yet. Run a benchmark first "
                      "(runs are saved automatically).")
        return 0
    console.print(history_table(runs[: args.limit]))
    return 0


def cmd_diff(args, console: Console) -> int:
    from .history import HistoryError, diff_table, resolve_ref

    try:
        if args.a is None and args.b is None:
            base, new = resolve_ref("prev"), resolve_ref("latest")
        elif args.b is None:
            base, new = resolve_ref(args.a), resolve_ref("latest")
        else:
            base, new = resolve_ref(args.a), resolve_ref(args.b)
    except HistoryError as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1
    console.print(diff_table(base, new))
    from .report import env_summary
    a_env, b_env = env_summary(base.environment), env_summary(new.environment)
    if a_env and b_env and a_env != b_env:
        console.print("\n[yellow]note:[/yellow] these runs were captured in "
                      "different environments — not strictly apples-to-apples:")
        console.print(f"  base: [dim]{a_env}[/dim]")
        console.print(f"  new:  [dim]{b_env}[/dim]")

    if getattr(args, "fail_on_regression", False):
        from .history import regressions
        regs = regressions(base, new, args.quality_threshold, args.speed_threshold)
        if regs:
            console.print("\n[red]✗ regressions detected:[/red]")
            for r in regs:
                if r.metric == "quality":
                    console.print(f"  [bold]{r.model}[/bold]: quality "
                                  f"{r.base:.0f}% → {r.new:.0f}% "
                                  f"(−{r.drop:g} pts > {r.threshold:g})")
                else:
                    console.print(f"  [bold]{r.model}[/bold]: tok/s "
                                  f"{r.base:.1f} → {r.new:.1f} "
                                  f"(−{r.drop:g}% > {r.threshold:g}%)")
            return 1
        console.print("\n[green]✓ no regressions beyond thresholds[/green]")
    return 0


def cmd_doctor(args, console: Console) -> int:
    from rich.table import Table

    from .doctor import run_checks, summarize

    checks = run_checks()
    icons = {"ok": "[green]✓[/green]", "warn": "[yellow]⚠[/yellow]",
             "fail": "[red]✗[/red]", "info": "[blue]i[/blue]"}
    t = Table(title="homebench doctor", header_style="bold cyan", show_header=False)
    t.add_column(justify="center")
    t.add_column(style="bold")
    t.add_column()
    for c in checks:
        t.add_row(icons.get(c.status, "?"), c.name, c.detail)
    console.print(t)

    fails, warns = summarize(checks)
    if fails:
        msg = f"[red]{fails} problem(s) to fix[/red]"
        if warns:
            msg += f", {warns} warning(s)"
        console.print(f"\n{msg}")
    elif warns:
        console.print(f"\n[yellow]{warns} warning(s)[/yellow] — you can still benchmark")
    else:
        console.print("\n[green]All good — you're ready to benchmark.[/green]")
    return 1 if fails else 0


def cmd_report(args, console: Console) -> int:
    from .history import HistoryError, resolve_ref
    from .models import BenchmarkResult
    from .report import env_summary, to_html, to_markdown

    if not args.html_out and not args.md:
        console.print("[red]error:[/red] choose an output — --html FILE and/or --md FILE")
        return 1
    try:
        rec = resolve_ref(args.ref)
    except HistoryError as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1
    result = BenchmarkResult.from_dict(rec.data)
    title = f"homebench report — {rec.when}"
    if args.html_out:
        with open(args.html_out, "w") as f:
            f.write(to_html(result, title=title))
        console.print(f"[green]✓[/green] wrote HTML report to [bold]{args.html_out}[/bold]")
    if args.md:
        with open(args.md, "w") as f:
            f.write(to_markdown(result))
        console.print(f"[green]✓[/green] wrote Markdown report to [bold]{args.md}[/bold]")
    env = env_summary(result.environment)
    if env:
        console.print(f"[dim]{rec.filename} · {env}[/dim]")
    return 0


def cmd_fit(args, console: Console) -> int:
    from .catalog import (
        CATALOG, CatalogError, QUANT_GB_PER_B, evaluate_catalog, load_catalog,
    )
    from .hardware import GPUInfo, capture
    from .report import fit_table, hardware_table

    if args.quant and args.quant not in QUANT_GB_PER_B:
        console.print(f"[red]error:[/red] unknown quant {args.quant!r}. "
                      f"Choices: {', '.join(QUANT_GB_PER_B)}")
        return 1
    if args.context < 256:
        console.print("[red]error:[/red] --context must be >= 256")
        return 1

    source_note = "built-in catalog"
    if getattr(args, "online", False):
        from .hub import HubError, top_models
        try:
            models, source_note = top_models(limit=args.top, sort=args.sort,
                                             refresh=args.refresh)
        except HubError as exc:
            console.print(f"[yellow]warning:[/yellow] {exc}")
            console.print("[dim]falling back to the built-in catalog.[/dim]")
            models = list(CATALOG)
    else:
        models = list(CATALOG)

    if getattr(args, "catalog", None):
        try:
            for path in args.catalog:
                models += load_catalog(path)
        except CatalogError as exc:
            console.print(f"[red]catalog error:[/red] {exc}")
            return 1

    hw = capture()
    if args.ram is not None:
        hw.ram_total_bytes = int(args.ram * 1_000_000_000)
        hw.ram_available_bytes = min(hw.ram_available_bytes, hw.ram_total_bytes)
    if args.vram is not None:
        hw.gpu = GPUInfo(name="(assumed)", vram_bytes=int(args.vram * 1_000_000_000),
                         kind="nvidia")

    budget, _label = hw.memory_budget()
    console.print(hardware_table(hw))
    console.print(f"[dim]Models: {source_note} · {len(models)} candidates[/dim]\n")
    results = evaluate_catalog(budget, context=args.context, quant=args.quant,
                               catalog=models)
    console.print(fit_table(results, show_all=args.all))

    fits = [r for r in results if r.status in ("fits", "tight")]
    console.print()
    if fits:
        biggest = max(fits, key=lambda r: r.model.params_b)
        console.print(
            f"[green]{len(fits)}[/green]/{len(results)} catalogued models fit — "
            f"largest: [bold]{biggest.model.name}[/bold] at {biggest.quant}."
        )
    else:
        console.print("[yellow]No catalogued models fit the detected budget.[/yellow]")
    console.print("[dim]Estimates = weights + KV cache + overhead; real usage "
                  "varies with context and backend. HuggingFace repos also work "
                  "with LM Studio and vLLM.[/dim]")
    if not args.all and any(r.status == "no" for r in results):
        console.print("[dim]Pass --all to include models that don't fit.[/dim]")

    if getattr(args, "json_out", None):
        import json

        with open(args.json_out, "w") as f:
            json.dump({"hardware": hw.to_dict(),
                       "budget_bytes": budget,
                       "context": args.context,
                       "results": [r.to_dict() for r in results]}, f, indent=2)
        console.print(f"[green]✓[/green] wrote hardware + fit to "
                      f"[bold]{args.json_out}[/bold]")
    return 0


def cmd_throughput(args, console: Console) -> int:
    from .metrics import measure_throughput, parse_levels
    from .report import throughput_table

    try:
        provider = _resolve_provider(args, console)
        models = _select_models(provider, args, console)
        levels = parse_levels(args.concurrency)
    except (ProviderError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1

    console.print(
        f"[bold]homebench throughput[/bold] · provider [cyan]{provider.name}[/cyan] "
        f"· concurrency {levels}\n"
    )
    if provider.name in ("ollama", "lmstudio"):
        console.print(
            "[dim]note: aggregate throughput only scales if the server batches "
            "concurrent requests (vLLM, llama.cpp continuous batching, or Ollama "
            "with OLLAMA_NUM_PARALLEL>1).[/dim]\n"
        )

    results = []
    for m in models:
        def on_start(c, n, name=m.name):
            console.print(f"  [bold]{name}[/bold]: concurrency {c} ({n} requests)…")

        res = measure_throughput(
            provider, m.name, concurrency_levels=levels, requests=args.requests,
            max_tokens=args.max_tokens, temperature=0.0, seed=args.seed,
            timeout=args.timeout, warmup=not args.no_warmup, on_level_start=on_start,
        )
        results.append(res)
        console.print(throughput_table(res))
        console.print()
        provider.unload(m.name)

    if getattr(args, "json_out", None):
        import json

        with open(args.json_out, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        console.print(f"[green]✓[/green] wrote throughput results to "
                      f"[bold]{args.json_out}[/bold]")
    return 0


def cmd_run(args, console: Console) -> int:
    from .quality import PackError

    try:
        provider = _resolve_provider(args, console)
        models = _select_models(provider, args, console)
        runner = _build_runner(provider, args)
    except ProviderError as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1
    except PackError as exc:
        console.print(f"[red]task pack error:[/red] {exc}")
        return 1

    use_tui = _should_use_tui(args, console)
    if use_tui:
        from .tui.app import run_tui

        result = run_tui(runner, models)
    else:
        from .plainui import run_plain

        result = run_plain(runner, models, console)

    if result is None:
        return 1
    from .score import value_verdict
    verdict = value_verdict(result.reports)
    if verdict:
        console.print(f"\n[bold green]🏆 Best value for your laptop:[/bold green] {verdict}")
    if runner.cache is not None and runner.cache.hits:
        console.print(f"[dim]Reused {runner.cache.hits} cached quality "
                      "response(s) — pass --refresh-cache to recompute.[/dim]")
    _export(result, args, console)
    if not getattr(args, "no_save", False):
        from .history import save_run

        try:
            path = save_run(result, label=getattr(args, "label", None))
            console.print(f"[green]✓[/green] saved run to [bold]{path}[/bold]  "
                          "(see `homebench history` / `homebench diff`)")
        except OSError as exc:
            console.print(f"[yellow]warning:[/yellow] could not save run: {exc}")
    return 0


def _should_use_tui(args, console: Console) -> bool:
    if getattr(args, "tui", False):
        return True
    if getattr(args, "no_tui", False):
        return False
    # Auto: only when we have a real interactive terminal.
    return console.is_terminal and sys.stdin.isatty()


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(_inject_default_command(argv))
    console = Console()

    command = getattr(args, "command", None)
    if command == "list":
        return cmd_list(args, console)
    if command == "tasks":
        return cmd_tasks(args, console)
    if command == "doctor":
        return cmd_doctor(args, console)
    if command == "history":
        return cmd_history(args, console)
    if command == "diff":
        return cmd_diff(args, console)
    if command == "report":
        return cmd_report(args, console)
    if command == "throughput":
        return cmd_throughput(args, console)
    if command == "fit":
        return cmd_fit(args, console)
    # "run" (explicit or injected default) runs the benchmark
    return cmd_run(args, console)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
