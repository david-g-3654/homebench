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
                        help="comma-separated model names (default: all discovered)")
        sp.add_argument("--limit", type=int, default=None,
                        help="benchmark at most N models")
        sp.add_argument("--provider", default=None,
                        help="ollama|lmstudio|llamacpp|vllm|openai "
                             "(default: auto-detect)")
        sp.add_argument("--host", default=None,
                        help="provider host URL (default: the provider's env var "
                             "or its standard localhost port)")
        sp.add_argument("--max-tokens", type=int, default=256,
                        help="max tokens per quality task (default: 256)")
        sp.add_argument("--speed-tokens", type=int, default=200,
                        help="tokens to generate for the speed probe (default: 200)")
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

    diff_p = sub.add_parser("diff", help="diff two saved runs (base -> new)")
    diff_p.add_argument("a", nargs="?", default=None,
                        help="base run: 'latest'/'prev', a 1-based index, or a path")
    diff_p.add_argument("b", nargs="?", default=None,
                        help="new run (default: latest)")

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
    return p


_COMMANDS = {"run", "list", "tasks", "history", "diff", "throughput"}


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
        selected = chosen or available
    else:
        selected = available
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
    )
    return Runner(provider, config, judge=judge)


def _export(result, args, console: Console) -> None:
    from .report import to_json, to_markdown

    if getattr(args, "md", None):
        with open(args.md, "w") as f:
            f.write(to_markdown(result))
        console.print(f"[green]✓[/green] wrote Markdown report to [bold]{args.md}[/bold]")
    if getattr(args, "json_out", None):
        with open(args.json_out, "w") as f:
            f.write(to_json(result))
        console.print(f"[green]✓[/green] wrote JSON results to [bold]{args.json_out}[/bold]")


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
    for task in suite:
        grading = "deterministic" if task.grader is not None else "LLM judge"
        t.add_row(task.id, task.category, grading)
    console.print(t)
    console.print(f"[dim]{len(suite)} tasks[/dim]")
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
    if command == "history":
        return cmd_history(args, console)
    if command == "diff":
        return cmd_diff(args, console)
    if command == "throughput":
        return cmd_throughput(args, console)
    # "run" (explicit or injected default) runs the benchmark
    return cmd_run(args, console)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
