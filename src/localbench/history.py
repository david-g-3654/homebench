"""Historical runs: save results, list past runs, and diff two of them.

Runs are stored as JSON under ``$LOCALBENCH_HOME/runs`` (default
``~/.localbench/runs``). Each file is a full :meth:`BenchmarkResult.to_dict`
plus a little metadata (label, version, saved-at).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.table import Table
from rich.text import Text

from . import __version__
from .models import BenchmarkResult
from .report import fmt_bytes, fmt_quality, fmt_tps, fmt_ttft


class HistoryError(RuntimeError):
    pass


# ---- locations -------------------------------------------------------------
def default_home() -> str:
    return os.environ.get("LOCALBENCH_HOME") or os.path.expanduser("~/.localbench")


def runs_dir(home: Optional[str] = None) -> str:
    return os.path.join(home or default_home(), "runs")


# ---- saving ----------------------------------------------------------------
def save_run(
    result: BenchmarkResult,
    home: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    d = runs_dir(home)
    os.makedirs(d, exist_ok=True)
    payload: Dict[str, Any] = result.to_dict()
    payload["label"] = label
    payload["version"] = __version__

    stamp = datetime.fromtimestamp(result.started_at or time.time())
    base = "run-" + stamp.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(d, base + ".json")
    n = 1
    while os.path.exists(path):  # avoid clobbering same-second runs
        path = os.path.join(d, f"{base}-{n}.json")
        n += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


# ---- records ---------------------------------------------------------------
@dataclass
class RunRecord:
    path: str
    data: Dict[str, Any]

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    @property
    def started_at(self) -> float:
        return float(self.data.get("started_at", 0) or 0)

    @property
    def when(self) -> str:
        if not self.started_at:
            return "?"
        return datetime.fromtimestamp(self.started_at).strftime("%Y-%m-%d %H:%M")

    @property
    def provider(self) -> str:
        return self.data.get("provider", "?")

    @property
    def label(self) -> Optional[str]:
        return self.data.get("label")

    @property
    def reports(self) -> List[Dict[str, Any]]:
        return self.data.get("reports", []) or []

    @property
    def model_names(self) -> List[str]:
        return [r.get("model", {}).get("name", "?") for r in self.reports]

    def best(self) -> Optional[Dict[str, Any]]:
        scored = [r for r in self.reports if r.get("quality_score") is not None]
        if not scored:
            return self.reports[0] if self.reports else None
        return max(scored, key=lambda r: r.get("quality_score", -1))


def _read_record(path: str) -> Optional[RunRecord]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return RunRecord(path=path, data=json.load(f))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def list_runs(home: Optional[str] = None) -> List[RunRecord]:
    d = runs_dir(home)
    if not os.path.isdir(d):
        return []
    records: List[RunRecord] = []
    for name in os.listdir(d):
        if not name.endswith(".json"):
            continue
        rec = _read_record(os.path.join(d, name))
        if rec is not None:
            records.append(rec)
    records.sort(key=lambda r: r.started_at, reverse=True)  # newest first
    return records


def resolve_ref(ref: str, home: Optional[str] = None) -> RunRecord:
    """Resolve a run reference: 'latest', 'prev', a 1-based index, or a path."""
    runs = list_runs(home)
    key = str(ref).strip().lower()
    if key in ("latest", "last"):
        _need(runs, 1)
        return runs[0]
    if key in ("prev", "previous"):
        _need(runs, 2)
        return runs[1]
    if key.isdigit():
        idx = int(key)
        _need(runs, idx)
        return runs[idx - 1]
    # treat as a path (absolute, relative, or a filename in the runs dir)
    for candidate in (ref, os.path.join(runs_dir(home), ref)):
        if os.path.isfile(candidate):
            rec = _read_record(candidate)
            if rec is None:
                raise HistoryError(f"could not read run file {candidate!r}")
            return rec
    raise HistoryError(f"no run matching {ref!r} (try `localbench history`)")


def _need(runs: List[RunRecord], n: int) -> None:
    if len(runs) < n:
        raise HistoryError(
            f"need at least {n} saved run(s), found {len(runs)}. "
            "Run a benchmark first (runs are saved automatically)."
        )


# ---- history table ---------------------------------------------------------
def history_table(runs: List[RunRecord]) -> Table:
    t = Table(title="Run history", header_style="bold cyan")
    t.add_column("#", justify="right", style="dim")
    t.add_column("When")
    t.add_column("Provider")
    t.add_column("Models", justify="right")
    t.add_column("Best", overflow="ellipsis")
    t.add_column("Label", style="italic")
    t.add_column("File", style="dim")
    for i, r in enumerate(runs, start=1):
        best = r.best()
        best_txt = "–"
        if best:
            q = best.get("quality_score")
            name = best.get("model", {}).get("name", "?")
            best_txt = f"{name} ({fmt_quality(q)})" if q is not None else name
        t.add_row(
            str(i), r.when, r.provider, str(len(r.reports)),
            best_txt, r.label or "", r.filename,
        )
    return t


# ---- diffing ---------------------------------------------------------------
def _metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "quality": report.get("quality_score"),
        "tps": (report.get("speed") or {}).get("tokens_per_sec", 0.0),
        "ttft": (report.get("speed") or {}).get("ttft_s", 0.0),
        "mem": (report.get("memory") or {}).get("size_bytes", 0)
        or (report.get("model") or {}).get("size_bytes", 0),
    }


def _delta(new: Optional[float], old: Optional[float], suffix: str = "",
           pct: bool = False, invert: bool = False) -> Text:
    if new is None or old is None:
        return Text("–", style="dim")
    d = new - old
    if abs(d) < 1e-9:
        return Text("±0", style="dim")
    good = (d < 0) if invert else (d > 0)
    style = "green" if good else "red"
    sign = "+" if d > 0 else ""
    body = f"{sign}{d:.0f}{suffix}" if pct else f"{sign}{d:.1f}{suffix}"
    return Text(body, style=style)


def diff_table(base: RunRecord, new: RunRecord) -> Table:
    """Compare two runs model-by-model (base = older, new = newer)."""
    by_name_a = {r.get("model", {}).get("name"): r for r in base.reports}
    by_name_b = {r.get("model", {}).get("name"): r for r in new.reports}
    names = list(by_name_b) + [n for n in by_name_a if n not in by_name_b]

    title = f"Diff: {base.filename} → {new.filename}"
    t = Table(title=title, header_style="bold cyan")
    t.add_column("Model", style="bold")
    t.add_column("Quality", justify="right")
    t.add_column("ΔQual", justify="right")
    t.add_column("tok/s", justify="right")
    t.add_column("Δtok/s", justify="right")
    t.add_column("Memory", justify="right")
    t.add_column("", justify="left")  # status

    for name in names:
        ra, rb = by_name_a.get(name), by_name_b.get(name)
        if rb is None:  # only in base -> removed
            ma = _metrics(ra)
            t.add_row(name, fmt_quality(ma["quality"]), Text("–", style="dim"),
                      fmt_tps(ma["tps"]), Text("–", style="dim"),
                      fmt_bytes(ma["mem"]), Text("removed", style="red"))
            continue
        mb = _metrics(rb)
        if ra is None:  # only in new -> added
            t.add_row(name, fmt_quality(mb["quality"]), Text("new", style="green"),
                      fmt_tps(mb["tps"]), Text("new", style="green"),
                      fmt_bytes(mb["mem"]), Text("added", style="green"))
            continue
        ma = _metrics(ra)
        t.add_row(
            name,
            fmt_quality(mb["quality"]),
            _delta(mb["quality"], ma["quality"], suffix="%", pct=True),
            fmt_tps(mb["tps"]),
            _delta(mb["tps"], ma["tps"]),
            fmt_bytes(mb["mem"]),
            "",
        )
    return t
