"""`homebench doctor` — diagnose the local setup.

Checks Python, hardware, reachable providers + pulled models, the home/cache
directory, optional deps, and whether a newer release is available. Every check
returns ok / warn / fail / info so first-time setup problems are obvious.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Check:
    name: str
    status: str          # "ok" | "warn" | "fail" | "info"
    detail: str = ""


def _ver(v: str) -> tuple:
    out = []
    for part in str(v).split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def _latest_pypi(timeout: float = 3.0) -> Optional[str]:
    try:
        import httpx
        r = httpx.get("https://pypi.org/pypi/homebench/json", timeout=timeout)
        r.raise_for_status()
        return r.json()["info"]["version"]
    except Exception:
        return None


def run_checks(home: Optional[str] = None, check_pypi: bool = True) -> List[Check]:
    from . import __version__
    from .report import fmt_bytes

    checks: List[Check] = []

    # Python
    py = platform.python_version()
    ok_py = sys.version_info >= (3, 9)
    checks.append(Check("Python", "ok" if ok_py else "fail",
                        py if ok_py else f"{py} — homebench needs 3.9+"))

    # Hardware
    try:
        from .hardware import capture
        hw = capture()
        budget, label = hw.memory_budget()
        checks.append(Check(
            "Hardware", "ok",
            f"{hw.cpu} · {fmt_bytes(hw.ram_total_bytes)} RAM · "
            f"model budget {fmt_bytes(budget)} ({label})"))
    except Exception as exc:  # pragma: no cover - defensive
        checks.append(Check("Hardware", "warn", f"could not read: {exc}"))

    # Providers + models
    from .providers import _AUTO_DETECT, _PROVIDERS
    reachable = []
    for name in _AUTO_DETECT:
        try:
            provider = _PROVIDERS[name]()
            if provider.is_available():
                try:
                    n = len(provider.list_models())
                except Exception:
                    n = 0
                reachable.append((name, n))
        except Exception:
            continue
    if reachable:
        for name, n in reachable:
            if n > 0:
                checks.append(Check(f"Provider: {name}", "ok", f"reachable · {n} model(s)"))
            else:
                checks.append(Check(f"Provider: {name}", "warn",
                                    "reachable but no models — pull one, "
                                    "e.g. `ollama pull llama3.2`"))
    else:
        checks.append(Check("Providers", "fail",
                            "none reachable — start one (e.g. `ollama serve`), "
                            "then `ollama pull llama3.2`"))

    # Home / cache / history
    from .history import default_home, list_runs
    h = home or default_home()
    try:
        os.makedirs(h, exist_ok=True)
        probe = os.path.join(h, ".doctor-probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        runs = len(list_runs(home=h))
        cached = os.path.exists(os.path.join(h, "response-cache.json"))
        checks.append(Check("Home directory", "ok",
                            f"{h} · writable · {runs} saved run(s) · "
                            f"{'cache present' if cached else 'no cache yet'}"))
    except OSError as exc:
        checks.append(Check("Home directory", "fail", f"{h} not writable: {exc}"))

    # Optional deps
    try:
        import yaml  # noqa: F401
        checks.append(Check("YAML task packs", "ok", "pyyaml installed"))
    except ImportError:
        checks.append(Check("YAML task packs", "info",
                            "pyyaml not installed — JSON packs still work "
                            "(`pip install \"homebench[yaml]\"` for YAML)"))

    # Version
    latest = _latest_pypi() if check_pypi else None
    if latest and _ver(latest) > _ver(__version__):
        checks.append(Check("Version", "info",
                            f"{__version__} installed — {latest} available "
                            "(`pip install -U homebench`)"))
    elif latest and _ver(latest) < _ver(__version__):
        checks.append(Check("Version", "ok",
                            f"{__version__} (ahead of PyPI {latest} — dev build)"))
    else:
        checks.append(Check("Version", "ok",
                            __version__ + (" (latest)" if latest else "")))
    return checks


def summarize(checks: List[Check]) -> tuple:
    fails = sum(1 for c in checks if c.status == "fail")
    warns = sum(1 for c in checks if c.status == "warn")
    return fails, warns
