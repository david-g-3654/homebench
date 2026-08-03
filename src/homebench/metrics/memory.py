"""Best-effort peak-RSS sampling for a provider's processes.

On unified-memory machines (Apple Silicon) the provider's ``/api/ps`` view is
the reliable footprint number; this sampler is a supplementary signal that
watches the resident set of the backend's processes while a model generates.
It degrades to zero if psutil can't see the processes.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

try:
    import psutil
except Exception:  # pragma: no cover - psutil is a hard dep, but be defensive
    psutil = None  # type: ignore


def _matching_procs(name_contains: str) -> List["psutil.Process"]:
    if psutil is None:
        return []
    procs = []
    needle = name_contains.lower()
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if needle in name or needle in cmd:
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs


def _total_rss(name_contains: str) -> int:
    total = 0
    for p in _matching_procs(name_contains):
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


class RSSSampler:
    """Samples summed RSS of matching processes in a background thread.

    Use as a context manager around a generation call; afterwards read
    :attr:`peak_bytes` (peak observed) and :attr:`delta_bytes` (peak minus the
    baseline captured on entry).
    """

    def __init__(self, name_contains: str = "ollama", interval: float = 0.1):
        self.name_contains = name_contains
        self.interval = interval
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def available(self) -> bool:
        return psutil is not None

    @property
    def delta_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)

    def __enter__(self) -> "RSSSampler":
        self.baseline_bytes = _total_rss(self.name_contains)
        self.peak_bytes = self.baseline_bytes
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            rss = _total_rss(self.name_contains)
            if rss > self.peak_bytes:
                self.peak_bytes = rss

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # one last sample in case the peak happened right at the end
        rss = _total_rss(self.name_contains)
        if rss > self.peak_bytes:
            self.peak_bytes = rss
