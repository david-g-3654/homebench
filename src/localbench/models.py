"""Data models shared across localbench.

Everything the runner produces is a plain dataclass so results serialize
cleanly to JSON and are trivial to test.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelInfo:
    """A model discovered from a provider."""

    name: str
    provider: str
    size_bytes: int = 0
    parameter_size: str = ""
    quantization: str = ""
    family: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpeedMetrics:
    """Timing for a single generation.

    Durations are seconds. ``tokens_per_sec`` is the pure generation rate
    (output tokens / eval time), which is the number people quote for local
    models and excludes prompt processing and model-load time.
    """

    ttft_s: float = 0.0            # time to first token (wall clock)
    tokens_per_sec: float = 0.0    # output eval rate
    prompt_tokens: int = 0
    output_tokens: int = 0
    prompt_eval_s: float = 0.0
    eval_s: float = 0.0
    load_s: float = 0.0            # model load time (excluded from other metrics)
    total_s: float = 0.0           # wall-clock request duration

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryMetrics:
    """Memory footprint of a loaded model.

    ``size_bytes`` / ``vram_bytes`` come from the provider's view of the
    resident model. ``rss_peak_bytes`` is the peak resident-set delta we
    sampled across the provider's processes during generation (best-effort).
    """

    size_bytes: int = 0
    vram_bytes: int = 0
    rss_peak_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskResult:
    """Outcome of one quality task for one model."""

    task_id: str
    category: str
    score: float                 # 0.0 .. 1.0
    passed: bool
    latency_s: float = 0.0
    output_tokens: int = 0
    response: str = ""
    detail: str = ""             # grader explanation / judge rationale

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelReport:
    """Everything measured for a single model in one benchmark run."""

    model: ModelInfo
    speed: SpeedMetrics = field(default_factory=SpeedMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    task_results: List[TaskResult] = field(default_factory=list)
    error: Optional[str] = None

    # ---- derived views -------------------------------------------------
    @property
    def quality_score(self) -> Optional[float]:
        """Mean task score in 0..100, or None if no tasks were graded."""
        if not self.task_results:
            return None
        return 100.0 * sum(t.score for t in self.task_results) / len(self.task_results)

    @property
    def tasks_passed(self) -> int:
        return sum(1 for t in self.task_results if t.passed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "speed": self.speed.to_dict(),
            "memory": self.memory.to_dict(),
            "quality_score": self.quality_score,
            "tasks_passed": self.tasks_passed,
            "tasks_total": len(self.task_results),
            "task_results": [t.to_dict() for t in self.task_results],
            "error": self.error,
        }


@dataclass
class BenchmarkResult:
    """A complete benchmark run across one or more models."""

    reports: List[ModelReport] = field(default_factory=list)
    provider: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": self.config,
            "reports": [r.to_dict() for r in self.reports],
        }
