"""The benchmark orchestrator.

Runs, per model: an optional warmup, a repeated speed probe (tokens/sec +
time-to-first-token), a memory reading, then the quality suite. Progress is
reported through a lightweight observer callback so both the TUI and the
plain-CLI renderer can subscribe to the same events.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .metrics.memory import RSSSampler
from .models import (
    BenchmarkResult,
    MemoryMetrics,
    ModelInfo,
    ModelReport,
    SpeedMetrics,
    TaskResult,
)
from .providers.base import Provider, ProviderError
from .quality import LLMJudge, Task, default_suite

# ---- observer events -------------------------------------------------------
EV_RUN_START = "run_start"
EV_MODEL_START = "model_start"
EV_PHASE = "phase"            # data: model, phase, note
EV_TASK_DONE = "task_done"    # data: model, task_id, category, score, passed
EV_MODEL_DONE = "model_done"  # data: report
EV_RUN_DONE = "run_done"      # data: result

Observer = Optional[Callable[..., None]]


def _emit(observer: Observer, event: str, **data) -> None:
    if observer is not None:
        observer(event, **data)


DEFAULT_SPEED_PROMPT = (
    "Write three detailed paragraphs explaining how a computer's CPU executes "
    "a program, from source code to running instructions."
)


@dataclass
class RunConfig:
    max_tokens: int = 256
    speed_max_tokens: int = 200
    temperature: float = 0.0
    seed: Optional[int] = 42
    speed_prompt: str = DEFAULT_SPEED_PROMPT
    repeat: int = 1                 # speed-probe repetitions (best is kept)
    warmup: bool = True
    unload_between: bool = True
    sample_rss: bool = True
    timeout: float = 300.0
    include_open: bool = False
    judge_model: Optional[str] = None
    run_quality: bool = True
    run_speed: bool = True

    def to_dict(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "speed_max_tokens": self.speed_max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "repeat": self.repeat,
            "warmup": self.warmup,
            "unload_between": self.unload_between,
            "sample_rss": self.sample_rss,
            "include_open": self.include_open,
            "judge_model": self.judge_model,
            "run_quality": self.run_quality,
            "run_speed": self.run_speed,
        }


class Runner:
    def __init__(
        self,
        provider: Provider,
        config: Optional[RunConfig] = None,
        judge: Optional[LLMJudge] = None,
    ):
        self.provider = provider
        self.config = config or RunConfig()
        self.judge = judge

    # ------------------------------------------------------------------
    def run(self, models: List[ModelInfo], observer: Observer = None) -> BenchmarkResult:
        result = BenchmarkResult(
            provider=self.provider.name,
            config=self.config.to_dict(),
        )
        _emit(observer, EV_RUN_START, models=[m.name for m in models])
        for model in models:
            _emit(observer, EV_MODEL_START, model=model.name)
            report = self._run_model(model, observer)
            result.reports.append(report)
            _emit(observer, EV_MODEL_DONE, report=report)
        result.finished_at = time.time()
        _emit(observer, EV_RUN_DONE, result=result)
        return result

    # ------------------------------------------------------------------
    def _run_model(self, model: ModelInfo, observer: Observer) -> ModelReport:
        report = ModelReport(model=model)
        cfg = self.config
        try:
            if cfg.warmup:
                _emit(observer, EV_PHASE, model=model.name, phase="warmup")
                self.provider.warmup(model.name, timeout=cfg.timeout)

            if cfg.run_speed:
                _emit(observer, EV_PHASE, model=model.name, phase="speed")
                report.speed, report.memory = self._measure_speed(model.name)

            if cfg.run_quality:
                _emit(observer, EV_PHASE, model=model.name, phase="quality")
                report.task_results = self._run_quality(model.name, observer)

            if cfg.unload_between:
                self.provider.unload(model.name)
        except ProviderError as exc:
            report.error = str(exc)
            _emit(observer, EV_PHASE, model=model.name, phase="error", note=str(exc))
        return report

    # ------------------------------------------------------------------
    def _measure_speed(self, model: str):
        cfg = self.config
        runs: List[SpeedMetrics] = []
        mem = MemoryMetrics()
        rss_peak_delta = 0
        for _ in range(max(1, cfg.repeat)):
            sampler = RSSSampler(name_contains=self.provider.name) if cfg.sample_rss else None
            if sampler is not None:
                sampler.__enter__()
            gen = self.provider.generate(
                model,
                cfg.speed_prompt,
                max_tokens=cfg.speed_max_tokens,
                temperature=cfg.temperature,
                seed=cfg.seed,
                timeout=cfg.timeout,
            )
            if sampler is not None:
                sampler.__exit__()
                rss_peak_delta = max(rss_peak_delta, sampler.delta_bytes)
            runs.append(gen.speed)

        # Keep the best (highest tokens/sec) run as the representative sample;
        # report median TTFT to smooth out first-call noise.
        best = max(runs, key=lambda s: s.tokens_per_sec)
        best.ttft_s = statistics.median(r.ttft_s for r in runs)

        # Memory: provider view is authoritative; RSS delta supplements it.
        pmem = self.provider.memory(model)
        mem.size_bytes = pmem.size_bytes
        mem.vram_bytes = pmem.vram_bytes
        mem.rss_peak_bytes = rss_peak_delta
        return best, mem

    # ------------------------------------------------------------------
    def _run_quality(self, model: str, observer: Observer) -> List[TaskResult]:
        cfg = self.config
        include_open = cfg.include_open and self.judge is not None
        suite = default_suite(include_open=include_open)
        results: List[TaskResult] = []
        for task in suite:
            tr = self._run_task(model, task)
            results.append(tr)
            _emit(
                observer, EV_TASK_DONE, model=model, task_id=task.id,
                category=task.category, score=tr.score, passed=tr.passed,
            )
        return results

    def _run_task(self, model: str, task: Task) -> TaskResult:
        cfg = self.config
        start = time.perf_counter()
        try:
            gen = self.provider.generate(
                model, task.prompt,
                max_tokens=task.max_tokens,
                temperature=cfg.temperature,
                seed=cfg.seed,
                timeout=cfg.timeout,
            )
        except ProviderError as exc:
            return TaskResult(
                task.id, task.category, 0.0, False,
                latency_s=time.perf_counter() - start,
                detail=f"generation error: {exc}",
            )
        latency = time.perf_counter() - start
        response = gen.text

        if task.grader is not None:
            grade = task.grader(response)
        elif self.judge is not None:
            grade = self.judge.score(task.prompt, response, task.reference)
        else:
            # open-ended task with no judge available: skip scoring
            grade = None

        if grade is None:
            return TaskResult(
                task.id, task.category, 0.0, False, latency_s=latency,
                output_tokens=gen.speed.output_tokens, response=response,
                detail="skipped (no grader/judge)",
            )
        return TaskResult(
            task.id, task.category, grade.score, grade.passed,
            latency_s=latency, output_tokens=gen.speed.output_tokens,
            response=response, detail=grade.detail,
        )
