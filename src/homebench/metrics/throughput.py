"""Concurrency / batch-throughput measurement.

Single-stream tok/s undersells servers that batch requests (vLLM especially,
also llama.cpp's continuous batching, or Ollama with ``OLLAMA_NUM_PARALLEL``).
This fires N requests at a fixed concurrency C and measures *aggregate*
throughput — total output tokens ÷ wall-clock time — then sweeps C to show how
throughput scales.

Requests run in a thread pool: the provider's HTTP calls release the GIL while
waiting on the server, so real batching happens server-side.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..providers.base import Provider, ProviderError

DEFAULT_PROMPT = (
    "Write a few detailed sentences explaining how modern CPUs use pipelining "
    "and out-of-order execution to run instructions faster."
)


def parse_levels(spec: str) -> List[int]:
    """Parse a concurrency spec like '1,2,4,8' into a sorted unique int list."""
    levels = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            raise ValueError(f"invalid concurrency level {part!r}")
        if v >= 1:
            levels.append(v)
    if not levels:
        raise ValueError("no valid concurrency levels")
    return sorted(set(levels))


@dataclass
class ConcurrencyPoint:
    concurrency: int
    requests: int
    completed: int = 0
    errors: int = 0
    total_output_tokens: int = 0
    wall_s: float = 0.0
    aggregate_tps: float = 0.0      # total output tokens / wall time
    mean_req_tps: float = 0.0       # mean per-request output rate
    mean_latency_s: float = 0.0
    p95_latency_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThroughputResult:
    model: str
    provider: str
    points: List[ConcurrencyPoint] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def peak_aggregate_tps(self) -> float:
        return max((p.aggregate_tps for p in self.points), default=0.0)

    def speedup(self, point: ConcurrencyPoint) -> Optional[float]:
        """Aggregate throughput relative to the lowest concurrency measured."""
        if not self.points:
            return None
        base = self.points[0].aggregate_tps
        if base <= 0:
            return None
        return point.aggregate_tps / base

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "peak_aggregate_tps": self.peak_aggregate_tps,
            "points": [p.to_dict() for p in self.points],
            "error": self.error,
        }


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _auto_requests(concurrency: int) -> int:
    # Enough in-flight work to reach steady state without dragging on.
    return max(4, concurrency * 3)


def measure_throughput(
    provider: Provider,
    model: str,
    *,
    concurrency_levels: List[int],
    requests: Optional[int] = None,
    max_tokens: int = 128,
    prompt: str = DEFAULT_PROMPT,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    timeout: float = 300.0,
    warmup: bool = True,
    on_level_start: Optional[Callable[[int, int], None]] = None,
    on_level_done: Optional[Callable[[ConcurrencyPoint], None]] = None,
) -> ThroughputResult:
    result = ThroughputResult(model=model, provider=provider.name)
    try:
        if warmup:
            provider.warmup(model, timeout=timeout)
    except ProviderError as exc:
        result.error = str(exc)
        return result

    for c in concurrency_levels:
        n = requests if requests is not None else _auto_requests(c)
        if on_level_start is not None:
            on_level_start(c, n)
        point = _run_level(
            provider, model, concurrency=c, n=n, max_tokens=max_tokens,
            prompt=prompt, temperature=temperature, seed=seed, timeout=timeout,
        )
        result.points.append(point)
        if on_level_done is not None:
            on_level_done(point)
    return result


def _one_request(provider, model, prompt, max_tokens, temperature, seed, timeout):
    start = time.perf_counter()
    gen = provider.generate(
        model, prompt, max_tokens=max_tokens, temperature=temperature,
        seed=seed, timeout=timeout,
    )
    latency = time.perf_counter() - start
    return gen.speed.output_tokens, latency


def _run_level(provider, model, *, concurrency, n, max_tokens, prompt,
               temperature, seed, timeout) -> ConcurrencyPoint:
    point = ConcurrencyPoint(concurrency=concurrency, requests=n)
    latencies: List[float] = []
    req_tps: List[float] = []
    total_tokens = 0
    errors = 0

    # Vary each prompt slightly so servers with prefix caching don't dedupe.
    prompts = [f"{prompt}\n\n(request {i + 1})" for i in range(n)]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_one_request, provider, model, p, max_tokens,
                        temperature, seed, timeout)
            for p in prompts
        ]
        for fut in as_completed(futures):
            try:
                out_tokens, latency = fut.result()
            except ProviderError:
                errors += 1
                continue
            total_tokens += out_tokens
            latencies.append(latency)
            if latency > 0:
                req_tps.append(out_tokens / latency)
    wall = time.perf_counter() - start

    point.completed = n - errors
    point.errors = errors
    point.total_output_tokens = total_tokens
    point.wall_s = wall
    point.aggregate_tps = (total_tokens / wall) if wall > 0 else 0.0
    point.mean_req_tps = (sum(req_tps) / len(req_tps)) if req_tps else 0.0
    point.mean_latency_s = (sum(latencies) / len(latencies)) if latencies else 0.0
    point.p95_latency_s = _percentile(latencies, 0.95)
    return point
