"""LM Studio provider.

Talks to LM Studio's local server. Model discovery and generation go through
the OpenAI-compatible endpoints (``/v1``), which are the most stable across
LM Studio versions; when the richer native REST API (``/api/v0``) is present
we use it to enrich model metadata and read the loaded-model footprint.

Timings (tok/s, TTFT) are measured client-side from the streamed response,
since the OpenAI-compatible API does not report server-side eval durations.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

import httpx

from ..models import MemoryMetrics, ModelInfo, SpeedMetrics
from .base import GenerationResult, Provider, ProviderError, TokenCallback


def _normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host:
        host = "http://localhost:1234"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


class LMStudioProvider(Provider):
    name = "lmstudio"

    def __init__(self, host: Optional[str] = None):
        env = os.environ.get("LMSTUDIO_HOST")
        self.host = _normalize_host(host or env or "http://localhost:1234")

    @property
    def process_hint(self) -> str:
        # Matches "LM Studio Helper" and the bundled runner processes.
        return "LM Studio"

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.host}/v1/models", timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> List[ModelInfo]:
        # Prefer the native API for quantization/arch details; fall back to /v1.
        native = self._native_models()
        if native:
            return native
        try:
            r = httpx.get(f"{self.host}/v1/models", timeout=10.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach LM Studio at {self.host}: {exc}")
        models = [
            ModelInfo(name=m.get("id", ""), provider=self.name)
            for m in r.json().get("data", [])
            if m.get("id")
        ]
        models.sort(key=lambda x: x.name)
        return models

    def _native_models(self) -> List[ModelInfo]:
        try:
            r = httpx.get(f"{self.host}/api/v0/models", timeout=10.0)
            r.raise_for_status()
        except httpx.HTTPError:
            return []
        out: List[ModelInfo] = []
        for m in r.json().get("data", []):
            if m.get("type") not in (None, "llm", "vlm"):
                continue  # skip embeddings
            if not m.get("id"):
                continue
            out.append(ModelInfo(
                name=m["id"],
                provider=self.name,
                size_bytes=int(m.get("size", 0) or 0),
                parameter_size=m.get("params_string", "") or m.get("params", ""),
                quantization=m.get("quantization", ""),
                family=m.get("arch", ""),
            ))
        out.sort(key=lambda x: x.name)
        return out

    # ------------------------------------------------------------------
    def generate(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        on_token: TokenCallback = None,
        timeout: float = 300.0,
    ) -> GenerationResult:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream_options": {"include_usage": True},
        }
        if seed is not None:
            payload["seed"] = seed

        speed = SpeedMetrics()
        chunks: List[str] = []
        delta_count = 0
        start = time.perf_counter()
        first_token_at: Optional[float] = None
        usage = None

        try:
            with httpx.stream(
                "POST", f"{self.host}/v1/chat/completions", json=payload, timeout=timeout
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for choice in obj.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            chunks.append(piece)
                            delta_count += 1
                            if on_token is not None:
                                on_token(piece)
        except httpx.HTTPError as exc:
            raise ProviderError(f"LM Studio generate failed for {model!r}: {exc}")

        end = time.perf_counter()
        speed.total_s = end - start
        if usage:
            speed.prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            speed.output_tokens = int(usage.get("completion_tokens", 0) or 0)
        else:
            speed.output_tokens = delta_count  # best-effort: ~1 token per delta
        if first_token_at is not None:
            speed.ttft_s = first_token_at - start
            speed.eval_s = max(0.0, end - first_token_at)
            if speed.eval_s > 0 and speed.output_tokens > 0:
                speed.tokens_per_sec = speed.output_tokens / speed.eval_s
        return GenerationResult(text="".join(chunks), speed=speed)

    # ------------------------------------------------------------------
    def memory(self, model: str) -> MemoryMetrics:
        try:
            r = httpx.get(f"{self.host}/api/v0/models", timeout=5.0)
            r.raise_for_status()
        except httpx.HTTPError:
            return MemoryMetrics()
        for m in r.json().get("data", []):
            if m.get("id") == model and m.get("state") == "loaded":
                size = int(m.get("size", 0) or 0)
                return MemoryMetrics(size_bytes=size, vram_bytes=size)
        return MemoryMetrics()
