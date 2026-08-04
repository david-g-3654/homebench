"""Ollama provider — talks to a local ``ollama serve`` over its REST API."""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

import httpx

from ..models import MemoryMetrics, ModelInfo, SpeedMetrics
from .base import GenerationResult, Provider, ProviderError, TokenCallback

_NS = 1_000_000_000  # nanoseconds per second (Ollama reports durations in ns)


def _normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host:
        host = "http://localhost:11434"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, host: Optional[str] = None):
        env = os.environ.get("OLLAMA_HOST")
        self.host = _normalize_host(host or env or "http://localhost:11434")

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> List[ModelInfo]:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=10.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach Ollama at {self.host}: {exc}")
        models: List[ModelInfo] = []
        for m in r.json().get("models", []):
            details = m.get("details") or {}
            models.append(
                ModelInfo(
                    name=m.get("name", ""),
                    provider=self.name,
                    size_bytes=int(m.get("size", 0) or 0),
                    parameter_size=details.get("parameter_size", ""),
                    quantization=details.get("quantization_level", ""),
                    family=details.get("family", ""),
                    digest=(m.get("digest", "") or "")[:16],
                )
            )
        models.sort(key=lambda x: x.name)
        return models

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
        options = {"num_predict": max_tokens, "temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }

        speed = SpeedMetrics()
        chunks: List[str] = []
        start = time.perf_counter()
        first_token_at: Optional[float] = None

        try:
            with httpx.stream(
                "POST", f"{self.host}/api/generate", json=payload, timeout=timeout
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise ProviderError(str(data["error"]))
                    piece = data.get("response", "")
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks.append(piece)
                        if on_token is not None:
                            on_token(piece)
                    if data.get("done"):
                        self._fill_timings(speed, data)
                        break
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama generate failed for {model!r}: {exc}")

        end = time.perf_counter()
        speed.total_s = end - start
        if first_token_at is not None:
            # Subtract server-reported load time so TTFT reflects prompt
            # processing + first token, not a cold model load.
            speed.ttft_s = max(0.0, (first_token_at - start) - speed.load_s)
        return GenerationResult(text="".join(chunks), speed=speed)

    @staticmethod
    def _fill_timings(speed: SpeedMetrics, data: dict) -> None:
        speed.prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        speed.output_tokens = int(data.get("eval_count", 0) or 0)
        speed.load_s = (data.get("load_duration", 0) or 0) / _NS
        speed.prompt_eval_s = (data.get("prompt_eval_duration", 0) or 0) / _NS
        speed.eval_s = (data.get("eval_duration", 0) or 0) / _NS
        if speed.eval_s > 0 and speed.output_tokens > 0:
            speed.tokens_per_sec = speed.output_tokens / speed.eval_s

    # ------------------------------------------------------------------
    def memory(self, model: str) -> MemoryMetrics:
        try:
            r = httpx.get(f"{self.host}/api/ps", timeout=5.0)
            r.raise_for_status()
        except httpx.HTTPError:
            return MemoryMetrics()
        for m in r.json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return MemoryMetrics(
                    size_bytes=int(m.get("size", 0) or 0),
                    vram_bytes=int(m.get("size_vram", 0) or 0),
                )
        return MemoryMetrics()

    def unload(self, model: str) -> None:
        try:
            httpx.post(
                f"{self.host}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=15.0,
            )
        except httpx.HTTPError:
            pass
