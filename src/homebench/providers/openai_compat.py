"""Generic OpenAI-compatible provider.

Most local runners (LM Studio, llama.cpp's ``llama-server``, vLLM, Jan,
LocalAI, text-generation-webui, …) expose an OpenAI-compatible HTTP API:
``GET /v1/models`` for discovery and ``POST /v1/chat/completions`` for
generation. This base implements both; concrete providers just set the
default host / env var / process hint (and can enrich metadata).

Timings (tok/s, TTFT) are measured client-side from the streamed response,
since the OpenAI-compatible API doesn't report server-side eval durations.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import httpx

from ..models import MemoryMetrics, ModelInfo, SpeedMetrics
from .base import GenerationResult, Provider, ProviderError, TokenCallback


def normalize_host(host: str, default: str) -> str:
    host = (host or "").strip().rstrip("/")
    if not host:
        host = default
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


class OpenAICompatibleProvider(Provider):
    """Talks to any server implementing the OpenAI ``/v1`` chat API."""

    name = "openai"
    default_host = "http://localhost:8000"
    host_env: Optional[str] = "OPENAI_BASE_URL"
    api_key_env: Optional[str] = "OPENAI_API_KEY"
    _process_hint: Optional[str] = None

    def __init__(self, host: Optional[str] = None, api_key: Optional[str] = None):
        env_host = os.environ.get(self.host_env) if self.host_env else None
        self.host = normalize_host(host or env_host or "", self.default_host)
        env_key = os.environ.get(self.api_key_env) if self.api_key_env else None
        self.api_key = api_key or env_key

    @property
    def process_hint(self) -> str:
        return self._process_hint or self.name

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.host}/v1/models", headers=self._headers(), timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> List[ModelInfo]:
        try:
            r = httpx.get(f"{self.host}/v1/models", headers=self._headers(), timeout=10.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach {self.name} at {self.host}: {exc}")
        models = [
            ModelInfo(name=m.get("id", ""), provider=self.name)
            for m in r.json().get("data", [])
            if m.get("id")
        ]
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
                "POST", f"{self.host}/v1/chat/completions",
                json=payload, headers=self._headers(), timeout=timeout,
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
            raise ProviderError(f"{self.name} generate failed for {model!r}: {exc}")

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
        # OpenAI-compatible servers don't expose memory; rely on RSS sampling.
        return MemoryMetrics()
