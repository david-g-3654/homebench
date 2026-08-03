"""LM Studio provider.

Model discovery and generation go through the OpenAI-compatible endpoints
(inherited from :class:`OpenAICompatibleProvider`); when LM Studio's richer
native REST API (``/api/v0``) is present we use it to enrich model metadata
and read the loaded-model footprint.
"""

from __future__ import annotations

from typing import List

import httpx

from ..models import MemoryMetrics, ModelInfo
from .openai_compat import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    name = "lmstudio"
    default_host = "http://localhost:1234"
    host_env = "LMSTUDIO_HOST"
    api_key_env = None
    # Matches "LM Studio Helper" and the bundled runner processes.
    _process_hint = "LM Studio"

    def list_models(self) -> List[ModelInfo]:
        # Prefer the native API for quantization/arch details; fall back to /v1.
        native = self._native_models()
        return native if native else super().list_models()

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
