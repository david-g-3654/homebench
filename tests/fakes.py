"""A deterministic in-memory provider for tests (no backend required)."""

from __future__ import annotations

from typing import Dict, List, Optional

from localbench.models import MemoryMetrics, ModelInfo, SpeedMetrics
from localbench.providers.base import GenerationResult, Provider, TokenCallback
from localbench.quality import default_suite

# An "oracle": answer every task with its own reference. This keeps the
# canned-answer test meaningful as the suite grows, and only works because
# each task's reference is authored to satisfy its grader (see test_suite).
_ORACLE: Dict[str, str] = {t.prompt: t.reference for t in default_suite(include_open=True)}


class FakeProvider(Provider):
    name = "ollama"  # pose as ollama so RSS sampler name matches harmlessly

    def __init__(self, models: Optional[List[ModelInfo]] = None,
                 tps: Optional[Dict[str, float]] = None, host: str = ""):
        self._models = models or [
            ModelInfo("fast:1b", "ollama", size_bytes=1_000_000_000,
                      parameter_size="1B", quantization="Q4", family="test"),
            ModelInfo("smart:8b", "ollama", size_bytes=5_000_000_000,
                      parameter_size="8B", quantization="Q4", family="test"),
        ]
        self._tps = tps or {"fast:1b": 120.0, "smart:8b": 40.0}
        self.unloaded: List[str] = []

    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[ModelInfo]:
        return list(self._models)

    def _answer_for(self, prompt: str) -> str:
        if prompt in _ORACLE:
            return _ORACLE[prompt]
        if "strict grader" in prompt:      # the LLM-judge prompt
            return "SCORE: 5 correct and well-formed"
        return "A short generated paragraph for the speed probe."

    def generate(self, model, prompt, *, max_tokens=256, temperature=0.0,
                 seed=None, on_token: TokenCallback = None, timeout=300.0):
        text = self._answer_for(prompt)
        if on_token:
            on_token(text)
        tps = self._tps.get(model, 50.0)
        out_tokens = max(1, len(text.split()))
        speed = SpeedMetrics(
            ttft_s=0.05, tokens_per_sec=tps, prompt_tokens=len(prompt.split()),
            output_tokens=out_tokens, eval_s=out_tokens / tps, total_s=0.1,
        )
        return GenerationResult(text=text, speed=speed)

    def memory(self, model: str) -> MemoryMetrics:
        info = next((m for m in self._models if m.name == model), None)
        size = info.size_bytes if info else 0
        return MemoryMetrics(size_bytes=size, vram_bytes=size)

    def unload(self, model: str) -> None:
        self.unloaded.append(model)
