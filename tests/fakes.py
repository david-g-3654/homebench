"""A deterministic in-memory provider for tests (no Ollama required)."""

from __future__ import annotations

from typing import Dict, List, Optional

from localbench.models import MemoryMetrics, ModelInfo, SpeedMetrics
from localbench.providers.base import GenerationResult, Provider, TokenCallback


# Canned answers keyed by a distinctive substring of each task prompt.
_ANSWERS = {
    "17 multiplied by 23": "17 * 23 = 391\nThe final answer is 391",
    "48 apples": "48 * 3/4 = 36 sold, 12 left, +15 = 27\n27",
    "60 miles in 1.5 hours": "60 / 1.5 = 40\n40",
    "2, 6, 12, 20, 30": "B",
    "All roses are flowers": "C",
    "capital city of Australia": "Canberra",
    "symbol 'Fe'": "Iron",
    '"name" and "age"': '{"name": "Alice", "age": 30}',
    "support@example.com": "support@example.com",
    "len('hello') + 2": "len('hello') is 5, plus 2 is 7\n7",
    "sky appears blue": "Rayleigh scattering makes shorter blue wavelengths "
    "scatter more, so the sky looks blue.",
    "declining a meeting": "Thank you for the invitation. Unfortunately I have "
    "a scheduling conflict and must decline.",
    "strict grader": "SCORE: 5 looks correct and well-formed",  # judge prompt
}


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
        for needle, ans in _ANSWERS.items():
            if needle in prompt:
                return ans
        return "I don't know."

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
