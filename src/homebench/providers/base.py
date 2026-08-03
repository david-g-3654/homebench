"""Provider abstraction.

A provider knows how to enumerate locally available models, generate text
from one (streaming, so we can time the first token), and report the memory
footprint of a loaded model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..models import MemoryMetrics, ModelInfo, SpeedMetrics


class ProviderError(RuntimeError):
    """Raised when a provider is unreachable or returns an error."""


@dataclass
class GenerationResult:
    """Raw output of a single streamed generation."""

    text: str = ""
    speed: SpeedMetrics = field(default_factory=SpeedMetrics)


# Called with each text chunk as it streams in.
TokenCallback = Optional[Callable[[str], None]]


class Provider(ABC):
    """Base class for local model providers."""

    #: short, stable identifier (also the CLI ``--provider`` value)
    name: str = "base"

    @property
    def process_hint(self) -> str:
        """Substring used to find this backend's processes for RSS sampling."""
        return self.name

    @abstractmethod
    def is_available(self) -> bool:
        """True if the backend is reachable right now."""

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """Enumerate models the user already has installed."""

    @abstractmethod
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
        """Generate a completion, streaming tokens to ``on_token``."""

    def memory(self, model: str) -> MemoryMetrics:  # pragma: no cover - optional
        """Memory footprint of the (loaded) model. Best-effort; may be empty."""
        return MemoryMetrics()

    def warmup(self, model: str, *, timeout: float = 300.0) -> None:
        """Load a model into memory so later timings exclude load time."""
        self.generate(model, "ok", max_tokens=1, timeout=timeout)

    def unload(self, model: str) -> None:  # pragma: no cover - optional
        """Ask the backend to evict the model from memory. No-op by default."""
        return None
