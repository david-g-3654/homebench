"""Model providers — each wraps one local inference backend."""

from __future__ import annotations

from typing import Dict, List, Type

from .base import GenerationResult, Provider, ProviderError
from .lmstudio import LMStudioProvider
from .ollama import OllamaProvider

# Registry of known providers, keyed by their short name.
# Order matters for auto-detection (first reachable one wins).
_PROVIDERS: Dict[str, Type[Provider]] = {
    OllamaProvider.name: OllamaProvider,
    LMStudioProvider.name: LMStudioProvider,
}


def available_providers() -> List[str]:
    return list(_PROVIDERS)


def get_provider(name: str, **kwargs) -> Provider:
    try:
        cls = _PROVIDERS[name]
    except KeyError:
        raise ProviderError(
            f"Unknown provider {name!r}. Known providers: {', '.join(_PROVIDERS)}"
        )
    return cls(**kwargs)


def detect_provider(**kwargs) -> Provider:
    """Return the first provider that is reachable, or raise."""
    tried = []
    for name, cls in _PROVIDERS.items():
        provider = cls(**kwargs)
        if provider.is_available():
            return provider
        tried.append(name)
    raise ProviderError(
        "No local model provider is reachable. Tried: "
        + ", ".join(tried)
        + ". Is Ollama running? (try `ollama serve`)"
    )


__all__ = [
    "Provider",
    "ProviderError",
    "GenerationResult",
    "OllamaProvider",
    "LMStudioProvider",
    "available_providers",
    "get_provider",
    "detect_provider",
]
