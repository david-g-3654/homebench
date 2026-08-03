"""Model providers — each wraps one local inference backend."""

from __future__ import annotations

from typing import Dict, List, Type

from .base import GenerationResult, Provider, ProviderError
from .llamacpp import LlamaCppProvider
from .lmstudio import LMStudioProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatibleProvider
from .vllm import VLLMProvider

# All providers selectable via --provider / get_provider().
_PROVIDERS: Dict[str, Type[Provider]] = {
    OllamaProvider.name: OllamaProvider,
    LMStudioProvider.name: LMStudioProvider,
    LlamaCppProvider.name: LlamaCppProvider,
    VLLMProvider.name: VLLMProvider,
    OpenAICompatibleProvider.name: OpenAICompatibleProvider,
}

# Order tried during auto-detection (first reachable one wins). The generic
# "openai" provider is intentionally excluded — it has no canonical local
# port and is meant to be selected explicitly with --provider openai.
_AUTO_DETECT: List[str] = [
    OllamaProvider.name,
    LMStudioProvider.name,
    LlamaCppProvider.name,
    VLLMProvider.name,
]


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
    """Return the first auto-detectable provider that is reachable, or raise."""
    tried = []
    for name in _AUTO_DETECT:
        provider = _PROVIDERS[name](**kwargs)
        if provider.is_available():
            return provider
        tried.append(name)
    raise ProviderError(
        "No local model provider is reachable. Tried: "
        + ", ".join(tried)
        + ". Is one running? e.g. `ollama serve`, LM Studio's server, "
        "`llama-server`, or `vllm serve`."
    )


__all__ = [
    "Provider",
    "ProviderError",
    "GenerationResult",
    "OllamaProvider",
    "LMStudioProvider",
    "LlamaCppProvider",
    "VLLMProvider",
    "OpenAICompatibleProvider",
    "available_providers",
    "get_provider",
    "detect_provider",
]
