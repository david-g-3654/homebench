"""llama.cpp server provider (``llama-server`` from llama.cpp).

Uses the OpenAI-compatible ``/v1`` endpoints. ``llama-server`` typically
serves a single loaded model; discovery returns whatever ``/v1/models``
reports. Memory is left to RSS sampling.
"""

from __future__ import annotations

from .openai_compat import OpenAICompatibleProvider


class LlamaCppProvider(OpenAICompatibleProvider):
    name = "llamacpp"
    default_host = "http://localhost:8080"
    host_env = "LLAMACPP_HOST"
    api_key_env = "LLAMACPP_API_KEY"
    _process_hint = "llama-server"
