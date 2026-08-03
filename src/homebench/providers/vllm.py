"""vLLM provider.

vLLM's OpenAI-compatible server (``vllm serve``) listens on port 8000 by
default. If the server was started with ``--api-key``, set ``VLLM_API_KEY``
(or ``OPENAI_API_KEY``) and it will be sent as a bearer token.
"""

from __future__ import annotations

import os
from typing import Optional

from .openai_compat import OpenAICompatibleProvider


class VLLMProvider(OpenAICompatibleProvider):
    name = "vllm"
    default_host = "http://localhost:8000"
    host_env = "VLLM_HOST"
    api_key_env = "VLLM_API_KEY"
    _process_hint = "vllm"

    def __init__(self, host: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(host=host, api_key=api_key)
        # Fall back to the conventional OpenAI key if a vLLM-specific one isn't set.
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
