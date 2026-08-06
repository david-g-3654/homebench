"""MLX provider — Apple-Silicon-native models via ``mlx_lm.server``.

Start it with ``mlx_lm.server --model <hf-repo>`` (from the ``mlx-lm`` package);
it exposes an OpenAI-compatible API on ``127.0.0.1:8080`` by default, so this is
a thin subclass of :class:`OpenAICompatibleProvider`.

It shares llama.cpp's default port (8080), and two OpenAI-compatible servers on
the same port can't be told apart, so MLX is **explicit-only** — select it with
``--provider mlx`` (override the port with ``--host`` / ``MLX_HOST``). Timings
are client-side; memory falls back to RSS sampling of the server process.
"""

from __future__ import annotations

from .openai_compat import OpenAICompatibleProvider


class MLXProvider(OpenAICompatibleProvider):
    name = "mlx"
    default_host = "http://localhost:8080"
    host_env = "MLX_HOST"
    api_key_env = None
    _process_hint = "mlx_lm.server"
