import pytest

from homebench.providers import (
    LlamaCppProvider,
    OpenAICompatibleProvider,
    VLLMProvider,
    available_providers,
    get_provider,
)
from homebench.providers.openai_compat import normalize_host


def test_all_providers_registered():
    for name in ("ollama", "lmstudio", "llamacpp", "vllm", "mlx", "openai"):
        assert name in available_providers()


def test_mlx_provider():
    from homebench.providers import MLXProvider, get_provider
    from homebench.providers import _AUTO_DETECT

    p = get_provider("mlx")
    assert isinstance(p, MLXProvider)
    assert p.host == "http://localhost:8080"
    assert p.process_hint == "mlx_lm.server"
    # explicit-only: not auto-detected (shares llama.cpp's port)
    assert "mlx" not in _AUTO_DETECT
    # unreachable is graceful
    assert MLXProvider(host="http://127.0.0.1:9").is_available() is False


def test_unknown_provider_raises():
    from homebench.providers import ProviderError

    with pytest.raises(ProviderError):
        get_provider("nope")


def test_normalize_host():
    assert normalize_host("localhost:8080", "http://d:1") == "http://localhost:8080"
    assert normalize_host("", "http://localhost:8080") == "http://localhost:8080"
    assert normalize_host("https://x:1/", "http://d:1") == "https://x:1"


@pytest.mark.parametrize(
    "cls,host,hint",
    [
        (LlamaCppProvider, "http://localhost:8080", "llama-server"),
        (VLLMProvider, "http://localhost:8000", "vllm"),
    ],
)
def test_provider_defaults(cls, host, hint):
    p = cls()
    assert p.host == host
    assert p.process_hint == hint


def test_host_env_override(monkeypatch):
    monkeypatch.setenv("LLAMACPP_HOST", "http://box:9999")
    assert LlamaCppProvider().host == "http://box:9999"


def test_api_key_header(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    p = VLLMProvider()
    assert p._headers() == {"Authorization": "Bearer secret"}
    assert OpenAICompatibleProvider()._headers() == {}


def test_unavailable_is_graceful():
    assert VLLMProvider(host="http://127.0.0.1:9").is_available() is False


class _FakeStream:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


def test_streaming_generate_parses_content_and_usage(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
        "data: [DONE]",
    ]
    import homebench.providers.openai_compat as mod
    monkeypatch.setattr(mod.httpx, "stream", lambda *a, **k: _FakeStream(lines))

    seen = []
    result = OpenAICompatibleProvider().generate(
        "m", "hi", on_token=seen.append
    )
    assert result.text == "Hello world"
    assert result.speed.prompt_tokens == 5
    assert result.speed.output_tokens == 2
    assert seen == ["Hello", " world"]


def test_streaming_generate_without_usage_counts_deltas(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"a"}}]}',
        'data: {"choices":[{"delta":{"content":"b"}}]}',
        'data: {"choices":[{"delta":{"content":"c"}}]}',
        "data: [DONE]",
    ]
    import homebench.providers.openai_compat as mod
    monkeypatch.setattr(mod.httpx, "stream", lambda *a, **k: _FakeStream(lines))

    result = OpenAICompatibleProvider().generate("m", "hi")
    assert result.text == "abc"
    assert result.speed.output_tokens == 3  # fell back to delta count


# =====================================================================
# reasoning tokens count as generated output
# =====================================================================
def _stream(monkeypatch, lines, clock=None):
    """Point generate() at a canned SSE stream; optionally script the clock."""
    import homebench.providers.openai_compat as mod
    monkeypatch.setattr(mod.httpx, "stream", lambda *a, **k: _FakeStream(lines))
    if clock is not None:
        ticks = iter(clock)
        monkeypatch.setattr(mod.time, "perf_counter", lambda: next(ticks))


# a reasoning model: every token arrives in reasoning_content, content is null
REASONING_ONLY = [
    'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}',
    'data: {"choices":[{"delta":{"reasoning_content":"The"}}]}',
    'data: {"choices":[{"delta":{"reasoning_content":" user"}}]}',
    'data: {"choices":[{"delta":{"reasoning_content":" asks"}}]}',
    'data: {"choices":[],"usage":{"prompt_tokens":22,"completion_tokens":3}}',
    "data: [DONE]",
]

MIXED = [
    'data: {"choices":[{"delta":{"reasoning_content":"Think"}}]}',
    'data: {"choices":[{"delta":{"reasoning_content":" more"}}]}',
    'data: {"choices":[{"delta":{"reasoning_content":" still"}}]}',
    'data: {"choices":[{"delta":{"content":"42"}}]}',
    'data: {"choices":[{"delta":{"content":"."}}]}',
    "data: [DONE]",
]


def test_reasoning_only_stream_reports_a_real_rate(monkeypatch):
    _stream(monkeypatch, REASONING_ONLY)

    result = OpenAICompatibleProvider().generate("m", "hi")

    assert result.speed.tokens_per_sec > 0     # was 0.00 before
    assert result.speed.ttft_s > 0
    assert result.speed.output_tokens == 3
    assert result.text == ""                   # reasoning is not the answer


def test_ttft_is_taken_at_the_first_reasoning_token(monkeypatch):
    # clock: start=10.0, first token=10.5, end=12.5
    _stream(monkeypatch, REASONING_ONLY, clock=[10.0, 10.5, 12.5])

    speed = OpenAICompatibleProvider().generate("m", "hi").speed

    assert speed.ttft_s == 0.5
    assert speed.eval_s == 2.0
    assert speed.tokens_per_sec == 1.5         # 3 tokens / 2.0 s


def test_generation_text_carries_content_only(monkeypatch):
    _stream(monkeypatch, MIXED)

    result = OpenAICompatibleProvider().generate("m", "hi")

    assert result.text == "42."                # what the grader receives


def test_mixed_stream_without_usage_counts_both_kinds(monkeypatch):
    # clock: start=0.0, first token (reasoning)=1.0, end=6.0
    _stream(monkeypatch, MIXED, clock=[0.0, 1.0, 6.0])

    speed = OpenAICompatibleProvider().generate("m", "hi").speed

    assert speed.output_tokens == 5            # 3 reasoning + 2 content
    assert speed.eval_s == 5.0                 # window opens at the reasoning
    assert speed.tokens_per_sec == 1.0         # 5 tokens / 5.0 s


def test_vllm_reasoning_field_counts_as_output(monkeypatch):
    # newer vLLM streams reasoning under "reasoning", not "reasoning_content"
    _stream(monkeypatch, [
        'data: {"choices":[{"delta":{"reasoning":"Hmm"}}]}',
        'data: {"choices":[{"delta":{"reasoning":" ok"}}]}',
        'data: {"choices":[{"delta":{"content":"4"}}]}',
        "data: [DONE]",
    ])

    result = OpenAICompatibleProvider().generate("m", "hi")

    assert result.text == "4"
    assert result.speed.output_tokens == 3
    assert result.speed.tokens_per_sec > 0


# =====================================================================
# an error streamed inside a 200 response is a failure, not 0 tokens
# =====================================================================
def test_error_streamed_in_200_raises(monkeypatch):
    from homebench.providers import ProviderError

    _stream(monkeypatch, [
        'data: {"error":{"code":400,"message":"the request exceeds the '
        'available context size","type":"exceed_context_size_error"}}',
        "data: [DONE]",
    ])

    with pytest.raises(ProviderError, match="exceeds the available context"):
        OpenAICompatibleProvider().generate("m", "hi")


def test_string_error_streamed_in_200_raises(monkeypatch):
    from homebench.providers import ProviderError

    _stream(monkeypatch, ['data: {"error":"model crashed"}', "data: [DONE]"])

    with pytest.raises(ProviderError, match="model crashed"):
        OpenAICompatibleProvider().generate("m", "hi")
