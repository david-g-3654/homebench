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
    for name in ("ollama", "lmstudio", "llamacpp", "vllm", "openai"):
        assert name in available_providers()


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
