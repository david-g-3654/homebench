from homebench.providers import available_providers, get_provider
from homebench.providers.lmstudio import LMStudioProvider


def test_lmstudio_registered():
    assert "lmstudio" in available_providers()
    assert isinstance(get_provider("lmstudio"), LMStudioProvider)


def test_lmstudio_defaults():
    p = LMStudioProvider()
    assert p.host == "http://localhost:1234"
    assert p.process_hint == "LM Studio"


def test_lmstudio_unavailable_is_graceful():
    # Nothing is serving on this port in the test env; must not raise.
    p = LMStudioProvider(host="http://127.0.0.1:9")
    assert p.is_available() is False


def test_native_model_parsing(monkeypatch):
    sample = {"data": [
        {"id": "qwen2.5-7b-instruct", "type": "llm", "arch": "qwen2",
         "quantization": "Q4_K_M", "params_string": "7B", "size": 4_500_000_000,
         "state": "loaded"},
        {"id": "nomic-embed", "type": "embeddings"},  # should be skipped
    ]}

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return sample

    import homebench.providers.lmstudio as mod
    monkeypatch.setattr(mod.httpx, "get", lambda *a, **k: _Resp())

    models = LMStudioProvider().list_models()
    assert len(models) == 1
    m = models[0]
    assert m.name == "qwen2.5-7b-instruct"
    assert m.quantization == "Q4_K_M"
    assert m.parameter_size == "7B"
    assert m.family == "qwen2"
    assert m.size_bytes == 4_500_000_000
