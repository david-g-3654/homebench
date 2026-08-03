from localbench.models import SpeedMetrics
from localbench.providers.ollama import OllamaProvider, _normalize_host


def test_normalize_host():
    assert _normalize_host("localhost:11434") == "http://localhost:11434"
    assert _normalize_host("http://x:1/") == "http://x:1"
    assert _normalize_host("") == "http://localhost:11434"
    assert _normalize_host("https://remote:443") == "https://remote:443"


def test_fill_timings_computes_tps():
    speed = SpeedMetrics()
    OllamaProvider._fill_timings(speed, {
        "prompt_eval_count": 31,
        "eval_count": 100,
        "load_duration": 2_000_000_000,       # 2s
        "prompt_eval_duration": 500_000_000,  # 0.5s
        "eval_duration": 2_000_000_000,       # 2s
    })
    assert speed.output_tokens == 100
    assert speed.eval_s == 2.0
    assert speed.load_s == 2.0
    assert abs(speed.tokens_per_sec - 50.0) < 1e-9


def test_fill_timings_zero_eval_is_safe():
    speed = SpeedMetrics()
    OllamaProvider._fill_timings(speed, {"eval_count": 0, "eval_duration": 0})
    assert speed.tokens_per_sec == 0.0
