import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Point HOMEBENCH_HOME at a throwaway dir so tests never touch the real
    ~/.homebench (response cache, run history, HF model cache)."""
    monkeypatch.setenv("HOMEBENCH_HOME", str(tmp_path / "home"))
