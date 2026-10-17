import pytest

from hackeval.config import get_settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CSV_PATH", str(tmp_path / "data" / "evaluation.csv"))
    monkeypatch.setenv("RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("MAX_RETRIES", "1")
    return get_settings()
