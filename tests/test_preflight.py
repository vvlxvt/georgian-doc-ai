import pytest

from app.config import Settings
from app.preflight import (
    PreflightError,
    validate_ollama,
    validate_qdrant,
    validate_tesseract,
)


class FakeResponse:
    def __init__(self, models):
        self.models = models

    def raise_for_status(self):
        pass

    def json(self):
        return {"models": self.models}


def test_validate_ollama_accepts_configured_model(monkeypatch):
    monkeypatch.setattr(
        "app.preflight.httpx.get",
        lambda *args, **kwargs: FakeResponse([{"name": "test-model"}]),
    )

    validate_ollama(Settings(ollama_model="test-model", _env_file=None))


def test_validate_ollama_rejects_missing_model(monkeypatch):
    monkeypatch.setattr(
        "app.preflight.httpx.get",
        lambda *args, **kwargs: FakeResponse([]),
    )

    with pytest.raises(PreflightError, match="not installed"):
        validate_ollama(Settings(ollama_model="test-model", _env_file=None))


def test_validate_qdrant_wraps_connection_error(monkeypatch):
    class BrokenClient:
        def __init__(self, url):
            pass

        def get_collections(self):
            raise RuntimeError("connection refused")

    monkeypatch.setattr("app.preflight.QdrantClient", BrokenClient)

    with pytest.raises(PreflightError, match="Qdrant is unavailable"):
        validate_qdrant(Settings(_env_file=None))


def test_validate_tesseract_rejects_missing_georgian_language(monkeypatch):
    monkeypatch.setattr(
        "app.preflight.pytesseract.get_languages",
        lambda config: ["eng"],
    )

    with pytest.raises(PreflightError, match="kat"):
        validate_tesseract(Settings(tesseract_languages="kat+eng", _env_file=None))
