import pytest

from app.config import Settings, settings


def test_settings_have_expected_defaults():
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "documents"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_size == 1024
    assert settings.ollama_model == "llama3.2:3b"
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.database_url == "sqlite:///data/app.db"
    assert settings.tesseract_languages == "kat+eng"


def test_settings_read_environment_overrides(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.example:6333")
    monkeypatch.setenv("EMBEDDING_SIZE", "768")

    config = Settings(_env_file=None)

    assert config.qdrant_url == "http://qdrant.example:6333"
    assert config.embedding_size == 768


def test_settings_are_immutable():
    config = Settings()

    with pytest.raises(Exception):
        config.qdrant_url = "http://example.com"
