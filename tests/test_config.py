from pathlib import Path

import pytest

from app.config import Settings


def test_settings_have_expected_defaults(monkeypatch):
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)

    config = Settings(_env_file=None)

    assert config.qdrant_url == "http://localhost:6333"
    assert config.qdrant_api_key == ""
    assert config.qdrant_collection == "documents"
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.embedding_size == 1024
    assert config.ollama_model == "llama3.2:3b"
    assert config.ollama_url == "http://localhost:11434"
    assert config.database_url == "sqlite:///data/app.db"
    assert config.tesseract_languages == "kat+eng"
    assert config.chunk_max_length == 1000
    assert config.chunk_overlap == 100
    assert config.min_text_length == 20


def test_settings_read_environment_overrides(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.example:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-api-key")
    monkeypatch.setenv("EMBEDDING_SIZE", "768")

    config = Settings(_env_file=None)

    assert config.qdrant_url == "http://qdrant.example:6333"
    assert config.qdrant_api_key == "test-api-key"
    assert config.embedding_size == 768


def test_settings_read_ingestion_overrides(monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_LENGTH", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    monkeypatch.setenv("MIN_TEXT_LENGTH", "40")

    config = Settings(_env_file=None)

    assert config.chunk_max_length == 500
    assert config.chunk_overlap == 50
    assert config.min_text_length == 40


def test_env_example_documents_every_setting():
    env_example = Path(__file__).resolve().parents[1] / ".env.example"

    documented = {
        line.split("=", 1)[0].strip().lower()
        for line in env_example.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }

    assert documented == set(Settings.model_fields)


def test_settings_are_immutable():
    config = Settings()

    with pytest.raises(Exception):
        config.qdrant_url = "http://example.com"
