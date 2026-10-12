from pathlib import Path

import pytest
from pydantic import ValidationError

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
    # default LLM provider and OpenRouter defaults
    assert config.llm_provider == "openrouter"
    assert config.openrouter_url == "https://api.openrouter.ai"
    assert config.openrouter_api_key == ""
    assert config.openrouter_model == "gpt-4o-mini"
    assert config.openrouter_connect_timeout == 10.0
    assert config.openrouter_read_timeout == 120.0
    assert not config.openrouter_supports_structured_output
    assert not config.openrouter_supports_json_mode
    assert not config.openrouter_supports_streaming
    assert config.llm_max_attempts == 2
    assert config.llm_total_timeout == 180.0
    assert config.database_url == "sqlite:///data/app.db"
    assert config.tesseract_languages == "kat+eng"
    assert config.chunk_max_length == 1000
    assert config.chunk_overlap == 100
    assert config.min_text_length == 20


def test_settings_read_environment_overrides(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.example:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-api-key")
    monkeypatch.setenv("EMBEDDING_SIZE", "768")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OPENROUTER_CONNECT_TIMEOUT", "3.5")
    monkeypatch.setenv("OPENROUTER_SUPPORTS_JSON_MODE", "true")
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("LLM_TOTAL_TIMEOUT", "90")

    config = Settings(_env_file=None)

    assert config.qdrant_url == "http://qdrant.example:6333"
    assert config.qdrant_api_key == "test-api-key"
    assert config.embedding_size == 768
    assert config.llm_provider == "ollama"
    assert config.openrouter_connect_timeout == 3.5
    assert config.openrouter_supports_json_mode
    assert config.llm_max_attempts == 3
    assert config.llm_total_timeout == 90.0


def test_settings_reject_unknown_llm_provider():
    with pytest.raises(ValidationError):
        Settings(llm_provider="unknown", _env_file=None)


def test_settings_reject_nonpositive_openrouter_timeout():
    with pytest.raises(ValidationError):
        Settings(openrouter_connect_timeout=0, _env_file=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [("llm_max_attempts", 0), ("llm_max_attempts", 4), ("llm_total_timeout", 0)],
)
def test_settings_reject_invalid_llm_retry_limits(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value}, _env_file=None)


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
