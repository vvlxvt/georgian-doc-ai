import pytest

from app.config import Settings
from app.preflight import (
    REQUIRED_TABLES,
    PreflightError,
    validate_llm_provider,
    validate_ollama,
    validate_qdrant,
    validate_schema,
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


def test_validate_llm_provider_checks_ollama_only_when_selected(monkeypatch):
    validated = []
    monkeypatch.setattr(
        "app.preflight.validate_ollama",
        lambda config: validated.append(config.ollama_model),
    )

    validate_llm_provider(Settings(llm_provider="ollama", _env_file=None))
    assert validated == ["llama3.2:3b"]


def test_validate_llm_provider_requires_openrouter_key_without_network(monkeypatch):
    monkeypatch.setattr(
        "app.preflight.validate_ollama",
        lambda config: pytest.fail("Ollama must not be checked"),
    )

    with pytest.raises(PreflightError, match="OPENROUTER_API_KEY is required"):
        validate_llm_provider(Settings(llm_provider="openrouter", _env_file=None))


def test_validate_llm_provider_accepts_openrouter_key_without_live_request():
    validate_llm_provider(
        Settings(
            llm_provider="openrouter",
            openrouter_api_key="test-key",
            _env_file=None,
        )
    )


def test_validate_qdrant_wraps_connection_error(monkeypatch):
    class BrokenClient:
        def __init__(self, url, api_key=None):
            pass

        def get_collections(self):
            raise RuntimeError("connection refused")

    monkeypatch.setattr("app.preflight.QdrantClient", BrokenClient)

    with pytest.raises(PreflightError, match="Qdrant is unavailable"):
        validate_qdrant(Settings(_env_file=None))


def test_validate_qdrant_uses_configured_api_key(monkeypatch):
    client_args = {}

    class FakeClient:
        def __init__(self, **kwargs):
            client_args.update(kwargs)

        def get_collections(self):
            return None

    monkeypatch.setattr("app.preflight.QdrantClient", FakeClient)

    validate_qdrant(
        Settings(
            qdrant_url="https://qdrant.example",
            qdrant_api_key="test-api-key",
            _env_file=None,
        )
    )

    assert client_args == {
        "url": "https://qdrant.example",
        "api_key": "test-api-key",
    }


def test_validate_tesseract_rejects_missing_georgian_language(monkeypatch):
    monkeypatch.setattr(
        "app.preflight.pytesseract.get_languages",
        lambda config: ["eng"],
    )

    with pytest.raises(PreflightError, match="kat"):
        validate_tesseract(Settings(tesseract_languages="kat+eng", _env_file=None))


class FakeInspector:
    def __init__(self, table_names):
        self.table_names = list(table_names)

    def get_table_names(self):
        return self.table_names


def test_validate_schema_accepts_migrated_database(monkeypatch):
    monkeypatch.setattr(
        "app.preflight.inspect",
        lambda engine: FakeInspector(REQUIRED_TABLES),
    )

    validate_schema(Settings(_env_file=None))


def test_validate_schema_rejects_database_without_migrations(monkeypatch):
    monkeypatch.setattr(
        "app.preflight.inspect",
        lambda engine: FakeInspector(["documents"]),
    )

    with pytest.raises(PreflightError, match="alembic upgrade head"):
        validate_schema(Settings(_env_file=None))


def test_validate_schema_wraps_unreachable_database(monkeypatch):
    def broken_inspect(engine):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("app.preflight.inspect", broken_inspect)

    with pytest.raises(PreflightError, match="is unavailable"):
        validate_schema(Settings(_env_file=None))
