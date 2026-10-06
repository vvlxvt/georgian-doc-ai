from app.config import Settings, settings


def test_settings_have_expected_defaults():
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "documents"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_size == 1024
    assert settings.ollama_model == "llama3.2:3b"
    assert settings.ollama_url == "http://localhost:11434"


def test_settings_are_immutable():
    config = Settings()

    try:
        config.qdrant_url = "http://example.com"
    except AttributeError:
        pass
    else:
        raise AssertionError("Settings should be immutable")