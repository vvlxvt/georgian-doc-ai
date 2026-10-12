import app.dependencies as dependencies


def test_rag_pipeline_reuses_process_scoped_services(monkeypatch):
    calls = {
        "embedding": 0,
        "qdrant": 0,
        "store": 0,
        "llm": 0,
    }

    class FakeEmbeddingService:
        def __init__(self, model_name):
            calls["embedding"] += 1

    class FakeQdrantClient:
        def __init__(self, url, api_key=None):
            calls["qdrant"] += 1

    class FakeVectorStore:
        def __init__(self, client, collection_name, vector_size):
            calls["store"] += 1

    class FakeLLMService:
        def __init__(self, *args, **kwargs):
            calls["llm"] += 1

    monkeypatch.setattr(
        dependencies,
        "HuggingFaceEmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(dependencies, "QdrantClient", FakeQdrantClient)
    monkeypatch.setattr(dependencies, "QdrantVectorStore", FakeVectorStore)
    # tests run with default provider 'openrouter' so patch both LLM implementations
    monkeypatch.setattr(dependencies, "OllamaLLMService", FakeLLMService)
    monkeypatch.setattr(dependencies, "OpenRouterLLMService", FakeLLMService)
    dependencies.reset_application_services()

    first_pipeline = dependencies.create_rag_pipeline()
    second_pipeline = dependencies.create_rag_pipeline()

    assert first_pipeline is second_pipeline
    assert calls == {
        "embedding": 1,
        "qdrant": 1,
        "store": 1,
        "llm": 1,
    }

    dependencies.reset_application_services()


def test_reset_application_services_creates_fresh_pipeline(monkeypatch):
    created_embeddings = []

    class FakeEmbeddingService:
        def __init__(self, model_name):
            created_embeddings.append(self)

    class FakeQdrantClient:
        def __init__(self, url, api_key=None):
            pass

    class FakeVectorStore:
        def __init__(self, client, collection_name, vector_size):
            pass

    class FakeLLMService:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        dependencies,
        "HuggingFaceEmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(dependencies, "QdrantClient", FakeQdrantClient)
    monkeypatch.setattr(dependencies, "QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr(dependencies, "OllamaLLMService", FakeLLMService)
    monkeypatch.setattr(dependencies, "OpenRouterLLMService", FakeLLMService)
    dependencies.reset_application_services()

    first_pipeline = dependencies.create_rag_pipeline()
    dependencies.reset_application_services()
    second_pipeline = dependencies.create_rag_pipeline()

    assert first_pipeline is not second_pipeline
    assert len(created_embeddings) == 2

    dependencies.reset_application_services()


def test_qdrant_client_uses_configured_remote_credentials(monkeypatch):
    from app.config import Settings

    client_args = {}

    class FakeQdrantClient:
        def __init__(self, **kwargs):
            client_args.update(kwargs)

    monkeypatch.setattr(dependencies, "QdrantClient", FakeQdrantClient)
    monkeypatch.setattr(
        dependencies,
        "settings",
        Settings(
            qdrant_url="https://qdrant.example",
            qdrant_api_key="test-api-key",
            _env_file=None,
        ),
    )
    dependencies.reset_application_services()

    dependencies.get_qdrant_client()

    assert client_args == {
        "url": "https://qdrant.example",
        "api_key": "test-api-key",
    }

    dependencies.reset_application_services()
