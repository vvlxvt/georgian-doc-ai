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
        def __init__(self, url):
            calls["qdrant"] += 1

    class FakeVectorStore:
        def __init__(self, client, collection_name, vector_size):
            calls["store"] += 1

    class FakeLLMService:
        def __init__(self, model, base_url):
            calls["llm"] += 1

    monkeypatch.setattr(
        dependencies,
        "HuggingFaceEmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(dependencies, "QdrantClient", FakeQdrantClient)
    monkeypatch.setattr(dependencies, "QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr(dependencies, "OllamaLLMService", FakeLLMService)
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
        def __init__(self, url):
            pass

    class FakeVectorStore:
        def __init__(self, client, collection_name, vector_size):
            pass

    class FakeLLMService:
        def __init__(self, model, base_url):
            pass

    monkeypatch.setattr(
        dependencies,
        "HuggingFaceEmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(dependencies, "QdrantClient", FakeQdrantClient)
    monkeypatch.setattr(dependencies, "QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr(dependencies, "OllamaLLMService", FakeLLMService)
    dependencies.reset_application_services()

    first_pipeline = dependencies.create_rag_pipeline()
    dependencies.reset_application_services()
    second_pipeline = dependencies.create_rag_pipeline()

    assert first_pipeline is not second_pipeline
    assert len(created_embeddings) == 2

    dependencies.reset_application_services()
