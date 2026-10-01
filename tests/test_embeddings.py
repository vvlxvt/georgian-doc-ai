from app.embeddings import EmbeddingService


def test_embedding_service_is_abstract():
    try:
        EmbeddingService()
    except TypeError:
        return

    raise AssertionError(
        "EmbeddingService should not be instantiated directly"
    )


def test_embedding_service_requires_embed():
    class TestEmbeddingService(EmbeddingService):
        def embed_many(self, texts):
            return []

    try:
        TestEmbeddingService()
    except TypeError:
        return

    raise AssertionError(
        "EmbeddingService must require embed() implementation"
    )


def test_embedding_service_requires_embed_many():
    class TestEmbeddingService(EmbeddingService):
        def embed(self, text):
            return []

    try:
        TestEmbeddingService()
    except TypeError:
        return

    raise AssertionError(
        "EmbeddingService must require embed_many() implementation"
    )

class FakeEmbeddingService(EmbeddingService):
    def embed(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 2.0, 3.0]
            for _ in texts
        ]


def test_embed_returns_vector():
    service = FakeEmbeddingService()

    result = service.embed("თბილისი")

    assert isinstance(result, list)
    assert all(isinstance(value, float) for value in result)
    assert len(result) == 3


def test_embed_many_returns_vector_for_each_text():
    service = FakeEmbeddingService()

    texts = [
        "თბილისი",
        "ბათუმი",
        "ქუთაისი",
    ]

    result = service.embed_many(texts)

    assert len(result) == len(texts)

    for vector in result:
        assert isinstance(vector, list)
        assert all(isinstance(value, float) for value in vector)
        assert len(vector) == 3