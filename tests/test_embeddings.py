import os

import pytest

from app.config import settings
from app.embeddings import EmbeddingService, HuggingFaceEmbeddingService


def require_embedding_model() -> HuggingFaceEmbeddingService:
    """Load the real model, skipping when the caller did not opt in.

    Loading BAAI/bge-m3 on Windows can abort the interpreter inside torch with
    "Windows fatal exception: access violation" during ``encode``. That is a
    hard crash, not an exception, so it cannot be caught: the test has to be
    skipped before the model is touched. Set RUN_REAL_MODEL_TESTS=1 to opt in.
    """
    if os.getenv("RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip(
            "set RUN_REAL_MODEL_TESTS=1 to run against the real "
            f"'{settings.embedding_model}' model"
        )

    return HuggingFaceEmbeddingService(model_name=settings.embedding_model)


def test_embedding_service_is_abstract():
    try:
        EmbeddingService()
    except TypeError:
        return

    raise AssertionError("EmbeddingService should not be instantiated directly")


def test_embedding_service_requires_embed():
    class TestEmbeddingService(EmbeddingService):
        def embed_many(self, texts):
            return []

    try:
        TestEmbeddingService()
    except TypeError:
        return

    raise AssertionError("EmbeddingService must require embed() implementation")


def test_embedding_service_requires_embed_many():
    class TestEmbeddingService(EmbeddingService):
        def embed(self, text):
            return []

    try:
        TestEmbeddingService()
    except TypeError:
        return

    raise AssertionError("EmbeddingService must require embed_many() implementation")


class FakeEmbeddingService(EmbeddingService):
    def embed(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]


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


@pytest.mark.integration
def test_huggingface_embedding_returns_1024_dimensions():
    service = require_embedding_model()

    embedding = service.embed("თბილისი საქართველოს დედაქალაქია")

    assert len(embedding) == settings.embedding_size
    assert all(isinstance(value, float) for value in embedding)


@pytest.mark.integration
def test_huggingface_embedding_many_returns_one_vector_per_text():
    service = require_embedding_model()

    texts = [
        "თბილისი საქართველოს დედაქალაქია",
        "ბათუმი აჭარის მთავარი ქალაქია",
        "თბილისში მდებარეობს პარლამენტი",
    ]

    embeddings = service.embed_many(texts)

    assert len(embeddings) == len(texts)
    assert all(len(embedding) == settings.embedding_size for embedding in embeddings)
    assert all(
        isinstance(value, float) for embedding in embeddings for value in embedding
    )
