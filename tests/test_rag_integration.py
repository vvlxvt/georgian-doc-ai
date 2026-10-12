import os
from uuid import uuid4

import httpx
import pytest
from qdrant_client import QdrantClient

from app.answer import AnswerGenerator
from app.config import settings
from app.context import ContextBuilder
from app.embeddings import EmbeddingService
from app.llm import LLMRequest, LLMRetryPolicy, OllamaLLMService, OpenRouterLLMService
from app.qdrant_store import QdrantVectorStore
from app.rag import RAGPipeline
from app.retrieval import Retriever


class FakeEmbeddingService(EmbeddingService):
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def require_ollama_model() -> None:
    try:
        response = httpx.get(f"{settings.ollama_url}/api/tags", timeout=2.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"Ollama is unavailable: {exc}")

    model_names = {model["name"] for model in response.json().get("models", [])}
    if settings.ollama_model not in model_names:
        pytest.skip(f"Ollama model '{settings.ollama_model}' is unavailable")


@pytest.mark.integration
def test_rag_pipeline_with_real_ollama():
    require_ollama_model()
    client = QdrantClient(":memory:")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="integration_test",
        vector_size=3,
    )

    embedding_service = FakeEmbeddingService()

    vector_store.upsert(
        vector_id=str(uuid4()),
        vector=[1.0, 0.0, 0.0],
        payload={
            "document_id": "doc-1",
            "page_id": "page-1",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Договор действует один год с даты подписания.",
        },
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    llm = OllamaLLMService(
        model=settings.ollama_model,
        base_url=settings.ollama_url,
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=ContextBuilder(),
        answer_generator=AnswerGenerator(llm),
    )

    answer = pipeline.answer(
        question="Каков срок действия договора?",
    )

    assert "год" in answer.lower()
    assert "[стр. 1]" in answer


@pytest.mark.integration
def test_openrouter_provider_smoke():
    if os.environ.get("RUN_OPENROUTER_TESTS") != "1":
        pytest.skip("Set RUN_OPENROUTER_TESTS=1 to allow a billable OpenRouter call")
    if not settings.openrouter_api_key.strip():
        pytest.skip("OPENROUTER_API_KEY is not configured")

    llm = OpenRouterLLMService(
        model=settings.openrouter_model,
        base_url=settings.openrouter_url,
        api_key=settings.openrouter_api_key,
        connect_timeout=settings.openrouter_connect_timeout,
        read_timeout=settings.openrouter_read_timeout,
        retry_policy=LLMRetryPolicy(
            max_attempts=settings.llm_max_attempts,
            total_timeout=settings.llm_total_timeout,
        ),
    )
    response = llm.call(
        LLMRequest(
            system_prompt="Reply briefly in Russian.",
            prompt="Ответь одним словом: готово.",
        )
    )

    assert response.text.strip()
