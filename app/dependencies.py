from functools import lru_cache

from qdrant_client import QdrantClient

from app.answer import AnswerGenerator
from app.config import settings
from app.context import ContextBuilder
from app.embeddings import HuggingFaceEmbeddingService
from app.llm import OllamaLLMService
from app.qdrant_store import QdrantVectorStore
from app.rag import RAGPipeline
from app.retrieval import Retriever


@lru_cache(maxsize=1)
def get_embedding_service() -> HuggingFaceEmbeddingService:
    return HuggingFaceEmbeddingService(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_size,
    )


@lru_cache(maxsize=1)
def get_llm_service() -> OllamaLLMService:
    return OllamaLLMService(
        model=settings.ollama_model,
        base_url=settings.ollama_url,
    )


@lru_cache(maxsize=1)
def create_rag_pipeline() -> RAGPipeline:
    embedding_service = get_embedding_service()

    vector_store = get_vector_store()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    context_builder = ContextBuilder()

    answer_generator = AnswerGenerator(llm=get_llm_service())

    return RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        answer_generator=answer_generator,
    )


def reset_application_services() -> None:
    create_rag_pipeline.cache_clear()
    get_llm_service.cache_clear()
    get_vector_store.cache_clear()
    get_qdrant_client.cache_clear()
    get_embedding_service.cache_clear()
