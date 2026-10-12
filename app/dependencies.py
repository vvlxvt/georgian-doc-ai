from functools import lru_cache

from qdrant_client import QdrantClient

from app.answer import AnswerGenerator
from app.chunking import TextChunker
from app.config import settings
from app.context import ContextBuilder
from app.embeddings import HuggingFaceEmbeddingService
from app.indexing import ChunkIndexer
from app.llm import OllamaLLMService, OpenRouterLLMService
from app.ocr import TesseractOCREngine
from app.pipeline import DocumentPipeline
from app.qdrant_store import QdrantVectorStore
from app.rag import RAGPipeline
from app.retrieval import Retriever
from app.text import TextNormalizer


@lru_cache(maxsize=1)
def get_embedding_service() -> HuggingFaceEmbeddingService:
    return HuggingFaceEmbeddingService(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_size,
    )


@lru_cache(maxsize=1)
def get_llm_service() -> OllamaLLMService | OpenRouterLLMService:
    """Select LLM implementation based on settings.llm_provider.

    Defaults to 'openrouter' when configured so in Settings.
    """
    provider = (settings.llm_provider or "").strip().lower()

    if provider == "openrouter":
        return OpenRouterLLMService(
            model=settings.openrouter_model,
            base_url=settings.openrouter_url,
            api_key=settings.openrouter_api_key or None,
            timeout=getattr(settings, "openrouter_timeout", 120.0),
        )

    # fallback to Ollama for backward compatibility
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


@lru_cache(maxsize=1)
def get_chunk_indexer() -> ChunkIndexer:
    return ChunkIndexer(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


def create_document_pipeline(session) -> DocumentPipeline:
    """Build the ingestion pipeline. The caller owns the session lifecycle."""
    return DocumentPipeline(
        session=session,
        ocr_processor=TesseractOCREngine(languages=settings.tesseract_languages),
        text_normalizer=TextNormalizer(),
        text_chunker=TextChunker(
            max_length=settings.chunk_max_length,
            overlap=settings.chunk_overlap,
        ),
        chunk_indexer=get_chunk_indexer(),
        min_text_length=settings.min_text_length,
    )


def reset_application_services() -> None:
    create_rag_pipeline.cache_clear()
    get_chunk_indexer.cache_clear()
    get_llm_service.cache_clear()
    get_vector_store.cache_clear()
    get_qdrant_client.cache_clear()
    get_embedding_service.cache_clear()
