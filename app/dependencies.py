from qdrant_client import QdrantClient

from app.answer import AnswerGenerator
from app.config import settings
from app.context import ContextBuilder
from app.embeddings import HuggingFaceEmbeddingService
from app.llm import OllamaLLMService
from app.qdrant_store import QdrantVectorStore
from app.rag import RAGPipeline
from app.retrieval import Retriever


def create_rag_pipeline() -> RAGPipeline:
    embedding_service = HuggingFaceEmbeddingService(
        model_name=settings.embedding_model
    )

    qdrant_client = QdrantClient(url=settings.qdrant_url)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_size,
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    context_builder = ContextBuilder()

    llm = OllamaLLMService(
        model=settings.ollama_model,
        base_url=settings.ollama_url,
    )

    answer_generator = AnswerGenerator(llm=llm)

    return RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        answer_generator=answer_generator,
    )