from qdrant_client import QdrantClient
from uuid import uuid4

from app.answer import AnswerGenerator
from app.context import ContextBuilder
from app.embeddings import EmbeddingService
from app.qdrant_store import QdrantVectorStore
from app.retrieval import Retriever
from app.rag import RAGPipeline
from app.llm import OllamaLLMService


class FakeEmbeddingService(EmbeddingService):
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_rag_pipeline_with_real_ollama():
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
            "text": (
                "The contract is valid for one year "
                "from the date of signing."
            ),
        },
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    llm = OllamaLLMService(
        model="llama3.2:3b",
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=ContextBuilder(),
        answer_generator=AnswerGenerator(llm),
    )

    answer = pipeline.answer(
        question="How long is the contract valid?",
    )

    assert answer
    assert isinstance(answer, str)