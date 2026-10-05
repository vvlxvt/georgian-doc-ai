from app.answer import AnswerGenerator
from app.context import ContextBuilder
from app.embeddings import EmbeddingService
from app.llm import LLMService
from app.rag import RAGPipeline
from app.retrieval import Retriever


class FakeEmbeddingService(EmbeddingService):
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    def search(
        self,
        vector,
        limit=5,
        score_threshold=None,
        document_id=None,
    ):
        return [
            {
                "id": "chunk-1",
                "score": 0.9,
                "payload": {
                    "document_id": "doc-1",
                    "page_number": 3,
                    "text": "The contract is valid for one year.",
                },
            }
        ]

    def upsert(self, vector_id, vector, payload):
        pass

    def delete(self, vector_id):
        pass

    def delete_by_page(self, page_id, exclude_ids=None):
        pass


class FakeLLMService(LLMService):
    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "The contract is valid for one year."


def test_rag_pipeline():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    llm = FakeLLMService()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=ContextBuilder(),
        answer_generator=AnswerGenerator(llm),
    )

    answer = pipeline.answer(question="How long is the contract valid?")

    assert answer == "The contract is valid for one year."

    assert len(llm.prompts) == 1

    prompt = llm.prompts[0]

    assert "How long is the contract valid?" in prompt
    assert "The contract is valid for one year." in prompt
