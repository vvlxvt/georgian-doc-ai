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
        document_ids=None,
    ):
        self.searched_document_ids = document_ids
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
        return "Договор действует один год. [стр. 3]"


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

    answer = pipeline.answer(question="Каков срок действия договора?")

    assert answer == "Договор действует один год. [стр. 3]"
    assert vector_store.searched_document_ids is None

    assert len(llm.prompts) == 1

    prompt = llm.prompts[0]

    assert "Каков срок действия договора?" in prompt
    assert "The contract is valid for one year." in prompt
    assert "[стр. 3]" in prompt


def test_rag_pipeline_forwards_selected_document_ids():
    vector_store = FakeVectorStore()
    pipeline = RAGPipeline(
        retriever=Retriever(FakeEmbeddingService(), vector_store),
        context_builder=ContextBuilder(),
        answer_generator=AnswerGenerator(FakeLLMService()),
    )

    pipeline.answer(
        question="Каков срок действия договора?",
        document_ids=["doc-1", "doc-2"],
    )

    assert vector_store.searched_document_ids == ["doc-1", "doc-2"]
