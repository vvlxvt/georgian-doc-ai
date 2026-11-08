import httpx

from app.answer import (
    ANSWER_PROMPT_VERSION,
    AnswerGenerator,
    ChatAnswerResult,
    ConversationMessage,
)
from app.context import ContextBuilder
from app.embeddings import EmbeddingService
from app.llm import (
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    LLMRetryPolicy,
    LLMService,
    OpenRouterLLMService,
)
from app.rag import RAGPipeline
from app.retrieval import Retriever


class FakeEmbeddingService(EmbeddingService):
    def __init__(self):
        self.embedded_texts = []

    def embed(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
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
        self.requests = []
        self.provider = "fake"
        self.model = "test-model"
        self.capabilities = LLMCapabilities()

    def call(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            text="Договор действует один год. [стр. 3]",
            provider=self.provider,
            model=self.model,
        )


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

    assert len(llm.requests) == 1

    request = llm.requests[0]

    assert "Каков срок действия договора?" in request.prompt
    assert "The contract is valid for one year." in request.prompt
    assert "[стр. 3]" in request.prompt
    assert "ID источника: S1" in request.prompt
    assert request.prompt_version == ANSWER_PROMPT_VERSION
    assert request.response_schema is None


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


def test_rag_pipeline_uses_openrouter_adapter_with_mocked_http(monkeypatch):
    calls = []
    response_text = "Договор действует один год. [стр. 3]"

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": response_text}}],
                "id": "request-1",
            },
            headers={"x-request-id": "request-header-1"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("app.llm.httpx.post", fake_post)
    llm = OpenRouterLLMService(
        model="test-model",
        base_url="https://openrouter.test",
        api_key="test-key",
        retry_policy=LLMRetryPolicy(max_attempts=1),
    )
    pipeline = RAGPipeline(
        retriever=Retriever(FakeEmbeddingService(), FakeVectorStore()),
        context_builder=ContextBuilder(),
        answer_generator=AnswerGenerator(llm),
    )

    answer = pipeline.answer(question="Каков срок действия договора?")

    assert answer == "Договор действует один год. [стр. 3]"
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://openrouter.test/api/v1/chat/completions"
    messages = kwargs["json"]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Каков срок действия договора?" in messages[-1]["content"]
    assert "The contract is valid for one year." in messages[-1]["content"]
    assert "ID источника: S1" in messages[-1]["content"]
    assert "обычный текст" in messages[0]["content"]
    assert "response_format" not in kwargs["json"]


def test_rag_pipeline_answer_with_sources():
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

    result = pipeline.answer_with_sources(question="Каков срок действия договора?")

    assert isinstance(result, ChatAnswerResult)
    assert result.answer == "Договор действует один год. [стр. 3]"
    assert result.model_name == "test-model"
    assert len(result.sources) == 1
    assert result.sources[0].document_id == "doc-1"
    assert result.sources[0].page_number == 3
    assert result.sources[0].chunk_id == "chunk-1"


def test_rag_pipeline_uses_history_for_retrieval_and_answer():
    embedding_service = FakeEmbeddingService()
    llm = FakeLLMService()
    pipeline = RAGPipeline(
        retriever=Retriever(embedding_service, FakeVectorStore()),
        context_builder=ContextBuilder(),
        answer_generator=AnswerGenerator(llm),
    )
    retrieval_query = "Предыдущий вопрос: срок договора\nТекущий вопрос: А у второго?"
    history = [
        ConversationMessage(role="user", content="Каков срок первого договора?"),
        ConversationMessage(role="assistant", content="Один год. [стр. 1]"),
    ]

    pipeline.answer_with_sources(
        question="А у второго?",
        retrieval_query=retrieval_query,
        conversation_history=history,
        document_ids=["doc-1"],
    )

    assert embedding_service.embedded_texts == [retrieval_query]
    assert "Текущий вопрос:\nА у второго?" in llm.requests[0].prompt
    assert "Каков срок первого договора?" in llm.requests[0].prompt
