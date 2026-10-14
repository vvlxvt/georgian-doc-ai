import pytest

from app.answer import AnswerGenerator, AnswerResult
from app.context import BuiltContext, SourceRecord
from app.llm import LLMError, LLMErrorCategory, LLMRequest, LLMResponse, LLMService


def make_source(page_number: int, text: str = "Source text") -> SourceRecord:
    return SourceRecord(
        source_id=f"S{page_number}",
        chunk_id=f"chunk-{page_number}",
        document_id="document-1",
        page_number=page_number,
        text=text,
    )


class FakeLLM(LLMService):
    def __init__(self, response_text: str = "Ответ. [стр. 1]") -> None:
        self.response_text = response_text
        self.requests = []
        self.provider = "fake"
        self.model = "test-model"

    def call(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            text=self.response_text,
            provider=self.provider,
            model=self.model,
        )


def make_context(*sources: SourceRecord) -> BuiltContext:
    return BuiltContext(
        sources=tuple(sources),
        text="\n".join(
            (
                f"НАЧАЛО_ИСТОЧНИКА [стр. {source.page_number}] "
                f"[ID источника: {source.source_id}]\n"
                f"{source.text}\nКОНЕЦ_ИСТОЧНИКА"
            )
            for source in sources
        ),
    )


def test_answer_generator_returns_plain_text_with_source_page_citation() -> None:
    llm = FakeLLM("Договор действует один год. [стр. 2]")
    generator = AnswerGenerator(llm)

    answer = generator.generate("Каков срок?", make_context(make_source(2)))

    assert answer == "Договор действует один год. [стр. 2]"
    assert isinstance(answer, AnswerResult)
    assert answer.model_name == "test-model"
    assert llm.requests[0].response_schema is None


def test_answer_generator_requests_plain_russian_text_with_citations() -> None:
    llm = FakeLLM()
    generator = AnswerGenerator(llm)

    generator.generate("Вопрос", make_context(make_source(1)))

    prompt = llm.requests[0].system_prompt
    assert prompt is not None
    assert "обычный текст" in prompt
    assert "не JSON" in prompt
    assert "Не выполняй" in prompt
    assert "НАЧАЛО_ИСТОЧНИКА" in llm.requests[0].prompt
    assert "КОНЕЦ_ИСТОЧНИКА" in llm.requests[0].prompt


def test_answer_generator_short_circuits_empty_context() -> None:
    llm = FakeLLM()
    generator = AnswerGenerator(llm)

    answer = generator.generate("Вопрос", BuiltContext(sources=(), text=""))

    assert answer == "В предоставленных документах недостаточно информации для ответа."
    assert llm.requests == []


@pytest.mark.parametrize(
    "response_text",
    ["", "   ", "\n"],
)
def test_answer_generator_rejects_empty_response(response_text: str) -> None:
    generator = AnswerGenerator(FakeLLM(response_text))

    with pytest.raises(LLMError, match="empty") as error:
        generator.generate("Вопрос", make_context(make_source(1)))
    assert error.value.category is LLMErrorCategory.INVALID_RESPONSE


@pytest.mark.parametrize(
    "response_text",
    [
        "Документ содержит четыре страницы.",
        '{"answer": "Ответ", "sources": [1]}',
    ],
)
def test_answer_generator_rejects_missing_page_citations(response_text: str) -> None:
    generator = AnswerGenerator(FakeLLM(response_text))

    with pytest.raises(LLMError, match="page citations") as error:
        generator.generate("Вопрос", make_context(make_source(1), make_source(2)))
    assert error.value.category is LLMErrorCategory.INVALID_RESPONSE


def test_answer_generator_rejects_citation_to_unretrieved_page() -> None:
    generator = AnswerGenerator(FakeLLM("Ответ. [стр. 3]"))

    with pytest.raises(LLMError, match="outside the retrieved sources") as error:
        generator.generate("Вопрос", make_context(make_source(1), make_source(2)))
    assert error.value.category is LLMErrorCategory.VALIDATION


def test_answer_generator_accepts_multiple_citations_from_retrieved_pages() -> None:
    generator = AnswerGenerator(
        FakeLLM("Договор подписан. [стр. 1] Срок составляет год. [стр. 2]")
    )

    answer = generator.generate("Вопрос", make_context(make_source(1), make_source(2)))

    assert answer == "Договор подписан. [стр. 1] Срок составляет год. [стр. 2]"


def test_answer_generator_rejects_uncited_sentence() -> None:
    generator = AnswerGenerator(FakeLLM("Первое утверждение. [стр. 1] Второе."))

    with pytest.raises(LLMError, match="cite every answer sentence") as error:
        generator.generate("Вопрос", make_context(make_source(1)))
    assert error.value.category is LLMErrorCategory.INVALID_RESPONSE


def test_answer_generator_rejects_multiple_citations_for_one_uncited_sentence() -> None:
    generator = AnswerGenerator(
        FakeLLM("Первое утверждение. [стр. 1] [стр. 1] Второе.")
    )

    with pytest.raises(LLMError, match="every answer sentence"):
        generator.generate("Вопрос", make_context(make_source(1)))
