import httpx
import pytest

from app.llm import (
    LLMCapabilities,
    LLMError,
    LLMErrorCategory,
    LLMRequest,
    LLMResponse,
    LLMService,
    OllamaLLMService,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"Server error '{self.status_code}'",
                request=httpx.Request("POST", "http://testserver/api/generate"),
                response=None,
            )

    def json(self):
        return self._payload


def test_ollama_llm_service_configuration():
    llm = OllamaLLMService(
        model="llama3.2:3b",
        base_url="http://localhost:11434/",
    )

    assert llm.model == "llama3.2:3b"
    assert llm.base_url == "http://localhost:11434"


def test_ollama_llm_service_is_abstract_contract():
    assert issubclass(OllamaLLMService, LLMService)


def test_ollama_llm_service_returns_response_text(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")

    monkeypatch.setattr(
        llm,
        "_post",
        lambda payload: FakeResponse({"response": "Векторная база хранит эмбеддинги."}),
    )

    answer = llm.generate("Что такое векторная база?")

    assert answer == "Векторная база хранит эмбеддинги."


def test_ollama_llm_service_returns_provider_neutral_response(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")
    monkeypatch.setattr(
        llm,
        "_post",
        lambda payload: FakeResponse({"response": "Готово."}),
    )

    response = llm.call(LLMRequest(prompt="Вопрос"))

    assert response == LLMResponse(
        text="Готово.",
        provider="ollama",
        model="test-model",
    )
    assert llm.capabilities == LLMCapabilities()


def test_ollama_llm_service_maps_available_usage_metadata(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")
    monkeypatch.setattr(
        llm,
        "_post",
        lambda payload: FakeResponse(
            {
                "response": "Готово.",
                "prompt_eval_count": 7,
                "eval_count": 3,
            }
        ),
    )

    response = llm.call(LLMRequest(prompt="Вопрос"))

    assert response.usage is not None
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 10


def test_ollama_llm_service_combines_system_and_user_prompts(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")
    payloads = []
    monkeypatch.setattr(
        llm,
        "_post",
        lambda payload: (
            payloads.append(payload) or FakeResponse({"response": "Ответ"})
        ),
    )

    llm.call(
        LLMRequest(
            system_prompt="System instructions",
            prompt="Question and evidence",
            prompt_version="answer-v1",
        )
    )

    assert payloads[0]["prompt"] == "System instructions\n\nQuestion and evidence"


@pytest.mark.parametrize(
    "payload",
    [{}, {"response": None}, {"response": ""}, []],
)
def test_ollama_llm_service_rejects_missing_or_malformed_text(monkeypatch, payload):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")
    monkeypatch.setattr(llm, "_post", lambda request: FakeResponse(payload))

    with pytest.raises(LLMError) as exc_info:
        llm.call(LLMRequest(prompt="Вопрос"))

    assert exc_info.value.category is LLMErrorCategory.INVALID_RESPONSE


def test_ollama_llm_service_maps_connection_failure(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")

    def fail(payload):
        raise httpx.ConnectError("sensitive request details")

    monkeypatch.setattr(llm, "_post", fail)

    with pytest.raises(LLMError) as exc_info:
        llm.call(LLMRequest(prompt="Вопрос"))

    assert exc_info.value.category is LLMErrorCategory.CONNECTION
    assert "sensitive request details" not in str(exc_info.value)


def test_ollama_llm_service_maps_invalid_json(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")
    response = FakeResponse({})

    def invalid_json():
        raise ValueError("raw response")

    response.json = invalid_json
    monkeypatch.setattr(llm, "_post", lambda payload: response)

    with pytest.raises(LLMError) as exc_info:
        llm.call(LLMRequest(prompt="Вопрос"))

    assert exc_info.value.category is LLMErrorCategory.INVALID_RESPONSE
    assert "raw response" not in str(exc_info.value)


def test_ollama_llm_service_propagates_http_error(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")

    def fail(payload):
        raise httpx.HTTPStatusError("Server error '500'", request=None, response=None)

    monkeypatch.setattr(llm, "_post", fail)

    with pytest.raises(LLMError) as exc_info:
        llm.generate("Любой вопрос")
    assert exc_info.value.category is LLMErrorCategory.UPSTREAM


def test_ollama_llm_service_is_integration_only():
    """The real service is only exercised under the integration marker."""
    assert OllamaLLMService().base_url.endswith("11434")
