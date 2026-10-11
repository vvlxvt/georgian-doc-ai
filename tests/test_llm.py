import httpx
import pytest

from app.llm import LLMService, OllamaLLMService


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


def test_ollama_llm_service_propagates_http_error(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")

    def fail(payload):
        raise httpx.HTTPStatusError("Server error '500'", request=None, response=None)

    monkeypatch.setattr(llm, "_post", fail)

    with pytest.raises(httpx.HTTPStatusError):
        llm.generate("Любой вопрос")


def test_ollama_llm_service_is_integration_only():
    """The real service is only exercised under the integration marker."""
    assert OllamaLLMService().base_url.endswith("11434")
