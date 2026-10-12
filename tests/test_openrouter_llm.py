import httpx

from app.llm import (
    LLMError,
    LLMErrorCategory,
    LLMRequest,
    LLMUsage,
    OpenRouterLLMService,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"Server error '{self.status_code}'",
                request=httpx.Request(
                    "POST", "http://testserver/api/v1/chat/completions"
                ),
                response=None,
            )

    def json(self):
        return self._payload


def test_openrouter_llm_service_configuration():
    llm = OpenRouterLLMService(
        model="test-model",
        base_url="http://localhost:9999/",
        api_key="abc",
        timeout=10.0,
    )

    assert llm.model == "test-model"
    assert llm.base_url == "http://localhost:9999"
    assert llm.api_key == "abc"


def test_openrouter_llm_service_returns_text(monkeypatch):
    llm = OpenRouterLLMService(
        model="test-model",
        base_url="http://testserver",
        api_key="k",
    )

    monkeypatch.setattr(
        llm,
        "_post",
        lambda payload: FakeResponse(
            {
                "choices": [{"message": {"content": "Ответ от OpenRouter"}}],
                "id": "r1",
                "usage": {"prompt_tokens": 1},
            }
        ),
    )

    resp = llm.call(LLMRequest(prompt="Привет"))

    assert resp.text == "Ответ от OpenRouter"
    assert resp.provider_request_id == "r1"
    assert resp.provider == "openrouter"
    assert resp.model == "test-model"
    assert resp.usage == LLMUsage(input_tokens=1)


def test_openrouter_llm_service_rejects_missing_answer_text(monkeypatch):
    llm = OpenRouterLLMService(model="test-model", base_url="http://testserver")
    monkeypatch.setattr(llm, "_post", lambda payload: FakeResponse({"choices": []}))

    try:
        llm.call(LLMRequest(prompt="Привет"))
    except LLMError as exc:
        assert exc.category is LLMErrorCategory.INVALID_RESPONSE
        assert exc.provider == "openrouter"
    else:
        raise AssertionError("LLMError was not raised")


def test_openrouter_llm_service_maps_authentication_error(monkeypatch):
    llm = OpenRouterLLMService(model="test-model", base_url="http://testserver")
    monkeypatch.setattr(
        llm,
        "_post",
        lambda payload: FakeResponse({}, status_code=401),
    )

    try:
        llm.call(LLMRequest(prompt="Привет"))
    except LLMError as exc:
        assert exc.category is LLMErrorCategory.AUTHENTICATION
        assert exc.status_code == 401
    else:
        raise AssertionError("LLMError was not raised")
