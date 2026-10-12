from decimal import Decimal

import httpx
import pytest

from app.llm import (
    LLMCapabilities,
    LLMError,
    LLMErrorCategory,
    LLMRequest,
    LLMUsage,
    OpenRouterLLMService,
)


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

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
        connect_timeout=3.0,
        read_timeout=10.0,
        capabilities=LLMCapabilities(supports_json_mode=True),
    )

    assert llm.model == "test-model"
    assert llm.base_url == "http://localhost:9999"
    assert llm.api_key == "abc"
    assert llm.connect_timeout == 3.0
    assert llm.read_timeout == 10.0
    assert llm.capabilities.supports_json_mode


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
                "usage": {"prompt_tokens": 1, "cost": 0.002},
            }
        ),
    )

    resp = llm.call(LLMRequest(prompt="Привет"))

    assert resp.text == "Ответ от OpenRouter"
    assert resp.provider_request_id == "r1"
    assert resp.provider == "openrouter"
    assert resp.model == "test-model"
    assert resp.usage == LLMUsage(input_tokens=1)
    assert resp.cost == Decimal("0.002")


def test_openrouter_llm_service_rejects_missing_answer_text(monkeypatch):
    llm = OpenRouterLLMService(
        model="test-model",
        base_url="http://testserver",
        api_key="test-key",
    )
    monkeypatch.setattr(llm, "_post", lambda payload: FakeResponse({"choices": []}))

    try:
        llm.call(LLMRequest(prompt="Привет"))
    except LLMError as exc:
        assert exc.category is LLMErrorCategory.INVALID_RESPONSE
        assert exc.provider == "openrouter"
    else:
        raise AssertionError("LLMError was not raised")


def test_openrouter_llm_service_maps_authentication_error(monkeypatch):
    llm = OpenRouterLLMService(
        model="test-model",
        base_url="http://testserver",
        api_key="test-key",
    )
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


def test_openrouter_llm_service_requires_key_before_network(monkeypatch):
    llm = OpenRouterLLMService(model="test-model", base_url="http://testserver")
    monkeypatch.setattr(
        llm,
        "_post",
        lambda payload: pytest.fail("Request must not be sent without an API key"),
    )

    with pytest.raises(LLMError) as exc_info:
        llm.call(LLMRequest(prompt="Привет"))

    assert exc_info.value.category is LLMErrorCategory.AUTHENTICATION


@pytest.mark.parametrize(
    ("status_code", "category"),
    [
        (402, LLMErrorCategory.INSUFFICIENT_CREDITS),
        (408, LLMErrorCategory.OUTCOME_UNKNOWN),
        (413, LLMErrorCategory.CONTEXT_LENGTH),
        (429, LLMErrorCategory.RATE_LIMIT),
        (500, LLMErrorCategory.UPSTREAM),
    ],
)
def test_openrouter_llm_service_maps_provider_status(
    monkeypatch,
    status_code,
    category,
):
    llm = OpenRouterLLMService(
        model="test-model",
        base_url="http://testserver",
        api_key="test-key",
    )
    monkeypatch.setattr(
        llm,
        "_post",
        lambda payload: FakeResponse({}, status_code=status_code),
    )

    with pytest.raises(LLMError) as exc_info:
        llm.call(LLMRequest(prompt="Привет"))

    assert exc_info.value.category is category
    assert exc_info.value.status_code == status_code


def test_openrouter_llm_service_serializes_request_and_timeout(monkeypatch):
    call = {}

    def fake_post(url, **kwargs):
        call["url"] = url
        call.update(kwargs)
        return FakeResponse(
            {"choices": [{"message": {"content": "Ответ"}}]},
            headers={"x-request-id": "request-header-id"},
        )

    monkeypatch.setattr("app.llm.httpx.post", fake_post)
    llm = OpenRouterLLMService(
        model="test-model",
        base_url="https://openrouter.example/",
        api_key="test-key",
        connect_timeout=2.5,
        read_timeout=18.0,
    )

    response = llm.call(LLMRequest(prompt="Тестовый вопрос"))

    assert call["url"] == ("https://openrouter.example/api/v1/chat/completions")
    assert call["json"] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Тестовый вопрос"}],
        "stream": False,
    }
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["timeout"].connect == 2.5
    assert call["timeout"].read == 18.0
    assert response.provider_request_id == "request-header-id"


@pytest.mark.parametrize(
    ("exception", "category"),
    [
        (httpx.ConnectTimeout("connect"), LLMErrorCategory.TIMEOUT),
        (httpx.ReadTimeout("read"), LLMErrorCategory.OUTCOME_UNKNOWN),
    ],
)
def test_openrouter_llm_service_maps_network_timeouts(
    monkeypatch,
    exception,
    category,
):
    llm = OpenRouterLLMService(
        model="test-model",
        base_url="http://testserver",
        api_key="test-key",
    )

    def fail(payload):
        raise exception

    monkeypatch.setattr(llm, "_post", fail)

    with pytest.raises(LLMError) as exc_info:
        llm.call(LLMRequest(prompt="Привет"))

    assert exc_info.value.category is category
