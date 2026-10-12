import httpx

from app.llm import OpenRouterLLMService


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"Server error '{self.status_code}'",
                request=httpx.Request("POST", "http://testserver/api/v1/chat/completions"),
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
    llm = OpenRouterLLMService(model="test-model", base_url="http://testserver", api_key="k")

    monkeypatch.setattr(
        llm,
        "_post",
        lambda payload: FakeResponse({"choices": [{"message": {"content": "Ответ от OpenRouter"}}], "id": "r1", "usage": {"prompt_tokens": 1}}),
    )

    resp = llm.call("Привет")

    assert resp.text == "Ответ от OpenRouter"
    assert resp.provider_request_id == "r1"
    assert resp.usage == {"prompt_tokens": 1}
