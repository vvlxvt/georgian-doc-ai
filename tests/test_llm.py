from datetime import UTC

import httpx
import pytest

from app.llm import (
    LLMCapabilities,
    LLMError,
    LLMErrorCategory,
    LLMRequest,
    LLMRetryPolicy,
    LLMService,
    OllamaLLMService,
    _post_with_error_mapping,
    _retry_after_seconds,
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
        lambda payload, timeout: FakeResponse(
            {"response": "Векторная база хранит эмбеддинги."}
        ),
    )

    answer = llm.generate("Что такое векторная база?")

    assert answer == "Векторная база хранит эмбеддинги."


def test_ollama_llm_service_returns_provider_neutral_response(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")
    monkeypatch.setattr(
        llm,
        "_post",
        lambda payload, timeout: FakeResponse({"response": "Готово."}),
    )

    response = llm.call(LLMRequest(prompt="Вопрос"))

    assert response.text == "Готово."
    assert response.provider == "ollama"
    assert response.model == "test-model"
    assert response.attempts == 1
    assert response.duration_seconds is not None
    assert llm.capabilities == LLMCapabilities()


def test_ollama_llm_service_maps_available_usage_metadata(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")
    monkeypatch.setattr(
        llm,
        "_post",
        lambda payload, timeout: FakeResponse(
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
        lambda payload, timeout: (
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
    monkeypatch.setattr(
        llm,
        "_post",
        lambda request, timeout: FakeResponse(payload),
    )

    with pytest.raises(LLMError) as exc_info:
        llm.call(LLMRequest(prompt="Вопрос"))

    assert exc_info.value.category is LLMErrorCategory.INVALID_RESPONSE


def test_ollama_llm_service_maps_connection_failure(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")

    def fail(payload, timeout):
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
    monkeypatch.setattr(llm, "_post", lambda payload, timeout: response)

    with pytest.raises(LLMError) as exc_info:
        llm.call(LLMRequest(prompt="Вопрос"))

    assert exc_info.value.category is LLMErrorCategory.INVALID_RESPONSE
    assert "raw response" not in str(exc_info.value)


def test_ollama_llm_service_propagates_http_error(monkeypatch):
    llm = OllamaLLMService(model="test-model", base_url="http://testserver")

    def fail(payload, timeout):
        raise httpx.HTTPStatusError("Server error '500'", request=None, response=None)

    monkeypatch.setattr(llm, "_post", fail)

    with pytest.raises(LLMError) as exc_info:
        llm.generate("Любой вопрос")
    assert exc_info.value.category is LLMErrorCategory.UPSTREAM


def test_ollama_llm_service_is_integration_only():
    """The real service is only exercised under the integration marker."""
    assert OllamaLLMService().base_url.endswith("11434")


def test_retry_policy_retries_safe_connection_failure_with_bounded_jitter():
    now = [10.0]
    calls = []
    sleeps = []

    def post(payload, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            now[0] += 0.1
            raise httpx.ConnectError("private connection detail")
        return FakeResponse({"response": "ok"})

    result = _post_with_error_mapping(
        post,
        {},
        provider="test",
        model="test-model",
        retry_policy=LLMRetryPolicy(
            max_attempts=2,
            total_timeout=5.0,
            base_delay=1.0,
            max_delay=2.0,
        ),
        sleep=lambda delay: (sleeps.append(delay), now.__setitem__(0, now[0] + delay)),
        monotonic=lambda: now[0],
        jitter=lambda: 0.25,
    )

    assert result.attempts == 2
    assert result.duration_seconds == pytest.approx(0.35)
    assert calls == pytest.approx([5.0, 4.65])
    assert sleeps == [0.25]


def test_retry_policy_honors_retry_after_for_rejected_rate_limited_request():
    now = [0.0]
    calls = []
    sleeps = []

    def post(payload, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            return FakeResponse(
                {},
                status_code=429,
                headers={"retry-after": "2"},
            )
        return FakeResponse({"response": "ok"})

    result = _post_with_error_mapping(
        post,
        {},
        provider="openrouter",
        model="test-model",
        retry_policy=LLMRetryPolicy(max_attempts=2, total_timeout=5.0),
        sleep=lambda delay: (sleeps.append(delay), now.__setitem__(0, now[0] + delay)),
        monotonic=lambda: now[0],
        jitter=lambda: 0.1,
    )

    assert result.attempts == 2
    assert calls == [5.0, 3.0]
    assert sleeps == [2.0]


@pytest.mark.parametrize(
    ("status_code", "category"),
    [
        (400, LLMErrorCategory.INVALID_REQUEST),
        (401, LLMErrorCategory.AUTHENTICATION),
        (402, LLMErrorCategory.INSUFFICIENT_CREDITS),
        (500, LLMErrorCategory.UPSTREAM),
    ],
)
def test_retry_policy_does_not_retry_nonretryable_http_status(status_code, category):
    calls = []

    def post(payload, timeout):
        calls.append(timeout)
        return FakeResponse({}, status_code=status_code)

    with pytest.raises(LLMError) as exc_info:
        _post_with_error_mapping(
            post,
            {},
            provider="openrouter",
            model="test-model",
            retry_policy=LLMRetryPolicy(max_attempts=3),
            sleep=lambda _: pytest.fail("Non-retryable responses must not sleep"),
            jitter=lambda: 0.5,
        )

    assert len(calls) == 1
    assert exc_info.value.category is category
    assert exc_info.value.attempts == 1


@pytest.mark.parametrize(
    "exception",
    [
        httpx.ReadTimeout("read timeout"),
        httpx.ReadError("read failure"),
        httpx.WriteError("write failure"),
        httpx.RemoteProtocolError("protocol failure"),
    ],
)
def test_retry_policy_does_not_retry_ambiguous_network_outcomes(exception):
    calls = []

    def post(payload, timeout):
        calls.append(timeout)
        raise exception

    with pytest.raises(LLMError) as exc_info:
        _post_with_error_mapping(
            post,
            {},
            provider="openrouter",
            model="test-model",
            retry_policy=LLMRetryPolicy(max_attempts=3),
            sleep=lambda _: pytest.fail("Ambiguous outcomes must not sleep"),
        )

    assert len(calls) == 1
    assert exc_info.value.category is LLMErrorCategory.OUTCOME_UNKNOWN
    assert exc_info.value.attempts == 1
    assert exception.args[0] not in str(exc_info.value)


def test_retry_after_supports_http_dates_and_rejects_invalid_values():
    from datetime import datetime

    current_time = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)

    assert _retry_after_seconds("3.5") == 3.5
    assert (
        _retry_after_seconds(
            "Mon, 12 Oct 2026 12:00:05 GMT",
            now=current_time,
        )
        == 5.0
    )
    assert _retry_after_seconds("not-a-date") is None


def test_retry_policy_does_not_exceed_remaining_total_time():
    calls = []
    sleeps = []
    now = [0.0]

    def post(payload, timeout):
        calls.append(timeout)
        now[0] = 2.0
        return FakeResponse(
            {},
            status_code=429,
            headers={"retry-after": "4"},
        )

    with pytest.raises(LLMError) as exc_info:
        _post_with_error_mapping(
            post,
            {},
            provider="openrouter",
            model="test-model",
            retry_policy=LLMRetryPolicy(max_attempts=3, total_timeout=5.0),
            sleep=sleeps.append,
            monotonic=lambda: now[0],
        )

    assert len(calls) == 1
    assert sleeps == []
    assert exc_info.value.category is LLMErrorCategory.RATE_LIMIT
    assert exc_info.value.attempts == 1
