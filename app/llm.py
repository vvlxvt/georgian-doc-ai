from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any

import httpx


class LLMErrorCategory(StrEnum):
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    INSUFFICIENT_CREDITS = "insufficient_credits"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    CONTEXT_LENGTH = "context_length"
    INVALID_RESPONSE = "invalid_response"
    REFUSAL = "refusal"
    UPSTREAM = "upstream"
    OUTCOME_UNKNOWN = "outcome_unknown"
    VALIDATION = "validation"


class LLMError(Exception):
    def __init__(
        self,
        category: LLMErrorCategory,
        message: str,
        *,
        provider: str,
        model: str,
        provider_request_id: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        retry_after: str | None = None,
        attempts: int = 0,
        duration_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.provider = provider
        self.model = model
        self.provider_request_id = provider_request_id
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.attempts = attempts
        self.duration_seconds = duration_seconds


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    system_prompt: str | None = None
    prompt_version: str | None = None
    response_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMCapabilities:
    supports_structured_output: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = False
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class LLMRetryPolicy:
    max_attempts: int = 2
    total_timeout: float = 180.0
    base_delay: float = 0.5
    max_delay: float = 4.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be greater than 0")
        if not math.isfinite(self.total_timeout) or self.total_timeout <= 0:
            raise ValueError("total_timeout must be greater than 0")
        if (
            not math.isfinite(self.base_delay)
            or not math.isfinite(self.max_delay)
            or self.base_delay < 0
            or self.max_delay < 0
        ):
            raise ValueError("retry delays must be finite and nonnegative")


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: LLMUsage | None = None
    cost: Decimal | None = None
    provider_request_id: str | None = None
    refused: bool = False
    attempts: int = 1
    duration_seconds: float | None = None


@dataclass(frozen=True)
class _PostResult:
    response: httpx.Response
    attempts: int
    duration_seconds: float


class LLMService(ABC):
    """Provider-neutral LLM contract with a text-only compatibility method."""

    provider: str
    model: str
    capabilities: LLMCapabilities

    @abstractmethod
    def call(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def generate(self, prompt: str) -> str:
        return self.call(LLMRequest(prompt=prompt)).text


def _post_with_error_mapping(
    post: Callable[[dict[str, Any], float], httpx.Response],
    payload: dict[str, Any],
    *,
    provider: str,
    model: str,
    retry_policy: LLMRetryPolicy = LLMRetryPolicy(),
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
    jitter: Callable[[], float] | None = None,
) -> _PostResult:
    sleep = sleep or time.sleep
    monotonic = monotonic or time.monotonic
    jitter = jitter or random.random
    started_at = monotonic()
    last_error: LLMError | None = None

    for attempt in range(1, retry_policy.max_attempts + 1):
        remaining = retry_policy.total_timeout - (monotonic() - started_at)
        if remaining <= 0:
            if last_error is not None:
                raise _with_attempt_details(
                    last_error,
                    attempts=attempt - 1,
                    started_at=started_at,
                    monotonic=monotonic,
                )
            raise LLMError(
                LLMErrorCategory.TIMEOUT,
                "The total LLM request time limit was reached.",
                provider=provider,
                model=model,
                attempts=attempt - 1,
                duration_seconds=monotonic() - started_at,
            )

        try:
            response = _post_once(
                post,
                payload,
                timeout=remaining,
                provider=provider,
                model=model,
            )
        except LLMError as exc:
            last_error = exc
            if not exc.retryable or attempt >= retry_policy.max_attempts:
                raise _with_attempt_details(
                    exc,
                    attempts=attempt,
                    started_at=started_at,
                    monotonic=monotonic,
                ) from exc
            delay = _retry_delay(
                exc,
                attempt=attempt,
                policy=retry_policy,
                jitter=jitter,
            )
            if delay >= remaining:
                raise _with_attempt_details(
                    exc,
                    attempts=attempt,
                    started_at=started_at,
                    monotonic=monotonic,
                ) from exc
            sleep(delay)
            continue

        duration_seconds = monotonic() - started_at
        if duration_seconds > retry_policy.total_timeout:
            raise LLMError(
                LLMErrorCategory.OUTCOME_UNKNOWN,
                "The total LLM request time limit was reached after transmission.",
                provider=provider,
                model=model,
                attempts=attempt,
                duration_seconds=duration_seconds,
            )
        return _PostResult(
            response=response,
            attempts=attempt,
            duration_seconds=duration_seconds,
        )

    raise AssertionError("Retry loop exited without a response or LLM error.")


def _post_once(
    post: Callable[[dict[str, Any], float], httpx.Response],
    payload: dict[str, Any],
    *,
    timeout: float,
    provider: str,
    model: str,
) -> httpx.Response:
    try:
        response = post(payload, timeout)
    except httpx.HTTPStatusError as exc:
        raise _mapped_http_error(
            exc,
            provider=provider,
            model=model,
        ) from exc
    except httpx.ConnectError as exc:
        raise LLMError(
            LLMErrorCategory.CONNECTION,
            "Could not connect to the LLM provider.",
            provider=provider,
            model=model,
            retryable=True,
        ) from exc
    except httpx.ConnectTimeout as exc:
        raise LLMError(
            LLMErrorCategory.CONNECTION,
            "Could not connect to the LLM provider before the connection timeout.",
            provider=provider,
            model=model,
            retryable=True,
        ) from exc
    except (httpx.ReadTimeout, httpx.WriteTimeout) as exc:
        raise LLMError(
            LLMErrorCategory.OUTCOME_UNKNOWN,
            "The provider request outcome is unknown after a network timeout.",
            provider=provider,
            model=model,
        ) from exc
    except (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError) as exc:
        raise LLMError(
            LLMErrorCategory.OUTCOME_UNKNOWN,
            "The provider request outcome is unknown after a network failure.",
            provider=provider,
            model=model,
        ) from exc
    except httpx.TimeoutException as exc:
        raise LLMError(
            LLMErrorCategory.TIMEOUT,
            "The request to the LLM provider timed out.",
            provider=provider,
            model=model,
        ) from exc
    except httpx.RequestError as exc:
        raise LLMError(
            LLMErrorCategory.CONNECTION,
            "The request to the LLM provider failed.",
            provider=provider,
            model=model,
        ) from exc

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise _mapped_http_error(
            exc,
            provider=provider,
            model=model,
            response=response,
        ) from exc

    return response


def _with_attempt_details(
    error: LLMError,
    *,
    attempts: int,
    started_at: float,
    monotonic: Callable[[], float],
) -> LLMError:
    return LLMError(
        error.category,
        str(error),
        provider=error.provider,
        model=error.model,
        provider_request_id=error.provider_request_id,
        status_code=error.status_code,
        retryable=error.retryable,
        retry_after=error.retry_after,
        attempts=attempts,
        duration_seconds=max(0.0, monotonic() - started_at),
    )


def _retry_delay(
    error: LLMError,
    *,
    attempt: int,
    policy: LLMRetryPolicy,
    jitter: Callable[[], float],
) -> float:
    retry_after = _retry_after_seconds(error.retry_after)
    if retry_after is not None:
        return retry_after

    ceiling = min(policy.max_delay, policy.base_delay * (2 ** (attempt - 1)))
    jitter_value = jitter()
    if not math.isfinite(jitter_value):
        jitter_value = 0.0
    return ceiling * min(1.0, max(0.0, jitter_value))


def _retry_after_seconds(
    value: str | None,
    *,
    now: datetime | None = None,
) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        current_time = now or datetime.now(UTC)
        seconds = (retry_at - current_time).total_seconds()
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _mapped_http_error(
    exc: httpx.HTTPStatusError,
    *,
    provider: str,
    model: str,
    response: httpx.Response | None = None,
) -> LLMError:
    status_code = (
        exc.response.status_code
        if exc.response is not None
        else getattr(response, "status_code", None)
    )
    headers = getattr(response, "headers", {})
    if exc.response is not None:
        headers = exc.response.headers
    retry_after = headers.get("retry-after") if headers else None
    provider_request_id = headers.get("x-request-id") if headers else None
    return LLMError(
        _http_error_category(status_code),
        f"The LLM provider returned HTTP {status_code or 'error'}.",
        provider=provider,
        model=model,
        status_code=status_code,
        retryable=status_code == 429,
        retry_after=retry_after,
        provider_request_id=provider_request_id,
    )


def _http_error_category(status_code: int | None) -> LLMErrorCategory:
    if status_code in (401, 403):
        return LLMErrorCategory.AUTHENTICATION
    if status_code == 402:
        return LLMErrorCategory.INSUFFICIENT_CREDITS
    if status_code == 413:
        return LLMErrorCategory.CONTEXT_LENGTH
    if status_code == 429:
        return LLMErrorCategory.RATE_LIMIT
    if status_code in (408, 504):
        return LLMErrorCategory.OUTCOME_UNKNOWN
    if status_code in (400, 422):
        return LLMErrorCategory.INVALID_REQUEST
    return LLMErrorCategory.UPSTREAM


def _response_data(
    response: httpx.Response,
    *,
    provider: str,
    model: str,
    attempts: int = 1,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise LLMError(
            LLMErrorCategory.INVALID_RESPONSE,
            "The LLM provider returned invalid JSON.",
            provider=provider,
            model=model,
            provider_request_id=getattr(response, "headers", {}).get("x-request-id"),
            attempts=attempts,
            duration_seconds=duration_seconds,
        ) from exc

    if not isinstance(data, dict):
        raise LLMError(
            LLMErrorCategory.INVALID_RESPONSE,
            "The LLM provider returned an invalid response.",
            provider=provider,
            model=model,
            provider_request_id=getattr(response, "headers", {}).get("x-request-id"),
            attempts=attempts,
            duration_seconds=duration_seconds,
        )

    return data


def _usage_from_payload(value: Any) -> LLMUsage | None:
    if not isinstance(value, dict):
        return None

    def token_count(name: str) -> int | None:
        count = value.get(name)
        return count if isinstance(count, int) and not isinstance(count, bool) else None

    usage = LLMUsage(
        input_tokens=token_count("prompt_tokens"),
        output_tokens=token_count("completion_tokens"),
        total_tokens=token_count("total_tokens"),
    )
    if (
        usage.input_tokens is None
        and usage.output_tokens is None
        and usage.total_tokens is None
    ):
        return None
    return usage


def _ollama_usage_from_payload(data: dict[str, Any]) -> LLMUsage | None:
    input_tokens = data.get("prompt_eval_count")
    output_tokens = data.get("eval_count")
    input_tokens = (
        input_tokens
        if isinstance(input_tokens, int) and not isinstance(input_tokens, bool)
        else None
    )
    output_tokens = (
        output_tokens
        if isinstance(output_tokens, int) and not isinstance(output_tokens, bool)
        else None
    )
    if input_tokens is None and output_tokens is None:
        return None
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _cost_from_payload(value: Any) -> Decimal | None:
    if not isinstance(value, (str, int, float, Decimal)) or isinstance(value, bool):
        return None
    try:
        cost = Decimal(str(value))
    except InvalidOperation:
        return None
    return cost if cost.is_finite() else None


class OllamaLLMService(LLMService):
    provider = "ollama"

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        retry_policy: LLMRetryPolicy | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_policy = retry_policy or LLMRetryPolicy()
        self.capabilities = LLMCapabilities()

    def _post(
        self,
        payload: dict[str, Any],
        timeout_budget: float,
    ) -> httpx.Response:
        """Single network seam, replaced by tests to avoid a live Ollama."""
        return httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=min(self.timeout, timeout_budget),
        )

    def call(self, request: LLMRequest) -> LLMResponse:
        prompt = "\n\n".join(
            part for part in (request.system_prompt, request.prompt) if part
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if request.response_schema is not None:
            if self.capabilities.supports_structured_output:
                payload["format"] = request.response_schema
            elif self.capabilities.supports_json_mode:
                payload["format"] = "json"
        response = _post_with_error_mapping(
            self._post,
            payload,
            provider=self.provider,
            model=self.model,
            retry_policy=self.retry_policy,
        )
        post_result = response
        response = post_result.response
        data = _response_data(
            response,
            provider=self.provider,
            model=self.model,
            attempts=post_result.attempts,
            duration_seconds=post_result.duration_seconds,
        )
        text = data.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMError(
                LLMErrorCategory.INVALID_RESPONSE,
                "The LLM provider response did not contain answer text.",
                provider=self.provider,
                model=self.model,
                attempts=post_result.attempts,
                duration_seconds=post_result.duration_seconds,
            )

        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            usage=_ollama_usage_from_payload(data),
            attempts=post_result.attempts,
            duration_seconds=post_result.duration_seconds,
        )


class OpenRouterLLMService(LLMService):
    provider = "openrouter"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openrouter.ai",
        api_key: str | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        capabilities: LLMCapabilities | None = None,
        retry_policy: LLMRetryPolicy | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.capabilities = capabilities or LLMCapabilities()
        self.retry_policy = retry_policy or LLMRetryPolicy()

    def _post(
        self,
        payload: dict[str, Any],
        timeout_budget: float,
    ) -> httpx.Response:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = httpx.Timeout(
            connect=min(self.connect_timeout, timeout_budget),
            read=min(self.read_timeout, timeout_budget),
            write=min(self.read_timeout, timeout_budget),
            pool=min(self.connect_timeout, timeout_budget),
        )
        return httpx.post(
            f"{self.base_url}/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )

    def call(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key or not self.api_key.strip():
            raise LLMError(
                LLMErrorCategory.AUTHENTICATION,
                "OPENROUTER_API_KEY is required to use the OpenRouter provider.",
                provider=self.provider,
                model=self.model,
            )
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if request.response_schema is not None:
            if self.capabilities.supports_structured_output:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": True,
                        "schema": request.response_schema,
                    },
                }
                payload["provider"] = {"require_parameters": True}
            elif self.capabilities.supports_json_mode:
                payload["response_format"] = {"type": "json_object"}
        response = _post_with_error_mapping(
            self._post,
            payload,
            provider=self.provider,
            model=self.model,
            retry_policy=self.retry_policy,
        )
        post_result = response
        response = post_result.response
        data = _response_data(
            response,
            provider=self.provider,
            model=self.model,
            attempts=post_result.attempts,
            duration_seconds=post_result.duration_seconds,
        )
        choices = data.get("choices")
        content: Any = None
        refused = False
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
                refused = bool(message.get("refusal"))
        if refused:
            return LLMResponse(
                text="",
                provider=self.provider,
                model=self.model,
                provider_request_id=_request_id(data, response),
                usage=_usage_from_payload(data.get("usage")),
                refused=True,
                attempts=post_result.attempts,
                duration_seconds=post_result.duration_seconds,
            )
        if isinstance(content, dict):
            content = content.get("text")
        if not isinstance(content, str) or not content.strip():
            raise LLMError(
                LLMErrorCategory.INVALID_RESPONSE,
                "The LLM provider response did not contain answer text.",
                provider=self.provider,
                model=self.model,
                provider_request_id=_request_id(data, response),
                attempts=post_result.attempts,
                duration_seconds=post_result.duration_seconds,
            )

        request_id = _request_id(data, response)
        usage = data.get("usage")
        cost = usage.get("cost") if isinstance(usage, dict) else None

        return LLMResponse(
            text=content,
            provider=self.provider,
            model=self.model,
            usage=_usage_from_payload(usage),
            cost=_cost_from_payload(cost),
            provider_request_id=request_id,
            attempts=post_result.attempts,
            duration_seconds=post_result.duration_seconds,
        )


def _request_id(data: dict[str, Any], response: httpx.Response) -> str | None:
    for value in (
        data.get("id"),
        data.get("request_id"),
        getattr(response, "headers", {}).get("x-request-id"),
    ):
        if isinstance(value, str) and value:
            return value
    return None
