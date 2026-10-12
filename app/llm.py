from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
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
    ) -> None:
        super().__init__(message)
        self.category = category
        self.provider = provider
        self.model = model
        self.provider_request_id = provider_request_id
        self.status_code = status_code


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    system_prompt: str | None = None
    prompt_version: str | None = None


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
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: LLMUsage | None = None
    cost: Decimal | None = None
    provider_request_id: str | None = None


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
    post: Callable[[dict[str, Any]], httpx.Response],
    payload: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> httpx.Response:
    try:
        response = post(payload)
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
        ) from exc
    except (httpx.ReadTimeout, httpx.WriteTimeout) as exc:
        raise LLMError(
            LLMErrorCategory.OUTCOME_UNKNOWN,
            "The provider request outcome is unknown after a network timeout.",
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
    return LLMError(
        _http_error_category(status_code),
        f"The LLM provider returned HTTP {status_code or 'error'}.",
        provider=provider,
        model=model,
        status_code=status_code,
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
) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise LLMError(
            LLMErrorCategory.INVALID_RESPONSE,
            "The LLM provider returned invalid JSON.",
            provider=provider,
            model=model,
        ) from exc

    if not isinstance(data, dict):
        raise LLMError(
            LLMErrorCategory.INVALID_RESPONSE,
            "The LLM provider returned an invalid response.",
            provider=provider,
            model=model,
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
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.capabilities = LLMCapabilities()

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """Single network seam, replaced by tests to avoid a live Ollama."""
        return httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
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
        response = _post_with_error_mapping(
            self._post,
            payload,
            provider=self.provider,
            model=self.model,
        )
        data = _response_data(response, provider=self.provider, model=self.model)
        text = data.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMError(
                LLMErrorCategory.INVALID_RESPONSE,
                "The LLM provider response did not contain answer text.",
                provider=self.provider,
                model=self.model,
            )

        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            usage=_ollama_usage_from_payload(data),
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
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.capabilities = capabilities or LLMCapabilities()

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.read_timeout,
            pool=self.connect_timeout,
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
        response = _post_with_error_mapping(
            self._post,
            payload,
            provider=self.provider,
            model=self.model,
        )
        data = _response_data(response, provider=self.provider, model=self.model)
        choices = data.get("choices")
        content: Any = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
        if isinstance(content, dict):
            content = content.get("text")
        if not isinstance(content, str) or not content.strip():
            raise LLMError(
                LLMErrorCategory.INVALID_RESPONSE,
                "The LLM provider response did not contain answer text.",
                provider=self.provider,
                model=self.model,
            )

        request_id = (
            data.get("id")
            or data.get("request_id")
            or getattr(response, "headers", {}).get("x-request-id")
        )
        if not isinstance(request_id, str):
            request_id = None
        usage = data.get("usage")
        cost = usage.get("cost") if isinstance(usage, dict) else None

        return LLMResponse(
            text=content,
            provider=self.provider,
            model=self.model,
            usage=_usage_from_payload(usage),
            cost=_cost_from_payload(cost),
            provider_request_id=request_id,
        )
