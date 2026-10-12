from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import httpx
from typing import Any


@dataclass
class LLMResponse:
    text: str
    usage: dict | None = None
    provider_request_id: str | None = None
    raw: dict | None = None


class LLMService(ABC):
    """Provider-neutral LLM service contract.

    Implementations should provide `call(prompt)` returning an LLMResponse.
    For compatibility `generate(prompt)` remains and returns the textual answer.
    """

    @abstractmethod
    def call(self, prompt: str) -> LLMResponse:
        raise NotImplementedError

    def generate(self, prompt: str) -> str:
        return self.call(prompt).text


class OllamaLLMService(LLMService):
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, payload: dict) -> httpx.Response:
        """Single network seam, replaced by tests to avoid a live Ollama."""
        return httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )

    def call(self, prompt: str) -> LLMResponse:
        response = self._post(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
        )

        response.raise_for_status()

        data = response.json()

        return LLMResponse(text=data.get("response", ""), raw=data)


class OpenRouterLLMService(LLMService):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openrouter.ai",
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, payload: dict, headers: dict | None = None) -> httpx.Response:
        headers = headers or {}
        if self.api_key:
            headers = {**headers, "Authorization": f"Bearer {self.api_key}"}
        return httpx.post(
            f"{self.base_url}/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

    def call(self, prompt: str) -> LLMResponse:
        # Minimal compatibility wrapper; keep the payload conservative.
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        response = self._post(payload)
        response.raise_for_status()
        data = response.json()

        # best-effort extraction: prefer chat-style choices -> message -> content
        text = ""
        try:
            choices = data.get("choices") or []
            text = ""
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, dict):
                    text = content.get("text", "")
                elif isinstance(content, str):
                    text = content
        except Exception:
            text = ""

        provider_request_id = data.get("id") or data.get("request_id")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None

        return LLMResponse(text=text, usage=usage, provider_request_id=provider_request_id, raw=data)
