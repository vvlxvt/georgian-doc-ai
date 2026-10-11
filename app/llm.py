from abc import ABC, abstractmethod

import httpx


class LLMService(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


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

    def generate(self, prompt: str) -> str:
        response = self._post(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
        )

        response.raise_for_status()

        data = response.json()

        return data["response"]
