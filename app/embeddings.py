from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError