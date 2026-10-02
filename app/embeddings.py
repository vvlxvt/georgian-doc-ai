from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer


class EmbeddingService(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HuggingFaceEmbeddingService(EmbeddingService):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
    ):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()
