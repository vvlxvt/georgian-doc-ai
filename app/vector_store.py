from abc import ABC, abstractmethod


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        vector_id: str,
        vector: list[float],
        payload: dict,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        vector: list[float],
        limit: int = 5,
    ) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, vector_id: str) -> None:
        raise NotImplementedError
