from dataclasses import dataclass

from app.embeddings import EmbeddingService
from app.vector_store import VectorStore


@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict


class Retriever:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        if limit < 1:
            raise ValueError("limit must be greater than 0")

        vector = self.embedding_service.embed(query)

        if document_ids is None:
            results = self.vector_store.search(
                vector=vector,
                limit=limit,
                document_id=document_id,
            )
        else:
            results = self.vector_store.search(
                vector=vector,
                limit=limit,
                document_id=document_id,
                document_ids=document_ids,
            )

        search_results = [
            SearchResult(
                id=result["id"],
                score=result["score"],
                payload=result["payload"],
            )
            for result in results
        ]

        if score_threshold is not None:
            search_results = [
                result for result in search_results if result.score >= score_threshold
            ]

        return search_results
