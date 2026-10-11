from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.vector_store import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections()

        exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if exists:
            self._validate_collection()
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def _validate_collection(self) -> None:
        collection = self.client.get_collection(self.collection_name)
        vectors = collection.config.params.vectors

        if not isinstance(vectors, VectorParams):
            raise ValueError(
                f"Collection '{self.collection_name}' must use a single unnamed vector"
            )

        if vectors.size != self.vector_size or vectors.distance != Distance.COSINE:
            raise ValueError(
                f"Collection '{self.collection_name}' is incompatible: expected "
                f"size={self.vector_size}, distance={Distance.COSINE.value}; got "
                f"size={vectors.size}, distance={vectors.distance.value}"
            )

    def upsert(
        self,
        vector_id: str,
        vector: list[float],
        payload: dict,
    ) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def search(
        self,
        vector: list[float],
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        query_filter = None

        if document_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    ),
                ],
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
        ).points

        return [
            {
                "id": str(result.id),
                "score": result.score,
                "payload": result.payload,
            }
            for result in results
        ]

    def delete(self, vector_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[vector_id],
        )

    def delete_by_page(
        self,
        page_id: str,
        exclude_ids: set[str] | None = None,
    ) -> None:
        exclude_ids = exclude_ids or set()

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="page_id",
                        match=MatchValue(value=page_id),
                    ),
                ],
                must_not=(
                    [
                        HasIdCondition(
                            has_id=list(exclude_ids),
                        ),
                    ]
                    if exclude_ids
                    else None
                ),
            ),
        )
