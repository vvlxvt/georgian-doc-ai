from types import SimpleNamespace
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.qdrant_store import QdrantVectorStore

VECTOR_SIZE = 3


def create_store():
    client = QdrantClient(":memory:")

    return QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_size=VECTOR_SIZE,
    )


def test_qdrant_store_creates_collection(monkeypatch):
    client = QdrantClient(":memory:")
    indexed_fields = []
    monkeypatch.setattr(
        client,
        "create_payload_index",
        lambda **kwargs: indexed_fields.append(kwargs),
    )
    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_size=VECTOR_SIZE,
    )

    collections = store.client.get_collections()

    names = [collection.name for collection in collections.collections]

    assert "test_chunks" in names
    assert {item["field_name"] for item in indexed_fields} == {
        "document_id",
        "page_id",
    }


def test_qdrant_store_accepts_compatible_existing_collection():
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_chunks",
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_size=VECTOR_SIZE,
    )

    assert store.collection_name == "test_chunks"


def test_qdrant_store_accepts_remote_unnamed_vector_config(monkeypatch):
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_chunks",
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    collection = client.get_collection("test_chunks")
    monkeypatch.setattr(
        client,
        "get_collection",
        lambda collection_name: SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={"": collection.config.params.vectors},
                )
            )
        ),
    )

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_size=VECTOR_SIZE,
    )

    assert store.collection_name == "test_chunks"


def test_qdrant_store_rejects_named_remote_vector_config(monkeypatch):
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_chunks",
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    collection = client.get_collection("test_chunks")
    monkeypatch.setattr(
        client,
        "get_collection",
        lambda collection_name: SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={"named": collection.config.params.vectors},
                )
            )
        ),
    )

    with pytest.raises(ValueError, match="single unnamed vector"):
        QdrantVectorStore(
            client=client,
            collection_name="test_chunks",
            vector_size=VECTOR_SIZE,
        )


@pytest.mark.parametrize(
    ("existing_size", "existing_distance"),
    [
        (4, Distance.COSINE),
        (VECTOR_SIZE, Distance.EUCLID),
    ],
)
def test_qdrant_store_rejects_incompatible_existing_collection(
    existing_size,
    existing_distance,
):
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_chunks",
        vectors_config=VectorParams(size=existing_size, distance=existing_distance),
    )

    with pytest.raises(ValueError, match="is incompatible"):
        QdrantVectorStore(
            client=client,
            collection_name="test_chunks",
            vector_size=VECTOR_SIZE,
        )


def test_qdrant_store_upsert_and_search():
    store = create_store()

    vector_id = str(uuid4())

    store.upsert(
        vector_id=vector_id,
        vector=[1.0, 0.0, 0.0],
        payload={
            "document_id": "doc-1",
            "page_number": 1,
            "text": "თბილისი საქართველოს დედაქალაქია",
        },
    )

    results = store.search(
        vector=[1.0, 0.0, 0.0],
        limit=5,
    )

    assert len(results) == 1

    result = results[0]

    assert result["id"] == vector_id
    assert result["payload"]["document_id"] == "doc-1"
    assert result["payload"]["page_number"] == 1
    assert result["payload"]["text"] == "თბილისი საქართველოს დედაქალაქია"


def test_qdrant_store_returns_most_similar_vector_first():
    store = create_store()

    similar_id = str(uuid4())
    different_id = str(uuid4())

    store.upsert(
        vector_id=similar_id,
        vector=[1.0, 0.0, 0.0],
        payload={"text": "similar"},
    )

    store.upsert(
        vector_id=different_id,
        vector=[0.0, 1.0, 0.0],
        payload={"text": "different"},
    )

    results = store.search(
        vector=[1.0, 0.0, 0.0],
        limit=2,
    )

    assert results[0]["id"] == similar_id
    assert results[0]["payload"]["text"] == "similar"
    assert results[0]["score"] > results[1]["score"]


def test_qdrant_store_delete():
    store = create_store()

    vector_id = str(uuid4())

    store.upsert(
        vector_id=vector_id,
        vector=[1.0, 0.0, 0.0],
        payload={"text": "test"},
    )

    store.delete(vector_id)

    results = store.search(
        vector=[1.0, 0.0, 0.0],
        limit=5,
    )

    assert results == []


def test_delete_by_page():
    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_size=3,
    )

    store.upsert(
        vector_id=str(uuid4()),
        vector=[1.0, 0.0, 0.0],
        payload={
            "page_id": "page-1",
            "text": "first",
        },
    )

    store.upsert(
        vector_id=str(uuid4()),
        vector=[0.0, 1.0, 0.0],
        payload={
            "page_id": "page-1",
            "text": "second",
        },
    )

    third_id = str(uuid4())

    store.upsert(
        vector_id=third_id,
        vector=[0.0, 0.0, 1.0],
        payload={
            "page_id": "page-2",
            "text": "third",
        },
    )

    store.delete_by_page("page-1")

    results = store.search(
        vector=[0.0, 0.0, 1.0],
        limit=10,
    )

    assert len(results) == 1
    assert results[0]["id"] == third_id
