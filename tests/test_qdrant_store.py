from uuid import uuid4

from qdrant_client import QdrantClient

from app.qdrant_store import QdrantVectorStore

VECTOR_SIZE = 3


def create_store():
    client = QdrantClient(":memory:")

    return QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_size=VECTOR_SIZE,
    )


def test_qdrant_store_creates_collection():
    store = create_store()

    collections = store.client.get_collections()

    names = [collection.name for collection in collections.collections]

    assert "test_chunks" in names


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
