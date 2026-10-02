from app.vector_store import VectorStore


def test_vector_store_is_abstract():
    try:
        VectorStore()
    except TypeError:
        return

    raise AssertionError("VectorStore should not be instantiated directly")


def test_vector_store_requires_upsert():
    class TestVectorStore(VectorStore):
        def search(self, vector, limit=5):
            return []

        def delete(self, vector_id):
            pass

    try:
        TestVectorStore()
    except TypeError:
        return

    raise AssertionError("VectorStore must require upsert() implementation")


def test_vector_store_requires_search():
    class TestVectorStore(VectorStore):
        def upsert(self, vector_id, vector, payload):
            pass

        def delete(self, vector_id):
            pass

    try:
        TestVectorStore()
    except TypeError:
        return

    raise AssertionError("VectorStore must require search() implementation")


def test_vector_store_requires_delete():
    class TestVectorStore(VectorStore):
        def upsert(self, vector_id, vector, payload):
            pass

        def search(self, vector, limit=5):
            return []

    try:
        TestVectorStore()
    except TypeError:
        return

    raise AssertionError("VectorStore must require delete() implementation")


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.items = {}

    def upsert(
        self,
        vector_id: str,
        vector: list[float],
        payload: dict,
    ) -> None:
        self.items[vector_id] = {
            "vector": vector,
            "payload": payload,
        }

    def search(
        self,
        vector: list[float],
        limit: int = 5,
    ) -> list[dict]:
        return list(self.items.values())[:limit]

    def delete(self, vector_id: str) -> None:
        self.items.pop(vector_id, None)


def test_vector_store_upsert_and_search():
    store = FakeVectorStore()

    vector = [0.1, 0.2, 0.3]
    payload = {
        "document_id": "doc-1",
        "page_number": 1,
        "text": "თბილისი საქართველოს დედაქალაქია",
    }

    store.upsert(
        vector_id="chunk-1",
        vector=vector,
        payload=payload,
    )

    results = store.search(
        vector=[0.1, 0.2, 0.3],
        limit=5,
    )

    assert len(results) == 1
    assert results[0]["vector"] == vector
    assert results[0]["payload"] == payload


def test_vector_store_delete():
    store = FakeVectorStore()

    store.upsert(
        vector_id="chunk-1",
        vector=[0.1, 0.2, 0.3],
        payload={"text": "test"},
    )

    store.delete("chunk-1")

    results = store.search(
        vector=[0.1, 0.2, 0.3],
    )

    assert results == []
