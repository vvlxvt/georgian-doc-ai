from app.embeddings import EmbeddingService
from app.retrieval import Retriever
from app.vector_store import VectorStore


class FakeEmbeddingService(EmbeddingService):
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.items = {}

    def upsert(self, vector_id, vector, payload):
        self.items[vector_id] = {
            "vector": vector,
            "payload": payload,
        }

    def search(
        self,
        vector,
        limit=5,
        score_threshold=None,
        document_id=None,
        document_ids=None,
    ):
        results = [
            {
                "id": vector_id,
                "score": item["score"],
                "payload": item["payload"],
            }
            for vector_id, item in self.items.items()
        ]

        if document_id is not None:
            results = [
                result
                for result in results
                if result["payload"].get("document_id") == document_id
            ]
        if document_ids is not None:
            results = [
                result
                for result in results
                if result["payload"].get("document_id") in document_ids
            ]

        if score_threshold is not None:
            results = [
                result for result in results if result["score"] >= score_threshold
            ]

        return results[:limit]

    def delete(self, vector_id):
        self.items.pop(vector_id, None)

    def delete_by_page(
        self,
        page_id,
        exclude_ids=None,
    ):
        exclude_ids = exclude_ids or set()

        self.items = {
            vector_id: item
            for vector_id, item in self.items.items()
            if (item["payload"].get("page_id") != page_id or vector_id in exclude_ids)
        }


def test_retriever_search():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    vector_store.upsert(
        vector_id="chunk-1",
        vector=[1.0, 0.0, 0.0],
        payload={
            "page_id": "page-1",
            "text": "საქართველოს კანონი",
        },
    )

    vector_store.items["chunk-1"]["score"] = 1.0

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = retriever.search(
        query="Что говорит закон?",
        limit=5,
    )

    assert len(results) == 1
    assert results[0].id == "chunk-1"
    assert results[0].score is not None
    assert results[0].payload["text"] == "საქართველოს კანონი"


def test_retriever_filters_by_score_threshold():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    vector_store.upsert(
        vector_id="relevant",
        vector=[1.0, 0.0, 0.0],
        payload={"text": "Relevant text"},
    )
    vector_store.items["relevant"]["score"] = 0.9

    vector_store.upsert(
        vector_id="weak",
        vector=[0.0, 1.0, 0.0],
        payload={"text": "Weak match"},
    )
    vector_store.items["weak"]["score"] = 0.3

    retriever = Retriever(embedding_service, vector_store)

    results = retriever.search(
        query="question",
        score_threshold=0.7,
    )

    assert len(results) == 1
    assert results[0].id == "relevant"


def test_retriever_filters_by_document_id():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    vector_store.upsert(
        vector_id="chunk-1",
        vector=[1.0, 0.0, 0.0],
        payload={
            "document_id": "doc-1",
            "text": "Text from document 1",
        },
    )
    vector_store.upsert(
        vector_id="chunk-2",
        vector=[1.0, 0.0, 0.0],
        payload={
            "document_id": "doc-2",
            "text": "Text from document 2",
        },
    )

    vector_store.items["chunk-1"]["score"] = 1.0
    vector_store.items["chunk-2"]["score"] = 1.0

    retriever = Retriever(embedding_service, vector_store)

    results = retriever.search(
        query="question",
        document_id="doc-1",
    )

    assert len(results) == 1
    assert results[0].id == "chunk-1"


def test_retriever_filters_by_multiple_document_ids():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    for index, document_id in enumerate(("doc-1", "doc-2", "doc-3")):
        vector_id = f"chunk-{document_id}"
        vector_store.upsert(
            vector_id=vector_id,
            vector=[1.0, 0.0, 0.0],
            payload={"document_id": document_id, "text": document_id},
        )
        vector_store.items[vector_id]["score"] = 1.0

    retriever = Retriever(embedding_service, vector_store)

    results = retriever.search(
        query="question",
        document_ids=["doc-1", "doc-3"],
    )

    assert {result.payload["document_id"] for result in results} == {
        "doc-1",
        "doc-3",
    }
