import pytest
from qdrant_client import QdrantClient

from app.embeddings import EmbeddingService, HuggingFaceEmbeddingService
from app.indexing import ChunkIndexer
from app.models import Document, DocumentChunk, DocumentPage
from app.qdrant_store import QdrantVectorStore
from app.vector_store import VectorStore


class FakeEmbeddingService(EmbeddingService):
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]


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
        document_id: str | None = None,
    ) -> list[dict]:
        results = []

        for vector_id, item in self.items.items():
            payload = item["payload"]

            if document_id is not None and payload.get("document_id") != document_id:
                continue

            results.append(
                {
                    "id": vector_id,
                    "score": item.get("score", 1.0),
                    "payload": payload,
                }
            )

            if len(results) >= limit:
                break

        return results

    def delete(self, vector_id: str) -> None:
        self.items.pop(vector_id, None)

    def delete_by_page(
        self,
        page_id: str,
        exclude_ids: set[str] | None = None,
    ) -> None:
        exclude_ids = exclude_ids or set()

        self.items = {
            vector_id: item
            for vector_id, item in self.items.items()
            if (item["payload"].get("page_id") != page_id or vector_id in exclude_ids)
        }

    def delete_by_document(self, document_id: str) -> None:
        self.items = {
            vector_id: item
            for vector_id, item in self.items.items()
            if item["payload"].get("document_id") != document_id
        }


def test_index_chunks(
    session,
):
    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.flush()

    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Test document",
    )
    session.add(page)
    session.flush()

    chunk = DocumentChunk(
        page_id=page.id,
        page_number=1,
        chunk_index=0,
        text="Georgian document text",
    )
    session.add(chunk)
    session.flush()

    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    indexer = ChunkIndexer(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    indexer.index_chunks([chunk])

    assert chunk.id in vector_store.items

    item = vector_store.items[chunk.id]

    assert item["vector"] == [1.0, 0.0, 0.0]

    assert item["payload"] == {
        "document_id": document.id,
        "page_id": page.id,
        "page_number": 1,
        "chunk_index": 0,
        "text": "Georgian document text",
    }


def test_index_chunks_reports_embedding_and_qdrant_stage_changes(session):
    document = Document(filename="test.pdf", file_hash="document-hash")
    session.add(document)
    session.flush()
    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Test document",
    )
    session.add(page)
    session.flush()
    chunk = DocumentChunk(
        page_id=page.id,
        page_number=1,
        chunk_index=0,
        text="Georgian document text",
    )
    session.add(chunk)
    session.flush()

    events = []
    indexer = ChunkIndexer(FakeEmbeddingService(), FakeVectorStore())
    indexer.index_chunks(
        [chunk],
        progress_callback=lambda step, status: events.append((step, status)),
    )

    assert events == [
        ("embeddings", "processing"),
        ("embeddings", "completed"),
        ("indexing", "processing"),
        ("indexing", "completed"),
    ]


def test_index_chunks_creates_one_vector_per_chunk(
    session,
):
    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.flush()

    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Test document",
    )
    session.add(page)
    session.flush()

    chunks = [
        DocumentChunk(
            page_id=page.id,
            page_number=1,
            chunk_index=index,
            text=f"Chunk {index}",
        )
        for index in range(3)
    ]

    session.add_all(chunks)
    session.flush()

    vector_store = FakeVectorStore()

    indexer = ChunkIndexer(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    indexer.index_chunks(chunks)

    assert len(vector_store.items) == 3

    for chunk in chunks:
        assert chunk.id in vector_store.items


def test_index_chunks_does_nothing_for_empty_list():
    vector_store = FakeVectorStore()

    indexer = ChunkIndexer(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    indexer.index_chunks([])

    assert vector_store.items == {}


def test_index_chunks_is_idempotent(session):
    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.flush()

    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Test document",
    )
    session.add(page)
    session.flush()

    chunk = DocumentChunk(
        page_id=page.id,
        page_number=1,
        chunk_index=0,
        text="Original text",
    )
    session.add(chunk)
    session.flush()

    vector_store = FakeVectorStore()

    indexer = ChunkIndexer(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    indexer.index_chunks([chunk])
    indexer.index_chunks([chunk])

    assert len(vector_store.items) == 1
    assert chunk.id in vector_store.items


def test_index_chunks_with_real_services(session):
    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.flush()

    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Test document",
    )
    session.add(page)
    session.flush()

    chunk = DocumentChunk(
        page_id=page.id,
        page_number=1,
        chunk_index=0,
        text="საქართველოს კანონი",
    )
    session.add(chunk)
    session.flush()

    embedding_service = HuggingFaceEmbeddingService()

    client = QdrantClient(":memory:")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_size=1024,
    )

    indexer = ChunkIndexer(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    indexer.index_chunks([chunk])

    results = vector_store.search(
        vector=embedding_service.embed(chunk.text),
        limit=1,
    )

    assert len(results) == 1
    assert results[0]["id"] == chunk.id
    assert results[0]["payload"]["text"] == chunk.text
    assert results[0]["payload"]["page_number"] == 1


"""повторная индексация удаляет старые vectors перед созданием новых"""


def test_index_chunks_replaces_old_vectors(session):
    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.flush()

    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Test document",
    )
    session.add(page)
    session.flush()

    old_chunks = [
        DocumentChunk(
            page_id=page.id,
            page_number=1,
            chunk_index=0,
            text="Old chunk 1",
        ),
        DocumentChunk(
            page_id=page.id,
            page_number=1,
            chunk_index=1,
            text="Old chunk 2",
        ),
    ]

    session.add_all(old_chunks)
    session.flush()

    vector_store = FakeVectorStore()

    indexer = ChunkIndexer(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    indexer.index_chunks(old_chunks)

    assert len(vector_store.items) == 2

    new_chunks = [
        DocumentChunk(
            page_id=page.id,
            page_number=1,
            chunk_index=0,
            text="New chunk 1",
        ),
        DocumentChunk(
            page_id=page.id,
            page_number=1,
            chunk_index=1,
            text="New chunk 2",
        ),
        DocumentChunk(
            page_id=page.id,
            page_number=1,
            chunk_index=2,
            text="New chunk 3",
        ),
    ]

    session.add_all(new_chunks)
    session.flush()

    indexer.index_chunks(new_chunks)

    assert len(vector_store.items) == 3

    for chunk in old_chunks:
        assert chunk.id not in vector_store.items

    for chunk in new_chunks:
        assert chunk.id in vector_store.items


class BrokenEmbeddingService(EmbeddingService):
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0]]


def test_index_chunks_raises_when_embedding_count_does_not_match(
    session,
):
    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.flush()

    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Test document",
    )
    session.add(page)
    session.flush()

    chunks = [
        DocumentChunk(
            page_id=page.id,
            page_number=1,
            chunk_index=0,
            text="Chunk 1",
        ),
        DocumentChunk(
            page_id=page.id,
            page_number=1,
            chunk_index=1,
            text="Chunk 2",
        ),
    ]

    session.add_all(chunks)
    session.flush()

    indexer = ChunkIndexer(
        embedding_service=BrokenEmbeddingService(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="Number of embeddings does not match number of chunks",
    ):
        indexer.index_chunks(chunks)


class FailingEmbeddingService(EmbeddingService):
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("Embedding service failed")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Embedding service failed")


def test_index_chunks_keeps_old_vectors_when_embedding_fails(
    session,
):
    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.flush()

    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Test document",
    )
    session.add(page)
    session.flush()

    old_chunk = DocumentChunk(
        page_id=page.id,
        page_number=1,
        chunk_index=0,
        text="Old chunk",
    )
    session.add(old_chunk)
    session.flush()

    vector_store = FakeVectorStore()

    working_indexer = ChunkIndexer(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    working_indexer.index_chunks([old_chunk])

    assert len(vector_store.items) == 1
    assert old_chunk.id in vector_store.items

    new_chunk = DocumentChunk(
        page_id=page.id,
        page_number=1,
        chunk_index=0,
        text="New chunk",
    )
    session.add(new_chunk)
    session.flush()

    failing_indexer = ChunkIndexer(
        embedding_service=FailingEmbeddingService(),
        vector_store=vector_store,
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding service failed",
    ):
        failing_indexer.index_chunks([new_chunk])

    assert len(vector_store.items) == 1
    assert old_chunk.id in vector_store.items
    assert new_chunk.id not in vector_store.items


class FailingVectorStore(FakeVectorStore):
    def __init__(self, fail_on_upsert: int):
        super().__init__()
        self.fail_on_upsert = fail_on_upsert
        self.upsert_count = 0

    def upsert(
        self,
        vector_id: str,
        vector: list[float],
        payload: dict,
    ) -> None:
        self.upsert_count += 1

        if self.upsert_count == self.fail_on_upsert:
            raise RuntimeError("Vector store upsert failed")

        super().upsert(
            vector_id=vector_id,
            vector=vector,
            payload=payload,
        )


def test_index_chunks_rolls_back_new_vectors_when_upsert_fails(
    session,
):
    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.flush()

    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Test document",
    )
    session.add(page)
    session.flush()

    old_chunk = DocumentChunk(
        page_id=page.id,
        page_number=1,
        chunk_index=0,
        text="Old chunk",
    )
    session.add(old_chunk)
    session.flush()

    new_chunks = [
        DocumentChunk(
            page_id=page.id,
            page_number=1,
            chunk_index=index,
            text=f"New chunk {index}",
        )
        for index in range(3)
    ]

    session.add_all(new_chunks)
    session.flush()

    failing_store = FailingVectorStore(
        fail_on_upsert=2,
    )

    # Создаём старый vector.
    failing_store.upsert(
        vector_id=old_chunk.id,
        vector=[1.0, 0.0, 0.0],
        payload={
            "document_id": document.id,
            "page_id": page.id,
            "page_number": 1,
            "chunk_index": 0,
            "text": old_chunk.text,
        },
    )

    # Не учитываем создание старого vector
    # при подсчёте ошибки.
    failing_store.upsert_count = 0

    indexer = ChunkIndexer(
        embedding_service=FakeEmbeddingService(),
        vector_store=failing_store,
    )

    with pytest.raises(
        RuntimeError,
        match="Vector store upsert failed",
    ):
        indexer.index_chunks(new_chunks)

    # Старый vector должен сохраниться.
    assert old_chunk.id in failing_store.items

    # Новые vectors должны быть полностью удалены rollback'ом.
    assert new_chunks[0].id not in failing_store.items
    assert new_chunks[1].id not in failing_store.items
    assert new_chunks[2].id not in failing_store.items

    assert len(failing_store.items) == 1
