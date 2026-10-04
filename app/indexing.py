from app.embeddings import EmbeddingService
from app.models import DocumentChunk
from app.vector_store import VectorStore


class ChunkIndexer:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        if not chunks:
            return

        page_id = chunks[0].page_id

        if any(chunk.page_id != page_id for chunk in chunks):
            raise ValueError(
                "All chunks must belong to the same page"
            )

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedding_service.embed_many(texts)

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings does not match number of chunks"
            )

        self.vector_store.delete_by_page(page_id)

        for chunk, embedding in zip(chunks, embeddings):
            self.vector_store.upsert(
                vector_id=chunk.id,
                vector=embedding,
                payload={
                    "document_id": chunk.page.document_id,
                    "page_id": chunk.page_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )