from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import DocumentChunk


class DocumentChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        page_id: str,
        page_number: int,
        chunk_index: int,
        text: str,
    ) -> DocumentChunk:
        chunk = DocumentChunk(
            page_id=page_id,
            page_number=page_number,
            chunk_index=chunk_index,
            text=text,
        )

        self.session.add(chunk)

        return chunk

    def create_many(
        self,
        page_id: str,
        page_number: int,
        chunks,
    ) -> list[DocumentChunk]:
        db_chunks = []

        for chunk in chunks:
            db_chunk = self.create(
                page_id=page_id,
                page_number=page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
            )

            db_chunks.append(db_chunk)

        return db_chunks

    def get_by_page(
        self,
        page_id: str,
    ) -> list[DocumentChunk]:
        return self.session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.page_id == page_id)
            .order_by(DocumentChunk.chunk_index)
        ).all()

    def delete_by_page(
        self,
        page_id: str,
    ) -> None:
        self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.page_id == page_id)
        )

    def delete_by_ids(self, chunk_ids: set[str]) -> None:
        if not chunk_ids:
            return

        self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
        )
