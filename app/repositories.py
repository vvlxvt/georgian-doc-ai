from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app.models import Chat, Document, DocumentChunk, Message, MessageRole


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


class ChatRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        document_ids: list[str],
        title: str = "New chat",
    ) -> Chat:
        selected_ids = list(dict.fromkeys(document_ids))
        if not selected_ids:
            raise ValueError("at least one document must be selected")

        documents = self.session.scalars(
            select(Document).where(Document.id.in_(selected_ids))
        ).all()
        documents_by_id = {document.id: document for document in documents}
        if len(documents_by_id) != len(selected_ids):
            raise ValueError("one or more selected documents were not found")

        chat = Chat(
            documents=[documents_by_id[document_id] for document_id in selected_ids],
            title=title,
        )
        self.session.add(chat)
        self.session.flush()
        return chat

    def list_for_document(self, document_id: str) -> list[Chat]:
        return self.session.scalars(
            select(Chat)
            .where(Chat.documents.any(Document.id == document_id))
            .options(selectinload(Chat.documents))
            .order_by(Chat.updated_at.desc(), Chat.id.desc())
        ).all()

    def list_all(self) -> list[Chat]:
        return self.session.scalars(
            select(Chat)
            .options(selectinload(Chat.documents))
            .order_by(Chat.updated_at.desc(), Chat.id.desc())
        ).all()

    def get(self, chat_id: str) -> Chat | None:
        return self.session.scalar(
            select(Chat)
            .options(
                selectinload(Chat.documents),
                selectinload(Chat.messages),
            )
            .where(Chat.id == chat_id)
        )

    def update_title(self, chat: Chat, title: str) -> None:
        chat.title = title
        chat.updated_at = datetime.now(UTC)

    def delete(self, chat: Chat) -> None:
        self.session.delete(chat)

    def add_message(
        self,
        chat_id: str,
        role: MessageRole,
        content: str,
        sources_json: str | None = None,
    ) -> Message:
        message = Message(
            chat_id=chat_id,
            role=role,
            content=content,
            sources_json=sources_json,
        )
        self.session.add(message)
        chat = self.session.get(Chat, chat_id)
        if chat is not None:
            chat.updated_at = datetime.now(UTC)
        return message

    def get_recent_messages(
        self,
        chat_id: str,
        limit: int,
    ) -> list[Message]:
        if limit < 1:
            raise ValueError("limit must be greater than 0")

        latest = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
            .subquery()
        )
        return self.session.scalars(
            select(Message)
            .join(latest, Message.id == latest.c.id)
            .order_by(Message.created_at, Message.id)
        ).all()
