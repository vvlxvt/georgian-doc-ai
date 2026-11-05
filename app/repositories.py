from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app.models import Conversation, DocumentChunk, Message, MessageRole


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


class ConversationRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, document_id: str, title: str = "New conversation") -> Conversation:
        conversation = Conversation(document_id=document_id, title=title)
        self.session.add(conversation)
        self.session.flush()
        return conversation

    def list_for_document(self, document_id: str) -> list[Conversation]:
        return self.session.scalars(
            select(Conversation)
            .where(Conversation.document_id == document_id)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        ).all()

    def list_all(self) -> list[Conversation]:
        return self.session.scalars(
            select(Conversation)
            .options(selectinload(Conversation.document))
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        ).all()

    def get(self, conversation_id: str) -> Conversation | None:
        return self.session.scalar(
            select(Conversation)
            .options(
                selectinload(Conversation.document),
                selectinload(Conversation.messages),
            )
            .where(Conversation.id == conversation_id)
        )

    def update_title(self, conversation: Conversation, title: str) -> None:
        conversation.title = title
        conversation.updated_at = datetime.now(UTC)

    def delete(self, conversation: Conversation) -> None:
        self.session.delete(conversation)

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        sources_json: str | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources_json=sources_json,
        )
        self.session.add(message)
        conversation = self.session.get(Conversation, conversation_id)
        if conversation is not None:
            conversation.updated_at = datetime.now(UTC)
        return message

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int,
    ) -> list[Message]:
        if limit < 1:
            raise ValueError("limit must be greater than 0")

        latest = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
            .subquery()
        )
        return self.session.scalars(
            select(Message)
            .join(latest, Message.id == latest.c.id)
            .order_by(Message.created_at, Message.id)
        ).all()
