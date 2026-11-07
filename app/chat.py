from __future__ import annotations

import json
from dataclasses import dataclass

from app.answer import ChatAnswerResult
from app.context import SourceReference
from app.models import Message, MessageRole
from app.rag import RAGPipeline
from app.repositories import ChatRepository

CHAT_MAX_HISTORY_TURNS = 5


@dataclass(frozen=True)
class ChatSendResult:
    """Result of sending a message in a chat.

    `error` is set when the LLM provider failed; the user message is still
    persisted and no assistant message is fabricated.
    """

    user_message: Message
    assistant_message: Message | None
    sources: tuple[SourceReference, ...]
    error: str | None


class ChatService:
    """Orchestrates a multi-document chat turn.

    Responsibilities:
      * validate the chat and its selected documents,
      * persist the user message,
      * load a bounded number of recent turns (history, kept out of evidence),
      * call the existing RAG pipeline scoped to the chat's documents,
      * persist the assistant answer together with validated source references.

    Source text and prompts are never persisted.
    """

    def __init__(
        self,
        session,
        chat_repository: ChatRepository,
        rag_pipeline: RAGPipeline,
        max_history_turns: int = CHAT_MAX_HISTORY_TURNS,
    ) -> None:
        self.session = session
        self.chat_repository = chat_repository
        self.rag_pipeline = rag_pipeline
        self.max_history_turns = max_history_turns

    def send_message(self, chat_id: str, content: str) -> ChatSendResult:
        content = content.strip()
        if not content:
            raise ValueError("message content must not be empty")

        chat = self.chat_repository.get(chat_id)
        if chat is None:
            raise ValueError(f"Chat '{chat_id}' not found")

        document_ids = [document.id for document in chat.documents]

        # Persist the user message first: it survives even if the LLM call fails.
        user_message = self.chat_repository.add_message(
            chat_id=chat_id,
            role=MessageRole.USER,
            content=content,
        )
        self.session.flush()

        # Load bounded recent history. Kept separate from evidence; untrusted.
        self.chat_repository.get_recent_messages(
            chat_id=chat_id,
            limit=self.max_history_turns * 2 + 1,
        )

        # MVP: retrieval query is the current user message as-is.
        retrieval_query = content

        try:
            answer_result: ChatAnswerResult = self.rag_pipeline.answer_with_sources(
                question=retrieval_query,
                document_ids=document_ids,
            )
        except Exception as exc:
            # LLM failure: keep the user message, do not fabricate an assistant answer.
            return ChatSendResult(
                user_message=user_message,
                assistant_message=None,
                sources=(),
                error=str(exc),
            )

        sources_payload = (
            json.dumps(
                [
                    {
                        "document_id": source.document_id,
                        "page_number": source.page_number,
                        "chunk_id": source.chunk_id,
                    }
                    for source in answer_result.sources
                ]
            )
            if answer_result.sources
            else None
        )

        assistant_message = self.chat_repository.add_message(
            chat_id=chat_id,
            role=MessageRole.ASSISTANT,
            content=answer_result.answer,
            sources_json=sources_payload,
        )
        self.session.flush()

        return ChatSendResult(
            user_message=user_message,
            assistant_message=assistant_message,
            sources=tuple(
                SourceReference(
                    document_id=source.document_id,
                    page_number=source.page_number,
                    chunk_id=source.chunk_id,
                )
                for source in answer_result.sources
            ),
            error=None,
        )
