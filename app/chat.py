from __future__ import annotations

import json
from dataclasses import dataclass

from app.answer import ChatAnswerResult, ConversationMessage
from app.context import SourceReference
from app.models import Message, MessageRole
from app.rag import RAGPipeline
from app.repositories import ChatRepository

CHAT_MAX_HISTORY_TURNS = 5
CHAT_MAX_HISTORY_CHARS = 6000


def _build_conversation_history(
    messages: list[Message],
) -> list[ConversationMessage]:
    history = []
    remaining_chars = CHAT_MAX_HISTORY_CHARS

    for message in reversed(messages):
        if remaining_chars <= 0:
            break

        content = message.content
        if len(content) > remaining_chars:
            content = content[:remaining_chars]
            history.append(
                ConversationMessage(
                    role="user" if message.role == MessageRole.USER else "assistant",
                    content=content,
                )
            )
            break

        history.append(
            ConversationMessage(
                role="user" if message.role == MessageRole.USER else "assistant",
                content=content,
            )
        )
        remaining_chars -= len(content)

    return list(reversed(history))


@dataclass(frozen=True)
class ChatSendResult:
    """Result of sending a message in a chat.

    Messages are persisted only when an answer is available.
    """

    user_message: Message | None
    assistant_message: Message | None
    sources: tuple[SourceReference, ...]
    error: Exception | None


class ChatService:
    """Orchestrates a multi-document chat turn.

    Responsibilities:
      * validate the chat and its selected documents,
      * load a bounded number of recent turns (history, kept out of evidence),
      * call the existing RAG pipeline scoped to the chat's documents,
      * persist the user message and assistant answer together with validated
        source references.

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

        # Keep history bounded and separate from document evidence.
        recent_messages = self.chat_repository.get_recent_messages(
            chat_id=chat_id,
            limit=self.max_history_turns * 2,
        )
        conversation_history = _build_conversation_history(recent_messages)

        history_query = "\n".join(
            f"{'Предыдущий вопрос' if message.role == 'user' else 'Предыдущий ответ'}: "
            f"{message.content}"
            for message in conversation_history
        )
        retrieval_query = (
            f"{history_query}\nТекущий вопрос: {content}" if history_query else content
        )

        try:
            answer_result: ChatAnswerResult = self.rag_pipeline.answer_with_sources(
                question=content,
                document_ids=document_ids,
                retrieval_query=retrieval_query,
                conversation_history=conversation_history,
            )
        except Exception as exc:
            return ChatSendResult(
                user_message=None,
                assistant_message=None,
                sources=(),
                error=exc,
            )

        user_message = self.chat_repository.add_message(
            chat_id=chat_id,
            role=MessageRole.USER,
            content=content,
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
