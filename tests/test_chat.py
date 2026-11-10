import pytest

from app.answer import ChatAnswerResult, ConversationMessage
from app.chat import CHAT_MAX_HISTORY_CHARS, ChatSendResult, ChatService
from app.context import SourceReference
from app.models import Chat, Document, Message, MessageRole
from app.repositories import ChatRepository


class FakeRAGPipeline:
    def __init__(self, answer_text: str = "Ответ. [стр. 1]", should_fail: bool = False):
        self.answer_text = answer_text
        self.should_fail = should_fail
        self.calls = []
        self.last_kwargs = {}

    def answer_with_sources(self, question: str, document_ids=None, **kwargs):
        self.calls.append((question, document_ids))
        self.last_kwargs = kwargs
        if self.should_fail:
            raise RuntimeError("LLM provider unavailable")
        return ChatAnswerResult(
            answer=self.answer_text,
            model_name="test-model",
            sources=(
                SourceReference(
                    document_id="doc-1",
                    page_number=1,
                    chunk_id="chunk-1",
                ),
            ),
        )


class FakeChatRepository(ChatRepository):
    def __init__(self, session, chats=None):
        super().__init__(session)
        self._chats = chats or {}
        self.added_messages = []
        self.history_messages = []
        self.recent_calls = []

    def get(self, chat_id: str):
        return self._chats.get(chat_id)

    def add_message(self, chat_id, role, content, sources_json=None):
        message = Message(
            id=f"msg-{len(self.added_messages)}",
            chat_id=chat_id,
            role=role,
            content=content,
            sources_json=sources_json,
        )
        self.added_messages.append(message)
        return message

    def get_recent_messages(self, chat_id: str, limit: int):
        self.recent_calls.append((chat_id, limit))
        return [*self.history_messages, *self.added_messages][-limit:]


def make_chat(chat_id: str, *document_ids: str) -> Chat:
    return Chat(
        id=chat_id,
        documents=[
            Document(
                id=document_id,
                filename=f"{document_id}.pdf",
                file_hash=document_id.ljust(64, "0"),
            )
            for document_id in document_ids
        ],
    )


def test_send_message_persists_user_and_assistant(session):
    repository = FakeChatRepository(session)
    chat = make_chat("chat-1", "doc-1")
    repository._chats["chat-1"] = chat
    session.add(chat)
    session.commit()

    rag = FakeRAGPipeline()
    service = ChatService(session, repository, rag)

    result = service.send_message("chat-1", "Вопрос про договор")

    assert isinstance(result, ChatSendResult)
    assert result.error is None
    assert result.user_message.role == MessageRole.USER
    assert result.user_message.content == "Вопрос про договор"
    assert result.assistant_message is not None
    assert result.assistant_message.role == MessageRole.ASSISTANT
    assert result.assistant_message.content == "Ответ. [стр. 1]"
    assert len(result.sources) == 1
    assert result.sources[0].document_id == "doc-1"
    assert result.sources[0].page_number == 1

    assert result.assistant_message.sources_json is not None
    import json

    sources_data = json.loads(result.assistant_message.sources_json)
    assert sources_data[0]["page_number"] == 1


def test_send_message_rag_scoped_to_chat_documents(session):
    repository = FakeChatRepository(session)
    chat = make_chat("chat-1", "doc-42", "doc-43")
    repository._chats["chat-1"] = chat
    session.add(chat)
    session.commit()

    rag = FakeRAGPipeline()
    service = ChatService(session, repository, rag)

    service.send_message("chat-1", "Каков срок?")

    assert len(rag.calls) == 1
    question, document_ids = rag.calls[0]
    assert question == "Каков срок?"
    assert document_ids == ["doc-42", "doc-43"]


def test_send_message_chat_not_found(session):
    repository = FakeChatRepository(session)
    rag = FakeRAGPipeline()
    service = ChatService(session, repository, rag)

    with pytest.raises(ValueError, match="not found"):
        service.send_message("missing-conv", "Вопрос")


def test_send_message_empty_content_raises(session):
    repository = FakeChatRepository(session)
    rag = FakeRAGPipeline()
    service = ChatService(session, repository, rag)

    with pytest.raises(ValueError, match="must not be empty"):
        service.send_message("chat-1", "   ")


def test_send_message_llm_failure_keeps_user_message(session):
    repository = FakeChatRepository(session)
    chat = make_chat("chat-1", "doc-1")
    repository._chats["chat-1"] = chat
    session.add(chat)
    session.commit()

    rag = FakeRAGPipeline(should_fail=True)
    service = ChatService(session, repository, rag)

    result = service.send_message("chat-1", "Вопрос после сбоя")

    assert result.error is not None
    assert isinstance(result.error, RuntimeError)
    assert "LLM provider unavailable" in str(result.error)
    assert result.user_message is not None
    assert result.user_message.content == "Вопрос после сбоя"
    assert result.user_message.role == MessageRole.USER
    assert result.assistant_message is None
    assert result.sources == ()


def test_send_message_loads_bounded_history(session):
    repository = FakeChatRepository(session)
    chat = make_chat("chat-1", "doc-1")
    repository._chats["chat-1"] = chat
    session.add(chat)
    session.commit()

    rag = FakeRAGPipeline()
    service = ChatService(session, repository, rag, max_history_turns=3)

    service.send_message("chat-1", "Вопрос")

    assert len(repository.recent_calls) == 1
    _, limit = repository.recent_calls[0]
    assert limit == 3 * 2 + 1


def test_send_message_uses_recent_history_for_retrieval_and_answer(session):
    repository = FakeChatRepository(session)
    chat = make_chat("chat-1", "doc-1")
    repository._chats["chat-1"] = chat
    repository.history_messages = [
        Message(
            id="previous-user",
            role=MessageRole.USER,
            content="Какой срок у первого договора?",
        ),
        Message(
            id="previous-assistant",
            role=MessageRole.ASSISTANT,
            content="Первый договор действует один год. [стр. 1]",
        ),
    ]
    session.add(chat)
    session.commit()

    rag = FakeRAGPipeline()
    service = ChatService(session, repository, rag)

    service.send_message("chat-1", "А у второго?")

    assert rag.calls == [("А у второго?", ["doc-1"])]
    assert rag.last_kwargs["retrieval_query"] == (
        "Предыдущий вопрос: Какой срок у первого договора?\n"
        "Предыдущий ответ: Первый договор действует один год. [стр. 1]\n"
        "Текущий вопрос: А у второго?"
    )
    assert rag.last_kwargs["conversation_history"] == [
        ConversationMessage(
            role="user",
            content="Какой срок у первого договора?",
        ),
        ConversationMessage(
            role="assistant",
            content="Первый договор действует один год. [стр. 1]",
        ),
    ]


def test_send_message_caps_history_chars_and_keeps_most_recent_context(session):
    repository = FakeChatRepository(session)
    chat = make_chat("chat-1", "doc-1")
    repository._chats["chat-1"] = chat
    repository.history_messages = [
        Message(id="old-user", role=MessageRole.USER, content="old " * 1000),
        Message(
            id="old-assistant",
            role=MessageRole.ASSISTANT,
            content="old answer " * 1000,
        ),
        Message(
            id="recent-user",
            role=MessageRole.USER,
            content="Recent question: " + "details " * 400,
        ),
        Message(
            id="recent-assistant",
            role=MessageRole.ASSISTANT,
            content="Recent answer: " + "details " * 400,
        ),
    ]
    session.add(chat)
    session.commit()

    rag = FakeRAGPipeline()
    ChatService(session, repository, rag).send_message("chat-1", "And next?")

    history = rag.last_kwargs["conversation_history"]
    assert sum(len(message.content) for message in history) <= CHAT_MAX_HISTORY_CHARS
    assert history[-1].content.startswith("Recent answer:")
    assert history[0].content.startswith("Recent question:")
    assert all("old " not in message.content for message in history)
