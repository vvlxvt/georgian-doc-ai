import pytest

from app.answer import ChatAnswerResult
from app.chat import ChatSendResult, ChatService
from app.context import SourceReference
from app.models import Conversation, Message, MessageRole
from app.repositories import ConversationRepository


class FakeRAGPipeline:
    def __init__(self, answer_text: str = "Ответ. [стр. 1]", should_fail: bool = False):
        self.answer_text = answer_text
        self.should_fail = should_fail
        self.calls = []

    def answer_with_sources(self, question: str, document_id: str = None, **kwargs):
        self.calls.append((question, document_id))
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


class FakeConversationRepository(ConversationRepository):
    def __init__(self, session, conversations=None):
        super().__init__(session)
        self._conversations = conversations or {}
        self.added_messages = []
        self.recent_calls = []

    def get(self, conversation_id: str):
        return self._conversations.get(conversation_id)

    def add_message(self, conversation_id, role, content, sources_json=None):
        message = Message(
            id=f"msg-{len(self.added_messages)}",
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources_json=sources_json,
        )
        self.added_messages.append(message)
        return message

    def get_recent_messages(self, conversation_id: str, limit: int):
        self.recent_calls.append((conversation_id, limit))
        return []


def make_conversation(conversation_id: str, document_id: str) -> Conversation:
    return Conversation(id=conversation_id, document_id=document_id)


def test_send_message_persists_user_and_assistant(session):
    repository = FakeConversationRepository(session)
    conversation = make_conversation("conv-1", "doc-1")
    repository._conversations["conv-1"] = conversation
    session.add(conversation)
    session.commit()

    rag = FakeRAGPipeline()
    service = ChatService(session, repository, rag)

    result = service.send_message("conv-1", "Вопрос про договор")

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


def test_send_message_rag_scoped_to_conversation_document(session):
    repository = FakeConversationRepository(session)
    conversation = make_conversation("conv-1", "doc-42")
    repository._conversations["conv-1"] = conversation
    session.add(conversation)
    session.commit()

    rag = FakeRAGPipeline()
    service = ChatService(session, repository, rag)

    service.send_message("conv-1", "Каков срок?")

    assert len(rag.calls) == 1
    question, document_id = rag.calls[0]
    assert question == "Каков срок?"
    assert document_id == "doc-42"


def test_send_message_conversation_not_found(session):
    repository = FakeConversationRepository(session)
    rag = FakeRAGPipeline()
    service = ChatService(session, repository, rag)

    with pytest.raises(ValueError, match="not found"):
        service.send_message("missing-conv", "Вопрос")


def test_send_message_empty_content_raises(session):
    repository = FakeConversationRepository(session)
    rag = FakeRAGPipeline()
    service = ChatService(session, repository, rag)

    with pytest.raises(ValueError, match="must not be empty"):
        service.send_message("conv-1", "   ")


def test_send_message_llm_failure_keeps_user_message(session):
    repository = FakeConversationRepository(session)
    conversation = make_conversation("conv-1", "doc-1")
    repository._conversations["conv-1"] = conversation
    session.add(conversation)
    session.commit()

    rag = FakeRAGPipeline(should_fail=True)
    service = ChatService(session, repository, rag)

    result = service.send_message("conv-1", "Вопрос после сбоя")

    assert result.error is not None
    assert "LLM provider unavailable" in result.error
    assert result.user_message is not None
    assert result.user_message.content == "Вопрос после сбоя"
    assert result.user_message.role == MessageRole.USER
    assert result.assistant_message is None
    assert result.sources == ()


def test_send_message_loads_bounded_history(session):
    repository = FakeConversationRepository(session)
    conversation = make_conversation("conv-1", "doc-1")
    repository._conversations["conv-1"] = conversation
    session.add(conversation)
    session.commit()

    rag = FakeRAGPipeline()
    service = ChatService(session, repository, rag, max_history_turns=3)

    service.send_message("conv-1", "Вопрос")

    assert len(repository.recent_calls) == 1
    _, limit = repository.recent_calls[0]
    assert limit == 3 * 2 + 1
