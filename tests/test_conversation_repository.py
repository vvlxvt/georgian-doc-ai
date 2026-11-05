from datetime import UTC, datetime, timedelta

from app.models import Document, MessageRole
from app.repositories import ConversationRepository


def create_document(session) -> Document:
    document = Document(filename="contract.pdf", file_hash="repository-document")
    session.add(document)
    session.commit()
    return document


def test_repository_creates_lists_and_updates_conversations(session):
    document = create_document(session)
    repository = ConversationRepository(session)

    conversation = repository.create(document.id)
    session.commit()
    repository.update_title(conversation, "Срок договора")
    session.commit()

    conversations = repository.list_for_document(document.id)

    assert [item.id for item in conversations] == [conversation.id]
    assert conversations[0].title == "Срок договора"
    assert repository.get(conversation.id).document_id == document.id


def test_repository_returns_recent_messages_in_chronological_order(session):
    document = create_document(session)
    repository = ConversationRepository(session)
    conversation = repository.create(document.id)
    session.flush()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(3):
        message = repository.add_message(
            conversation.id,
            MessageRole.USER,
            f"message {index}",
        )
        message.created_at = start + timedelta(seconds=index)
    session.commit()

    messages = repository.get_recent_messages(conversation.id, limit=2)

    assert [message.content for message in messages] == ["message 1", "message 2"]


def test_repository_deletes_conversation_and_messages(session):
    document = create_document(session)
    repository = ConversationRepository(session)
    conversation = repository.create(document.id)
    repository.add_message(conversation.id, MessageRole.USER, "question")
    session.commit()

    repository.delete(conversation)
    session.commit()

    assert repository.get(conversation.id) is None
