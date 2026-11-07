from datetime import UTC, datetime, timedelta

import pytest

from app.models import Document, MessageRole
from app.repositories import ChatRepository


def create_document(session) -> Document:
    document = Document(filename="contract.pdf", file_hash="repository-document")
    session.add(document)
    session.commit()
    return document


def test_repository_creates_lists_and_updates_chats(session):
    document = create_document(session)
    second_document = Document(
        filename="appendix.pdf",
        file_hash="repository-second-document",
    )
    session.add(second_document)
    session.commit()
    repository = ChatRepository(session)

    chat = repository.create([document.id, second_document.id])
    session.commit()
    repository.update_title(chat, "Срок договора")
    session.commit()

    chats = repository.list_for_document(document.id)

    assert [item.id for item in chats] == [chat.id]
    assert chats[0].title == "Срок договора"
    assert {item.id for item in repository.get(chat.id).documents} == {
        document.id,
        second_document.id,
    }


def test_repository_rejects_empty_or_unknown_document_ids(session):
    repository = ChatRepository(session)

    with pytest.raises(ValueError, match="at least one document"):
        repository.create([])

    with pytest.raises(ValueError, match="not found"):
        repository.create(["missing"])


def test_repository_returns_recent_messages_in_chronological_order(session):
    document = create_document(session)
    repository = ChatRepository(session)
    chat = repository.create([document.id])
    session.flush()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(3):
        message = repository.add_message(
            chat.id,
            MessageRole.USER,
            f"message {index}",
        )
        message.created_at = start + timedelta(seconds=index)
    session.commit()

    messages = repository.get_recent_messages(chat.id, limit=2)

    assert [message.content for message in messages] == ["message 1", "message 2"]


def test_repository_deletes_chat_and_messages(session):
    document = create_document(session)
    repository = ChatRepository(session)
    chat = repository.create([document.id])
    repository.add_message(chat.id, MessageRole.USER, "question")
    session.commit()

    repository.delete(chat)
    session.commit()

    assert repository.get(chat.id) is None
