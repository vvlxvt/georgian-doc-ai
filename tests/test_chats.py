import json
from datetime import UTC, datetime, timedelta

from app.models import Chat, Document, Message, MessageRole


def create_document(session, file_hash: str = "document-hash") -> Document:
    document = Document(filename="contract.pdf", file_hash=file_hash)
    session.add(document)
    session.commit()
    return document


def test_chat_persists_ordered_messages_and_sources(session):
    document = create_document(session)
    second_document = create_document(session, file_hash="second-document-hash")
    chat = Chat()
    session.add(chat)
    chat.documents.extend([document, second_document])
    start = datetime(2026, 1, 1, tzinfo=UTC)
    chat.messages.extend(
        [
            Message(
                role=MessageRole.USER,
                content="Каков срок договора?",
                created_at=start,
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="Один год. [стр. 2]",
                created_at=start + timedelta(seconds=1),
                sources_json=json.dumps(
                    [{"document_id": document.id, "page_number": 2, "chunk_id": "c-2"}]
                ),
            ),
        ]
    )
    session.add(chat)
    session.commit()

    saved = session.get(Chat, chat.id)

    assert saved.title == "New chat"
    assert [message.role for message in saved.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert json.loads(saved.messages[1].sources_json) == [
        {"document_id": document.id, "page_number": 2, "chunk_id": "c-2"}
    ]


def test_deleting_chat_deletes_its_messages(session):
    document = create_document(session)
    chat = Chat(documents=[document])
    chat.messages.append(Message(role=MessageRole.USER, content="Вопрос"))
    session.add(chat)
    session.commit()
    message_id = chat.messages[0].id

    session.delete(chat)
    session.commit()

    assert session.get(Chat, chat.id) is None
    assert session.get(Message, message_id) is None


def test_deleting_document_preserves_chat_and_messages(session):
    document = create_document(session)
    second_document = create_document(session, file_hash="second-document-hash")
    chat = Chat(documents=[document, second_document])
    chat.messages.append(Message(role=MessageRole.USER, content="Вопрос"))
    session.add(chat)
    session.commit()
    chat_id = chat.id
    message_id = chat.messages[0].id

    session.delete(document)
    session.commit()

    saved_chat = session.get(Chat, chat_id)
    assert saved_chat is not None
    assert [item.id for item in saved_chat.documents] == [second_document.id]
    assert session.get(Message, message_id) is not None
