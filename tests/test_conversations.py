import json

from app.models import Conversation, Document, Message, MessageRole


def create_document(session, file_hash: str = "document-hash") -> Document:
    document = Document(filename="contract.pdf", file_hash=file_hash)
    session.add(document)
    session.commit()
    return document


def test_conversation_persists_ordered_messages_and_sources(session):
    document = create_document(session)
    conversation = Conversation(document_id=document.id)
    conversation.messages.extend(
        [
            Message(role=MessageRole.USER, content="Каков срок договора?"),
            Message(
                role=MessageRole.ASSISTANT,
                content="Один год. [стр. 2]",
                sources_json=json.dumps(
                    [{"document_id": document.id, "page_number": 2, "chunk_id": "c-2"}]
                ),
            ),
        ]
    )
    session.add(conversation)
    session.commit()

    saved = session.get(Conversation, conversation.id)

    assert saved.title == "New conversation"
    assert [message.role for message in saved.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert json.loads(saved.messages[1].sources_json) == [
        {"document_id": document.id, "page_number": 2, "chunk_id": "c-2"}
    ]


def test_deleting_conversation_deletes_its_messages(session):
    document = create_document(session)
    conversation = Conversation(document_id=document.id)
    conversation.messages.append(Message(role=MessageRole.USER, content="Вопрос"))
    session.add(conversation)
    session.commit()
    message_id = conversation.messages[0].id

    session.delete(conversation)
    session.commit()

    assert session.get(Conversation, conversation.id) is None
    assert session.get(Message, message_id) is None


def test_deleting_document_deletes_conversations_and_messages(session):
    document = create_document(session)
    conversation = Conversation(document_id=document.id)
    conversation.messages.append(Message(role=MessageRole.USER, content="Вопрос"))
    session.add(conversation)
    session.commit()
    conversation_id = conversation.id
    message_id = conversation.messages[0].id

    session.delete(document)
    session.commit()

    assert session.get(Conversation, conversation_id) is None
    assert session.get(Message, message_id) is None
