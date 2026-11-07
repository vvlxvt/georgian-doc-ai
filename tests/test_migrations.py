from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from alembic import command
from app.models import Document, DocumentPage

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def upgrade_database(database_path: Path) -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    command.upgrade(config, "head")


def migration_config(database_path: Path) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    return config


def test_migration_upgrades_legacy_schema_with_document_chunks(tmp_path):
    database_path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{database_path}")

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE documents ("
                "id VARCHAR(36) PRIMARY KEY, filename VARCHAR(255) NOT NULL, "
                "file_hash VARCHAR(64) NOT NULL UNIQUE, created_at DATETIME NOT NULL)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE document_pages ("
                "id VARCHAR(36) PRIMARY KEY, document_id VARCHAR(36) NOT NULL, "
                "page_number INTEGER NOT NULL, content_hash VARCHAR(64) NOT NULL, "
                "status VARCHAR(30) NOT NULL, extraction_method VARCHAR(30), "
                "text TEXT, error TEXT)"
            )
        )

    upgrade_database(database_path)

    inspector = inspect(engine)
    assert {
        "document_chunks",
        "chats",
        "chat_documents",
        "messages",
    }.issubset(inspector.get_table_names())
    indexes = inspector.get_indexes("document_pages")
    assert any(
        index["name"] == "uq_document_pages_document_id_page_number" and index["unique"]
        for index in indexes
    )

    assert "document_id" not in {
        column["name"] for column in inspector.get_columns("chats")
    }
    association_foreign_keys = inspector.get_foreign_keys("chat_documents")
    assert {key["options"]["ondelete"] for key in association_foreign_keys} == {
        "CASCADE"
    }
    chat_title = next(
        column for column in inspector.get_columns("chats") if column["name"] == "title"
    )
    assert chat_title["default"].strip("'\"") == "New chat"
    message_foreign_keys = inspector.get_foreign_keys("messages")
    assert any(
        key["referred_table"] == "chats" and key["constrained_columns"] == ["chat_id"]
        for key in message_foreign_keys
    )
    assert any(
        index["name"] == "ix_messages_chat_id"
        for index in inspector.get_indexes("messages")
    )


def test_chat_migration_renames_legacy_schema_and_preserves_data(tmp_path):
    database_path = tmp_path / "chats.db"
    engine = create_engine(f"sqlite:///{database_path}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE documents ("
                "id VARCHAR(36) PRIMARY KEY, filename VARCHAR(255) NOT NULL, "
                "file_hash VARCHAR(64) NOT NULL UNIQUE, created_at DATETIME NOT NULL)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO documents (id, filename, file_hash, created_at) "
                "VALUES ('doc-1', 'contract.pdf', 'hash', '2026-11-07')"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE conversations ("
                "id VARCHAR(36) PRIMARY KEY, document_id VARCHAR(36) NOT NULL, "
                "title VARCHAR(255) NOT NULL, created_at DATETIME NOT NULL, "
                "updated_at DATETIME NOT NULL, "
                "FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO conversations "
                "(id, document_id, title, created_at, updated_at) "
                "VALUES ('conv-1', 'doc-1', 'New conversation', "
                "'2026-11-07', '2026-11-07')"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX ix_conversations_document_id "
                "ON conversations(document_id)"
            )
        )

    config = migration_config(database_path)
    command.upgrade(config, "20261107_01")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO messages "
                "(id, conversation_id, role, content, created_at) "
                "VALUES ('msg-1', 'conv-1', 'user', 'Question', '2026-11-07')"
            )
        )
    command.upgrade(config, "head")

    with engine.connect() as connection:
        assert connection.execute(
            text("SELECT chat_id, document_id FROM chat_documents")
        ).one() == ("conv-1", "doc-1")
        assert connection.execute(
            text("SELECT chat_id, content FROM messages WHERE id = 'msg-1'")
        ).one() == ("conv-1", "Question")
        assert (
            connection.execute(
                text("SELECT title FROM chats WHERE id = 'conv-1'")
            ).scalar_one()
            == "New chat"
        )

    command.downgrade(config, "20261104_01")

    inspector = inspect(engine)
    legacy_chat_columns = {
        column["name"] for column in inspector.get_columns("conversations")
    }
    assert "document_id" in legacy_chat_columns
    document_id_column = next(
        column
        for column in inspector.get_columns("conversations")
        if column["name"] == "document_id"
    )
    assert document_id_column["nullable"] is False
    restored_foreign_keys = inspector.get_foreign_keys("conversations")
    assert restored_foreign_keys[0]["options"]["ondelete"] == "CASCADE"
    assert "conversation_documents" not in inspector.get_table_names()
    assert any(
        index["name"] == "ix_conversations_document_id"
        for index in inspector.get_indexes("conversations")
    )
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT document_id FROM conversations WHERE id = 'conv-1'")
            ).scalar_one()
            == "doc-1"
        )
        assert (
            connection.execute(
                text("SELECT title FROM conversations WHERE id = 'conv-1'")
            ).scalar_one()
            == "New conversation"
        )
        assert connection.execute(
            text("SELECT conversation_id, content FROM messages WHERE id = 'msg-1'")
        ).one() == ("conv-1", "Question")


def test_page_number_is_unique_within_a_document(session):
    document = Document(filename="contract.pdf", file_hash="document-hash")
    session.add(document)
    session.flush()

    session.add_all(
        [
            DocumentPage(
                document_id=document.id,
                page_number=1,
                content_hash="first-page",
            ),
            DocumentPage(
                document_id=document.id,
                page_number=1,
                content_hash="duplicate-page",
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        session.commit()
