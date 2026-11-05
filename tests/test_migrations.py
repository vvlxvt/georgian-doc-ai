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
    assert {"document_chunks", "conversations", "messages"}.issubset(
        inspector.get_table_names()
    )
    indexes = inspector.get_indexes("document_pages")
    assert any(
        index["name"] == "uq_document_pages_document_id_page_number" and index["unique"]
        for index in indexes
    )

    conversation_foreign_keys = inspector.get_foreign_keys("conversations")
    assert conversation_foreign_keys[0]["options"]["ondelete"] == "CASCADE"


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
