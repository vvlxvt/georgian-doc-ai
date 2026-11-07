"""Rename persistent conversations to chats.

Revision ID: 20261108_01
Revises: 20261107_01
Create Date: 2026-11-08
"""

import sqlalchemy as sa

from alembic import op

revision = "20261108_01"
down_revision = "20261107_01"
branch_labels = None
depends_on = None


def _table_names() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _column_names(table_name: str) -> set[str]:
    return {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)
    }


def _rename_title_default(table_name: str, old: str, new: str) -> None:
    columns = sa.inspect(op.get_bind()).get_columns(table_name)
    title = next(column for column in columns if column["name"] == "title")
    current_default = title["default"]
    if current_default is None:
        return

    normalized_default = current_default.strip("'\"")
    if normalized_default == new:
        return
    if normalized_default != old:
        raise RuntimeError(f"Unexpected {table_name}.title default.")

    with op.batch_alter_table(table_name) as batch_op:
        batch_op.alter_column(
            "title",
            existing_type=sa.String(length=255),
            existing_nullable=False,
            existing_server_default=old,
            server_default=new,
        )


def upgrade() -> None:
    tables = _table_names()

    if "chats" not in tables:
        if "conversations" not in tables:
            raise RuntimeError("Neither chats nor conversations table exists.")
        op.rename_table("conversations", "chats")
        tables.remove("conversations")
        tables.add("chats")

    if "chat_documents" not in tables:
        if "conversation_documents" not in tables:
            raise RuntimeError(
                "Neither chat_documents nor conversation_documents table exists."
            )
        op.rename_table("conversation_documents", "chat_documents")

    if "conversation_id" in _column_names("chat_documents"):
        with op.batch_alter_table("chat_documents") as batch_op:
            batch_op.alter_column(
                "conversation_id",
                new_column_name="chat_id",
                existing_type=sa.String(length=36),
                existing_nullable=False,
            )

    if "messages" in _table_names():
        indexes = sa.inspect(op.get_bind()).get_indexes("messages")
        index_names = {index["name"] for index in indexes}
        if "ix_messages_conversation_id" in index_names:
            op.drop_index(
                "ix_messages_conversation_id",
                table_name="messages",
            )

        if "conversation_id" in _column_names("messages"):
            with op.batch_alter_table("messages") as batch_op:
                batch_op.alter_column(
                    "conversation_id",
                    new_column_name="chat_id",
                    existing_type=sa.String(length=36),
                    existing_nullable=False,
                )

        index_names = {
            index["name"] for index in sa.inspect(op.get_bind()).get_indexes("messages")
        }
        if "ix_messages_chat_id" not in index_names:
            op.create_index("ix_messages_chat_id", "messages", ["chat_id"])

    op.execute(
        sa.text("UPDATE chats SET title = 'New chat' WHERE title = 'New conversation'")
    )
    _rename_title_default("chats", "New conversation", "New chat")


def downgrade() -> None:
    tables = _table_names()

    if "chats" not in tables:
        return

    op.execute(
        sa.text("UPDATE chats SET title = 'New conversation' WHERE title = 'New chat'")
    )
    _rename_title_default("chats", "New chat", "New conversation")

    if "messages" in tables:
        indexes = sa.inspect(op.get_bind()).get_indexes("messages")
        index_names = {index["name"] for index in indexes}
        if "ix_messages_chat_id" in index_names:
            op.drop_index("ix_messages_chat_id", table_name="messages")

        if "chat_id" in _column_names("messages"):
            with op.batch_alter_table("messages") as batch_op:
                batch_op.alter_column(
                    "chat_id",
                    new_column_name="conversation_id",
                    existing_type=sa.String(length=36),
                    existing_nullable=False,
                )
            op.create_index(
                "ix_messages_conversation_id",
                "messages",
                ["conversation_id"],
            )

    if "chat_documents" in tables:
        if "chat_id" in _column_names("chat_documents"):
            with op.batch_alter_table("chat_documents") as batch_op:
                batch_op.alter_column(
                    "chat_id",
                    new_column_name="conversation_id",
                    existing_type=sa.String(length=36),
                    existing_nullable=False,
                )
        op.rename_table("chat_documents", "conversation_documents")

    op.rename_table("chats", "conversations")
