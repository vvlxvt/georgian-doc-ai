"""Add many-to-many conversation-documents table.

Revision ID: 20261107_01
Revises: 20261104_01
Create Date: 2026-11-07
"""

import sqlalchemy as sa

from alembic import op

revision = "20261107_01"
down_revision = "20261104_01"
branch_labels = None
depends_on = None


def _table_names() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _column_names(table_name: str) -> set[str]:
    return set(col["name"] for col in sa.inspect(op.get_bind()).get_columns(table_name))


def upgrade() -> None:
    table_names = _table_names()

    if "conversation_documents" not in table_names:
        op.create_table(
            "conversation_documents",
            sa.Column("conversation_id", sa.String(length=36), nullable=False),
            sa.Column("document_id", sa.String(length=36), nullable=False),
            sa.ForeignKeyConstraint(
                ["conversation_id"], ["conversations.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["document_id"], ["documents.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("conversation_id", "document_id"),
        )

    # Migrate existing document_id values from conversations into the m2m table.
    bind = op.get_bind()
    if "document_id" in _column_names("conversations"):
        bind.execute(
            sa.text(
                "INSERT OR IGNORE INTO conversation_documents "
                "(conversation_id, document_id) "
                "SELECT id, document_id FROM conversations "
                "WHERE document_id IS NOT NULL"
            )
        )

    # Drop the old document_id column from conversations.
    if "document_id" in _column_names("conversations"):
        # Drop the index that references the column being removed.
        indexes = sa.inspect(op.get_bind()).get_indexes("conversations")
        if any(index["name"] == "ix_conversations_document_id" for index in indexes):
            op.drop_index("ix_conversations_document_id", table_name="conversations")

        with op.batch_alter_table("conversations") as batch_op:
            batch_op.drop_column("document_id")


def downgrade() -> None:
    bind = op.get_bind()

    if "document_id" not in _column_names("conversations"):
        associations = bind.execute(
            sa.text(
                "SELECT conversation_id, document_id "
                "FROM conversation_documents ORDER BY conversation_id, document_id"
            )
        ).all()
        documents_by_conversation: dict[str, list[str]] = {}
        for conversation_id, document_id in associations:
            documents_by_conversation.setdefault(conversation_id, []).append(
                document_id
            )

        conversation_ids = {
            row[0] for row in bind.execute(sa.text("SELECT id FROM conversations"))
        }
        if any(
            len(documents_by_conversation.get(conversation_id, [])) != 1
            for conversation_id in conversation_ids
        ):
            raise RuntimeError(
                "Cannot downgrade conversations with zero or multiple documents."
            )

        op.drop_table("conversation_documents")
        with op.batch_alter_table("conversations") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "document_id",
                    sa.String(length=36),
                    sa.ForeignKey(
                        "documents.id",
                        ondelete="CASCADE",
                        name="fk_conversations_document_id_documents",
                    ),
                    nullable=True,
                )
            )

        for conversation_id, document_ids in documents_by_conversation.items():
            bind.execute(
                sa.text(
                    "UPDATE conversations SET document_id = :document_id "
                    "WHERE id = :conversation_id"
                ),
                {
                    "document_id": document_ids[0],
                    "conversation_id": conversation_id,
                },
            )

        with op.batch_alter_table("conversations") as batch_op:
            batch_op.alter_column(
                "document_id", existing_type=sa.String(36), nullable=False
            )

        op.create_index(
            "ix_conversations_document_id", "conversations", ["document_id"]
        )
