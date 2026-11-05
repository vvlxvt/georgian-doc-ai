"""Add persistent document conversations.

Revision ID: 20261104_01
Revises: 20261006_01
Create Date: 2026-11-04
"""

import sqlalchemy as sa

from alembic import op

revision = "20261104_01"
down_revision = "20261006_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    table_names = set(sa.inspect(bind).get_table_names())

    if "conversations" not in table_names:
        op.create_table(
            "conversations",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("document_id", sa.String(length=36), nullable=False),
            sa.Column(
                "title",
                sa.String(length=255),
                nullable=False,
                server_default="New conversation",
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(
                ["document_id"],
                ["documents.id"],
                ondelete="CASCADE",
            ),
        )
        op.create_index(
            "ix_conversations_document_id", "conversations", ["document_id"]
        )

    if "messages" not in table_names:
        op.create_table(
            "messages",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("conversation_id", sa.String(length=36), nullable=False),
            sa.Column("role", sa.String(length=16), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("sources_json", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(
                ["conversation_id"],
                ["conversations.id"],
                ondelete="CASCADE",
            ),
        )
        op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_table("conversations")
