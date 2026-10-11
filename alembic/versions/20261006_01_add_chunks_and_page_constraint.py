"""Add document chunks and enforce unique page numbers.

Revision ID: 20261006_01
Revises:
Create Date: 2026-10-06
"""

import sqlalchemy as sa

from alembic import op

revision = "20261006_01"
down_revision = None
branch_labels = None
depends_on = None


def _table_names() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    table_names = _table_names()

    if "documents" not in table_names:
        op.create_table(
            "documents",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("filename", sa.String(length=255), nullable=False),
            sa.Column("file_hash", sa.String(length=64), nullable=False, unique=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )

    if "document_pages" not in table_names:
        op.create_table(
            "document_pages",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("document_id", sa.String(length=36), nullable=False),
            sa.Column("page_number", sa.Integer(), nullable=False),
            sa.Column("content_hash", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=30), nullable=False),
            sa.Column("extraction_method", sa.String(length=30), nullable=True),
            sa.Column("text", sa.Text(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        )
        table_names.add("document_pages")

    indexes = sa.inspect(op.get_bind()).get_indexes("document_pages")
    if not any(
        index["name"] == "uq_document_pages_document_id_page_number"
        for index in indexes
    ):
        op.create_index(
            "uq_document_pages_document_id_page_number",
            "document_pages",
            ["document_id", "page_number"],
            unique=True,
        )

    if "document_chunks" not in table_names:
        op.create_table(
            "document_chunks",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("page_id", sa.String(length=36), nullable=False),
            sa.Column("page_number", sa.Integer(), nullable=False),
            sa.Column("chunk_index", sa.Integer(), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["page_id"], ["document_pages.id"]),
        )
        op.create_index("ix_document_chunks_page_id", "document_chunks", ["page_id"])


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_index(
        "uq_document_pages_document_id_page_number", table_name="document_pages"
    )
