from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PageStatus(StrEnum):
    PENDING = "pending"
    PARSED = "parsed"
    OCR = "ocr"
    OCR_FAILED = "ocr_failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    filename: Mapped[str] = mapped_column(String(255))
    file_hash: Mapped[str] = mapped_column(String(64), unique=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
    )

    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentPage(Base):
    __tablename__ = "document_pages"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "page_number",
            name="uq_document_pages_document_id_page_number",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id"),
        index=True,
    )

    page_number: Mapped[int] = mapped_column(Integer)

    content_hash: Mapped[str] = mapped_column(String(64))

    status: Mapped[str] = mapped_column(
        String(30),
        default=PageStatus.PENDING,
    )

    extraction_method: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    document: Mapped[Document] = relationship(
        back_populates="pages",
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="page",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    page_id: Mapped[str] = mapped_column(
        ForeignKey("document_pages.id"),
        index=True,
    )

    page_number: Mapped[int] = mapped_column(Integer)

    chunk_index: Mapped[int] = mapped_column(Integer)

    text: Mapped[str] = mapped_column(
        Text,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
    )

    page: Mapped["DocumentPage"] = relationship(
        back_populates="chunks",
    )
