from app.chunking import TextChunk
from app.models import Document, DocumentPage
from app.repositories import DocumentChunkRepository


def test_create_chunks(session):
    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )

    session.add(document)
    session.commit()

    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Some document text.",
    )

    session.add(page)
    session.commit()

    chunks = [
        TextChunk(
            page_number=1,
            chunk_index=0,
            text="First chunk.",
        ),
        TextChunk(
            page_number=1,
            chunk_index=1,
            text="Second chunk.",
        ),
    ]

    repository = DocumentChunkRepository(session)

    result = repository.create_many(
        page_id=page.id,
        page_number=page.page_number,
        chunks=chunks,
    )

    session.commit()

    assert len(result) == 2

    assert result[0].chunk_index == 0
    assert result[0].text == "First chunk."

    assert result[1].chunk_index == 1
    assert result[1].text == "Second chunk."


def test_get_chunks_by_page(session):
    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.commit()

    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Some document text.",
    )
    session.add(page)
    session.commit()

    repository = DocumentChunkRepository(session)

    repository.create(
        page_id=page.id,
        page_number=1,
        chunk_index=1,
        text="Second chunk.",
    )

    repository.create(
        page_id=page.id,
        page_number=1,
        chunk_index=0,
        text="First chunk.",
    )

    session.commit()

    result = repository.get_by_page(page.id)

    assert len(result) == 2
    assert result[0].chunk_index == 0
    assert result[0].text == "First chunk."
    assert result[1].chunk_index == 1
    assert result[1].text == "Second chunk."


def test_delete_chunks_by_page(session):
    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.commit()

    page = DocumentPage(
        document_id=document.id,
        page_number=1,
        content_hash="page-hash",
        status="parsed",
        extraction_method="native",
        text="Some document text.",
    )
    session.add(page)
    session.commit()

    repository = DocumentChunkRepository(session)

    repository.create(
        page_id=page.id,
        page_number=1,
        chunk_index=0,
        text="First chunk.",
    )

    repository.create(
        page_id=page.id,
        page_number=1,
        chunk_index=1,
        text="Second chunk.",
    )

    session.commit()

    repository.delete_by_page(page.id)
    session.commit()

    result = repository.get_by_page(page.id)

    assert result == []
