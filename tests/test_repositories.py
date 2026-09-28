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
