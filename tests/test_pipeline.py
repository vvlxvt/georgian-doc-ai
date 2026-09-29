from app.models import Document, DocumentPage, PageStatus
from app.pipeline import DocumentPipeline


class FakeOCR:
    def __init__(self, text="OCR recognized text."):
        self.text = text

    def recognize(self, image_path: str) -> str:
        return self.text


class FakePDFPage:
    def __init__(
        self,
        page_number=1,
        text="",
        content_hash="page-hash",
    ):
        self.page_number = page_number
        self.text = text
        self.content_hash = content_hash


class FakePDFReader:
    def __init__(self, pages):
        self._pages = pages

    def pages(self):
        yield from self._pages

    def render_page(self, page_number):
        return b"fake image"


def create_pipeline(
    session,
    ocr_processor,
):
    from app.chunking import TextChunker
    from app.text import TextNormalizer

    return DocumentPipeline(
        session=session,
        ocr_processor=ocr_processor,
        text_normalizer=TextNormalizer(),
        text_chunker=TextChunker(
            max_length=100,
            overlap=20,
        ),
    )


def test_process_native_text(session, monkeypatch):
    page = FakePDFPage(
        text=(
            "This is a sufficiently long native text extracted from the PDF document."
        ),
    )

    reader = FakePDFReader([page])

    monkeypatch.setattr(
        "app.pipeline.PDFReader",
        lambda path: reader,
    )

    monkeypatch.setattr(
        DocumentPipeline,
        "_hash_file",
        staticmethod(lambda path: "document-hash"),
    )

    pipeline = create_pipeline(
        session,
        FakeOCR(),
    )

    document_id = pipeline.process("test.pdf")

    document = session.get(Document, document_id)

    assert document is not None
    assert document.filename == "test.pdf"

    db_page = (
        session.query(DocumentPage)
        .filter_by(
            document_id=document.id,
            page_number=1,
        )
        .one()
    )

    assert db_page.status == PageStatus.PARSED
    assert db_page.extraction_method == "native"
    assert db_page.text == (
        "This is a sufficiently long native text extracted from the PDF document."
    )

    assert len(db_page.chunks) > 0
    assert db_page.chunks[0].text


def test_process_page_with_ocr(session):
    page = FakePDFPage(
        text="",
        content_hash="scanned-page-hash",
    )

    reader = FakePDFReader([page])

    pipeline = create_pipeline(
        session,
        FakeOCR(
            text=(
                "ეს არის საკმარისად გრძელი ტექსტი, რომელიც მიღებულია OCR-ის საშუალებით."
            ),
        ),
    )

    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.commit()

    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    db_page = (
        session.query(DocumentPage)
        .filter_by(
            document_id=document.id,
            page_number=1,
        )
        .one()
    )

    assert db_page.status == PageStatus.OCR
    assert db_page.extraction_method == "tesseract"
    assert db_page.text == (
        "ეს არის საკმარისად გრძელი ტექსტი, რომელიც მიღებულია OCR-ის საშუალებით."
    )

    assert len(db_page.chunks) > 0


def test_process_page_when_ocr_fails(session):
    class FailingOCR:
        def recognize(self, image_path: str) -> str:
            raise RuntimeError("Tesseract failed")

    page = FakePDFPage(
        text="",
        content_hash="scanned-page-hash",
    )

    reader = FakePDFReader([page])

    pipeline = create_pipeline(
        session,
        FailingOCR(),
    )

    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.commit()

    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    db_page = (
        session.query(DocumentPage)
        .filter_by(
            document_id=document.id,
            page_number=1,
        )
        .one()
    )

    assert db_page.status == PageStatus.OCR_FAILED
    assert db_page.extraction_method == "tesseract"
    assert "Tesseract failed" in db_page.error


def test_process_page_skips_already_processed_page(session):
    page = FakePDFPage(
        text=(
            "This is a sufficiently long native text extracted from the PDF document."
        ),
        content_hash="same-page-hash",
    )

    reader = FakePDFReader([page])

    pipeline = create_pipeline(
        session,
        FakeOCR(),
    )

    document = Document(
        filename="test.pdf",
        file_hash="document-hash",
    )
    session.add(document)
    session.commit()

    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    db_page = (
        session.query(DocumentPage)
        .filter_by(
            document_id=document.id,
            page_number=1,
        )
        .one()
    )

    first_chunk_count = len(db_page.chunks)

    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    session.refresh(db_page)

    assert len(db_page.chunks) == first_chunk_count
