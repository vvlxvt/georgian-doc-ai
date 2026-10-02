from unittest.mock import Mock

from app.chunking import TextChunker
from app.models import Document, DocumentPage, PageStatus
from app.pdf import PDFPage
from app.pipeline import DocumentPipeline
from app.text import TextNormalizer


def create_pipeline(session, ocr_processor=None):
    if ocr_processor is None:
        ocr_processor = Mock()

    return DocumentPipeline(
        session=session,
        ocr_processor=ocr_processor,
        text_normalizer=TextNormalizer(),
        text_chunker=TextChunker(
            max_length=500,
            overlap=50,
        ),
    )


def create_document(session):
    document = Document(
        filename="test.pdf",
        file_hash="a" * 64,
    )

    session.add(document)
    session.commit()

    return document


def create_page(
    page_number=1,
    text="This is a sufficiently long native text for testing.",
):
    return PDFPage(
        page_number=page_number,
        text=text,
        content_hash="",
        _doc_path="test.pdf",
    )


def get_document_page(session, document, page_number=1):
    return (
        session.query(DocumentPage)
        .filter_by(
            document_id=document.id,
            page_number=page_number,
        )
        .first()
    )


def test_process_native_text(session):
    document = create_document(session)
    pipeline = create_pipeline(session)

    page = create_page()

    reader = Mock()

    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    saved_page = get_document_page(
        session=session,
        document=document,
    )

    assert saved_page is not None
    assert saved_page.status == PageStatus.PARSED
    assert saved_page.extraction_method == "native"
    assert saved_page.text == page.text
    assert saved_page.content_hash == pipeline._hash_text(page.text)

    reader.render_page.assert_not_called()


def test_process_page_with_ocr(session):
    document = create_document(session)

    ocr_processor = Mock()
    ocr_processor.recognize.return_value = "Georgian OCR text with enough content."

    pipeline = create_pipeline(
        session=session,
        ocr_processor=ocr_processor,
    )

    page = create_page(
        text="",
    )

    image_bytes = b"test image bytes"

    reader = Mock()
    reader.render_page.return_value = image_bytes

    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    saved_page = get_document_page(
        session=session,
        document=document,
    )

    assert saved_page is not None
    assert saved_page.status == PageStatus.OCR
    assert saved_page.extraction_method == "tesseract"
    assert saved_page.text == ("Georgian OCR text with enough content.")
    assert saved_page.content_hash == pipeline._hash_bytes(
        image_bytes,
    )

    reader.render_page.assert_called_once_with(
        page.page_number,
    )

    ocr_processor.recognize.assert_called_once()


def test_process_page_when_ocr_fails(session):
    document = create_document(session)

    ocr_processor = Mock()
    ocr_processor.recognize.side_effect = RuntimeError(
        "Tesseract failed",
    )

    pipeline = create_pipeline(
        session=session,
        ocr_processor=ocr_processor,
    )

    page = create_page(
        text="",
    )

    image_bytes = b"test image bytes"

    reader = Mock()
    reader.render_page.return_value = image_bytes

    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    saved_page = get_document_page(
        session=session,
        document=document,
    )

    assert saved_page is not None
    assert saved_page.status == PageStatus.OCR_FAILED
    assert saved_page.extraction_method == "tesseract"
    assert saved_page.error == "OCR failed: Tesseract failed"
    assert saved_page.content_hash == pipeline._hash_bytes(
        image_bytes,
    )


def test_process_page_skips_already_processed_native_page(session):
    document = create_document(session)

    pipeline = create_pipeline(session)

    page = create_page()

    reader = Mock()

    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    reader.reset_mock()

    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    saved_page = get_document_page(
        session=session,
        document=document,
    )

    assert saved_page.status == PageStatus.PARSED
    assert saved_page.text == page.text

    reader.render_page.assert_not_called()


def test_native_page_with_different_text_is_reprocessed(session):
    document = create_document(session)

    pipeline = create_pipeline(session)

    first_page = create_page(
        text="First sufficiently long native text for testing.",
    )

    reader = Mock()

    pipeline._process_page(
        document=document,
        page=first_page,
        reader=reader,
    )

    saved_page = get_document_page(
        session=session,
        document=document,
    )

    first_hash = saved_page.content_hash

    second_page = create_page(
        text="Second sufficiently long native text for testing.",
    )

    pipeline._process_page(
        document=document,
        page=second_page,
        reader=reader,
    )

    saved_page = get_document_page(
        session=session,
        document=document,
    )

    assert saved_page.status == PageStatus.PARSED
    assert saved_page.text == second_page.text
    assert saved_page.content_hash != first_hash


def test_same_native_text_produces_same_hash(session):
    pipeline = create_pipeline(session)

    text = "Some sufficiently long text for hashing."

    first_hash = pipeline._hash_text(text)
    second_hash = pipeline._hash_text(text)

    assert first_hash == second_hash


def test_different_native_text_produces_different_hash(session):
    pipeline = create_pipeline(session)

    first_hash = pipeline._hash_text(
        "First sufficiently long text.",
    )

    second_hash = pipeline._hash_text(
        "Second sufficiently long text.",
    )

    assert first_hash != second_hash


def test_different_ocr_images_produce_different_hashes(session):
    pipeline = create_pipeline(session)

    first_hash = pipeline._hash_bytes(
        b"first image bytes",
    )

    second_hash = pipeline._hash_bytes(
        b"second image bytes",
    )

    assert first_hash != second_hash


def test_process_page_skips_already_processed_ocr_page(session):
    ocr_processor = Mock()
    ocr_processor.recognize.return_value = (
        "Georgian OCR text that is long enough for processing."
    )

    pipeline = create_pipeline(
        session=session,
        ocr_processor=ocr_processor,
    )

    document = create_document(session)

    page = create_page(
        page_number=1,
        text="",
    )

    reader = Mock()
    image_bytes = b"same image bytes"

    reader.render_page.return_value = image_bytes

    # First processing
    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    ocr_processor.recognize.assert_called_once()

    ocr_processor.reset_mock()

    # Second processing of the same page
    page = create_page(
        page_number=1,
        text="",
    )

    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    # OCR must not run again
    ocr_processor.recognize.assert_not_called()


def test_ocr_page_with_different_image_is_reprocessed(session):
    ocr_processor = Mock()
    ocr_processor.recognize.return_value = (
        "Georgian OCR text that is long enough for processing."
    )

    pipeline = create_pipeline(
        session=session,
        ocr_processor=ocr_processor,
    )

    document = create_document(session)

    reader = Mock()
    reader.render_page.return_value = b"first image"

    page = create_page(
        page_number=1,
        text="",
    )

    # First processing
    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    assert ocr_processor.recognize.call_count == 1

    # The page image has changed
    reader.render_page.return_value = b"different image"

    page = create_page(
        page_number=1,
        text="",
    )

    # Second processing
    pipeline._process_page(
        document=document,
        page=page,
        reader=reader,
    )

    # OCR must run again
    assert ocr_processor.recognize.call_count == 2
