from pathlib import Path

from app.database import SessionLocal, init_db
from app.models import Document, DocumentPage, PageStatus
from app.ocr import TesseractOCREngine
from app.pipeline import DocumentPipeline

PDF_PATH = Path("data/input/bad_text.pdf")


def main():
    init_db()

    session = SessionLocal()

    try:
        ocr_processor = TesseractOCREngine(
            languages="kat+eng",
            psm=3,
        )

        pipeline = DocumentPipeline(
            session=session,
            ocr_processor=ocr_processor,
        )

        document_id = pipeline.process(
            str(PDF_PATH),
        )

        document = session.get(
            Document,
            document_id,
        )

        assert document is not None

        pages = (
            session.query(DocumentPage)
            .filter_by(document_id=document.id)
            .order_by(DocumentPage.page_number)
            .all()
        )

        assert pages

        for page in pages:
            print(f"\nPage {page.page_number}: {page.status}")
            print(f"Method: {page.extraction_method}")
            print(page.text[:500])

        assert all(page.status == PageStatus.PARSED for page in pages)

        print("\nTest passed.")

    finally:
        session.close()


if __name__ == "__main__":
    main()
