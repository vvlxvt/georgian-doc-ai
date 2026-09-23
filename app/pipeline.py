# app/pipeline.py

from pathlib import Path

from sqlalchemy.orm import Session

from app.models import Document, DocumentPage, PageStatus
from app.pdf import PDFReader

MIN_TEXT_LENGTH = 20


class DocumentPipeline:

    def __init__(
        self,
        session: Session,
        ocr_processor,
    ):
        self.session = session
        self.ocr_processor = ocr_processor

    def process(self, path: str) -> str:
        path = Path(path)
        file_hash = self._hash_file(path)

        document = self.session.query(Document).filter_by(file_hash=file_hash).first()

        if not document:
            document = Document(
                filename=path.name,
                file_hash=file_hash,
            )
            self.session.add(document)
            self.session.commit()

        reader = PDFReader(str(path))
        pages = reader.pages()

        for page in pages:
            self._process_page(
                document=document,
                page=page,
            )

        return document.id

    def _process_page(self, document: Document, page):
        existing_page = (
            self.session.query(DocumentPage)
            .filter_by(
                document_id=document.id,
                page_number=page.page_number,
            )
            .first()
        )

        # Same page version was already processed successfully.
        if (
            existing_page
            and existing_page.status == PageStatus.PARSED
            and existing_page.content_hash == page.content_hash
        ):
            print(f"Page {page.page_number}: " "skipped (already parsed)")
            return

        # Native text extraction succeeded.
        if len(page.text) >= MIN_TEXT_LENGTH:
            if existing_page:
                existing_page.content_hash = page.content_hash
                existing_page.status = PageStatus.PARSED
                existing_page.extraction_method = "native"
                existing_page.text = page.text
                existing_page.error = None
            else:
                db_page = DocumentPage(
                    document_id=document.id,
                    page_number=page.page_number,
                    content_hash=page.content_hash,
                    status=PageStatus.PARSED,
                    extraction_method="native",
                    text=page.text,
                )
                self.session.add(db_page)

            self.session.commit()

            print(f"Page {page.page_number}: native text")
            return

        # Native text extraction was insufficient.
        print(f"Page {page.page_number}: OCR required")

    @staticmethod
    def _hash_file(path: Path) -> str:

        import hashlib

        hasher = hashlib.sha256()

        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                hasher.update(chunk)

        return hasher.hexdigest()
