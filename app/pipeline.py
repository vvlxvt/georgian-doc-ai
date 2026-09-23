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
        
        reader = PDFReader(str(path))
        pages = reader.pages()

        document = Document(
            filename=path.name,
            file_hash=self._hash_file(path),
        )

        self.session.add(document)
        self.session.commit()

        for page in pages:
            self._process_page(
                document=document,
                page=page,
            )

        return document.id

    def _process_page(self, document, page):

        existing = (
            self.session.query(DocumentPage)
            .filter_by(
                document_id=document.id,
                page_number=page.page_number,
                content_hash=page.content_hash,
            )
            .first()
        )

        if existing and existing.status == PageStatus.PARSED:
            print(f"Page {page.page_number}: skipped")
            return

        if len(page.text) >= MIN_TEXT_LENGTH:

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

        print(f"Page {page.page_number}: OCR required")

        # OCR добавим следующим шагом

    @staticmethod
    def _hash_file(path: Path) -> str:

        import hashlib

        hasher = hashlib.sha256()

        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                hasher.update(chunk)

        return hasher.hexdigest()
