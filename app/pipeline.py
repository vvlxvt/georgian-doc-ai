import hashlib
import tempfile
from pathlib import Path

from app.chunking import TextChunker
from app.models import Document, DocumentPage, PageStatus
from app.ocr import OCRProcessor
from app.pdf import PDFPage, PDFReader
from app.repositories import DocumentChunkRepository
from app.text import TextNormalizer

MIN_TEXT_LENGTH = 20


class DocumentPipeline:
    def __init__(
        self,
        session,
        ocr_processor: OCRProcessor,
        text_normalizer: TextNormalizer,
        text_chunker: TextChunker,
    ):
        self.session = session
        self.ocr_processor = ocr_processor
        self.text_normalizer = text_normalizer
        self.text_chunker = text_chunker
        self.chunk_repository = DocumentChunkRepository(session)

    def _get_page(
        self,
        document: Document,
        page_number: int,
    ) -> DocumentPage | None:
        return (
            self.session.query(DocumentPage)
            .filter_by(
                document_id=document.id,
                page_number=page_number,
            )
            .first()
        )

    def _is_already_processed(
        self,
        existing_page: DocumentPage | None,
        page: PDFPage,
    ) -> bool:
        return (
            existing_page is not None
            and existing_page.content_hash == page.content_hash
            and existing_page.status
            in (
                PageStatus.PARSED,
                PageStatus.OCR,
            )
        )


    def process(self, path: str) -> str:
        path = Path(path)
        file_hash = self._hash_file(path)

        document = (
            self.session.query(Document)
            .filter_by(file_hash=file_hash)
            .first()
        )

        if not document:
            document = Document(
                filename=path.name,
                file_hash=file_hash,
            )
            self.session.add(document)
            self.session.commit()

        reader = PDFReader(str(path))

        for page in reader.pages():
            self._process_page(
                document=document,
                page=page,
                reader=reader,
            )

        return document.id

    def _save_page_text(
        self,
        document: Document,
        page: PDFPage,
        text: str,
        status: PageStatus,
        extraction_method: str,
    ) -> None:
        text = self.text_normalizer.normalize(text)

        if not text:
            return

        db_page = self._get_or_create_page(
            document=document,
            page=page,
        )

        db_page.content_hash = page.content_hash
        db_page.status = status
        db_page.extraction_method = extraction_method
        db_page.text = text
        db_page.error = None

        self.session.flush()

        chunks = self.text_chunker.chunk(
            text=text,
            page_number=page.page_number,
        )

        self.chunk_repository.delete_by_page(db_page.id)

        self.chunk_repository.create_many(
            page_id=db_page.id,
            page_number=page.page_number,
            chunks=chunks,
        )

        self.session.commit()

        print(
            f"Page {page.page_number}: "
            f"{extraction_method}, {len(chunks)} chunks"
        )

    def _get_or_create_page(
        self,
        document: Document,
        page: PDFPage,
    ) -> DocumentPage:
        db_page = self._get_page(
            document=document,
            page_number=page.page_number,
        )

        if db_page:
            return db_page

        db_page = DocumentPage(
            document_id=document.id,
            page_number=page.page_number,
            content_hash=page.content_hash,
        )

        self.session.add(db_page)

        return db_page

    def _process_page(
        self,
        document: Document,
        page: PDFPage,
        reader: PDFReader,
    ) -> None:
        existing_page = self._get_page(
            document=document,
            page_number=page.page_number,
        )

        if len(page.text) >= MIN_TEXT_LENGTH:
            if self._is_already_processed(
                existing_page=existing_page,
                page=page,
            ):
                print(
                    f"Page {page.page_number}: "
                    "skipped (already processed)"
                )
                return

            self._save_page_text(
                document=document,
                page=page,
                text=page.text,
                status=PageStatus.PARSED,
                extraction_method="native",
            )
            return

        print(f"Page {page.page_number}: OCR required")

        text = self._run_ocr(
            document=document,
            page=page,
            reader=reader,
        )

        if text is None:
            return

        self._save_page_text(
            document=document,
            page=page,
            text=text,
            status=PageStatus.OCR,
            extraction_method="tesseract",
        )

    def _run_ocr(
        self,
        document: Document,
        page: PDFPage,
        reader: PDFReader,
    ) -> str | None:
        try:
            image_bytes = reader.render_page(page.page_number)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "page.png"
                temp_path.write_bytes(image_bytes)

                text = self.ocr_processor.recognize(
                    str(temp_path),
                )

        except Exception as exc:
            self._save_ocr_failure(
                document=document,
                page=page,
                error=f"OCR failed: {exc}",
            )
            return None

        if not text.strip():
            self._save_ocr_failure(
                document=document,
                page=page,
                error="OCR returned empty text",
            )
            return None

        return text

    def _save_ocr_failure(
        self,
        document: Document,
        page: PDFPage,
        error: str,
    ) -> None:
        db_page = self._get_or_create_page(
            document=document,
            page=page,
        )

        db_page.content_hash = page.content_hash
        db_page.status = PageStatus.OCR_FAILED
        db_page.extraction_method = "tesseract"
        db_page.error = error

        self.session.commit()

        print(f"Page {page.page_number}: {error}")

    @staticmethod
    def _hash_file(path: Path) -> str:
        hasher = hashlib.sha256()

        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                hasher.update(chunk)

        return hasher.hexdigest()
