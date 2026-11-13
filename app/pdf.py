import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class PDFPage:
    page_number: int
    text: str
    content_hash: str
    total_pages: int = 0


class PDFReader:

    def __init__(
        self,
        path: str,
        progress_callback: Callable[[str, str], None] | None = None,
    ):
        self.path = str(Path(path).resolve())
        self.progress_callback = progress_callback

    def pages(self) -> Iterator[PDFPage]:
        with pymupdf.open(self.path) as doc:
            total_pages = len(doc)
            for page_index in range(total_pages):
                self._report_progress("processing")
                page = doc.load_page(page_index)

                text = page.get_text("text") or ""

                pix = page.get_pixmap(
                    dpi=300,
                    alpha=False,
                )

                content_hash = hashlib.sha256(pix.tobytes("png")).hexdigest()

                self._report_progress("completed")
                yield PDFPage(
                    page_number=page_index + 1,
                    text=text,
                    content_hash=content_hash,
                    total_pages=total_pages,
                )

    def render_page(self, page_number: int) -> bytes:
        with pymupdf.open(self.path) as doc:
            page = doc.load_page(page_number - 1)

            pix = page.get_pixmap(
                dpi=300,
                alpha=False,
            )

            return pix.tobytes("png")

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _report_progress(self, status: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback("extraction", status)
