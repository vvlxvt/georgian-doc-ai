import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class PDFPage:
    page_number: int
    text: str
    content_hash: str


class PDFReader:

    def __init__(self, path: str):
        self.path = str(Path(path).resolve())

    def pages(self) -> Iterator[PDFPage]:
        with pymupdf.open(self.path) as doc:
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)

                text = page.get_text("text") or ""

                pix = page.get_pixmap(
                    dpi=300,
                    alpha=False,
                )

                content_hash = hashlib.sha256(pix.tobytes("png")).hexdigest()

                yield PDFPage(
                    page_number=page_index + 1,
                    text=text,
                    content_hash=content_hash,
                )

    def render_page(self, page_number: int) -> bytes:
        with pymupdf.open(self.path) as doc:
            page = doc.load_page(page_number - 1)

            pix = page.get_pixmap(
                dpi=300,
                alpha=False,
            )

            return pix.tobytes("png")
