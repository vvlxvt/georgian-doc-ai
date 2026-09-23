# app/pdf.py

from dataclasses import dataclass

import fitz


@dataclass
class PDFPage:
    page_number: int
    text: str
    content_hash: str


class PDFReader:

    def __init__(self, path: str):
        self.path = path

    def pages(self) -> list[PDFPage]:
        document = fitz.open(self.path)

        result = []

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            result.append(
                PDFPage(
                    page_number=page_number,
                    text=text,
                    content_hash=self._hash_page(page),
                )
            )

        document.close()

        return result

    @staticmethod
    def _hash_page(page) -> str:
        import hashlib

        pix = page.get_pixmap(
            matrix=fitz.Matrix(1, 1),
            alpha=False,
        )

        return hashlib.sha256(pix.tobytes("png")).hexdigest()
