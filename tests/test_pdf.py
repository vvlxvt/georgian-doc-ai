from app import pdf
from app.pdf import PDFReader


def test_pdf_reader_reports_native_text_extraction(monkeypatch):
    class FakePage:
        def get_text(self, mode):
            assert mode == "text"
            return "Native text"

        def get_pixmap(self, dpi, alpha):
            assert dpi == 300
            assert alpha is False
            return self

        def tobytes(self, image_format):
            assert image_format == "png"
            return b"rendered page"

    class FakeDocument:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __len__(self):
            return 1

        def load_page(self, page_index):
            assert page_index == 0
            return FakePage()

    monkeypatch.setattr(pdf.pymupdf, "open", lambda path: FakeDocument())
    events = []
    reader = PDFReader(
        "document.pdf",
        progress_callback=lambda step, status: events.append((step, status)),
    )

    pages = list(reader.pages())

    assert pages[0].text == "Native text"
    assert pages[0].total_pages == 1
    assert events == [
        ("extraction", "processing"),
        ("extraction", "completed"),
    ]
