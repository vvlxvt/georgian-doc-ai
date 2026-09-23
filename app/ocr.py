# app/ocr.py

from abc import ABC, abstractmethod


class OCRProcessor(ABC):

    @abstractmethod
    def recognize(self, image: bytes) -> str:
        pass


class FakeOCRProcessor(OCRProcessor):

    def recognize(self, image: bytes) -> str:
        return "ტესტური ტექსტი OCR-დან"
