from abc import ABC, abstractmethod

import pytesseract
from PIL import Image


class OCRProcessor(ABC):
    @abstractmethod
    def recognize(self, image_path: str) -> str:
        """Recognize text from an image."""
        raise NotImplementedError


class TesseractOCREngine(OCRProcessor):
    def __init__(
        self,
        languages: str = "kat+eng",
        psm: int = 3,
    ):
        self.languages = languages
        self.psm = psm

    def recognize(self, image_path: str) -> str:
        image = Image.open(image_path)

        text = pytesseract.image_to_string(
            image,
            lang=self.languages,
            config=f"--psm {self.psm}",
        )

        return text.strip()
