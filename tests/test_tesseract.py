from app.ocr import TesseractOCREngine


def main():
    engine = TesseractOCREngine()

    text = engine.recognize(
        "data/input/test_page.png",
    )

    print(text)


if __name__ == "__main__":
    main()
