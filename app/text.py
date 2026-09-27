from dataclasses import dataclass


class TextNormalizer:

    def normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        lines = [" ".join(line.split()) for line in text.split("\n")]

        normalized = "\n".join(lines)

        while "\n\n\n" in normalized:
            normalized = normalized.replace("\n\n\n", "\n\n")

        return normalized.strip()


@dataclass
class TextChunk:
    document_id: str
    page_number: int
    chunk_index: int
    text: str
