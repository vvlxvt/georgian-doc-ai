import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    page_number: int
    chunk_index: int
    text: str


class TextChunker:

    def __init__(self, max_length: int = 1000):
        self.max_length = max_length

    def chunk(
        self,
        text: str,
        page_number: int,
    ) -> list[TextChunk]:
        paragraphs = self._split_into_paragraphs(text)

        chunks = []
        current_parts = []
        current_length = 0
        chunk_index = 0

        for paragraph in paragraphs:
            # Paragraph itself is too large.
            if len(paragraph) > self.max_length:
                # First save the current chunk.
                if current_parts:
                    chunks.append(
                        self._create_chunk(
                            current_parts,
                            page_number,
                            chunk_index,
                        )
                    )
                    chunk_index += 1
                    current_parts = []
                    current_length = 0

                # Split the large paragraph separately.
                paragraph_chunks = self._chunk_large_paragraph(
                    paragraph,
                    page_number,
                    chunk_index,
                )

                chunks.extend(paragraph_chunks)
                chunk_index += len(paragraph_chunks)

                continue

            paragraph_length = len(paragraph)

            # Can this paragraph be added to the current chunk?
            additional_length = (
                paragraph_length
                if not current_parts
                else paragraph_length + 2
            )

            if (
                current_parts
                and current_length + additional_length > self.max_length
            ):
                chunks.append(
                    self._create_chunk(
                        current_parts,
                        page_number,
                        chunk_index,
                    )
                )

                chunk_index += 1
                current_parts = []
                current_length = 0

            current_parts.append(paragraph)

            current_length += (
                paragraph_length
                if len(current_parts) == 1
                else paragraph_length + 2
            )

        if current_parts:
            chunks.append(
                self._create_chunk(
                    current_parts,
                    page_number,
                    chunk_index,
                )
            )

        return chunks

    def _chunk_large_paragraph(
        self,
        paragraph: str,
        page_number: int,
        start_index: int,
    ) -> list[TextChunk]:
        sentences = self._split_into_sentences(paragraph)

        chunks = []
        current_parts = []
        current_length = 0
        chunk_index = start_index

        for sentence in sentences:
            sentence_length = len(sentence)

            if (
                current_parts
                and current_length + sentence_length + 1
                > self.max_length
            ):
                chunks.append(
                    self._create_chunk(
                        current_parts,
                        page_number,
                        chunk_index,
                    )
                )

                chunk_index += 1
                current_parts = []
                current_length = 0

            current_parts.append(sentence)
            current_length += (
                sentence_length
                if len(current_parts) == 1
                else sentence_length + 1
            )

        if current_parts:
            chunks.append(
                self._create_chunk(
                    current_parts,
                    page_number,
                    chunk_index,
                )
            )

        return chunks

    @staticmethod
    def _create_chunk(
        parts: list[str],
        page_number: int,
        chunk_index: int,
    ) -> TextChunk:
        return TextChunk(
            page_number=page_number,
            chunk_index=chunk_index,
            text="\n\n".join(parts),
        )

    @staticmethod
    def _split_into_paragraphs(text: str) -> list[str]:
        return [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                text,
            )
            if sentence.strip()
        ]