import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    page_number: int
    chunk_index: int
    text: str


class TextChunker:
    """Splits text into chunks whose maximum length is measured in characters."""

    def __init__(
        self,
        max_length: int = 1000,
        overlap: int = 100,
    ):
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0")

        if overlap < 0:
            raise ValueError("overlap must not be negative")

        self.max_length = max_length
        self.overlap = overlap

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

            # The paragraph itself is too large.
            if len(paragraph) > self.max_length:

                # Save the current chunk first.
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

                # Process the large paragraph separately.
                paragraph_chunks = self._chunk_large_paragraph(
                    paragraph,
                    page_number,
                    chunk_index,
                )

                chunks.extend(paragraph_chunks)
                chunk_index += len(paragraph_chunks)

                continue

            # Length of the paragraph plus the separator "\n\n".
            additional_length = (
                len(paragraph) if not current_parts else len(paragraph) + 2
            )

            # The paragraph does not fit into the current chunk.
            if current_parts and current_length + additional_length > self.max_length:
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
                len(paragraph) if len(current_parts) == 1 else len(paragraph) + 2
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

        if not sentences:
            return []

        chunks = []
        current_parts = []
        chunk_index = start_index

        for sentence in sentences:

            # A single sentence is already too large.
            if len(sentence) > self.max_length:

                if current_parts:
                    chunks.append(
                        self._create_chunk(
                            current_parts,
                            page_number,
                            chunk_index,
                            separator=" ",
                        )
                    )
                    chunk_index += 1

                sentence_chunks = self._chunk_long_sentence(
                    sentence,
                    page_number,
                    chunk_index,
                )

                chunks.extend(sentence_chunks)
                chunk_index += len(sentence_chunks)

                current_parts = []
                continue

            candidate = current_parts + [sentence]

            # The sentence does not fit.
            if current_parts and len(" ".join(candidate)) > self.max_length:
                chunks.append(
                    self._create_chunk(
                        current_parts,
                        page_number,
                        chunk_index,
                        separator=" ",
                    )
                )

                chunk_index += 1

                overlap_parts = self._get_overlap(current_parts)

                current_parts = overlap_parts + [sentence]

                # If overlap + sentence is already too large,
                # remove overlap until the new chunk fits.
                while len(" ".join(current_parts)) > self.max_length and overlap_parts:
                    overlap_parts.pop(0)
                    current_parts = overlap_parts + [sentence]

            else:
                current_parts = candidate

        if current_parts:
            chunks.append(
                self._create_chunk(
                    current_parts,
                    page_number,
                    chunk_index,
                    separator=" ",
                )
            )

        return chunks

    def _get_overlap(
        self,
        parts: list[str],
    ) -> list[str]:
        if self.overlap == 0:
            return []

        overlap_parts = []
        word_count = 0

        for part in reversed(parts):
            words = part.split()

            if word_count + len(words) > self.overlap:
                break

            overlap_parts.insert(0, part)
            word_count += len(words)

        return overlap_parts

    def _chunk_long_sentence(
        self,
        sentence: str,
        page_number: int,
        start_index: int,
    ) -> list[TextChunk]:
        words = sentence.split()

        chunks = []
        current_words = []
        chunk_index = start_index

        for word in words:
            if len(word) > self.max_length:
                if current_words:
                    chunks.append(
                        self._create_chunk(
                            current_words,
                            page_number,
                            chunk_index,
                            separator=" ",
                        )
                    )
                    chunk_index += 1
                    current_words = []

                for offset in range(0, len(word), self.max_length):
                    chunks.append(
                        TextChunk(
                            page_number=page_number,
                            chunk_index=chunk_index,
                            text=word[offset : offset + self.max_length],
                        )
                    )
                    chunk_index += 1

                continue

            candidate = current_words + [word]

            if current_words and len(" ".join(candidate)) > self.max_length:
                chunks.append(
                    self._create_chunk(
                        current_words,
                        page_number,
                        chunk_index,
                        separator=" ",
                    )
                )

                chunk_index += 1
                current_words = [word]

            else:
                current_words = candidate

        if current_words:
            chunks.append(
                self._create_chunk(
                    current_words,
                    page_number,
                    chunk_index,
                    separator=" ",
                )
            )

        return chunks

    @staticmethod
    def _create_chunk(
        parts: list[str],
        page_number: int,
        chunk_index: int,
        separator: str = "\n\n",
    ) -> TextChunk:
        return TextChunk(
            page_number=page_number,
            chunk_index=chunk_index,
            text=separator.join(parts),
        )

    @staticmethod
    def _split_into_paragraphs(text: str) -> list[str]:
        return [
            paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()
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
