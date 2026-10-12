from dataclasses import dataclass

from app.retrieval import SearchResult


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    chunk_id: str
    document_id: str
    page_number: int
    text: str


@dataclass(frozen=True)
class BuiltContext:
    sources: tuple[SourceRecord, ...]
    text: str


class ContextBuilder:
    def build(self, results: list[SearchResult]) -> str:
        return self.build_context(results).text

    def build_context(self, results: list[SearchResult]) -> BuiltContext:
        if not results:
            return BuiltContext(sources=(), text="")

        sources: list[SourceRecord] = []

        for result in results:
            chunk_id = result.id.strip() if isinstance(result.id, str) else ""
            document_id = result.payload.get("document_id")
            page_number = result.payload.get("page_number")
            raw_text = result.payload.get("text")
            text = raw_text.strip() if isinstance(raw_text, str) else ""

            if (
                not chunk_id
                or not isinstance(document_id, str)
                or not document_id.strip()
                or not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or page_number < 1
                or not text
            ):
                continue

            sources.append(
                SourceRecord(
                    source_id=f"S{len(sources) + 1}",
                    chunk_id=chunk_id,
                    document_id=document_id.strip(),
                    page_number=page_number,
                    text=text,
                )
            )

        sections = [
            (
                f"НАЧАЛО_ИСТОЧНИКА [стр. {source.page_number}] "
                f"[ID источника: {source.source_id}]\n"
                f"{source.text}\nКОНЕЦ_ИСТОЧНИКА"
            )
            for source in sources
        ]

        return BuiltContext(sources=tuple(sources), text="\n\n".join(sections))
