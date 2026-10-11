from app.retrieval import SearchResult


class ContextBuilder:
    def build(self, results: list[SearchResult]) -> str:
        if not results:
            return ""

        sections = []

        for result in results:
            page_number = result.payload.get("page_number")
            text = result.payload.get("text", "").strip()

            if not text:
                continue

            if page_number is not None:
                header = f"НАЧАЛО_ИСТОЧНИКА [стр. {page_number}]"
            else:
                header = "НАЧАЛО_ИСТОЧНИКА [страница неизвестна]"

            sections.append(f"{header}\n{text}\nКОНЕЦ_ИСТОЧНИКА")

        return "\n\n".join(sections)
