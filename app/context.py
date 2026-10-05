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
                header = f"[Page {page_number}]"
            else:
                header = "[Unknown page]"

            sections.append(f"{header}\n{text}")

        return "\n\n".join(sections)
