from app.context import ContextBuilder
from app.retrieval import SearchResult


def test_context_builder_builds_context():
    results = [
        SearchResult(
            id="chunk-1",
            score=0.91,
            payload={
                "document_id": "doc-1",
                "page_id": "page-1",
                "page_number": 3,
                "chunk_index": 0,
                "text": "საქართველოს კანონის პირველი მუხლი.",
            },
        ),
        SearchResult(
            id="chunk-2",
            score=0.82,
            payload={
                "document_id": "doc-1",
                "page_id": "page-2",
                "page_number": 7,
                "chunk_index": 1,
                "text": "მეორე მნიშვნელოვანი დებულება.",
            },
        ),
    ]

    builder = ContextBuilder()

    context = builder.build(results)

    assert "НАЧАЛО_ИСТОЧНИКА [стр. 3]" in context
    assert "НАЧАЛО_ИСТОЧНИКА [стр. 7]" in context
    assert context.count("КОНЕЦ_ИСТОЧНИКА") == 2
    assert "საქართველოს კანონის პირველი მუხლი." in context
    assert "მეორე მნიშვნელოვანი დებულება." in context


def test_context_builder_preserves_result_order():
    results = [
        SearchResult(
            id="chunk-1",
            score=0.9,
            payload={
                "page_number": 1,
                "text": "First chunk",
            },
        ),
        SearchResult(
            id="chunk-2",
            score=0.8,
            payload={
                "page_number": 2,
                "text": "Second chunk",
            },
        ),
    ]

    context = ContextBuilder().build(results)

    assert context.index("First chunk") < context.index("Second chunk")


def test_context_builder_returns_empty_string_for_no_results():
    context = ContextBuilder().build([])

    assert context == ""
