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
                "document_id": "doc-1",
                "page_number": 1,
                "text": "First chunk",
            },
        ),
        SearchResult(
            id="chunk-2",
            score=0.8,
            payload={
                "document_id": "doc-2",
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


def test_context_builder_assigns_deterministic_source_ids_and_metadata():
    results = [
        SearchResult(
            id="chunk-1",
            score=0.9,
            payload={
                "document_id": "doc-1",
                "page_number": 3,
                "text": "First source",
            },
        ),
        SearchResult(
            id="chunk-2",
            score=0.8,
            payload={
                "document_id": "doc-2",
                "page_number": 8,
                "text": "Second source",
            },
        ),
    ]

    built = ContextBuilder().build_context(results)

    assert [source.source_id for source in built.sources] == ["S1", "S2"]
    assert built.sources[0].chunk_id == "chunk-1"
    assert built.sources[0].document_id == "doc-1"
    assert built.sources[0].page_number == 3
    assert built.sources[1].document_id == "doc-2"
    assert "ID источника: S1" in built.text
    assert "ID источника: S2" in built.text
    assert "doc-1" not in built.text
    assert "chunk-1" not in built.text
    assert ContextBuilder().build_context(results).sources == built.sources


def test_context_builder_skips_results_with_invalid_source_metadata():
    results = [
        SearchResult(
            id="missing-document",
            score=0.9,
            payload={"page_number": 1, "text": "No document ID"},
        ),
        SearchResult(
            id="invalid-page",
            score=0.8,
            payload={
                "document_id": "doc-2",
                "page_number": True,
                "text": "Invalid page number",
            },
        ),
        SearchResult(
            id="empty-text",
            score=0.7,
            payload={
                "document_id": "doc-3",
                "page_number": 4,
                "text": " ",
            },
        ),
        SearchResult(
            id="valid-chunk",
            score=0.6,
            payload={
                "document_id": "doc-4",
                "page_number": 5,
                "text": "Valid source",
            },
        ),
    ]

    built = ContextBuilder().build_context(results)

    assert len(built.sources) == 1
    assert built.sources[0].source_id == "S1"
    assert built.sources[0].chunk_id == "valid-chunk"
    assert "Valid source" in built.text
    assert "No document ID" not in built.text
    assert "Invalid page number" not in built.text
