from app.chunking import TextChunker


def test_empty_text():
    chunker = TextChunker(max_length=100)

    result = chunker.chunk("", page_number=1)

    assert result == []


def test_single_paragraph():
    chunker = TextChunker(max_length=100)

    result = chunker.chunk(
        "This is a test paragraph.",
        page_number=2,
    )

    assert len(result) == 1
    assert result[0].text == "This is a test paragraph."
    assert result[0].page_number == 2
    assert result[0].chunk_index == 0


def test_multiple_paragraphs():
    chunker = TextChunker(max_length=100)

    text = (
        "First paragraph.\n\n"
        "Second paragraph.\n\n"
        "Third paragraph."
    )

    result = chunker.chunk(text, page_number=1)

    assert len(result) == 1

    assert result[0].text == (
        "First paragraph.\n\n"
        "Second paragraph.\n\n"
        "Third paragraph."
    )


def test_split_into_chunks():
    chunker = TextChunker(max_length=30)

    text = (
        "First paragraph is here.\n\n"
        "Second paragraph is here.\n\n"
        "Third paragraph is here."
    )

    result = chunker.chunk(text, page_number=3)

    assert len(result) == 3

    assert result[0].chunk_index == 0
    assert result[1].chunk_index == 1
    assert result[2].chunk_index == 2

    assert all(chunk.page_number == 3 for chunk in result)


def test_large_paragraph_is_split_by_sentences():
    chunker = TextChunker(max_length=50)

    text = (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence."
    )

    result = chunker.chunk(text, page_number=1)

    assert len(result) == 3
    assert all(len(chunk.text) <= 50 for chunk in result)


def test_multiple_paragraphs_split_by_length():
    chunker = TextChunker(max_length=35)

    text = (
        "First paragraph.\n\n"
        "Second paragraph.\n\n"
        "Third paragraph."
    )

    result = chunker.chunk(text, page_number=1)

    assert len(result) == 2

    assert result[0].text == (
        "First paragraph.\n\n"
        "Second paragraph."
    )

    assert result[1].text == "Third paragraph."