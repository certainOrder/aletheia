from app.utils.chunking import chunk_text


def test_chunk_text_sentence_aware_and_overlap():
    text = "One. Two three four. Five six seven eight nine. Ten."
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 2
    # Ensure max length constraint
    assert all(len(c) <= 20 for c in chunks)
    # Overlap property: start of second chunk should include tail of first
    if len(chunks) >= 2:
        assert chunks[1][:5] == chunks[0][-5:]


def test_chunk_text_fallback_naive():
    text = "x" * 55
    chunks = chunk_text(text, chunk_size=20, overlap=3)
    assert chunks and all(len(c) <= 20 for c in chunks)
    # Expect at least 3 chunks: 20, then 20-3 overlap, then remainder
    assert len(chunks) >= 3
