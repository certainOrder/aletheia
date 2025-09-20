"""Utilities for chunking long text into overlapping segments.

The chunker is character-based with a preference for sentence boundaries. It
ensures chunks are at most `chunk_size` characters, overlapping the previous
chunk by `overlap` characters. If sentence splitting fails (e.g., no punctuation),
we fall back to naive fixed-size chunks.
"""

from __future__ import annotations

import re


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences using a simple regex heuristic.

    This is intentionally lightweight and offline-friendly; it won't be perfect
    but is good enough for chunk boundaries. We keep punctuation attached.
    """
    text = text.strip()
    if not text:
        return []
    # Split on punctuation followed by whitespace/capitals; keep punctuation
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Normalize whitespace
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split `text` into overlapping chunks.

    Rules:
    - Each chunk length <= chunk_size
    - Consecutive chunks overlap by `overlap` characters (if possible)
    - Prefer to end chunks at sentence boundaries when it doesn't exceed chunk_size
    - Fallback to naive slicing when sentences are too long
    """
    if chunk_size <= 0:
        return []
    if overlap < 0:
        overlap = 0

    sentences = _sentence_split(text)
    if not sentences:
        # fallback: naive slicing
        naive_chunks: list[str] = []
        start = 0
        n = max(1, chunk_size)
        while start < len(text):
            end = min(len(text), start + n)
            naive_chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - overlap)
        return naive_chunks

    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if not buf:
            # start new buffer with sentence
            buf = sent
        elif len(buf) + 1 + len(sent) <= chunk_size:
            buf = f"{buf} {sent}"
        else:
            # flush current buffer
            chunks.append(buf)
            # start next buffer with overlap tail from previous
            if overlap > 0:
                tail = buf[-overlap:]
                buf = (tail + " " + sent).strip()
                # If the overlapped start already exceeds chunk size, trim from right
                # to preserve the overlap at the beginning of the chunk.
                if len(buf) > chunk_size:
                    buf = buf[:chunk_size]
            else:
                buf = sent

            # If a single sentence is longer than chunk_size, break it up
            while len(buf) > chunk_size:
                chunks.append(buf[:chunk_size])
                buf = buf[chunk_size - overlap :]
                buf = buf.strip()

    if buf:
        chunks.append(buf)

    # Final pass: ensure all are within limit
    out = []
    for c in chunks:
        if len(c) <= chunk_size:
            out.append(c)
        else:
            # last-ditch split
            start = 0
            while start < len(c):
                end = min(len(c), start + chunk_size)
                out.append(c[start:end])
                if end == len(c):
                    break
                start = max(0, end - overlap)
    return out
