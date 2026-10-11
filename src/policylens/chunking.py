from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Chunk, Page


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def fixed_size_chunks(
    pages: Iterable[Page], chunk_words: int = 180, overlap_words: int = 35
) -> list[Chunk]:
    if chunk_words <= 0 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Require chunk_words > overlap_words >= 0")

    chunks: list[Chunk] = []
    for page in pages:
        tokens = _words(page.text)
        step = chunk_words - overlap_words
        for offset in range(0, len(tokens), step):
            part = tokens[offset : offset + chunk_words]
            if not part:
                continue
            chunks.append(
                Chunk(
                    id=f"{page.document}:p{page.number}:fixed:{offset}",
                    document=page.document,
                    pages=(page.number,),
                    text=" ".join(part),
                )
            )
            if offset + chunk_words >= len(tokens):
                break
    return chunks


def structure_aware_chunks(
    pages: Iterable[Page], max_words: int = 220
) -> list[Chunk]:
    """Split at headings/paragraphs, combining small adjacent paragraphs."""
    chunks: list[Chunk] = []
    for page in pages:
        blocks = [block.strip() for block in re.split(r"\n\s*\n+", page.text) if block.strip()]
        if len(blocks) == 1:
            blocks = [line.strip() for line in page.text.splitlines() if line.strip()]

        heading: str | None = None
        buffer: list[str] = []
        count = 0
        index = 0

        def flush() -> None:
            nonlocal buffer, count, index
            if not buffer:
                return
            chunks.append(
                Chunk(
                    id=f"{page.document}:p{page.number}:section:{index}",
                    document=page.document,
                    pages=(page.number,),
                    text="\n\n".join(buffer),
                    heading=heading,
                )
            )
            index += 1
            buffer = []
            count = 0

        for block in blocks:
            block_words = _words(block)
            looks_like_heading = len(block_words) <= 10 and (
                block.isupper() or block.endswith(":") or re.match(r"^\d+(?:\.\d+)*\s", block)
            )
            if looks_like_heading:
                flush()
                heading = block.rstrip(":")
                continue
            if count and count + len(block_words) > max_words:
                flush()
            if len(block_words) > max_words:
                for start in range(0, len(block_words), max_words):
                    buffer = [" ".join(block_words[start : start + max_words])]
                    count = len(_words(buffer[0]))
                    flush()
                continue
            buffer.append(block)
            count += len(block_words)
        flush()
    return chunks

