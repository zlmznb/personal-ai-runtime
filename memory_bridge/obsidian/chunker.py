"""Deterministic Markdown chunking.

A whole note is a bad unit for memory formation: it is long, it mixes several
topics, and one edit would invalidate all of it. Chunks are the unit of
ingestion, and their content hashes are what make re-import idempotent.

Determinism is the requirement: the same file must always produce the same
chunks with the same boundaries and the same hashes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from memory_bridge.obsidian.provenance import hash_text, normalize_text

#: ``## Heading`` / ``###### Heading``
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

#: Fence markers that start/end a code block.
_FENCE_MARKERS = ("```", "~~~")

#: Blocks shorter than this are headings, rules or stray fragments.
MIN_CHUNK_CHARS = 40

#: A chunk larger than this is split further at paragraph boundaries.
MAX_CHUNK_CHARS = 2000


@dataclass(frozen=True)
class Chunk(object):
    """One ingestible unit of a document."""

    ordinal: int
    heading_path: Tuple[str, ...]
    text: str
    line_start: int
    line_end: int
    content_hash: str = ""

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "ordinal": self.ordinal,
            "heading_path": list(self.heading_path),
            "text": self.text,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_hash": self.content_hash,
        }


def _sections(lines):
    # type: (Sequence[str]) -> List[Tuple[Tuple[str, ...], int, int]]
    """Split into ``(heading_path, start_index, end_index)`` sections.

    Fenced code blocks are respected, so a ``#`` inside a code sample is not
    mistaken for a heading.
    """
    sections = []
    stack = []  # type: List[str]
    current_path = ()  # type: Tuple[str, ...]
    current_start = 0
    in_fence = False
    marker = ""

    for index, line in enumerate(lines):
        stripped = line.strip()
        if in_fence:
            if stripped.startswith(marker):
                in_fence = False
                marker = ""
            continue
        if stripped.startswith(_FENCE_MARKERS):
            in_fence = True
            marker = stripped[:3]
            continue
        match = _HEADING_RE.match(stripped)
        if match:
            sections.append((current_path, current_start, index))
            level = len(match.group(1))
            stack = stack[: level - 1]
            stack.append(match.group(2).strip())
            current_path = tuple(stack)
            current_start = index + 1

    sections.append((current_path, current_start, len(lines)))
    return sections


def _strip_code(indexed_lines):
    # type: (Sequence[Tuple[int, str]]) -> List[Tuple[int, str]]
    """Drop fenced code blocks, keeping original line numbers for the rest."""
    kept = []
    in_fence = False
    marker = ""
    for index, line in indexed_lines:
        stripped = line.strip()
        if in_fence:
            if stripped.startswith(marker):
                in_fence = False
                marker = ""
            continue
        if stripped.startswith(_FENCE_MARKERS):
            in_fence = True
            marker = stripped[:3]
            continue
        kept.append((index, line))
    return kept


def _paragraphs(indexed_lines):
    # type: (Sequence[Tuple[int, str]]) -> List[List[Tuple[int, str]]]
    paragraphs = []
    buffer = []  # type: List[Tuple[int, str]]
    for index, line in indexed_lines:
        if line.strip():
            buffer.append((index, line))
        elif buffer:
            paragraphs.append(buffer)
            buffer = []
    if buffer:
        paragraphs.append(buffer)
    return paragraphs


def _build(path, sentence_paragraphs, ordinal, min_chars, max_chars, boundary_prefixes=()):
    # type: (Tuple[str, ...], Sequence[Sequence[Tuple[int, str]]], int, int, int, Sequence[str]) -> List[Chunk]
    """Group paragraphs into chunks.

    A paragraph that starts with one of ``boundary_prefixes`` begins a new chunk
    AND bypasses the minimum length. That is what lets an explicit
    ``记住：...`` buried in a long note become its own verbatim chunk, so Memory
    Core's deterministic rule extractor actually sees it at the start of the
    text. Without this, explicit declarations in real notes never fire.
    """
    lowered_boundaries = tuple(prefix.lower() for prefix in boundary_prefixes)
    chunks = []
    current = []  # type: List[Tuple[int, str]]
    current_length = 0

    def starts_with_boundary(text):
        # type: (str) -> bool
        candidate = text.strip().lower()
        return any(candidate.startswith(prefix) for prefix in lowered_boundaries)

    def flush():
        # type: () -> None
        if not current:
            return
        text = normalize_text("\n".join(line for _, line in current))
        required = 1 if starts_with_boundary(text) else min_chars
        if len(text) < required:
            return
        chunks.append(
            Chunk(
                ordinal=len(chunks),
                heading_path=path,
                text=text,
                line_start=current[0][0] + 1,
                line_end=current[-1][0] + 1,
                content_hash=hash_text(text),
            )
        )

    for paragraph in sentence_paragraphs:
        paragraph_text = "\n".join(line for _, line in paragraph)
        if starts_with_boundary(paragraph_text):
            # An explicit declaration becomes exactly ONE paragraph on its own.
            # Memory Core's rule extractor takes the whole remainder of the text
            # as the memory content, so appending the next paragraph would
            # produce a non-atomic memory holding two unrelated statements.
            flush()
            current = list(paragraph)
            current_length = len(paragraph_text) + 2
            flush()
            current = []
            current_length = 0
            continue
        if current and max_chars > 0 and current_length + len(paragraph_text) + 2 > max_chars:
            flush()
            current = []
            current_length = 0
        current.extend(paragraph)
        current_length += len(paragraph_text) + 2
    flush()
    return chunks


def chunk_markdown(
    body,
    min_chars=MIN_CHUNK_CHARS,
    max_chars=MAX_CHUNK_CHARS,
    include_code_blocks=False,
    boundary_prefixes=(),
):
    # type: (str, int, int, bool, Sequence[str]) -> List[Chunk]
    """Split a document body into deterministic, hashed chunks."""
    text = normalize_text(body)
    if not text:
        return []
    lines = text.split("\n")

    chunks = []  # type: List[Chunk]
    for heading_path, start, end in _sections(lines):
        indexed = list(enumerate(lines[start:end], start=start))
        if not include_code_blocks:
            indexed = _strip_code(indexed)
        if not indexed:
            continue
        for chunk in _build(
            heading_path,
            _paragraphs(indexed),
            len(chunks),
            int(min_chars),
            int(max_chars),
            boundary_prefixes,
        ):
            chunks.append(chunk)
    # Re-number so ordinals are dense and ordered after filtering.
    return [
        Chunk(
            ordinal=index,
            heading_path=chunk.heading_path,
            text=chunk.text,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            content_hash=chunk.content_hash,
        )
        for index, chunk in enumerate(chunks)
    ]


__all__ = ["Chunk", "chunk_markdown", "MIN_CHUNK_CHARS", "MAX_CHUNK_CHARS"]
