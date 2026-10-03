"""Text normalisation for SQLite FTS5.

Why this module exists
----------------------
FTS5's built-in tokenisers treat a run of CJK characters as a *single* token.
With the default tokeniser, indexing "用户偏好本地优先" produces one token, so a
search for "偏好" matches nothing at all. That would make the memory core
unusable for Chinese content.

Fix: store a normalised form of the text in the FTS index (never in the source
tables) in which every CJK character is its own token. English words are
lowercased. Punctuation and whitespace are dropped, which usefully makes
"用户偏好：本地优先" and "用户偏好本地优先" normalise identically.

Queries use the same normalisation and are matched with OR, so BM25 ranks
documents matching more of the query higher. An exact-phrase bonus is applied
separately during scoring (see ``retrieval.fusion``).

This is a lexical approximation, not a segmentation model. Its limits are known
and are exactly what the deferred vector phase is meant to address.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

#: CJK-ish ranges tokenised per character.
_CJK_RANGES = (
    (0x3040, 0x30FF),    # Hiragana / Katakana
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0xAC00, 0xD7AF),    # Hangul syllables
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x20000, 0x2FA1F),  # CJK Unified Ideographs Extension B and beyond
)

CJK = "cjk"
WORD = "word"
SEPARATOR = "sep"


def _is_cjk(char):
    # type: (str) -> bool
    code = ord(char)
    for low, high in _CJK_RANGES:
        if low <= code <= high:
            return True
    return False


def classify(char):
    # type: (str) -> str
    """Classify a single character as ``cjk``, ``word`` or ``sep``."""
    if _is_cjk(char):
        return CJK
    if char.isalnum():
        return WORD
    return SEPARATOR


def runs(text):
    # type: (str) -> List[Tuple[str, str]]
    """Split text into maximal runs of the same class, dropping separators.

    A separator terminates the current run, so ``"Local-First"`` yields two word
    runs and ``"本地：优先"`` yields two CJK runs. For CJK this is invisible in
    the output (each character is already its own token), which is why
    punctuation differences do not affect CJK matching.
    """
    if not isinstance(text, str):
        return []
    grouped = []  # type: List[Tuple[str, str]]
    current_kind = None  # type: Optional[str]
    current = []  # type: List[str]
    for char in text:
        kind = classify(char)
        if kind == SEPARATOR:
            if current:
                grouped.append((current_kind, "".join(current)))  # type: ignore[arg-type]
                current = []
                current_kind = None
            continue
        if kind != current_kind:
            if current:
                grouped.append((current_kind, "".join(current)))  # type: ignore[arg-type]
            current = [char]
            current_kind = kind
        else:
            current.append(char)
    if current:
        grouped.append((current_kind, "".join(current)))  # type: ignore[arg-type]
    return grouped


def normalize_search_text(text):
    # type: (str) -> str
    """Produce the string that is stored in the FTS ``content`` column."""
    tokens = []  # type: List[str]
    for kind, value in runs(text):
        if kind == CJK:
            tokens.extend(value)
        else:
            tokens.append(value.lower())
    return " ".join(tokens)


def query_units(query):
    # type: (str) -> List[str]
    """Reduce a query to individual match units (CJK character or word)."""
    units = []  # type: List[str]
    for kind, value in runs(query):
        if kind == CJK:
            units.extend(value)
        else:
            units.append(value.lower())
    return units


def build_match_expression(query):
    # type: (str) -> Optional[str]
    """Build a safe FTS5 MATCH expression, or ``None`` if the query is empty.

    The expression is assembled only from characters we emitted ourselves, and
    every unit is quoted, so user input can never inject FTS5 syntax (``"``,
    ``*``, ``NEAR``, ``-`` and friends). Ranking is left to BM25.
    """
    units = query_units(query)
    if not units:
        return None
    quoted = []
    for unit in units:
        quoted.append('"' + unit.replace('"', '""') + '"')
    return " OR ".join(quoted)


def phrase_needle(query):
    # type: (str) -> str
    """Normalised needle used for the exact-phrase bonus.

    Must be produced with the same normalisation as the indexed text, or the
    substring test would never succeed for CJK content.
    """
    return normalize_search_text(query)


__all__ = [
    "normalize_search_text",
    "build_match_expression",
    "phrase_needle",
    "query_units",
    "runs",
    "classify",
    "CJK",
    "WORD",
    "SEPARATOR",
]
