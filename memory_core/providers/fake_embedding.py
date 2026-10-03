"""Deterministic, dependency-free embedding provider.

Purpose
-------
* Let the whole vector pipeline be tested with no model, no network and no
  third-party maths library.
* Provide a stable, reproducible embedding so that retrieval quality, hybrid
  fusion and index rebuilds can be asserted exactly.

How it works
------------
A signed feature-hashing bag-of-features embedding:

1. The text is reduced to features - word tokens, word character bigrams, and
   per-character plus bigram features for CJK.
2. Each feature is hashed with SHA-1 (never ``hash()``, which is randomised per
   process) into one of ``dim`` buckets, with a sign derived from the hash.
3. The vector is L2-normalised.

Texts sharing features therefore get a high cosine similarity, which is enough
for retrieval tests to be meaningful without being a learned model. This is a
*test double*, not a semantic model - it is named ``fake`` for that reason.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, List, Optional, Sequence

from memory_core.domain.errors import ProviderError
from memory_core.providers.base import (
    EmbeddingCapabilities,
    EmbeddingProvider,
    normalize_embedding_texts,
)

_CJK_RANGES = (
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
)


def _is_cjk(char):
    # type: (str) -> bool
    code = ord(char)
    for low, high in _CJK_RANGES:
        if low <= code <= high:
            return True
    return False


def _runs(text):
    # type: (str) -> List[List[str]]
    """Split into [kind, chars] runs where kind is 'word' or 'cjk'."""
    grouped = []  # type: List[List[Any]]
    for char in text.lower():
        if _is_cjk(char):
            kind = "cjk"
        elif char.isalnum():
            kind = "word"
        else:
            continue
        if grouped and grouped[-1][0] == kind:
            grouped[-1][1].append(char)
        else:
            grouped.append([kind, [char]])
    return grouped


def feature_list(text):
    # type: (str) -> List[str]
    """Extract hashing features from a text. Public so tests can inspect it."""
    if not isinstance(text, str):
        return []
    features = []
    for kind, chars in _runs(text):
        run = "".join(chars)
        if kind == "word":
            features.append("w:" + run)
            for index in range(len(run) - 1):
                features.append("g:" + run[index:index + 2])
        else:
            for char in run:
                features.append("c:" + char)
            for index in range(len(run) - 1):
                features.append("b:" + run[index:index + 2])
    return features


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic, offline embedding provider."""

    name = "fake-embedding"

    def __init__(
        self,
        model="fake-embed",
        dim=64,               # type: int
        fail_with=None,       # type: Optional[str]
        salt="",              # type: str
    ):
        # type: (...) -> None
        if int(dim) <= 0:
            raise ProviderError("embedding dim must be positive")
        self.model = model
        self.dim = int(dim)
        self.fail_with = fail_with
        #: Changing the salt produces a different (but still deterministic)
        #: embedding for the same text - used to simulate a different model.
        self.salt = salt
        self.calls = 0

    def capabilities(self):
        # type: () -> EmbeddingCapabilities
        return EmbeddingCapabilities(dim=self.dim, max_batch=512, supports_batching=True)

    def _vector(self, text):
        # type: (str) -> List[float]
        buckets = [0.0] * self.dim
        for feature in feature_list(text):
            digest = hashlib.sha1((self.salt + feature).encode("utf-8")).hexdigest()
            value = int(digest[:12], 16)
            index = value % self.dim
            sign = 1.0 if (value >> 40) & 1 else -1.0
            buckets[index] += sign
        norm = math.sqrt(sum(value * value for value in buckets))
        if norm == 0.0:
            return [0.0] * self.dim
        return [value / norm for value in buckets]

    def embed(self, texts):
        # type: (Sequence[str]) -> List[List[float]]
        items = normalize_embedding_texts(texts)
        if self.fail_with:
            raise ProviderError(self.fail_with, provider=self.name, model=self.model)
        self.calls += len(items)
        return [self._vector(text) for text in items]


__all__ = ["FakeEmbeddingProvider", "feature_list"]
