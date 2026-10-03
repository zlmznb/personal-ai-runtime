"""Retrieval facade: keyword, semantic, and hybrid modes.

This layer is read-only and never consults a chat model (ADR 0003). Semantic
mode does use the *embedding* provider, which is a different thing: swapping the
chat model still cannot change what is recalled.

Three modes:

``keyword``
    FTS5 BM25 over memories and preferences. Fully deterministic.

``semantic``
    Cosine similarity through the :class:`VectorIndex`. Returns memories only;
    preferences are key/value data and are deliberately not vectorised.

``hybrid``
    Reciprocal Rank Fusion of the two. When only one source is available the
    other is returned **unchanged**, so installing an embedding provider can only
    ever add recall, never reorder an existing result set.

No mode ever writes. Index construction is explicit (see ``SemanticIndexer``),
which is what keeps recall safe to call in a byte-identity assertion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from memory_core.domain.dto import RecallHit
from memory_core.domain.enums import ItemType
from memory_core.domain.errors import ProviderError
from memory_core.retrieval import fusion, tokenize

#: Supported retrieval modes.
MODES = ("keyword", "semantic", "hybrid")


class Retriever(object):
    """Deterministic retrieval over memories and preferences."""

    def __init__(self, store, weights=None, vector_index=None, embedding_provider=None):
        # type: (Any, Any, Any, Any) -> None
        self.store = store
        self.weights = weights or fusion.DEFAULT_WEIGHTS
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider

    @property
    def semantic_available(self):
        # type: () -> bool
        return self.vector_index is not None and self.embedding_provider is not None

    def modes(self):
        # type: () -> Sequence[str]
        return MODES

    # -- public API --------------------------------------------------------

    def recall(
        self,
        query,
        scope=None,                 # type: Optional[str]
        kinds=None,                 # type: Optional[Sequence[str]]
        limit=10,                   # type: int
        include_preferences=True,   # type: bool
        mode="hybrid",              # type: str
    ):
        # type: (...) -> List[RecallHit]
        """Return ranked hits for ``query``. An unsearchable query returns []."""
        if mode not in MODES:
            raise ValueError("unknown retrieval mode {0!r}; expected one of {1}".format(mode, list(MODES)))
        if limit is None or int(limit) <= 0:
            return []
        limit = int(limit)
        fetch = max(limit * 3, limit)

        expression = tokenize.build_match_expression(query)
        keyword_hits = (
            self._keyword_hits(expression, query, scope, kinds, fetch, include_preferences)
            if expression is not None
            else []
        )

        if mode == "keyword":
            return keyword_hits[:limit]

        semantic_hits = self._semantic_hits(query, scope, kinds, fetch) if self.semantic_available else []

        if mode == "semantic":
            return semantic_hits[:limit]

        # hybrid: one source present means the other adds nothing, so return the
        # available ranking untouched. This is what guarantees that enabling the
        # vector index cannot disturb existing keyword results.
        if not semantic_hits:
            return keyword_hits[:limit]
        if not keyword_hits:
            return semantic_hits[:limit]
        return fusion.fuse_hybrid(keyword_hits, semantic_hits, self.weights, limit=limit)

    # -- keyword -----------------------------------------------------------

    def _keyword_hits(self, expression, query, scope, kinds, limit, include_preferences):
        # type: (str, str, Optional[str], Optional[Sequence[str]], int, bool) -> List[RecallHit]
        needle = tokenize.phrase_needle(query)
        memory_rows = self.store.search_memories(
            expression, needle, scope=scope, kinds=kinds, statuses=("active",), limit=limit
        )
        preference_rows = (
            self.store.search_preferences(expression, needle, scope=scope, limit=limit)
            if include_preferences
            else []
        )
        return fusion.rank_hits(memory_rows, preference_rows, self.weights, limit=limit)

    # -- semantic ----------------------------------------------------------

    def _semantic_hits(self, query, scope, kinds, limit):
        # type: (str, Optional[str], Optional[Sequence[str]], int) -> List[RecallHit]
        """Vector search. A provider or index failure degrades to no hits."""
        if not tokenize.query_units(query):
            # A query with no searchable content ("!!!") has no meaningful
            # embedding direction. Returning nearest neighbours anyway would be
            # noise inconsistent with the other modes.
            return []
        try:
            vectors = self.embedding_provider.embed([query])
        except ProviderError:
            return []
        if not vectors:
            return []
        try:
            matches = self.vector_index.search(vectors[0], limit=limit)
        except (ProviderError, ValueError):
            return []
        if not matches:
            return []

        by_id = {
            memory.id: memory
            for memory in self.store.get_memories_by_ids([match.memory_id for match in matches])
        }
        hits = []
        for match in matches:
            memory = by_id.get(match.memory_id)
            if memory is None:
                continue
            # Embeddings outlive a status change, so filter here rather than
            # relying on the index to be pruned.
            if memory.status != "active":
                continue
            if scope is not None and memory.scope != scope:
                continue
            if kinds and memory.kind not in kinds:
                continue
            hits.append(
                RecallHit(
                    item=memory,
                    item_type=ItemType.MEMORY.value,
                    score=float(match.score),
                    matched_on="vector",
                )
            )
        return hits[:limit]

    # -- diagnostics -------------------------------------------------------

    def describe(self):
        # type: () -> Dict[str, Any]
        described = {
            "modes": list(MODES),
            "default_mode": "hybrid",
            "semantic_available": self.semantic_available,
            "weights": self.weights.as_dict(),
        }  # type: Dict[str, Any]
        if self.vector_index is not None:
            described["vector_index"] = self.vector_index.stats()
        if self.embedding_provider is not None:
            described["embedding_provider"] = self.embedding_provider.describe()
        return described


__all__ = ["Retriever", "MODES"]
