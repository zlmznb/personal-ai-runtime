"""Building and rebuilding the derived semantic index.

This is the only place that turns memories into vectors. It is never called
implicitly by retrieval: recall stays read-only, which is what preserves the
byte-identity guarantee of the model-swap tests.

The index is incremental (``build`` embeds only what is missing) and fully
rebuildable (``rebuild`` clears first). Both properties are tested.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from memory_core.domain.errors import MemoryCoreError, ProviderError
from memory_core.providers.base import EmbeddingProvider
from memory_core.retrieval.vector_index import SqliteVectorIndex, VectorIndex


class SemanticIndexer(object):
    """Keeps a vector index in step with the memories in storage."""

    def __init__(self, store, provider, index=None, batch_size=32):
        # type: (Any, EmbeddingProvider, Optional[VectorIndex], int) -> None
        if provider is None:
            raise MemoryCoreError("SemanticIndexer requires an embedding provider")
        self.store = store
        self.provider = provider
        self.batch_size = max(1, int(batch_size))
        self.index = index if index is not None else SqliteVectorIndex(store, provider.identifier())

    @property
    def model_id(self):
        # type: () -> str
        return self.provider.identifier()

    def build(self, scope=None, kinds=None, force=False, limit=None):
        # type: (Optional[str], Optional[Sequence[str]], bool, Optional[int]) -> Dict[str, Any]
        """Embed memories that are missing from the index.

        With ``force=True`` the index for this model is cleared first, which is
        exactly what an embedding-model change requires.
        """
        cleared = 0
        if force:
            cleared = self.index.clear()

        pending = self.store.memory_ids_missing_embedding(self.model_id)
        if scope is not None or kinds is not None:
            filtered = []
            for memory in self.store.list_memories(
                scope=scope, kinds=kinds, statuses=("active",), limit=1_000_000
            ):
                filtered.append(memory.id)
            allowed = set(filtered)
            pending = [memory_id for memory_id in pending if memory_id in allowed]
        if limit is not None:
            pending = pending[: int(limit)]

        embedded = 0
        failures = []  # type: List[Dict[str, str]]
        for start in range(0, len(pending), self.batch_size):
            chunk_ids = pending[start:start + self.batch_size]
            memories = self.store.get_memories_by_ids(chunk_ids)
            if not memories:
                continue
            texts = [self._document(memory) for memory in memories]
            try:
                vectors = self.provider.embed(texts)
            except ProviderError as exc:
                # One bad batch must not corrupt the index or the memories.
                for memory in memories:
                    failures.append({"memory_id": memory.id, "error": str(exc)})
                continue
            rows = [(memory.id, vector) for memory, vector in zip(memories, vectors)]
            embedded += self.index.upsert_many(rows)

        return {
            "model_id": self.model_id,
            "cleared": cleared,
            "embedded": embedded,
            "pending_before": len(pending),
            "failures": failures,
            "indexed": self.index.count(),
        }

    def rebuild(self, scope=None, kinds=None, limit=None):
        # type: (Optional[str], Optional[Sequence[str]], Optional[int]) -> Dict[str, Any]
        """Clear this model's vectors and regenerate them from the memories."""
        return self.build(scope=scope, kinds=kinds, force=True, limit=limit)

    def drop(self, all_models=False):
        # type: (bool) -> int
        """Delete vectors. The memories are untouched.

        With ``all_models=False`` (default) only this model's vectors go, so a
        coexisting index for another embedding model is preserved. With
        ``all_models=True`` every derived vector is removed - the index is then
        empty and fully rebuildable from SQLite.
        """
        if all_models:
            removed = self.store.delete_embeddings(None)
            # ``clear`` also invalidates the index's in-process cache, so the
            # next search cannot serve rows that no longer exist.
            self.index.clear()
            return removed
        return self.index.clear()

    def stats(self):
        # type: () -> Dict[str, Any]
        described = self.index.stats()
        described["stale"] = len(self.store.memory_ids_missing_embedding(self.model_id))
        described["provider"] = self.provider.describe()
        return described

    @staticmethod
    def _document(memory):
        # type: (Any) -> str
        """The text that represents a memory in vector space.

        Content first, then subject and tags, so that a query matching either
        the statement or its metadata can find it.
        """
        parts = [memory.content]
        if memory.subject:
            parts.append(memory.subject)
        if memory.tags:
            parts.extend(memory.tags)
        return "\n".join(parts)


__all__ = ["SemanticIndexer"]
