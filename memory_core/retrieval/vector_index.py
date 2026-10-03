"""Vector index: a derived, disposable, rebuildable similarity index.

Architecture position
---------------------
A vector index is **never** a source of truth. Everything it holds can be
rebuilt from ``memories`` plus an embedding provider. Two implementations ship
here, which is what makes that claim testable rather than aspirational:

* :class:`SqliteVectorIndex` - vectors persisted in the derived ``embeddings``
  table of the same SQLite file. Default. The index can be dropped with a single
  ``DELETE`` and rebuilt.
* :class:`InMemoryVectorIndex` - a process-local index with no persistence at
  all. Proof that the rest of the system does not depend on the index existing.

Vector maths lives here, never in ``storage``: storage persists opaque bytes.

Optional acceleration
---------------------
``numpy`` is used for the dot product when importable. It is never required -
:func:`set_numpy_enabled` exists so the pure-Python path is exercised in tests.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from operator import mul
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - presence depends on the environment
    import numpy as _numpy_original
except Exception:  # pragma: no cover
    _numpy_original = None

#: The active backend. Tests flip this to exercise the pure-Python path.
_numpy = _numpy_original

#: Bytes per element in the stored encoding (little-endian float32).
FLOAT32_ITEMSIZE = 4


def has_numpy():
    # type: () -> bool
    return _numpy is not None


def set_numpy_enabled(enabled):
    # type: (bool) -> bool
    """Force the pure-Python path on (False) or back to the default (True).

    Returns whether numpy was active *before* the call. Exists so tests can
    prove both paths behave identically and that nothing depends on numpy.
    """
    global _numpy
    previous = _numpy is not None
    _numpy = _numpy_original if enabled else None
    return previous


@dataclass(frozen=True)
class VectorHit(object):
    """One similarity result."""

    memory_id: str
    score: float

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {"memory_id": self.memory_id, "score": self.score}


def normalize(vector):
    # type: (Sequence[float]) -> List[float]
    """L2-normalise so that cosine similarity is a plain dot product."""
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def encode_vector(vector):
    # type: (Sequence[float]) -> Tuple[int, bytes]
    """Encode an already-normalised vector as little-endian float32."""
    values = array("f", [float(value) for value in vector])
    return len(values), values.tobytes()


def decode_vector(blob, dim):
    # type: (bytes, int) -> array
    values = array("f")
    values.frombytes(blob)
    if len(values) != int(dim):
        raise ValueError(
            "vector blob holds {0} values but dim says {1}".format(len(values), dim)
        )
    return values


def cosine_similarity(left, right):
    # type: (Sequence[float], Sequence[float]) -> float
    """Dot product of two unit vectors, clamped to [-1, 1]."""
    if len(left) != len(right):
        raise ValueError("dimension mismatch: {0} vs {1}".format(len(left), len(right)))
    if _numpy is not None and not isinstance(left, array) and not isinstance(right, array):
        return float(_numpy.dot(left, right))
    return float(sum(map(mul, left, right)))


class VectorIndex(object):
    """Interface every vector index implementation satisfies."""

    name = "vector-index"

    @property
    def model_id(self):
        # type: () -> str
        raise NotImplementedError

    def upsert(self, memory_id, vector):
        # type: (str, Sequence[float]) -> None
        raise NotImplementedError

    def upsert_many(self, rows):
        # type: (Iterable[Tuple[str, Sequence[float]]]) -> int
        raise NotImplementedError

    def delete(self, memory_id):
        # type: (str) -> bool
        raise NotImplementedError

    def clear(self):
        # type: () -> int
        """Delete the whole index. The memories themselves are untouched."""
        raise NotImplementedError

    def count(self):
        # type: () -> int
        raise NotImplementedError

    def search(self, query_vector, limit=10):
        # type: (Sequence[float], int) -> List[VectorHit]
        raise NotImplementedError

    def stats(self):
        # type: () -> Dict[str, Any]
        raise NotImplementedError


def _rank(scored, limit):
    # type: (List[Tuple[str, float]], int) -> List[VectorHit]
    """Total, deterministic ordering: score descending, then id ascending."""
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    if limit is not None and int(limit) > 0:
        scored = scored[: int(limit)]
    return [VectorHit(memory_id=item_id, score=score) for item_id, score in scored]


class InMemoryVectorIndex(VectorIndex):
    """A process-local index with no persistence.

    Useful in tests, and as a concrete demonstration that the system works with
    the index taken away: deleting it costs only recall quality, never data.
    """

    name = "in-memory"

    def __init__(self, model_id, use_numpy=None):
        # type: (str, Optional[bool]) -> None
        self._model_id = str(model_id)
        self._vectors = {}  # type: Dict[str, Tuple[float, ...]]
        self._dim = None  # type: Optional[int]
        self._use_numpy = has_numpy() if use_numpy is None else bool(use_numpy) and has_numpy()

    @property
    def model_id(self):
        # type: () -> str
        return self._model_id

    @property
    def dim(self):
        # type: () -> Optional[int]
        return self._dim

    def upsert(self, memory_id, vector):
        # type: (str, Sequence[float]) -> None
        normalized = normalize(vector)
        if self._dim is None:
            self._dim = len(normalized)
        elif len(normalized) != self._dim:
            raise ValueError(
                "dimension mismatch: index holds {0}-d vectors, got {1}".format(
                    self._dim, len(normalized)
                )
            )
        self._vectors[str(memory_id)] = tuple(normalized)

    def upsert_many(self, rows):
        # type: (Iterable[Tuple[str, Sequence[float]]]) -> int
        count = 0
        for memory_id, vector in rows:
            self.upsert(memory_id, vector)
            count += 1
        return count

    def delete(self, memory_id):
        # type: (str) -> bool
        return self._vectors.pop(str(memory_id), None) is not None

    def clear(self):
        # type: () -> int
        removed = len(self._vectors)
        self._vectors.clear()
        self._dim = None
        return removed

    def count(self):
        # type: () -> int
        return len(self._vectors)

    def search(self, query_vector, limit=10):
        # type: (Sequence[float], int) -> List[VectorHit]
        if not self._vectors:
            return []
        query = normalize(query_vector)
        if self._dim is not None and len(query) != self._dim:
            raise ValueError(
                "query has {0} dimensions but the index holds {1}".format(
                    len(query), self._dim
                )
            )
        items = list(self._vectors.items())
        if self._use_numpy:
            import numpy as _np  # local import keeps the dependency optional

            matrix = _np.array([vector for _, vector in items], dtype=_np.float32)
            similarities = matrix @ _np.array(query, dtype=_np.float32)
            scored = [
                (memory_id, float(similarity))
                for (memory_id, _), similarity in zip(items, similarities)
            ]
        else:
            scored = [(memory_id, float(sum(map(mul, query, vector)))) for memory_id, vector in items]
        return _rank(scored, limit)

    def stats(self):
        # type: () -> Dict[str, Any]
        return {
            "kind": self.name,
            "model_id": self._model_id,
            "count": self.count(),
            "dim": self._dim,
            "numpy": self._use_numpy,
        }


class SqliteVectorIndex(VectorIndex):
    """Persistent index backed by the derived ``embeddings`` table.

    Vectors live in the same SQLite file as the truth, under a separate table
    keyed by ``(memory_id, model_id)``. There is therefore no second store to
    keep in sync, and the index is still trivially droppable.
    """

    name = "sqlite"

    def __init__(self, store, model_id, use_numpy=None):
        # type: (Any, str, Optional[bool]) -> None
        self.store = store
        self._model_id = str(model_id)
        self._use_numpy = has_numpy() if use_numpy is None else bool(use_numpy) and has_numpy()
        self._cache = None  # type: Optional[Any]
        self._dim = None  # type: Optional[int]

    @property
    def model_id(self):
        # type: () -> str
        return self._model_id

    @property
    def dim(self):
        # type: () -> Optional[int]
        if self._dim is None:
            self._load()
        return self._dim

    def _invalidate(self):
        # type: () -> None
        self._cache = None
        self._dim = None

    def _load(self):
        # type: () -> Any
        """Load the whole index into memory, cached until the next write.

        Brute force is a deliberate v0.2 choice: at personal scale (thousands to
        tens of thousands of vectors) an exact scan is milliseconds, and it can
        never return an approximate answer. See docs/benchmarks.md.
        """
        if self._cache is not None:
            return self._cache
        ids = []  # type: List[str]
        vectors = []  # type: List[Any]
        dim = None  # type: Optional[int]
        for memory_id, _model, row_dim, blob in self.store.iter_embeddings(self._model_id):
            if dim is None:
                dim = row_dim
            elif row_dim != dim:
                raise ValueError(
                    "index holds mixed dimensions ({0} and {1})".format(dim, row_dim)
                )
            ids.append(memory_id)
            vectors.append(decode_vector(blob, row_dim))
        if self._use_numpy and ids:
            import numpy as _np  # local import keeps the dependency optional

            matrix = _np.array([list(vector) for vector in vectors], dtype=_np.float32)
        else:
            matrix = vectors
        self._dim = dim
        self._cache = (ids, matrix)
        return self._cache

    def reload(self):
        # type: () -> None
        """Drop the in-process cache so the next search re-reads from SQLite."""
        self._invalidate()

    def upsert(self, memory_id, vector):
        # type: (str, Sequence[float]) -> None
        dim, blob = encode_vector(normalize(vector))
        self.store.upsert_embedding(memory_id, self._model_id, dim, blob)
        self._invalidate()

    def upsert_many(self, rows):
        # type: (Iterable[Tuple[str, Sequence[float]]]) -> int
        prepared = []
        for memory_id, vector in rows:
            dim, blob = encode_vector(normalize(vector))
            prepared.append((str(memory_id), dim, blob))
        written = self.store.upsert_embeddings(prepared, model_id=self._model_id)
        if written:
            self._invalidate()
        return written

    def delete(self, memory_id):
        # type: (str) -> bool
        with self.store.transaction() as connection:
            removed = connection.execute(
                "DELETE FROM embeddings WHERE memory_id = ? AND model_id = ?",
                (str(memory_id), self._model_id),
            ).rowcount
        if removed:
            self._invalidate()
        return removed > 0

    def clear(self):
        # type: () -> int
        removed = self.store.delete_embeddings(self._model_id)
        self._invalidate()
        return removed

    def count(self):
        # type: () -> int
        return self.store.count_embeddings(self._model_id)

    def search(self, query_vector, limit=10):
        # type: (Sequence[float], int) -> List[VectorHit]
        ids, matrix = self._load()
        if not ids:
            return []
        query = normalize(query_vector)
        if self._dim is not None and len(query) != self._dim:
            raise ValueError(
                "query has {0} dimensions but the index holds {1}".format(
                    len(query), self._dim
                )
            )
        if self._use_numpy:
            import numpy as _np  # local import keeps the dependency optional

            similarities = matrix @ _np.array(query, dtype=_np.float32)
            scored = [
                (memory_id, float(similarity))
                for memory_id, similarity in zip(ids, similarities)
            ]
        else:
            scored = [
                (memory_id, float(sum(map(mul, query, vector))))
                for memory_id, vector in zip(ids, matrix)
            ]
        return _rank(scored, limit)

    def stats(self):
        # type: () -> Dict[str, Any]
        if self._dim is None:
            # Report the real dimension rather than "unknown" once vectors exist.
            try:
                self._load()
            except ValueError:
                pass
        return {
            "kind": self.name,
            "model_id": self._model_id,
            "count": self.count(),
            "dim": self._dim,
            "numpy": self._use_numpy,
        }


__all__ = [
    "VectorIndex",
    "VectorHit",
    "SqliteVectorIndex",
    "InMemoryVectorIndex",
    "normalize",
    "encode_vector",
    "decode_vector",
    "cosine_similarity",
    "has_numpy",
    "set_numpy_enabled",
    "FLOAT32_ITEMSIZE",
]
