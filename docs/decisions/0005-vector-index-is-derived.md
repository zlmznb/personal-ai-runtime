# ADR 0005 — The vector index is derived, disposable and rebuildable

Status: accepted (v0.2)

## Context

v0.2 needs semantic search. Two options were on the table:

1. A standalone vector database (Chroma, Qdrant) or an in-file ANN extension
   (`sqlite-vec`).
2. A derived table of vectors inside the same SQLite file, scanned exactly.

Chroma and Qdrant add a second store, which reintroduces the dual-write problem
ADR 0001 exists to prevent. `sqlite-vec` was evaluated: extension loading works
in this environment (`enable_load_extension` succeeds), but the extension binary
is not present and the project has a zero-dependency rule, so adopting it would
mean shipping a native artifact per platform.

## Decision

Vectors live in a derived `embeddings` table in the same SQLite file, keyed by
`(memory_id, model_id)`, and are scanned exactly by brute force.

The index is:

- **Derived** — never part of the source of truth. `export` excludes it.
- **Disposable** — `drop_index()` deletes it, and only recall quality changes.
- **Rebuildable** — `rebuild_index()` regenerates it from `memories` plus an
  embedding provider.
- **Replaceable** — `VectorIndex` has two implementations (`SqliteVectorIndex`,
  `InMemoryVectorIndex`), plus an optional numpy acceleration path that is
  proven equivalent to the pure-Python one.

## Consequences

- Deleting a memory cascades to its vectors, so the derived table cannot drift.
- Superseded or deleted memories are filtered at query time rather than relying
  on the index being pruned, because status changes do not touch vectors.
- Recall never writes. Index construction is always explicit, which is what
  keeps the byte-identity assertions of MSIT-1 and MSIT-2 possible.

## Evidence for brute force

Measured at 10,000 memories x 256 dimensions: exact vector scan **10.9 ms p50 /
13.4 ms p95**; hybrid retrieval **11.3 ms p50**. See `docs/benchmarks.md`.

An ANN index would buy a speedup that is not needed at personal scale, at the
cost of approximate results and a second thing to keep consistent. The revised
re-entry triggers are in `architecture.md` §9.
