# ADR 0003 — Retrieval must be deterministic; no LLM reranking in v0.1

Status: accepted (v0.1)

## Context

The project's headline acceptance test is a *hard* assertion: after swapping the
model, recall returns identical memories in identical order, and the database is
byte-identical.

That assertion is only achievable if the retrieval path never consults a model.

## Decision

- v0.1 retrieval is pure algorithm: SQLite FTS5 BM25, metadata/scope/time
  filters, deterministic tie-breaking.
- LLM reranking is not implemented in v0.1. When it is added it must be
  optional, disabled by default, and its output must be marked so that a
  deterministic result remains available.
- Retrieval is read-only and has no network access.

## Consequences

- MSIT-1 can assert 100% identity rather than a similarity threshold. A
  threshold would not be a guarantee.
- Retrieval quality is bounded by lexical matching until the vector phase; this
  is an accepted cost, measured on a labelled fixture set.
- Any future reranker must degrade to the deterministic ordering when the model
  is unavailable, so the offline guarantee (architecture.md red line 5) holds.
