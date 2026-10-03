# ADR 0002 — No vector database, no LiteLLM in v0.1

Status: accepted (v0.1)

## Context

The obvious instinct is to start with Chroma or Qdrant, plus LiteLLM for
provider abstraction. Both were evaluated and rejected for the first version.

## Decision

**Vector databases (Chroma, Qdrant, sqlite-vec): not used in v0.1.**

- Personal-scale memory (thousands to tens of thousands of rows) is served in
  milliseconds by SQLite + FTS5.
- A second store means dual writes: if SQLite succeeds and the vector store
  fails, the system no longer has a single answer to "what is remembered".
  That directly undermines ADR 0001.
- When scale genuinely demands it, the first step is an *in-process* index
  (sqlite-vec / hnswlib) behind a `VectorIndex` interface — no dual write, no
  service. A standalone vector database comes later, only for multi-device or
  concurrent-writer scenarios.

**LiteLLM: not used in v0.1.**

- There are two providers and both speak the OpenAI-compatible shape, so a thin
  in-repo adapter is the whole cost.
- The provider abstraction *is* this project's core asset; outsourcing it makes
  the most important layer third-party and version-dependent.
- It does not solve the actual hard problem here, which is that chat and
  embedding capabilities come from different vendors.

## Consequences

- `providers/` contains ~200 lines of hand-written adapter code, which we fully
  understand and can change at will.
- The internal DTO shape intentionally mirrors the OpenAI-compatible wire format,
  so adding a `LiteLLMGatewayProvider` later is additive and touches nothing
  above the provider layer.
- Re-entry triggers are recorded in architecture.md §9.
