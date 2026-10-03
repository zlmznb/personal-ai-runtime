# ADR 0001 — `memory.sqlite` is the only source of truth

Status: accepted (v0.1)

## Context

The project must guarantee that swapping the model does not lose long-term
identity, memories, preferences or project state. In v0.1 there are several
tempting places to put state: Ollama session state, DeepSeek conversation
history, a framework checkpointer, a cache, or an extra database.

Any of these turns "did we lose anything?" into a question with more than one
answer.

## Decision

There is exactly one long-term persistent artifact: `memory.sqlite`.

- Every other piece of persisted state is *derived* and must be either
  reconstructible from `memory.sqlite` or safely disposable.
- No model session state, framework checkpoint, or second database may hold
  long-term memory.
- `storage/` never imports a model SDK and never performs HTTP.

## Consequences

- Backup, restore and migration are single-file operations.
- The model-swap test can assert byte-identical databases.
- We defer vector databases until they are genuinely needed (§9 of
  architecture.md), because a second store would introduce dual-write
  consistency problems for no current benefit.
- An upper layer (e.g. LangGraph, if ever adopted) may *consume* Memory Core but
  must never *own* memory.
