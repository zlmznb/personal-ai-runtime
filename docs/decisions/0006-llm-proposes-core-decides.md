# ADR 0006 — The LLM proposes; Memory Core decides

Status: accepted (v0.2)

## Context

v0.2 lets a model participate in memory formation: read a turn, decide whether
something is worth remembering, and emit a structured proposal. That is a large
increase in the amount of untrusted input flowing toward the database.

## Decision

The model has exactly one capability: turn a string into a string. It cannot
reach storage, and three independent gates stand between its output and a write.

1. **Strict schema validation** (`formation/candidate.py`).
   The response must be a JSON object with a single `memories` array. Unknown
   top-level keys and unknown keys inside an item are rejected. Operations are
   restricted to ADD / UPDATE / SUPERSEDE / NOOP. Content, kind, tags,
   confidence and salience are each validated.

2. **Referential safety.**
   A SUPERSEDE proposal may only name an id that was actually shown to the model
   in the prompt. A hallucinated id is rejected at parse time, so the model
   cannot ask Memory Core to overwrite an unrelated memory.

3. **Acceptance policy** (`formation/policy.py`).
   A pure, model-free function decides accept or reject: NOOP, low confidence,
   too-short content, duplicates and non-existent supersede targets are all
   rejected with a reason.

Only then does `MemoryCore._commit` validate the candidate through the domain
guards and write it, inside a transaction. Any failure rolls back and is
recorded as a rejection.

## Consequences

- **Failure cannot pollute the database.** A provider error, malformed JSON, a
  schema violation or a rejected candidate all result in zero writes. The raw
  event is still appended, so the turn remains re-extractable.
- **The model can never be the only holder of anything.** It writes nothing
  anywhere, and every artefact it influences carries `source.model` and
  `generated_by` provenance.
- **LLM memories are never facts.** Their default confidence is 0.6, below the
  1.0 of a user's explicit statement, and `generated_by` records the model.
- Formation is testable without a model at all: `LlmExtractor` is driven by any
  `ChatProvider`, including the deterministic fake, so schema and policy
  behaviour is asserted exactly rather than sampled.
