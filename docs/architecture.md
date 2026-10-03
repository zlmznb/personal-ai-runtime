# Memory Core — Architecture

Version: v0.2 Memory Intelligence
Status: implemented and tested

*v0.1 (Phase 1)* shipped the durable core: a single SQLite source of truth,
events, memories, preferences, project state, FTS5 retrieval, and the model-swap
guarantee.

*v0.2* adds an intelligence layer **on top of** those guarantees without
weakening any of them: LLM-assisted memory formation, an independent
`EmbeddingProvider`, a derived and disposable vector index, and hybrid retrieval.

---

## 1. Objective

Build a **personal AI memory core** whose persistence layer is completely
decoupled from model inference, so that local Ollama, cloud DeepSeek, embedding
models, and any future model can be swapped at will without any loss of
long-term personal data, persona settings, project state or history.

The verification target is a single sentence:

> **After swapping the model - chat or embedding - the AI's long-term identity,
> memories and project state are still there.**

This is deliberately *not* "a chatbot with memory". A chatbot with memory can be
built with a framework in an afternoon and will fail this target the moment you
change vendors. This project builds the boring, durable part first, then adds
intelligence behind the same boundary.

### Still out of scope

Chroma · Qdrant · sqlite-vec · LiteLLM · Mem0 · LangGraph · MCP · Agent ·
Web UI · mobile · IoT · cloud sync · multi-user · scheduled or background
extraction · fine-tuning or training · approximate/ANN indexes · complex
identity system.

Each of these is listed with a *re-entry trigger* in §9. None is needed to test
the core hypothesis, and several would actively damage it.

---

## 2. Principle 0 and the constraints it forces

**Principle 0: Memory is the asset. Models are consumables.**

Everything below is *derived* from that sentence, not bolted on afterwards.

| # | Constraint | Consequence in the code |
|---|---|---|
| **P0-1** | Model output is a *candidate*, never a fact | Exactly one validated write gate: `MemoryCore.remember()`. No code path lets a provider write to storage. |
| **P0-2** | Exactly one source of truth, and it is one file | `memory.sqlite`. Everything else is derived and must be rebuildable or disposable. |
| **P0-3** | Events are immutable; memories are a derived view | `events` is append-only. Memories can always be regenerated from the event log. |
| **P0-4** | No model concept may leak into the schema | No `openai_*` / `deepseek_*` / `llama_*` columns. The single exception is `generated_by`, free-text, observability only, never read by logic. |

The litmus test for any proposed change:

> If Ollama and DeepSeek disappeared tomorrow, would `memory.sqlite` still be a
> complete, readable, usable personal memory?

---

## 3. Architecture red lines

Enforced by tests, not by convention.

1. **`memory.sqlite` is the only long-term source of truth.** No model session
   state, no LangGraph checkpoint, no second database may hold long-term memory.
   *Why this matters:* the moment memory lives in two places, "did we lose
   anything?" stops having a single answer, and the model-swap guarantee becomes
   unprovable.

2. **`storage/` and `domain/` must not import a model SDK and must not do HTTP.**
   Storage must be testable in a completely offline, model-free environment.
   `tests/test_offline_degradation.py` uses a static import-graph check to keep
   this true as the project grows.

3. **Retrieval is deterministic. No LLM reranking in v0.1.**
   This is the enabling condition for MSIT-1's 100% assertion (§7). If retrieval
   ever consults a model, the guarantee degrades from "identical" to "roughly
   similar", which is not a guarantee.

4. **All model output passes through Memory Core's own data structures and
   validators before Storage.** Providers return our DTOs, not vendor objects.
   `remember()` revalidates everything.

5. **Offline-first degradation.** With zero providers configured, `remember`,
   `recall`, `forget`, `export`, `snapshot`, preferences and project state must
   all still work. Models can only *add* capability, never gate access to data.

---

## 4. Layering

Dependencies point in one direction only. Retrieval and Formation depend on
Storage; they do not depend on each other; the API facade depends on all three.

```
        Consumers (CLI now; UI / Agent / mobile / IoT later)
                              |
                    MemoryCore facade (api.py)
                              |
        +---------------------+---------------------+
        |                     |                     |
   Retrieval             Formation              Storage
   (read-only,           (events ->            (SQLite,
    deterministic)        candidates)           source of truth)
        |                     |                     ^
        |                     +---------------------+
        |                     (validated write gate)
        |
   [no model]             [optional model]      [no model, no network]
```

| Layer | Responsibility | Model use | Network | Mutable |
|---|---|---|---|---|
| **Storage** | Persistence, transactions, schema migrations, CRUD, constraint enforcement | **Never** | **Never** | Source of truth |
| **Retrieval** | query + scope + filters → deterministic ranked list | **Never in v0.1** | Never | Read-only |
| **Formation** | events → validated memory candidates + ADD/SUPERSEDE/NOOP decision | Optional, degradable | Optional | Stateless |
| **Providers** | Translate our DTOs to/from a vendor API | Is the model | Is the network | Stateless |

### Why the three must stay separate

- **Storage being "dumb"** is what makes the whole system testable and
  verifiable offline. It computes no embeddings (vectors are passed in), makes
  no similarity judgements, and knows no provider types.
- **Retrieval being pure** is what makes MSIT-1 a hard assertion.
- **Formation not writing directly** is what makes the event log replayable: the
  same events can be re-run through a new extraction strategy, or a fixed one,
  without touching history.

---

## 5. Data model

Four tables plus a metadata table and two FTS indexes.

### `events` — append-only raw history (P0-3)

| Column | Notes |
|---|---|
| `id` | ULID-like, time-sortable |
| `kind` / `role` | `message`, `note`, `system` / `user`, `assistant` |
| `content` | verbatim text |
| `session_id`, `scope` | conversation grouping, namespace |
| `metadata` | JSON |
| `created_at` | ISO-8601 UTC |
| `source` | `cli`, `import`, ... |

Never updated, never deleted by normal operation. This is what makes memories
regenerable.

### `memories` — long-term memory

| Column | Purpose | Design note |
|---|---|---|
| `id` | ULID-like | time-sortable |
| `kind` | `semantic` / `episodic` / `identity` | see §5.1 |
| `scope` | `global`, `project:<id>`, `device:<id>` | reserved for edge/cloud later; v0.1 uses `global` + `project:*` |
| `subject` | `user:self`, `person:alice`, ... | who/what it is about |
| `content` | one atomic, human-readable statement | atomicity enables supersede + dedupe |
| `structured` | JSON | machine-usable form |
| `tags` | JSON array | FTS-indexed |
| `source` | JSON: `event_ids`, `origin` | provenance is mandatory |
| `confidence` | 0.0–1.0 | explicit user statement = 1.0; anything model-derived < 1.0 |
| `salience` | importance weight | ranking input |
| `status` | `active` / `superseded` / `archived` / `deleted` | logical delete |
| `valid_from` / `valid_to` | temporal validity | "I used Python in 2024, Rust in 2026" |
| `superseded_by` | FK to the replacing row | history is never overwritten |
| `generated_by` | free-text label | observability only; never read by logic (P0-4) |
| `created_at` / `updated_at` / `revision` | | |

**Append-mostly + supersede, never physical overwrite.** Changing a memory
writes a new row and marks the old one `superseded`. History stays queryable, so
nothing is ever truly lost.

### `preferences` — explicit, exact-key user declarations

`id`, `scope`, `key`, `value` (JSON), `statement`, `confidence`, `source`,
`created_at`, `superseded_by`.

A **partial unique index** guarantees exactly one active row per
`(scope, key)`:

```sql
CREATE UNIQUE INDEX ... ON preferences(scope, key) WHERE superseded_by IS NULL;
```

Preferences are read by exact key, not by similarity. This is deliberate: a
preference must apply *immediately, deterministically, every time*. If
preferences depended on vector recall, the same question could yield different
answers on different runs — unacceptable for a personal assistant.

### `project_state` — "what is true now"

`id`, `project_id`, `scope`, `state` (JSON), `note`, `source`, `created_at`,
`superseded_by`.

Current state = the row with `superseded_by IS NULL`. History = all rows.
This is not a workflow engine; it is a versioned current-value store.

### `meta` and FTS

`meta` holds `schema_version` and the config fingerprint.
`memories_fts` and `preferences_fts` are standalone FTS5 tables maintained
inside the same transaction as their source rows, indexed by BM25.

### 5.1 Distinguishing the five kinds of memory

| Type | Nature | Written | Lifetime | Retrieval | Mutable |
|---|---|---|---|---|---|
| **Working / short-term** | current conversation context | per turn | minutes | not in the store | volatile |
| **Episodic** (`kind=episodic`) | *what happened* | once | permanent | time + entity + FTS | **immutable** |
| **Semantic** (`kind=semantic`) | *stable facts about the world* | distilled from events | long | FTS (+ vectors later) | supersedable |
| **Preference** (separate table) | *binding constraints* | explicit declaration | long, versioned | **exact key** | overwritten, history kept |
| **Project state** (separate table) | *what is true now* | explicit update | per project | by `project_id` | overwritten, history kept |
| **Identity** (`kind=identity`) | *who the assistant is* | explicit / curated | permanent | FTS, high salience | supersedable |

The distinctions that matter most:

- **Episodic vs semantic.** Episodic is "at time T, X happened" (immutable,
  timestamped). Semantic is "therefore, Y is true" (can be overturned by new
  evidence). The same information is often stored twice, at both levels.
- **Preference vs semantic.** Preferences are structured key/value with a single
  current value. Semantic memories are fuzzy and similarity-retrieved. Mixing
  them means preferences become probabilistic, which users experience as the AI
  "forgetting" something it was told.
- **Project state vs memory.** A memory is a statement *about the past*; project
  state is *the current snapshot*. If you ever have to ask "which of these
  memories is the latest project status?", the model is wrong. v0.1 keeps them
  in separate tables so the question cannot arise.

---

## 6. ModelProvider abstraction

**Memory Core code never imports a vendor SDK.** Only `providers/` may contain
HTTP and vendor specifics.

### v0.1: one protocol — `ChatProvider`

- `ChatProvider.capabilities` → `Capabilities` (`supports_json_mode`,
  `max_context`, ...). Callers degrade based on declared capability instead of
  assuming every model can do structured output.
- `ChatProvider.chat(messages, **options) -> ChatResult` — takes our
  `ChatMessage` DTOs, returns our `ChatResult` DTO. No vendor objects cross the
  boundary.
- Errors are normalized to `ProviderError`. A provider failure is a *degradation*,
  not a crash.

Implementations in v0.1: `FakeProvider` (deterministic, offline),
`OllamaProvider` (local chat), `DeepSeekProvider` (cloud chat only).

### Where the model is allowed to act

Four roles, each with a mandatory deterministic fallback:

| Role | v0.1 | Fallback when unavailable |
|---|---|---|
| 1. Generate the final answer (`ask`) | yes | return the recalled memory context itself |
| 2. Distill memory candidates | **deferred** | rules-based formation already works |
| 3. Rerank retrieval results | **deferred** | deterministic BM25 ordering |
| 4. Compute embeddings | **deferred** | FTS5 keyword retrieval |

**The system is fully functional with all four unavailable.** This is a
tested property, not an aspiration.

### Deferred: `EmbeddingProvider` must be a *separate* protocol

When vector search arrives, embeddings get their own protocol rather than being
added to `ChatProvider`. The reason is load-bearing and concrete: **as of this
writing, the DeepSeek API offers no embeddings endpoint** — only chat/reasoner
style models. So the chat provider and the embedding provider are, by
construction, *different vendors*. A design that merges them into one interface
breaks the first time a model is swapped.

Ollama happens to serve both chat and embeddings, but that is a coincidence, not
a design premise. This decision is recorded now precisely so it is not lost
before the vector phase begins.

### Configuration: role-based binding

Chat, formation, embedding and reranking are independent roles in
`config/models.json`, so a single run can mix providers
(`chat=deepseek-chat`, `embedding=ollama:nomic-embed-text`, ...).

### Preventing leakage

`generated_by` records which model produced an artifact, for observability and
evaluation. It is never read by any logic, and it is excluded from the canonical
digest used by the model-swap test.

---

## 7. The verification: Model-Swap Isolation Tests (MSIT)

Three tests, escalating. MSIT-1 and MSIT-2 are implemented in v0.2.

### MSIT-1 — swap the inference model, database untouched *(hard 100%)*

1. Run the scripted session in `fixtures/scripted_session.json`, writing
   identity, semantic, episodic, preferences and project state.
2. Hash `memory.sqlite` at the byte level.
3. Swap `ChatProvider`: FakeProvider A → FakeProvider B → unreachable Ollama →
   DeepSeek without a key → no provider at all.
4. Assert for 24 recall queries spanning every memory kind:
   - **identical memory IDs, in identical order, with identical scores**;
   - the database hash is **byte-identical**.
5. Assert identity memories, every preference, and project state are unchanged.

Only possible because retrieval never consults a chat model (red line 3). If
this test fails, the architecture has been compromised.

### MSIT-2 — swap the chat model and the embedding model, drop the index *(v0.2)*

1. Create memories with chat model A, index them with embedding model A.
2. Close, then reopen with **chat model B and embedding model B**.
3. Assert the memories, preferences and project state are unchanged.
4. Assert the new embedding model simply has no vectors, so semantic recall
   degrades to nothing and hybrid degrades cleanly to keyword.
5. **Drop the entire vector index.** Assert every memory is still there, still
   exported, still keyword-searchable.
6. **Rebuild the index** from SQLite. Assert semantic and hybrid search work
   again and find the original memories.
7. Assert `export_jsonl()` - the source tables only - is byte-identical
   throughout.

The database *hash* cannot be used here, because re-embedding legitimately
rewrites the derived `embeddings` table. Export is the correct instrument
precisely because it covers the source of truth and excludes every derived
table.

### MSIT-3 — swap the distillation model *(future)*

Replay the same events through a different LLM extractor; assert identical
`scope`/`kind`/`subject` sets, ≥90% `structured` field agreement, ≥0.85 mean
semantic similarity of `content`, and zero loss of provenance.

### Supporting acceptance criteria

- Offline: with all providers removed, `recall`/`export`/`forget` work.
- All providers failing: raw events still persist and `learn` writes nothing.
- Static import check: `domain/` and `storage/` import no HTTP and no provider
  code; `retrieval/` may import only the provider *interface* and numpy.
- No vendor-specific field name in any schema.
- Exactly one persistent artifact: `memory.sqlite`.
- Export is deterministic: two exports of the same database are byte-identical.
- Every derived table can be dropped and rebuilt.

---

## 8. Why long-term memory must never be trained into the weights

Recorded here because it is the reason this project exists.

1. It defeats the project's own top priority: swapping the model destroys the
   memory. That is the most extreme form of the failure we are preventing.
2. **Not editable.** "I changed my mind" / "that's wrong" cannot be fixed.
3. **Not deletable.** Privacy, forgetting, deleting a chapter of your life —
   impossible with weights.
4. **Not attributable.** You cannot answer "where did this come from, how sure
   are you, when did you learn it".
5. **Latency mismatch.** Training is batch, hours-to-days; memory writes must be
   sub-second. Today's conversation cannot wait for the next fine-tune.
6. **Privacy and compliance.** Personal and future IoT data must not be sent
   into training.
7. **Catastrophic forgetting and interference** between old and new knowledge.
8. **No access control.** Weights are monolithic; per-scope isolation
   (`project:X` visible only here, not synced to that device) is impossible.
9. **Cost and irreversibility.** Every memory update becomes a training run,
   with no rollback.

**The right mental model: the model is a CPU (stateless reasoning engine); the
Memory Core is the disk and filesystem (stateful source of truth).** You do not
compile your filesystem into the CPU.

**The one permitted exception**, which must stay explicit:

- ✅ may be trained into the model: **style, tone, phrasing** — *how it speaks*.
- ❌ never trained into the model: **facts, preferences, project state, history,
  identity content** — *what it says, and who it is*.

`Persona-Style` may enter the weights. `Persona-Content` and `Memory` never do.

---

## 9. Deferred components and their re-entry triggers

| Component | Solves | Re-entry trigger |
|---|---|---|
| **ANN index** (sqlite-vec / hnswlib) | sub-linear similarity search | measured vector p95 > 50 ms at the real working set. At 10,000 memories it is currently 13.4 ms, so this is years away |
| **Qdrant** | service-ized vector search, multi-client | multi-device / edge-cloud coordination, or concurrent writers. SQLite serialises writers, so this is the real trigger |
| **Chroma** | prototype retrieval quality | only as a throwaway evaluation harness, never as a store |
| **LiteLLM** | multi-provider normalization, fallback, cost | provider count > 4, or fallback/cost accounting needed. Adds a `ChatProvider` implementation; changes nothing above it |
| **Mem0** | distillation pipeline | as a *reference implementation* for A/B evaluation of our own extractor, never as a dependency |
| **LangGraph** | stateful workflow orchestration | an *upper layer* (Agent) decision. It must never own memory |
| **Scheduled / background extraction** | automatic learning | when an upper layer can supply reliable triggers; formation is explicit by design in v0.2 |

Two rules apply to every item on this list:

- It must sit behind an interface that already exists (`VectorIndex`,
  `ChatProvider`, `EmbeddingProvider`), and
- it must be reconstructible from `memory.sqlite` alone.

### The most dangerous temptation

If a future upper layer adopts LangGraph, it will be tempting to keep memory in
its checkpointer or Store. **This is forbidden.** It creates a second source of
truth and silently destroys the model-swap guarantee. LangGraph may consume
Memory Core; it may never hold it.

### The second temptation: promoting the vector index

The vector index now holds real data (10,000 memories → 14 MB of vectors), and
it will be tempting to treat it as a store: to write to it first, to query it
directly, to treat "the index is stale" as "the data is missing".

It is not a store. `tests/test_model_swap_isolation.py` drops the entire index
mid-lifecycle and asserts that every memory is still there, still exported, and
still keyword-searchable. Any change that makes that test fail is a change that
has promoted a cache to a source of truth.

---

## 10. Deliberate simplifications

Recorded so they are understood as choices, not oversights.

- **Brute-force vector search, exact, no ANN.** At 10,000 memories it is 13 ms
  p95. Approximate search would buy a speedup nobody needs, at the cost of a
  second store to keep consistent (ADR 0005, `docs/benchmarks.md`).
- **Rules-first formation, LLM second.** Explicit `记住: ...` instructions become
  memories deterministically with `confidence = 1.0`. The LLM path is additive
  and can be disabled entirely with `use_llm_formation=False`.
- **The LLM extractor proposes memories only** — not preferences and not project
  state. Those have exact-key, single-current-value semantics that a
  probabilistic model should not be writing (ADR 0006).
- **Formation is explicit, never scheduled.** There is no background worker, so
  there is no way for a model to write while nobody is looking.
- **No import command.** `export` is implemented (and deterministic), import is
  not. Snapshot/restore is done at the SQLite level via `snapshot`.
- **`project_state` is a versioned current value, not a workflow engine.**
- **Identity is a memory kind**, not a separate persona subsystem.
- **numpy is optional.** It accelerates the vector scan ~5.5× at 10,000
  memories; everything works without it, and both paths are asserted equal.

---

## 11. v0.2 — the intelligence layer

### 11.1 Updated layering

```
        Consumers (CLI now; UI / Agent / mobile / IoT later)
                              |
                    MemoryCore facade (api.py)
                              |
        +---------------------+---------------------+
        |                     |                     |
   Retrieval             Formation              Storage
   keyword|semantic|     candidates ->         SQLite
   hybrid (read-only)    verdict                source of truth
        |                     |                     ^
        |                     +---------------------+
        |                        validated write gate
        v
   VectorIndex (derived, disposable)
        ^
        |
   EmbeddingProvider  <- an independent choice from ChatProvider
```

Two boundaries were added in v0.2, and both are one-directional:

- **EmbeddingProvider** is a peer of `ChatProvider`, not a member of it
  (ADR 0004). A chat model and an embedding model are separate decisions because
  in practice they are separate vendors.
- **VectorIndex** hangs off retrieval and is *never* reachable from storage. It
  is derived data: droppable, rebuildable, replaceable (ADR 0005).

### 11.2 LLM memory formation

```
utterance
  -> event appended to `events` (always, first)
  -> prompt = utterance + the existing memories the model may supersede
  -> ChatProvider.chat(...)
  -> strict schema validation          (formation/candidate.py)
  -> Candidate objects (untrusted)     (formation/base.py)
  -> acceptance policy                 (formation/policy.py)
  -> domain guards                     (domain/guards.py)
  -> committed in one transaction      (api.MemoryCore._commit)
```

Operations are ADD, UPDATE/SUPERSEDE and NOOP. Three independent gates stand
between the model and the database, described in ADR 0006. The property that
matters: **any failure at any stage results in zero writes**, while the raw event
is still recorded, so the turn can be re-extracted later.

### 11.3 Candidate lifecycle

```
candidate -> validate -> accept / reject -> commit
```

Every outcome is recorded, with the candidate, the verdict, the reason, and the
id of anything written. A rejection is a normal result, not an error, and it
never means data was lost.

`propose()` runs the whole pipeline **without writing anything at all** - not
even the event - so it is safe to call on arbitrary text.

Provenance carried on every LLM-formed memory: source event ids, the originating
model (`source.model`), the model that produced the row (`generated_by`), the
creation time, the memory kind, and a confidence strictly below 1.0.

### 11.4 EmbeddingProvider

A separate protocol with its own capability declaration, its own registry types,
and its own configuration role. Implementations: `FakeEmbeddingProvider`
(deterministic, offline), `OllamaEmbeddingProvider` (real, local).

`identifier()` returns `"<name>:<model>"` and is stored on every vector, so
vectors from different models are never compared with each other.

### 11.5 VectorIndex

`VectorIndex` has two implementations and an optional acceleration path:

| Implementation | Storage | Purpose |
|---|---|---|
| `SqliteVectorIndex` | derived `embeddings` table | default, persistent |
| `InMemoryVectorIndex` | none | proves replaceability; used in tests |

Both are exact (brute-force cosine), and both are deterministic: ordering is
score descending, then id ascending.

Vectors are stored L2-normalised as little-endian float32, so cosine similarity
is a plain dot product. Storage persists opaque bytes and validates only the
length contract; the vector maths never leaves `retrieval/`.

### 11.6 Hybrid retrieval

| mode | mechanism | model use |
|---|---|---|
| `keyword` | FTS5 BM25 over memories and preferences | none |
| `semantic` | cosine similarity through the VectorIndex | embedding only |
| `hybrid` | Reciprocal Rank Fusion of the two | embedding only |

RRF is used because BM25 and cosine have no common scale; only rank is fused.
In fused mode, bonuses are *multiplicative* (`PHRASE_MULTIPLIER`,
`preference_weight`) because RRF flattens score differences and an additive
bonus would swamp the fusion.

**The degradation rule matters more than the fusion rule:** when only one source
is available, the other is returned **unchanged**. So installing an embedding
provider, building an index, or dropping one can never *reorder* an existing
keyword result set. This is what keeps MSIT-1's byte-identity guarantee intact
when the semantic stack is present.

Semantic mode covers memories only. Preferences are exact-key data and are
deliberately not vectorised - a preference that applies only when a similarity
search happens to surface it is not a preference.

### 11.7 Schema v2

One table added, purely additive:

```sql
CREATE TABLE embeddings (
    memory_id  TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model_id   TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (memory_id, model_id)
);
```

Derived, not source. `SOURCE_TABLES` and `DERIVED_TABLES` are now distinct
constants, export covers only the former, and a migration mechanism plus a
`_verify_schema()` check runs on open, so a migration that silently does nothing
fails loudly.

The composite primary key is the load-bearing detail: an embedding is a
*relationship* between a memory and a model, never a column on the memory. That
is why swapping the embedding model cannot touch the memories.

### 11.8 Performance

Full results in `docs/benchmarks.md`. Headline numbers at 10,000 memories
(256-dimensional embeddings, exact scan):

| metric | value |
|---|---|
| keyword p95 | 7.0 ms |
| vector p95 | 13.4 ms |
| hybrid p95 | 12.8 ms |
| database size | 21.6 MB (14.1 MB of it vectors) |
| index build | 3.9 s |
| single-row insert | 5.95 ms (one transaction + fsync) |

### 11.9 Still not done in v0.2

Scheduled/background extraction · LLM-written preferences and project state ·
ANN indexes · multi-device sync · concurrent writers · encryption · `import` ·
a labelled IR evaluation of retrieval quality · a complex identity subsystem.
None of them is needed for the guarantees v0.2 claims, and each is listed with a
re-entry trigger in §9.
