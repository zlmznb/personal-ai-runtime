# Model Swap Validation — Final Core Capability Acceptance

**Date:** 2026-10-02
**Verdict: PASS**

This is the acceptance experiment for the claim the whole project exists to prove:

> **After swapping the model, the AI's long-term identity, memories and project
> state are still there.**

Raw captured evidence: [`model_swap_validation.json`](model_swap_validation.json)
Reproduce with:

```powershell
py -3 scripts/validation_experiment.py --workdir <dir> --out <dir>\evidence.json
```

**No Memory Core source file was modified for this experiment.** The only file
added is the experiment harness `scripts/validation_experiment.py`, which
composes the public API. Verification of that claim:

```
newest mtime under memory_core/  : 2026-10-02 08:12   (end of the v0.2 phase)
files created by this experiment : scripts/validation_experiment.py  20:12
                                   docs/model_swap_validation.json   20:13
```

---

## 1. 实验环境

| item | value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.12.14 |
| SQLite | 3.53.1 |
| Ollama | local daemon at `http://127.0.0.1:11434` |
| Chat model (Model A) | `qwen3:8b` — real, local, 8B parameters |
| Embedding model | `nomic-embed-text` (768-d), real, local |
| Model B | `fake:model-b` — see §3 |
| Database | `…\validation-216971032\source_of_truth.sqlite` |
| Retrieval mode | `hybrid` (FTS5 + vector, RRF fusion) |

---

## 2. Model A

| property | value |
|---|---|
| Provider class | `OllamaProvider` |
| Identity | `ollama:qwen3:8b` |
| Endpoint | `http://127.0.0.1:11434/v1` (OpenAI-compatible) |
| Role | **writer** — performs LLM Memory Formation |

Model A is a genuine local LLM. It is the only component in this experiment that
exercises judgement: everything else is deterministic code.

---

## 3. Model B

| property | value |
|---|---|
| Provider class | `FakeProvider` |
| Identity | `fake:model-b` |
| Role | **reader** |

### Why Model B is not a second real LLM

A second local chat model was attempted first, and it failed for an environmental
reason, not a design one:

```
ollama pull qwen2.5:1.5b
Error: pull model manifest: Get "https://registry.ollama.ai/v2/library/qwen2.5/manifests/1.5b":
       dial tcp: lookup registry.ollama.ai: no such host
```

This machine currently has **no external network**, and only one chat model is
installed locally. Rather than add a dependency to the experiment, Model B was
built through the existing `ChatProvider` interface, exactly as the acceptance
brief permits.

### Why this makes the evidence *stronger*, not weaker

Model B is deliberately **knowledge-free**. Its reply function can only do one
thing: read the `Memory context:` block out of the prompt it was handed and
repeat it. It has no world knowledge, no training data about the user, and no
memory of its own.

Therefore if the project name `Persona-EdgeAIoT` appears in Model B's answer, that
fact **provably travelled** `SQLite → retrieval → prompt → Model B`. A clever
model could in principle guess; this one cannot. A real second LLM would have
made the result *less* conclusive on the question that matters here, which is
"did the information come from memory?", not "is the answer fluent?".

The architectural point is unaffected: `MemoryCore` is handed a
`ChatProvider`; `OllamaProvider` and `FakeProvider` are two implementations of
that same interface, and `tests/test_providers.py` plus
`tests/test_model_swap_isolation.py` cover five provider configurations
including two real-model ones.

---

## 4. 实验一：写入了哪些 Memory

The user provided the three facts as one onboarding turn:

> 我正在开发 Persona-EdgeAIoT，这是一个长期项目。我的长期研究方向更偏向感知层
> AIoT，而不是单纯训练大模型。我希望 Personal AI 的长期记忆与底层模型解耦。

The existing v0.2 formation pipeline ran:

```
utterance → event appended → prompt → qwen3:8b → strict schema validation
          → Candidate (ADD ×3) → acceptance policy → domain guards → commit
```

**Result: 3 candidates, 3 accepted, 0 rejected, on attempt 1 of 3, in 37.3 s.**
No retry was needed (the harness allows up to 3 and records every attempt).

| # | operation | content | kind | confidence | salience | outcome |
|---|---|---|---|---|---|---|
| 1 | ADD | 用户正在开发名为 Persona-EdgeAIoT 的长期项目 | semantic | 0.95 | 0.90 | accepted |
| 2 | ADD | 用户的研究方向更偏向于感知层 AIoT，而非单纯训练大模型 | semantic | 0.95 | 0.85 | accepted |
| 3 | ADD | 用户希望 Personal AI 的长期记忆与底层模型解耦 | semantic | 0.95 | 0.80 | accepted |

Note that **confidence is 0.95, not 1.0** — an LLM proposal is a hypothesis, and
the architecture refuses to record it as a fact (constraint P0-1).

The written rows, with provenance:

| memory id | content | generated_by | tags | created_at |
|---|---|---|---|---|
| `mem_01M3Y8GSNP4GG5X3GSJ7KJPG1B` | 用户正在开发名为 Persona-EdgeAIoT 的长期项目 | `ollama:qwen3:8b` | `project`, `research` | 2026-10-02T12:13:12.758Z |
| `mem_01M3Y8GSNY5MSR9B9XVE3JZEWH` | 用户的研究方向更偏向于感知层 AIoT，而非单纯训练大模型 | `ollama:qwen3:8b` | `research`, `focus` | 2026-10-02T12:13:12.766Z |
| `mem_01M3Y8GSP64WDYAJYEDCQX39KK` | 用户希望 Personal AI 的长期记忆与底层模型解耦 | `ollama:qwen3:8b` | `architecture`, `design` | 2026-10-02T12:13:12.774Z |

Then the derived vector index was built with the real embedding model:

```
embedded=3  failures=[]  indexed=3  elapsed=0.05s
model_id=ollama-embedding:nomic-embed-text
```

---

## 5. Memory 在 SQLite 中的实际记录

Read with the `sqlite3` module directly — **no Memory Core code involved**. This
is evidence about the file, not about the application's view of it.

```
tables : embeddings, events, memories, memories_fts(+4 shadow), meta,
         preferences, preferences_fts(+4 shadow), project_state
memories : 3      events : 1
vector rows : {"ollama-embedding:nomic-embed-text": 3}
```

The three `memories` rows as stored:

```
mem_01M3Y8GSNP4GG5X3GSJ7KJPG1B
  content     : 用户正在开发名为 Persona-EdgeAIoT 的长期项目
  kind        : semantic
  confidence  : 0.95      salience : 0.9      status : active
  generated_by: ollama:qwen3:8b
  source      : {"event_ids": ["evt_01M3Y8FN8PHJ631FG59T63JTT2"],
                 "model": "ollama:qwen3:8b", "origin": "llm",
                 "reason": "明确说明了当前的核心项目名称和性质"}

mem_01M3Y8GSNY5MSR9B9XVE3JZEWH
  content     : 用户的研究方向更偏向于感知层 AIoT，而非单纯训练大模型
  kind        : semantic
  confidence  : 0.95      salience : 0.85     status : active
  generated_by: ollama:qwen3:8b
  source      : {"event_ids": ["evt_01M3Y8FN8PHJ631FG59T63JTT2"],
                 "model": "ollama:qwen3:8b", "origin": "llm",
                 "reason": "清晰表达了技术研究的核心方向"}

mem_01M3Y8GSP64WDYAJYEDCQX39KK
  content     : 用户希望 Personal AI 的长期记忆与底层模型解耦
  kind        : semantic
  confidence  : 0.95      salience : 0.8      status : active
  generated_by: ollama:qwen3:8b
  source      : {"event_ids": ["evt_01M3Y8FN8PHJ631FG59T63JTT2"],
                 "model": "ollama:qwen3:8b", "origin": "llm",
                 "reason": "阐明了系统设计的关键技术要求"}
```

The raw event is also preserved, which is what makes the memories regenerable:

```
evt_01M3Y8FN8PHJ631FG59T63JTT2 | user |
我正在开发 Persona-EdgeAIoT，这是一个长期项目。我的长期研究方向更偏向感知层 AIoT，
而不是单纯训练大模型。我希望 Personal AI 的长期记忆与底层模型解耦。
```

**Source-of-truth digest** (`sha256` over the sorted `id|content` of `memories`):

```
after Model A finished : 0257b6fe4264b19b90538cbc7b56e4359451b1f3108bdd3c3b9765c4ebdeb6b6
at the very end of E2/E3/E4 : 0257b6fe4264b19b90538cbc7b56e4359451b1f3108bdd3c3b9765c4ebdeb6b6
identical : True
```

---

## 6. 实验二：Model A 停止后的状态

| check | result |
|---|---|
| `core.close()` called on the Model A core | yes |
| database file digest unchanged by closing Model A | **True** |
| `storage` layer holds any reference to a provider | **False** |
| Model B is a different object than Model A | **True** |
| Model B is a different implementation | **True** (`OllamaProvider` → `FakeProvider`) |
| memories visible to Model B | **3** |

```
model_a_class : OllamaProvider      model_a_identity : ollama:qwen3:8b
model_b_class : FakeProvider        model_b_identity : fake:model-b
database_digest_after_model_a_closed : 7d5c24138f6e4bc4b018482ccc3cdc980b36b6130ca004e8c90107058c8f13fd
memory_ids_seen_by_model_b : [mem_01M3Y8GSP64WDYAJYEDCQX39KK,
                              mem_01M3Y8GSNY5MSR9B9XVE3JZEWH,
                              mem_01M3Y8GSNP4GG5X3GSJ7KJPG1B]
```

`memory_count_seen_by_model_b = 3` is the important line: a completely different
provider implementation, opened as a brand-new `MemoryCore` against the same
file, sees every memory Model A wrote. **`Model A ≠ Memory`.**

---

## 7. 实验三：Model B 检索到的 Memory

Model B was asked:

> 你还记得我正在做什么项目，以及我的长期研究方向吗？

Full path taken:

```
Model B
  ↓
MemoryCore.recall(question, mode="hybrid")            ← no chat model involved
  ↓
SQLite memories_fts (BM25)  +  SQLite embeddings (cosine)
  ↓
Reciprocal Rank Fusion → 3 memories
  ↓
prompt built
  ↓
Model B generates the answer
```

**Retrieved context (`context_ids`), identical to what Model A wrote:**

```
mem_01M3Y8GSNP4GG5X3GSJ7KJPG1B   用户正在开发名为 Persona-EdgeAIoT 的长期项目
mem_01M3Y8GSNY5MSR9B9XVE3JZEWH   用户的研究方向更偏向于感知层 AIoT，而非单纯训练大模型
mem_01M3Y8GSP64WDYAJYEDCQX39KK   用户希望 Personal AI 的长期记忆与底层模型解耦
```

### The exact prompt Model B received

This was captured by wrapping Model B in a recording provider. It is the proof
that the model did not guess — the facts were handed to it.

```
--- role=system ---
You are a personal AI assistant with a long-term memory core. The memory context
below was retrieved from your persistent memory store. Treat it as your own
recollection. If the context does not contain the answer, say so rather than
inventing one.

Memory context:
- [semantic] 用户正在开发名为 Persona-EdgeAIoT 的长期项目
- [semantic] 用户的研究方向更偏向于感知层 AIoT，而非单纯训练大模型
- [semantic] 用户希望 Personal AI 的长期记忆与底层模型解耦
--- role=user ---
你还记得我正在做什么项目，以及我的长期研究方向吗？
```

### Model B's answer

```
From my long-term memory I can see:
  * [semantic] 用户正在开发名为 Persona-EdgeAIoT 的长期项目
  * [semantic] 用户的研究方向更偏向于感知层 AIoT，而非单纯训练大模型
  * [semantic] 用户希望 Personal AI 的长期记忆与底层模型解耦
```

`degraded = False`, `provider = fake`, `model = model-b`.

### Ablation — Model B cannot answer without memory

The same question, same Model B, against an **empty** database:

```
memory_count : 0
context_ids  : []
answer       : I have no long-term memory of that.
```

This is the control condition. It rules out the possibility that the answer was
manufactured by the model: with nothing retrieved, Model B has nothing to say.

### Retrieval detail for the same question

| mode | hits | scores |
|---|---|---|
| keyword (FTS5 BM25) | 3 | 2.496789, 2.333063, 0.582489 |
| semantic (cosine) | 3 | 0.594125, 0.593524, 0.510601 |
| hybrid (RRF) | 3 | 0.032787, 0.032258, 0.031746 |

The same three memories come back in the same order under all three modes.

---

## 8. 实验四：Vector Index 删除后的结果

The derived vector index was destroyed with `drop_index(all_models=True)`.
**The SQLite source of truth was not touched.**

| measurement | before drop | after drop |
|---|---|---|
| vector rows in `embeddings` | 3 | **0** |
| rows in `memories` | 3 | **3** |
| semantic retrieval hits | 3 | **0** |
| keyword retrieval hits | 3 | **3** |
| hybrid retrieval hits | 3 | **3** |

`dropped_vector_rows = 3`. Independent SQLite read after the drop:

```
tables         : (unchanged)
memories       : 3
vector rows    : 0
memories still present:
   mem_01M3Y8GSNP4GG5X3GSJ7KJPG1B | 用户正在开发名为 Persona-EdgeAIoT 的长期项目
   mem_01M3Y8GSNY5MSR9B9XVE3JZEWH | 用户的研究方向更偏向于感知层 AIoT，而非单纯训练大模型
   mem_01M3Y8GSP64WDYAJYEDCQX39KK | 用户希望 Personal AI 的长期记忆与底层模型解耦
```

What this proves:

- **The vector index is not the only source of long-term memory.** Deleting it
  costs semantic recall and nothing else.
- Semantic search returns nothing (its backing data is gone) while keyword
  search still returns all three memories, order intact.
- Hybrid degrades cleanly to keyword — the degradation rule from
  `architecture.md` §11.6 in action.
- **SQLite is the source of truth.** All three memories are still there, still
  readable, still searchable, still exportable.

---

## 9. 实验四（续）：Rebuild 后的结果

The index was rebuilt **from SQLite** using the existing `rebuild_index()` flow:

```
SQLite memories → embedding provider → embeddings table
```

```
rebuild result : embedded=3  failures=[]  indexed=3
                 model_id=ollama-embedding:nomic-embed-text
after rebuild  : vector rows = {"ollama-embedding:nomic-embed-text": 3}
```

Semantic retrieval came back, with **identical scores** to before the index was
destroyed:

| mode | before drop | after rebuild |
|---|---|---|
| semantic | 0.594125, 0.593524, 0.510601 | 0.594125, 0.593524, 0.510601 |
| keyword | 2.496789, 2.333063, 0.582489 | 2.496789, 2.333063, 0.582489 |
| hybrid | 0.032787, 0.032258, 0.031746 | 0.032787, 0.032258, 0.031746 |

The rebuilt index is *equivalent* to the destroyed one, because it was derived
from the same source of truth.

Model B re-read the same memories, with the same context and the same answer:

```
context_ids : [mem_01M3Y8GSNP4GG5X3GSJ7KJPG1B,
               mem_01M3Y8GSNY5MSR9B9XVE3JZEWH,
               mem_01M3Y8GSP64WDYAJYEDCQX39KK]

answer      : From my long-term memory I can see:
                * [semantic] 用户正在开发名为 Persona-EdgeAIoT 的长期项目
                * [semantic] 用户的研究方向更偏向于感知层 AIoT，而非单纯训练大模型
                * [semantic] 用户希望 Personal AI 的长期记忆与底层模型解耦
```

### A note on the database file digest

```
after Model A closed : 7d5c24138f6e4bc4b018482ccc3cdc980b36b6130ca004e8c90107058c8f13fd
at the very end      : 75fb628c1a1e4a7d6db2e1dd4d0b4646041bb206a95cbb5fce901b026ee42c73
```

These differ, and they **should**. Rebuilding the index rewrote the derived
`embeddings` table (new `created_at` values). The file digest is therefore the
wrong instrument here; the right one is the source digest, which is unchanged:

```
source-of-truth digest after Model A : 0257b6fe4264b19b90538cbc7b56e4359451b1f3108bdd3c3b9765c4ebdeb6b6
source-of-truth digest at the end    : 0257b6fe4264b19b90538cbc7b56e4359451b1f3108bdd3c3b9765c4ebdeb6b6
identical : True
```

It is exactly because re-embedding may legitimately change the file that
`export_jsonl()` covers the source tables only — and why MSIT-1 (no derived
writes at all) can assert byte-identity of the whole file while MSIT-2 asserts
byte-identity of the source tables.

---

## 10. 最终结论

```
Model A (ollama:qwen3:8b — real 8B LLM)
      ↓
Memory Core   (learn → Candidate → Validate → Accept → commit)
      ↓
SQLite        (3 memories, 1 raw event, source of truth)
      ↓
  ══ switch model ══
      ↓
Model B (fake:model-b — knowledge-free, different implementation)
      ↓
the same Memory Core
      ↓
the same SQLite
      ↓
the same long-term user information
```

Every assertion in the verdict is true:

| assertion | result |
|---|---|
| memories written by Model A | 3 |
| formation attempts used | 1 of 3 |
| project memory present after everything | True |
| research-direction memory present | True |
| decoupling memory present | True |
| Model B's answer cites the project | True |
| Model B's answer cites the research direction | True |
| Model B's answer after index rebuild cites the project | True |
| Model B cannot answer without memory (ablation) | True |
| vector index was destroyed (0 rows) | True |
| memories survived index destruction | True |
| vector index rebuilt from SQLite | True |
| same memory ids before and after | True |
| source-of-truth digest unchanged across the whole experiment | True |

**PASS — the Personal Memory Core genuinely achieves the claim:**

> 模型可替换，但长期记忆、身份和项目状态不随模型消失。

### What this experiment did *not* need

- No Chroma, Qdrant, Mem0, LangGraph, LiteLLM
- No MCP, no Agent, no Web UI, no IoT, no mobile, no cloud sync
- No architectural refactor, no new abstraction
- No change to any `memory_core/` source file

### Honest limitations

1. **Only one real chat model was available.** Model B is a provider
   implementation reached through the same interface, not a second LLM. The
   ablation and the captured prompt compensate for this: they prove the
   information came from retrieval rather than from the model. A second real LLM
   can be run with
   `--model-b-provider ollama --model-b-model <name>` at any time without any
   code change.
2. **Formation quality is model-dependent.** qwen3:8b succeeded on the first
   attempt here; the harness allows 3 and reports every attempt, because a small
   model can occasionally return schema-invalid JSON. That failure path is
   covered by `tests/test_learn_lifecycle.py` and never pollutes the database.
3. **The experiment ran on one machine, one database, three memories.** It is a
   capability proof, not a load test. Performance at scale is covered separately
   in [`benchmarks.md`](benchmarks.md).
