# Memory — Personal AI Memory Core

> **Memory is the asset. Models are consumables.**

A model-agnostic, local-first memory core for a personal AI. It stores long-term
identity, memories, preferences and project state in a single SQLite file, and
treats every model — chat *and* embedding — as an interchangeable, optional
external process.

The goal is **not** a "memory chatbot". The goal is to prove one thing:

> **After swapping the model, the AI's long-term identity, memories and project
> state are still there.**

---

## Status: v0.2 Memory Intelligence

**v0.1 (durable core)** — SQLite source of truth · event log · memories ·
preferences · project state · FTS5 retrieval · `remember`/`recall`/`forget`/
`export`/`snapshot` · FakeProvider · OllamaProvider · DeepSeekProvider · CLI ·
MSIT-1.

**v0.2 (intelligence layer)** — LLM-assisted memory formation (ADD / SUPERSEDE /
NOOP) with strict schema validation · candidate lifecycle with accept/reject
verdicts · an independent `EmbeddingProvider` · a derived, disposable,
rebuildable `VectorIndex` · hybrid retrieval (FTS5 + vectors via RRF) · MSIT-2 ·
measured performance at 100 / 1,000 / 10,000 memories.

Not implemented (deliberately): Chroma, Qdrant, sqlite-vec, LiteLLM, Mem0,
LangGraph, MCP, Agent, Web UI, mobile, IoT, cloud sync, multi-user, scheduled
extraction, fine-tuning, ANN indexes, complex identity system.

See [docs/architecture.md](docs/architecture.md), the recorded decisions in
[docs/decisions/](docs/decisions/), the measured numbers in
[docs/benchmarks.md](docs/benchmarks.md), and the end-to-end capability proof in
[docs/model_swap_validation.md](docs/model_swap_validation.md).

---

## Proof of the core claim

[docs/model_swap_validation.md](docs/model_swap_validation.md) is a recorded
acceptance experiment that walks the full lifecycle on a real local 8B model:

1. **Model A** (`ollama:qwen3:8b`, real) writes three long-term memories through
   LLM formation.
2. **Model A is stopped**; the storage layer holds no reference to any provider.
3. **Model B** (a knowledge-free provider, different implementation) opens the
   same database and answers a question about the user. The exact prompt it
   received is captured, proving the answer came from retrieval, and an ablation
   against an empty database proves it cannot answer otherwise.
4. **The entire vector index is destroyed.** All memories remain, still
   keyword-searchable. The index is then **rebuilt from SQLite**, and semantic
   retrieval returns identical scores.

Result: **PASS**. The source-of-truth digest is byte-identical from the moment
Model A finished to the end of the experiment. No `memory_core/` source file was
modified to run it.

---

## Architecture red lines

These are not preferences. They are invariants, and tests enforce them.

1. **`memory.sqlite` is the only long-term source of truth.**
   No model session state, no LangGraph checkpoint, no vector database, no
   second store may hold long-term memory.

2. **`storage/` and `domain/` never import a model SDK, and never perform HTTP.**
   Storage is "dumb": it persists bytes, computes no vectors, and makes no
   similarity judgements.

3. **The chat model can never affect what is recalled.**
   Retrieval consults no chat model. Semantic mode uses the *embedding* provider,
   which is an independent choice.

4. **Every model output passes through Memory Core validation before Storage.**
   The LLM proposes candidates; `MemoryCore` decides whether they are written.

5. **With no model available at all, the database must still be readable,
   searchable, exportable and deletable.**

6. **The vector index is derived data.** It can be deleted, rebuilt, or replaced
   with no loss of memory.

7. **The embedding model is independent of the chat model.** Separate protocol,
   separate configuration, separate swap.

> Litmus test: *if every model disappeared tomorrow, would `memory.sqlite` still
> be a complete, readable, usable personal memory?* If the answer is yes, the
> architecture is intact.

---

## Requirements

- Python **3.8+** (verified on 3.8.6 and 3.12.14)
- **Zero required dependencies.** Standard library only (`sqlite3`, `json`,
  `urllib`, `argparse`, `dataclasses`, `array`, `hashlib`).
- SQLite with FTS5 (bundled with CPython; verified 3.32.3 and 3.53.1).
- *Optional:* `numpy` accelerates the vector scan ~5.5× (10,000 memories:
  10.9 ms → 60.4 ms without it). Everything works without it, and both paths are
  asserted to return identical results.
- *Optional:* [Ollama](https://ollama.com) for local chat and local embeddings.

Nothing needs to be installed.

---

## Quick start

```powershell
$PY = "py"   # or "python"

& $PY -m memory_core init

# --- explicit, deterministic memory writes -------------------------------
& $PY -m memory_core remember "用户偏好本地优先，隐私敏感" --kind semantic --tag privacy
& $PY -m memory_core observe "记住：我偏好深色主题"
& $PY -m memory_core observe "记住偏好：theme = dark"
& $PY -m memory_core observe '项目状态：memory = {"phase": "v0.2"}'
& $PY -m memory_core observe "身份：我是一个本地优先的个人 AI"

# --- retrieval (never touches a chat model) ------------------------------
& $PY -m memory_core recall "偏好"                       # hybrid by default
& $PY -m memory_core recall "偏好" --search-mode keyword
& $PY -m memory_core recall "偏好" --search-mode semantic

# --- LLM-assisted formation ---------------------------------------------
& $PY -m memory_core --provider ollama --model qwen3:8b learn "我叫阿哲，主要用 Python 和 Rust"
& $PY -m memory_core --provider ollama --model qwen3:8b propose "我在重构检索层"   # dry run, writes nothing

# --- semantic index (derived; drop and rebuild freely) -------------------
& $PY -m memory_core --embedding-provider ollama-embedding --embedding-model nomic-embed-text index build
& $PY -m memory_core --embedding-provider ollama-embedding --embedding-model nomic-embed-text index stats
& $PY -m memory_core --embedding-provider ollama-embedding --embedding-model nomic-embed-text index drop --all
& $PY -m memory_core --embedding-provider ollama-embedding --embedding-model nomic-embed-text index rebuild

# --- preferences, project state, data management -------------------------
& $PY -m memory_core pref set editor vscode
& $PY -m memory_core project set memory '{"phase": "v0.2"}'
& $PY -m memory_core forget <memory_id>
& $PY -m memory_core export --out export.jsonl
& $PY -m memory_core snapshot --out backup.sqlite
& $PY -m memory_core stats
```

The database path defaults to `./memory.sqlite`. Override with `--db PATH` or
`MEMORY_DB`.

### Providers

Chat and embedding are independent; mix them freely.

```powershell
# offline deterministic, no network
& $PY -m memory_core --provider fake --embedding-provider fake-embedding ask "我偏好什么"

# local chat + local embeddings
& $PY -m memory_core --provider ollama --model qwen3:8b `
    --embedding-provider ollama-embedding --embedding-model nomic-embed-text ask "我偏好什么"

# cloud chat (no embeddings endpoint exists at DeepSeek) + local embeddings
$env:DEEPSEEK_API_KEY = "sk-..."
& $PY -m memory_core --provider deepseek --model deepseek-chat `
    --embedding-provider ollama-embedding --embedding-model nomic-embed-text ask "我偏好什么"
```

Or declare roles in `config/models.json` (copy `config/models.example.json`).

---

## Running the tests

No install and no test framework required — tests are stdlib `unittest` and also
run under pytest if you have it.

```powershell
py -3 -m unittest discover -s tests -t . -v
```

The default suite needs **no network and no model**: 536 tests, of which 11 are
opt-in integration tests that skip. Verified on Python 3.8.6 and 3.12.14.

### Opt-in real-model integration

```powershell
$env:MEMORY_TEST_OLLAMA = "1"
py -3 -m unittest tests.test_ollama_integration -v
```

With a local Ollama: 536 tests, 0 skipped, 0 failures.

Requires Ollama with a chat model (`qwen3:8b` by default) and at least one
embedding model (`ollama pull nomic-embed-text all-minilm`). Override with
`MEMORY_TEST_OLLAMA_MODEL` and `MEMORY_TEST_EMBED_MODELS`.

These cover real embeddings, real LLM formation, real embedding-model swapping,
and MSIT-1 against a real 8B model.

---

## The headline tests: swap the model, keep the memory

### MSIT-1 — the chat model cannot touch the data

`tests/test_model_swap_isolation.py`. A scripted session writes identity,
memories, preferences and project state; the database is hashed; the chat
provider is swapped across five configurations (no provider, FakeA, FakeB,
unreachable Ollama, keyless DeepSeek). Then, for all 24 queries, the memory IDs,
order **and scores** must be identical, and the database must be
**byte-identical**.

Only possible because retrieval never consults a chat model.

### MSIT-2 — the embedding model and the vector index cannot either

The same file. Memories are created with chat model A and indexed with embedding
model A. Then chat model B **and** embedding model B are swapped in, the entire
vector index is dropped, and it is rebuilt from SQLite. Throughout,
`export_jsonl()` — the source tables only — must be byte-identical, and every
memory must remain present, keyword-searchable and recoverable.

Prove it by hand:

```powershell
$db = "$env:TEMP\demo.sqlite"
py -3 -m memory_core --db $db remember "用户正在开发 Persona-EdgeAIoT，这是一个长期项目"
$h0 = (Get-FileHash $db -Algorithm SHA256).Hash

py -3 -m memory_core --db $db --provider fake recall "Persona"
py -3 -m memory_core --db $db --provider ollama --model qwen3:8b recall "Persona"
py -3 -m memory_core --db $db --provider deepseek --model deepseek-chat recall "Persona"

# different models -> different answers; identical memories; unchanged file
(Get-FileHash $db -Algorithm SHA256).Hash -eq $h0    # must be True
```

---

## Performance

Measured by `scripts/benchmark.py` (256-d embeddings, exact scan, numpy).
Full results and the pure-Python comparison: [docs/benchmarks.md](docs/benchmarks.md).

| memories | keyword p50 | vector p50 | hybrid p50 | db size |
|---:|---:|---:|---:|---:|
| 100 | 0.30 ms | 0.83 ms | 0.85 ms | 0.35 MB |
| 1,000 | 0.75 ms | 1.68 ms | 1.75 ms | 2.29 MB |
| 10,000 | 5.26 ms | 10.92 ms | 11.26 ms | 21.56 MB |

```powershell
py -3 scripts/benchmark.py --sizes 100,1000,10000 --queries 20
```

---

## Project layout

```
memory_core/
  api.py            Single public facade. The only validated write gate.
  cli.py            argparse CLI
  config.py         JSON config; CLI > env > file > defaults
  domain/           Pure data + validation. No IO, no network, no model SDK.
  storage/          SQLite only. Source of truth + derived vector table.
  retrieval/        Read-only retrieval: tokenize, fusion, vector_index, indexer
  formation/        Rules + LLM extractors, candidate schema, acceptance policy
  providers/        The ONLY place HTTP or a vendor protocol may live
tests/              11 modules; MSIT-1/MSIT-2 are the headline
fixtures/           Scripted session used by the model-swap tests
scripts/            benchmark.py
config/             Optional JSON config
docs/               architecture.md, benchmarks.md, decisions/
```

---

## License / ownership

Personal project. No data leaves the machine unless you explicitly configure a
cloud provider, and even then only prompt text is sent — never the database.
