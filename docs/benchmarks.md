# Benchmarks — v0.2

Measured with `scripts/benchmark.py` on the development machine (Windows,
Python 3.12.14, SQLite 3.53.1, local NVMe).

Reproduce with:

```powershell
py -3 scripts/benchmark.py --sizes 100,1000,10000 --queries 20 --out docs/benchmark-numpy.json
py -3 scripts/benchmark.py --sizes 100,1000,10000 --queries 3 --no-numpy --out docs/benchmark-pure-python.json
```

Embeddings are the deterministic offline `fake-embedding` provider at 256
dimensions. That is deliberate: it measures **this code** rather than a network
round trip. Real-model latency is a property of the model, not of the memory
core, and `tests/test_ollama_integration.py` covers real embeddings separately.

---

## Results — numpy acceleration (default when numpy is importable)

| memories | bulk insert | insert/s | keyword p50 | vector p50 | hybrid p50 | db size | index size |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.01 s | 8,679 | 0.296 ms | 0.833 ms | 0.851 ms | 0.35 MB | 0.14 MB |
| 1,000 | 0.07 s | 14,752 | 0.753 ms | 1.681 ms | 1.748 ms | 2.29 MB | 1.41 MB |
| 10,000 | 0.66 s | 15,214 | 5.261 ms | 10.924 ms | 11.261 ms | 21.56 MB | 14.13 MB |

### p95 latency

| memories | keyword | vector | hybrid |
|---:|---:|---:|---:|
| 100 | 0.517 ms | 1.361 ms | 1.219 ms |
| 1,000 | 1.097 ms | 2.376 ms | 2.394 ms |
| 10,000 | 7.025 ms | 13.408 ms | 12.827 ms |

### Index build and storage

| memories | build | vectors | raw vector bytes | bytes/vector incl. page overhead |
|---:|---:|---:|---:|---:|
| 100 | 0.04 s | 100 | 102,400 | ~1,470 |
| 1,000 | 0.38 s | 1,000 | 1,024,000 | ~1,480 |
| 10,000 | 3.86 s | 10,000 | 10,240,000 | ~1,480 |

Single-row insert (one transaction and one fsync each, the interactive path):
**p50 5.95 ms, p95 6.69 ms**. Bulk insert in one transaction is ~2,500× faster
per row, which is why `reindex` and bulk import use the batched path.

---

## Results — pure Python (no numpy)

Same workload with `--no-numpy`, i.e. the fallback that must work when numpy is
absent (Python 3.8 environments, minimal installs).

| memories | keyword p50 | vector p50 | hybrid p50 | vector p95 |
|---:|---:|---:|---:|---:|
| 100 | 0.373 ms | 1.335 ms | 1.342 ms | 1.571 ms |
| 1,000 | 0.746 ms | 6.923 ms | 6.922 ms | 7.687 ms |
| 10,000 | 5.364 ms | 60.381 ms | 59.992 ms | 64.444 ms |

**Keyword search is unaffected** — FTS5 runs inside SQLite and never touches
numpy. Only the brute-force vector scan changes, by roughly 5.5× at 10,000
memories.

Both paths are asserted to return identical results in
`tests/test_vector_index.py::BackendEquivalenceTests`.

---

## What the numbers mean

1. **The v0.2 architecture holds well past personal scale.** At 10,000 memories
   hybrid retrieval is ~11 ms p50 / ~13 ms p95 with numpy, well inside the
   v0.1 target of 50 ms p95. A personal memory core will not see this number for
   years.

2. **Brute force is the right choice, and this is the evidence.** An exact scan
   over 10,000 vectors costs single-digit milliseconds. An ANN index would trade
   exactness for a speedup that is not needed, and would add a second store to
   keep consistent — the thing ADR 0001 exists to prevent.

3. **Scaling is close to linear**, as expected for an exact scan: keyword grows
   0.30 → 0.75 → 5.26 ms and vector 0.83 → 1.68 → 10.92 ms for 10×/10×. There is
   no cliff, and no super-linear surprise.

4. **Storage is dominated by the vector index**, not by the memories: at 10,000
   memories the database is 21.6 MB, of which 14.1 MB is vectors. Raw float32
   data is 10.2 MB, so the overhead is ~1.4× — SQLite page granularity, not
   waste. Dropping the index reclaims it and loses no data.

5. **The index build is a batch operation**, ~3.9 s for 10,000 vectors with the
   deterministic fake provider. With a real embedding model this is dominated by
   the model; `all-minilm` embeds a small batch in under a second while
   `nomic-embed-text` takes ~25 s for a cold first call.

## Deliberately not measured

- **Concurrent writers.** SQLite in rollback-journal mode serialises writers.
  Multi-process access is out of scope for v0.2 and is the trigger that would
  justify a real vector service (see architecture.md §9).
- **Real embedding throughput at scale.** 10,000 real embeddings is a
  model-throughput question, not a memory-core question.
- **Recall quality on a labelled corpus.** Only targeted probes exist
  (documented below), not a full IR evaluation.

## A note on embedding-model quality (measured, not theoretical)

The same two memories were probed with both locally available models:

| model | dim | query `隐私优先` | verdict |
|---|---:|---|---|
| `nomic-embed-text` | 768 | relevant 0.7070 vs unrelated 0.7027 | correct, but a narrow margin |
| `all-minilm` | 384 | relevant 0.4410 vs unrelated 0.5409 | **wrong** — English-only model on Chinese text |

`all-minilm` is an English-only model; on Chinese content its rankings are close
to noise. This is worth stating plainly:

> **Embedding-model choice affects retrieval quality. It never affects
> durability.** Both models produce a working system in which every memory is
> still present, still exported, still keyword-searchable, and still recoverable
> from SQLite after the index is dropped.

That distinction is the whole point of keeping the embedding model behind a
separate protocol and the vectors in a disposable table.
