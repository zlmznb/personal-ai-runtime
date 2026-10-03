"""Performance benchmark for Memory Core.

Measures, at several memory counts:

* insert throughput (bulk) and single-insert latency
* index build time
* keyword (FTS5), vector and hybrid query latency
* database size, and the size the derived vector index adds

Run it directly::

    py -3 scripts/benchmark.py
    py -3 scripts/benchmark.py --sizes 100,1000,10000 --queries 30 --dim 256
    py -3 scripts/benchmark.py --no-numpy --out docs/benchmark-fallback.json

The default embedding provider is the deterministic offline fake, so the numbers
measure *this code* rather than a network. Swap in a real model with
``--embedding-provider ollama-embedding --embedding-model all-minilm`` when you
want end-to-end latency including the model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.api import MemoryCore  # noqa: E402
from memory_core.domain import guards  # noqa: E402
from memory_core.providers.fake_embedding import FakeEmbeddingProvider  # noqa: E402
from memory_core.retrieval import vector_index  # noqa: E402
from memory_core.storage.sqlite_store import SqliteStore  # noqa: E402

TOPICS = (
    "本地优先架构", "隐私边界", "模型替换", "向量检索", "长期记忆",
    "端云协同", "项目状态", "偏好设置", "事件日志", "索引重建",
    "determinism", "embedding model", "vector index", "hybrid retrieval",
    "source of truth", "schema validation", "candidate lifecycle", "rollback",
)

VERBS = (
    "用户正在开发", "系统记录了", "团队决定", "项目要求", "架构约束规定",
    "用户偏好", "必须保证", "实验表明", "文档说明了", "测试覆盖了",
)

QUERIES = (
    "本地优先",
    "模型替换",
    "向量索引",
    "privacy boundary",
    "hybrid retrieval",
    "source of truth",
    "候选校验",
    "rollback 回滚",
)


def synthetic_memories(count, scope="global"):
    # type: (int, str) -> list
    memories = []
    for index in range(count):
        topic = TOPICS[index % len(TOPICS)]
        verb = VERBS[(index // len(TOPICS)) % len(VERBS)]
        content = "{0}{1}，编号 {2}，这是第 {3} 条用于性能测试的长期记忆".format(
            verb, topic, index, index
        )
        memories.append(
            guards.build_memory(
                content,
                kind="semantic",
                scope=scope,
                tags=["benchmark", "topic:{0}".format(index % len(TOPICS))],
                salience=(index % 10) / 10.0,
            )
        )
    return memories


def percentile(values, fraction):
    # type: (list, float) -> float
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[position]


def timed(function, repeat=1):
    # type: (object, int) -> tuple
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return result, samples


def build_embedding_provider(args):
    # type: (argparse.Namespace) -> object
    if args.embedding_provider == "fake-embedding":
        return FakeEmbeddingProvider(model=args.embedding_model or "bench-embed", dim=args.dim)
    from memory_core.providers.registry import resolve_cli_embedding

    return resolve_cli_embedding({}, args.embedding_provider, args.embedding_model)


def measure_single_inserts(args):
    # type: (argparse.Namespace) -> dict
    """Per-row (one transaction each) insert latency, in its own database.

    Kept separate so the probe rows cannot inflate the index measurements.
    """
    workdir = tempfile.mkdtemp(prefix="memory-core-bench-latency-")
    try:
        store = SqliteStore(os.path.join(workdir, "memory.sqlite"))
        try:
            samples = []
            for memory in synthetic_memories(args.single_inserts):
                start = time.perf_counter()
                store.insert_memory(memory)
                samples.append((time.perf_counter() - start) * 1000.0)
        finally:
            store.close()
        return {
            "samples": len(samples),
            "p50_ms": round(percentile(samples, 0.50), 4),
            "p95_ms": round(percentile(samples, 0.95), 4),
            "mean_ms": round(statistics.fmean(samples), 4),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def benchmark_size(size, args):
    # type: (int, argparse.Namespace) -> dict
    workdir = tempfile.mkdtemp(prefix="memory-core-bench-")
    db_path = os.path.join(workdir, "memory.sqlite")
    try:
        provider = build_embedding_provider(args)
        store = SqliteStore(db_path)
        try:
            memories = synthetic_memories(size)
            start = time.perf_counter()
            store.insert_memories(memories)
            bulk_seconds = time.perf_counter() - start
        finally:
            store.close()

        without_index = os.path.getsize(db_path)

        core = MemoryCore.open(
            db_path=db_path, embedding_provider=provider, use_llm_formation=False
        )
        try:
            start = time.perf_counter()
            index_result = core.build_index()
            index_seconds = time.perf_counter() - start

            with_index = os.path.getsize(db_path)
            index_bytes = max(0, with_index - without_index)

            timings = {}  # type: dict
            for mode in ("keyword", "semantic", "hybrid"):
                # Warm up so we measure steady state, not index loading.
                core.recall(QUERIES[0], mode=mode, limit=10)

                samples = []
                hits = 0
                for _ in range(args.queries):
                    for query in QUERIES:
                        start = time.perf_counter()
                        result = core.recall(query, mode=mode, limit=10)
                        samples.append((time.perf_counter() - start) * 1000.0)
                        hits += len(result)
                timings[mode] = {
                    "mean_ms": round(statistics.fmean(samples), 4),
                    "p50_ms": round(percentile(samples, 0.50), 4),
                    "p95_ms": round(percentile(samples, 0.95), 4),
                    "max_ms": round(max(samples), 4),
                    "queries": len(samples),
                    "avg_hits": round(hits / float(len(samples)), 2),
                }

            stats = core.stats()
            index_stats = core.index_stats() or {}
        finally:
            core.close()

        rows = index_result["indexed"]
        dim = index_stats.get("dim") or 0
        return {
            "memories": size,
            "numpy": index_stats.get("numpy"),
            "embedding_model": provider.identifier(),
            "embedding_dim": dim,
            "insert": {
                "bulk_seconds": round(bulk_seconds, 4),
                "bulk_per_second": round(size / bulk_seconds, 1) if bulk_seconds else None,
            },
            "index": {
                "build_seconds": round(index_seconds, 4),
                "rows": rows,
                "pending_before": index_result["pending_before"],
                "failures": len(index_result["failures"]),
            },
            "size": {
                "db_bytes_without_index": without_index,
                "db_bytes_with_index": with_index,
                "index_bytes": index_bytes,
                "bytes_per_vector": (round(index_bytes / float(rows), 1) if rows else None),
                "raw_vector_bytes": rows * dim * 4,
                "page_bytes": stats["page_bytes"],
            },
            "queries": timings,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def render(report):
    # type: (dict) -> str
    lines = []
    lines.append("")
    lines.append("Memory Core performance benchmark")
    lines.append("=" * 100)
    lines.append(
        "embedding={0}  dim={1}  numpy={2}  queries/mode={3}".format(
            report["embedding_model"],
            report["embedding_dim"],
            report["numpy"],
            report["queries"],
        )
    )
    single = report.get("single_insert") or {}
    if single:
        lines.append(
            "single-row insert (one transaction each): "
            "p50={0:.2f}ms  p95={1:.2f}ms  mean={2:.2f}ms  n={3}".format(
                single["p50_ms"], single["p95_ms"], single["mean_ms"], single["samples"]
            )
        )
    lines.append("")
    header = "{0:>8} | {1:>10} | {2:>9} | {3:>9} | {4:>9} | {5:>9} | {6:>10} | {7:>10}".format(
        "memories", "bulk(s)", "ins/s", "kw p50", "vec p50", "hyb p50", "db(MB)", "index(MB)"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in report["results"]:
        queries = row["queries"]
        lines.append(
            "{0:>8} | {1:>10.2f} | {2:>9.0f} | {3:>8.3f}ms | {4:>8.3f}ms | {5:>8.3f}ms | {6:>10.2f} | {7:>10.2f}".format(
                row["memories"],
                row["insert"]["bulk_seconds"],
                row["insert"]["bulk_per_second"] or 0,
                queries["keyword"]["p50_ms"],
                queries["semantic"]["p50_ms"],
                queries["hybrid"]["p50_ms"],
                row["size"]["db_bytes_with_index"] / 1048576.0,
                row["size"]["index_bytes"] / 1048576.0,
            )
        )
    lines.append("")
    lines.append("p95 latency")
    lines.append("-" * 60)
    for row in report["results"]:
        lines.append(
            "  {0:>6} memories: keyword {1:>7.3f}ms  vector {2:>7.3f}ms  hybrid {3:>7.3f}ms".format(
                row["memories"],
                row["queries"]["keyword"]["p95_ms"],
                row["queries"]["semantic"]["p95_ms"],
                row["queries"]["hybrid"]["p95_ms"],
            )
        )
    lines.append("")
    lines.append("index build / storage")
    lines.append("-" * 60)
    for row in report["results"]:
        lines.append(
            "  {0:>6} memories: build {1:>7.2f}s  vectors {2:>6}  raw {3:>8} B  db {4:>8.2f} MB".format(
                row["memories"],
                row["index"]["build_seconds"],
                row["index"]["rows"],
                row["size"]["raw_vector_bytes"],
                row["size"]["db_bytes_with_index"] / 1048576.0,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    # type: (object) -> int
    parser = argparse.ArgumentParser(description="Memory Core performance benchmark")
    parser.add_argument("--sizes", default="100,1000,10000", help="comma-separated memory counts")
    parser.add_argument("--queries", type=int, default=20, help="query rounds per mode")
    parser.add_argument("--dim", type=int, default=256, help="fake embedding dimension")
    parser.add_argument("--single-inserts", type=int, default=50, help="single-insert latency samples")
    parser.add_argument("--embedding-provider", default="fake-embedding")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--no-numpy", action="store_true", help="force the pure-Python vector path")
    parser.add_argument("--out", default=None, help="write the JSON report to this path")
    args = parser.parse_args(argv)

    if args.no_numpy:
        vector_index.set_numpy_enabled(False)

    sizes = [int(value) for value in str(args.sizes).split(",") if value.strip()]
    single_insert = measure_single_inserts(args)
    results = []
    for size in sizes:
        sys.stderr.write("benchmarking {0} memories...\n".format(size))
        results.append(benchmark_size(size, args))

    report = {
        "sizes": sizes,
        "queries": args.queries,
        "numpy": vector_index.has_numpy(),
        "embedding_model": results[0]["embedding_model"] if results else None,
        "embedding_dim": results[0]["embedding_dim"] if results else None,
        "single_insert": single_insert,
        "results": results,
    }
    sys.stdout.write(render(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("JSON report written to {0}\n".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
