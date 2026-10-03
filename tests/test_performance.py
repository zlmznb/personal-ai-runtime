"""Bounded performance sanity checks.

This is not the benchmark - that is ``scripts/benchmark.py``, which produces the
numbers in ``docs/benchmarks.md``. These tests only assert that the system stays
*correct* and does not blow up at a modest scale.

The time bounds are deliberately loose (seconds, not milliseconds) so that they
cannot flake on a slow or loaded machine. Their job is to catch a genuine
complexity regression, not to measure throughput.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest

from memory_core.api import MemoryCore
from memory_core.domain import guards
from memory_core.providers.fake_embedding import FakeEmbeddingProvider

SIZE = 300
QUERY = "本地优先"
LOOSE_BOUND_SECONDS = 10.0

TOPICS = ("本地优先", "隐私边界", "模型替换", "向量检索", "长期记忆")


def synthetic(count):
    memories = []
    for index in range(count):
        memories.append(
            guards.build_memory(
                "第 {0} 条记忆，主题 {1}，编号 {2}".format(index, TOPICS[index % len(TOPICS)], index),
                tags=["bench", "t{0}".format(index % 5)],
            )
        )
    return memories


class ScaleTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-perf-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")

    def test_bulk_insert_is_fast_and_correct(self):
        core = MemoryCore.open(db_path=self.db_path)
        self.addCleanup(core.close)
        start = time.perf_counter()
        written = core.store.insert_memories(synthetic(SIZE))
        elapsed = time.perf_counter() - start

        self.assertEqual(written, SIZE)
        self.assertEqual(core.store.count_memories(), SIZE)
        self.assertLess(elapsed, LOOSE_BOUND_SECONDS, "bulk insert took {0:.1f}s".format(elapsed))

    def test_every_mode_stays_correct_and_bounded_at_scale(self):
        core = MemoryCore.open(
            db_path=self.db_path, embedding_provider=FakeEmbeddingProvider(model="bench", dim=128)
        )
        self.addCleanup(core.close)
        core.store.insert_memories(synthetic(SIZE))
        core.build_index()

        expected_ids = sorted(memory.id for memory in core.list_memories(limit=SIZE * 2))
        self.assertEqual(len(expected_ids), SIZE)

        for mode in ("keyword", "semantic", "hybrid"):
            with self.subTest(mode=mode):
                core.recall(QUERY, mode=mode)  # warm up
                start = time.perf_counter()
                hits = core.recall(QUERY, mode=mode, limit=10)
                elapsed = time.perf_counter() - start
                self.assertTrue(hits, "{0} returned nothing".format(mode))
                self.assertLessEqual(len(hits), 10)
                self.assertLess(
                    elapsed,
                    LOOSE_BOUND_SECONDS,
                    "{0} took {1:.2f}s at {2} memories".format(mode, elapsed, SIZE),
                )

    def test_index_build_scales_linearly_enough(self):
        core = MemoryCore.open(
            db_path=self.db_path, embedding_provider=FakeEmbeddingProvider(model="bench", dim=64)
        )
        self.addCleanup(core.close)
        core.store.insert_memories(synthetic(SIZE))
        start = time.perf_counter()
        result = core.build_index()
        elapsed = time.perf_counter() - start

        self.assertEqual(result["embedded"], SIZE)
        self.assertEqual(result["failures"], [])
        self.assertLess(elapsed, LOOSE_BOUND_SECONDS)

    def test_rebuild_is_not_slower_than_the_first_build_by_an_order_of_magnitude(self):
        core = MemoryCore.open(
            db_path=self.db_path, embedding_provider=FakeEmbeddingProvider(model="bench", dim=64)
        )
        self.addCleanup(core.close)
        core.store.insert_memories(synthetic(SIZE))

        start = time.perf_counter()
        core.build_index()
        first = time.perf_counter() - start

        start = time.perf_counter()
        core.rebuild_index()
        second = time.perf_counter() - start

        self.assertLess(second, max(first * 10.0, LOOSE_BOUND_SECONDS))

    def test_index_size_is_proportional_to_the_number_of_vectors(self):
        provider = FakeEmbeddingProvider(model="bench", dim=64)
        core = MemoryCore.open(db_path=self.db_path, embedding_provider=provider)
        self.addCleanup(core.close)

        core.store.insert_memories(synthetic(100))
        core.build_index()
        small = core.store.page_size_bytes()

        core.store.insert_memories(synthetic(200))
        core.build_index()
        large = core.store.page_size_bytes()

        self.assertGreater(large, small)
        self.assertEqual(core.store.count_embeddings(), 300)

    def test_zero_memories_is_not_a_special_case_failure(self):
        core = MemoryCore.open(
            db_path=self.db_path, embedding_provider=FakeEmbeddingProvider(model="bench", dim=16)
        )
        self.addCleanup(core.close)
        self.assertEqual(core.build_index()["embedded"], 0)
        for mode in ("keyword", "semantic", "hybrid"):
            self.assertEqual(core.recall(QUERY, mode=mode), [])


if __name__ == "__main__":
    unittest.main()
