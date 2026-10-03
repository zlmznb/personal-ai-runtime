"""SemanticIndexer tests: building, rebuilding and failing safely."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from memory_core.domain import guards
from memory_core.providers.fake_embedding import FakeEmbeddingProvider
from memory_core.retrieval.indexer import SemanticIndexer
from memory_core.retrieval.vector_index import InMemoryVectorIndex, SqliteVectorIndex
from memory_core.storage.sqlite_store import SqliteStore


class IndexerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-indexer-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")
        self.store = SqliteStore(self.db_path)
        self.addCleanup(self.store.close)
        self.provider = FakeEmbeddingProvider(model="fake-embed", dim=64)
        self.indexer = SemanticIndexer(self.store, self.provider)

    def remember(self, content, **kwargs):
        memory = guards.build_memory(content, **kwargs)
        return self.store.insert_memory(memory)


class BuildTests(IndexerTestCase):
    def test_build_embeds_every_active_memory(self):
        for index in range(4):
            self.remember("memory number {0}".format(index))
        result = self.indexer.build()
        self.assertEqual(result["pending_before"], 4)
        self.assertEqual(result["embedded"], 4)
        self.assertEqual(result["indexed"], 4)
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["model_id"], "fake-embedding:fake-embed")

    def test_build_is_incremental(self):
        self.remember("first")
        self.remember("second")
        self.assertEqual(self.indexer.build()["embedded"], 2)

        self.remember("third")
        second_run = self.indexer.build()
        self.assertEqual(second_run["pending_before"], 1)
        self.assertEqual(second_run["embedded"], 1)
        self.assertEqual(second_run["indexed"], 3)

    def test_build_with_nothing_to_do(self):
        result = self.indexer.build()
        self.assertEqual(result["embedded"], 0)
        self.assertEqual(result["indexed"], 0)

    def test_scope_filter(self):
        self.remember("global memory")
        self.remember("project memory", scope="project:memory")
        result = self.indexer.build(scope="project:memory")
        self.assertEqual(result["embedded"], 1)

    def test_kind_filter(self):
        self.remember("a semantic fact", kind="semantic")
        self.remember("a remembered event", kind="episodic")
        self.assertEqual(self.indexer.build(kinds=("episodic",))["embedded"], 1)

    def test_limit(self):
        for index in range(5):
            self.remember("m{0}".format(index))
        self.assertEqual(self.indexer.build(limit=2)["embedded"], 2)

    def test_batch_size_does_not_change_the_result(self):
        for index in range(5):
            self.remember("m{0}".format(index))
        small = SemanticIndexer(self.store, self.provider, batch_size=1).build()
        self.assertEqual(small["indexed"], 5)

    def test_superseded_memories_are_not_indexed(self):
        original = self.remember("old statement")
        replacement = guards.build_memory("new statement")
        self.store.supersede_memory(original.id, replacement)
        self.indexer.build()
        indexed = {row[0] for row in self.store.iter_embeddings(self.indexer.model_id)}
        self.assertIn(replacement.id, indexed)
        self.assertNotIn(original.id, indexed)


class RebuildTests(IndexerTestCase):
    def test_rebuild_replaces_everything(self):
        for index in range(3):
            self.remember("m{0}".format(index))
        self.indexer.build()
        result = self.indexer.rebuild()
        self.assertEqual(result["cleared"], 3)
        self.assertEqual(result["indexed"], 3)

    def test_drop_leaves_the_memories_untouched(self):
        self.remember("survives the drop")
        self.indexer.build()
        removed = self.indexer.drop()
        self.assertEqual(removed, 1)
        self.assertEqual(self.store.count_memories(), 1)
        self.assertEqual(self.store.count_embeddings(), 0)

    def test_rebuild_after_drop_restores_search(self):
        self.remember("用户偏好本地优先")
        self.indexer.build()
        before = [hit.memory_id for hit in self.indexer.index.search(self.provider.embed_one("本地优先"))]
        self.indexer.drop()
        self.assertEqual(self.indexer.index.search(self.provider.embed_one("本地优先")), [])
        self.indexer.rebuild()
        after = [hit.memory_id for hit in self.indexer.index.search(self.provider.embed_one("本地优先"))]
        self.assertEqual(before, after)


class FailureTests(IndexerTestCase):
    def test_provider_failure_is_recorded_and_writes_nothing(self):
        self.remember("will not be embedded")
        failing = FakeEmbeddingProvider(model="fake-embed", dim=64, fail_with="embedding down")
        indexer = SemanticIndexer(self.store, failing)
        result = indexer.build()
        self.assertEqual(result["embedded"], 0)
        self.assertEqual(len(result["failures"]), 1)
        self.assertIn("embedding down", result["failures"][0]["error"])
        self.assertEqual(self.store.count_embeddings(), 0)
        # The memories are untouched.
        self.assertEqual(self.store.count_memories(), 1)

    def test_a_failed_batch_does_not_stop_later_batches(self):
        for index in range(4):
            self.remember("m{0}".format(index))

        calls = {"n": 0}

        class FlakyProvider(FakeEmbeddingProvider):
            def embed(self, texts):
                calls["n"] += 1
                if calls["n"] == 1:
                    from memory_core.domain.errors import ProviderError

                    raise ProviderError("first batch fails")
                return super(FlakyProvider, self).embed(texts)

        indexer = SemanticIndexer(self.store, FlakyProvider(model="f", dim=32), batch_size=2)
        result = indexer.build()
        self.assertEqual(len(result["failures"]), 2)
        self.assertEqual(result["embedded"], 2)

    def test_indexer_requires_a_provider(self):
        with self.assertRaises(Exception):
            SemanticIndexer(self.store, None)


class StatsTests(IndexerTestCase):
    def test_stats_reports_staleness(self):
        self.remember("one")
        self.remember("two")
        self.indexer.build()
        stats = self.indexer.stats()
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["stale"], 0)
        self.assertEqual(stats["model_id"], "fake-embedding:fake-embed")
        self.assertEqual(stats["provider"]["identifier"], "fake-embedding:fake-embed")

        self.remember("three")
        self.assertEqual(self.indexer.stats()["stale"], 1)

    def test_stats_before_any_build(self):
        self.remember("one")
        stats = self.indexer.stats()
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["stale"], 1)


class IndexBackendTests(IndexerTestCase):
    def test_an_in_memory_index_can_be_substituted(self):
        self.remember("用户偏好本地优先")
        index = InMemoryVectorIndex(self.provider.identifier())
        indexer = SemanticIndexer(self.store, self.provider, index=index)
        indexer.build()
        self.assertEqual(index.count(), 1)
        # Nothing was persisted: the substitute index is not the source of truth.
        self.assertEqual(self.store.count_embeddings(), 0)
        self.assertEqual(self.store.count_memories(), 1)

    def test_document_includes_subject_and_tags(self):
        memory = guards.build_memory("short", subject="user:self", tags=["privacy"])
        document = SemanticIndexer._document(memory)
        self.assertIn("short", document)
        self.assertIn("user:self", document)
        self.assertIn("privacy", document)

    def test_default_index_is_sqlite_backed(self):
        self.assertIsInstance(self.indexer.index, SqliteVectorIndex)


if __name__ == "__main__":
    unittest.main()
