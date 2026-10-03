"""VectorIndex tests.

The central property: the index is *derived*. It can be deleted, rebuilt, swapped
for a different implementation, or collapsed to the pure-Python path, and the
memories are unaffected.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import unittest

from memory_core.domain import guards
from memory_core.domain.errors import StorageError
from memory_core.retrieval.vector_index import (
    FLOAT32_ITEMSIZE,
    InMemoryVectorIndex,
    SqliteVectorIndex,
    cosine_similarity,
    decode_vector,
    encode_vector,
    has_numpy,
    normalize,
    set_numpy_enabled,
)
from memory_core.storage.sqlite_store import SqliteStore


def unit(*values):
    return normalize(list(values))


class EncodingTests(unittest.TestCase):
    def test_normalize_produces_a_unit_vector(self):
        vector = normalize([3.0, 4.0])
        self.assertAlmostEqual(math.hypot(*vector), 1.0, places=9)
        self.assertAlmostEqual(vector[0], 0.6, places=9)

    def test_normalize_handles_the_zero_vector(self):
        self.assertEqual(normalize([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])

    def test_encode_decode_round_trip(self):
        dim, blob = encode_vector([0.5, -0.25, 1.0])
        self.assertEqual(dim, 3)
        self.assertEqual(len(blob), 3 * FLOAT32_ITEMSIZE)
        decoded = decode_vector(blob, 3)
        self.assertAlmostEqual(decoded[0], 0.5, places=6)
        self.assertAlmostEqual(decoded[1], -0.25, places=6)

    def test_decode_rejects_a_dimension_mismatch(self):
        _dim, blob = encode_vector([1.0, 2.0])
        with self.assertRaises(ValueError):
            decode_vector(blob, 5)

    def test_cosine_of_identical_unit_vectors_is_one(self):
        vector = unit(1.0, 2.0, 3.0)
        self.assertAlmostEqual(cosine_similarity(vector, vector), 1.0, places=6)

    def test_cosine_of_orthogonal_vectors_is_zero(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, places=9)

    def test_cosine_rejects_mismatched_dimensions(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0])


class InMemoryIndexTests(unittest.TestCase):
    def test_upsert_and_search(self):
        index = InMemoryVectorIndex("model-a")
        index.upsert("a", [1.0, 0.0])
        index.upsert("b", [0.0, 1.0])
        hits = index.search([0.9, 0.1], limit=2)
        self.assertEqual([hit.memory_id for hit in hits], ["a", "b"])
        self.assertGreater(hits[0].score, hits[1].score)

    def test_count_delete_clear(self):
        index = InMemoryVectorIndex("model-a")
        index.upsert_many([("a", [1.0, 0.0]), ("b", [0.0, 1.0])])
        self.assertEqual(index.count(), 2)
        self.assertTrue(index.delete("a"))
        self.assertFalse(index.delete("a"))
        self.assertEqual(index.count(), 1)
        self.assertEqual(index.clear(), 1)
        self.assertEqual(index.count(), 0)

    def test_search_on_an_empty_index(self):
        self.assertEqual(InMemoryVectorIndex("m").search([1.0, 0.0]), [])

    def test_vectors_are_normalised_on_write(self):
        index = InMemoryVectorIndex("m")
        index.upsert("a", [10.0, 0.0])
        self.assertAlmostEqual(index.search([1.0, 0.0])[0].score, 1.0, places=6)

    def test_dimension_mismatch_is_rejected(self):
        index = InMemoryVectorIndex("m")
        index.upsert("a", [1.0, 0.0])
        with self.assertRaises(ValueError):
            index.upsert("b", [1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            index.search([1.0, 0.0, 0.0])

    def test_ordering_is_total_and_deterministic(self):
        index = InMemoryVectorIndex("m")
        for name in ("c", "a", "b"):
            index.upsert(name, [1.0, 0.0])
        first = [hit.memory_id for hit in index.search([1.0, 0.0])]
        second = [hit.memory_id for hit in index.search([1.0, 0.0])]
        self.assertEqual(first, second)
        self.assertEqual(first, ["a", "b", "c"])

    def test_limit_is_respected(self):
        index = InMemoryVectorIndex("m")
        for name in "abcd":
            index.upsert(name, [1.0, 0.0])
        self.assertEqual(len(index.search([1.0, 0.0], limit=2)), 2)

    def test_stats(self):
        index = InMemoryVectorIndex("model-a")
        index.upsert("a", [1.0, 0.0])
        stats = index.stats()
        self.assertEqual(stats["kind"], "in-memory")
        self.assertEqual(stats["model_id"], "model-a")
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["dim"], 2)


class SqliteIndexTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-vindex-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")
        self.store = SqliteStore(self.db_path)
        self.addCleanup(self.store.close)

    def make_memory(self, content):
        memory = guards.build_memory(content)
        return self.store.insert_memory(memory)


class SqliteIndexTests(SqliteIndexTestCase):
    def test_upsert_and_search(self):
        first = self.make_memory("alpha")
        second = self.make_memory("beta")
        index = SqliteVectorIndex(self.store, "model-a")
        index.upsert(first.id, [1.0, 0.0])
        index.upsert(second.id, [0.0, 1.0])

        hits = index.search([1.0, 0.0], limit=2)
        self.assertEqual([hit.memory_id for hit in hits], [first.id, second.id])
        self.assertEqual(index.count(), 2)

    def test_persists_across_a_reopen(self):
        memory = self.make_memory("durable")
        SqliteVectorIndex(self.store, "model-a").upsert(memory.id, [1.0, 0.0])
        self.store.close()

        reopened = SqliteStore(self.db_path)
        self.addCleanup(reopened.close)
        hits = SqliteVectorIndex(reopened, "model-a").search([1.0, 0.0])
        self.assertEqual([hit.memory_id for hit in hits], [memory.id])

    def test_clear_removes_only_this_models_vectors(self):
        memory = self.make_memory("shared")
        SqliteVectorIndex(self.store, "model-a").upsert(memory.id, [1.0, 0.0])
        SqliteVectorIndex(self.store, "model-b").upsert(memory.id, [0.0, 1.0])

        removed = SqliteVectorIndex(self.store, "model-a").clear()
        self.assertEqual(removed, 1)
        self.assertEqual(self.store.count_embeddings("model-a"), 0)
        self.assertEqual(self.store.count_embeddings("model-b"), 1)

    def test_models_do_not_mix(self):
        memory = self.make_memory("content")
        SqliteVectorIndex(self.store, "model-a").upsert(memory.id, [1.0, 0.0])
        # model-b has no vectors yet, so it must return nothing.
        self.assertEqual(SqliteVectorIndex(self.store, "model-b").search([1.0, 0.0]), [])

    def test_delete_removes_one_memory(self):
        memory = self.make_memory("x")
        index = SqliteVectorIndex(self.store, "model-a")
        index.upsert(memory.id, [1.0, 0.0])
        self.assertTrue(index.delete(memory.id))
        self.assertFalse(index.delete(memory.id))
        self.assertEqual(index.count(), 0)

    def test_deleting_a_memory_cascades_to_its_vectors(self):
        memory = self.make_memory("temporary")
        SqliteVectorIndex(self.store, "model-a").upsert(memory.id, [1.0, 0.0])
        self.store.delete_memory(memory.id)
        self.assertEqual(self.store.count_embeddings(), 0)

    def test_upsert_is_idempotent(self):
        memory = self.make_memory("x")
        index = SqliteVectorIndex(self.store, "model-a")
        index.upsert(memory.id, [1.0, 0.0])
        index.upsert(memory.id, [0.0, 1.0])
        self.assertEqual(index.count(), 1)
        self.assertAlmostEqual(index.search([0.0, 1.0])[0].score, 1.0, places=6)

    def test_reload_picks_up_external_writes(self):
        memory = self.make_memory("x")
        index = SqliteVectorIndex(self.store, "model-a")
        self.assertEqual(index.search([1.0, 0.0]), [])
        # A second index instance writes behind this one's cache.
        SqliteVectorIndex(self.store, "model-a").upsert(memory.id, [1.0, 0.0])
        self.assertEqual(index.search([1.0, 0.0]), [])
        index.reload()
        self.assertEqual(len(index.search([1.0, 0.0])), 1)

    def test_stats(self):
        memory = self.make_memory("x")
        index = SqliteVectorIndex(self.store, "model-a")
        index.upsert(memory.id, [1.0, 0.0])
        stats = index.stats()
        self.assertEqual(stats["kind"], "sqlite")
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["dim"], 2)

    def test_upsert_many(self):
        memories = [self.make_memory("m{0}".format(i)) for i in range(3)]
        index = SqliteVectorIndex(self.store, "model-a")
        written = index.upsert_many([(m.id, [1.0, float(i)]) for i, m in enumerate(memories)])
        self.assertEqual(written, 3)
        self.assertEqual(index.count(), 3)


class BackendEquivalenceTests(SqliteIndexTestCase):
    """The numpy path is an optimisation only: results must not depend on it."""

    def tearDown(self):
        set_numpy_enabled(True)

    def _search_with(self, use_numpy):
        set_numpy_enabled(use_numpy)
        index = SqliteVectorIndex(self.store, "model-a", use_numpy=use_numpy)
        return [(hit.memory_id, round(hit.score, 6)) for hit in index.search([1.0, 0.2, 0.4], limit=5)]

    def test_pure_python_and_numpy_agree(self):
        for index in range(5):
            memory = self.make_memory("memory {0}".format(index))
            SqliteVectorIndex(self.store, "model-a").upsert(
                memory.id, [float(index), 1.0, 0.5]
            )
        accelerated = self._search_with(True)
        pure = self._search_with(False)
        self.assertEqual(accelerated, pure)

    def test_works_with_numpy_disabled(self):
        memory = self.make_memory("only pure python")
        set_numpy_enabled(False)
        index = SqliteVectorIndex(self.store, "model-a")
        index.upsert(memory.id, [1.0, 0.0])
        hits = index.search([1.0, 0.0])
        self.assertEqual(len(hits), 1)
        self.assertFalse(index.stats()["numpy"])

    def test_in_memory_index_also_works_without_numpy(self):
        set_numpy_enabled(False)
        index = InMemoryVectorIndex("m", use_numpy=False)
        index.upsert("a", [1.0, 0.0])
        self.assertEqual(len(index.search([1.0, 0.0])), 1)

    def test_environment_report(self):
        self.assertIsInstance(has_numpy(), bool)


class StorageEmbeddingTests(SqliteIndexTestCase):
    def test_upsert_validates_the_encoding_length(self):
        memory = self.make_memory("x")
        with self.assertRaises(StorageError):
            self.store.upsert_embedding(memory.id, "m", 4, b"\x00" * 8)

    def test_upsert_rejects_non_positive_dim(self):
        memory = self.make_memory("x")
        with self.assertRaises(StorageError):
            self.store.upsert_embedding(memory.id, "m", 0, b"")

    def test_upsert_rejects_a_non_bytes_vector(self):
        memory = self.make_memory("x")
        with self.assertRaises(StorageError):
            self.store.upsert_embedding(memory.id, "m", 2, "not bytes")

    def test_embedding_for_an_unknown_memory_is_rejected(self):
        with self.assertRaises(StorageError):
            self.store.upsert_embedding("mem_missing", "m", 2, b"\x00" * 8)

    def test_get_and_iter(self):
        memory = self.make_memory("x")
        self.store.upsert_embedding(memory.id, "m", 2, b"\x00" * 8)
        dim, blob = self.store.get_embedding(memory.id, "m")
        self.assertEqual(dim, 2)
        self.assertEqual(len(blob), 8)
        self.assertIsNone(self.store.get_embedding(memory.id, "other"))
        self.assertEqual(len(list(self.store.iter_embeddings("m"))), 1)

    def test_model_ids(self):
        memory = self.make_memory("x")
        self.store.upsert_embedding(memory.id, "b-model", 2, b"\x00" * 8)
        self.store.upsert_embedding(memory.id, "a-model", 2, b"\x00" * 8)
        self.assertEqual(self.store.embedding_model_ids(), ["a-model", "b-model"])

    def test_memory_ids_missing_embedding(self):
        first = self.make_memory("first")
        second = self.make_memory("second")
        self.store.upsert_embedding(first.id, "m", 2, b"\x00" * 8)
        missing = self.store.memory_ids_missing_embedding("m")
        self.assertEqual(missing, [second.id])

    def test_superseded_memories_are_not_indexed_by_default(self):
        original = self.make_memory("old")
        replacement = guards.build_memory("new")
        self.store.supersede_memory(original.id, replacement)
        missing = self.store.memory_ids_missing_embedding("m")
        self.assertEqual(missing, [replacement.id])

    def test_get_memories_by_ids_preserves_order(self):
        memories = [self.make_memory("m{0}".format(i)) for i in range(3)]
        wanted = [memories[2].id, memories[0].id, memories[1].id]
        loaded = self.store.get_memories_by_ids(wanted)
        self.assertEqual([m.id for m in loaded], wanted)

    def test_get_memories_by_ids_ignores_unknown_ids(self):
        memory = self.make_memory("only one")
        self.assertEqual(
            [m.id for m in self.store.get_memories_by_ids(["nope", memory.id])], [memory.id]
        )
        self.assertEqual(self.store.get_memories_by_ids([]), [])


if __name__ == "__main__":
    unittest.main()
