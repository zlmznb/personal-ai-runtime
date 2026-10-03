"""Hybrid retrieval tests: keyword, semantic and fused modes.

The property that matters most here: adding an embedding provider and a vector
index may only ever *add* recall. Without one, hybrid must return exactly what
keyword returns - so the model-swap guarantee of MSIT-1 cannot be disturbed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from memory_core.api import MemoryCore
from memory_core.providers.fake_embedding import FakeEmbeddingProvider


class RetrievalTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-hybrid-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")

    def open_core(self, embedding=True, dim=128):
        provider = FakeEmbeddingProvider(model="fake-embed", dim=dim) if embedding else None
        core = MemoryCore.open(db_path=self.db_path, embedding_provider=provider)
        self.addCleanup(core.close)
        return core

    def ids(self, core, query, **kwargs):
        return [hit.id for hit in core.recall(query, **kwargs)]


class KeywordModeTests(RetrievalTestCase):
    def test_keyword_mode_works_with_nothing_configured(self):
        core = self.open_core(embedding=False)
        memory = core.remember("用户偏好本地优先，隐私敏感")
        self.assertEqual(self.ids(core, "本地优先", mode="keyword"), [memory.id])

    def test_keyword_mode_ignores_the_vector_index(self):
        core = self.open_core()
        memory = core.remember("用户偏好本地优先")
        core.build_index()
        self.assertEqual(self.ids(core, "本地优先", mode="keyword"), [memory.id])


class SemanticModeTests(RetrievalTestCase):
    def test_semantic_returns_nothing_without_an_embedding_provider(self):
        core = self.open_core(embedding=False)
        core.remember("用户偏好本地优先")
        self.assertEqual(self.ids(core, "本地优先", mode="semantic"), [])

    def test_semantic_returns_nothing_before_the_index_is_built(self):
        core = self.open_core()
        core.remember("用户偏好本地优先")
        self.assertEqual(self.ids(core, "本地优先", mode="semantic"), [])

    def test_semantic_finds_the_relevant_memory_after_indexing(self):
        core = self.open_core()
        relevant = core.remember("用户偏好本地优先，隐私敏感")
        core.remember("用户喜欢喝咖啡")
        core.remember("Deployment runs on Friday")
        core.build_index()

        hits = core.recall("本地优先", mode="semantic")
        self.assertTrue(hits)
        self.assertEqual(hits[0].id, relevant.id)
        self.assertEqual(hits[0].matched_on, "vector")

    def test_semantic_finds_a_memory_with_no_shared_keyword(self):
        # The whole point of vector search: no lexical overlap, still retrieved.
        core = self.open_core()
        core.remember("用户偏好本地优先，隐私敏感")
        core.remember("unrelated content about gardening")
        core.build_index()
        hits = core.recall("隐私", mode="semantic")
        self.assertTrue(hits)
        self.assertIn("本地优先", hits[0].item.content)

    def test_semantic_respects_scope(self):
        core = self.open_core()
        core.remember("用户偏好本地优先", scope="global")
        scoped = core.remember("用户偏好本地优先", scope="project:memory")
        core.build_index()
        hits = core.recall("本地优先", scope="project:memory", mode="semantic")
        self.assertEqual([hit.id for hit in hits], [scoped.id])

    def test_semantic_respects_kinds(self):
        core = self.open_core()
        core.remember("alpha fact statement", kind="semantic")
        episodic = core.remember("alpha fact statement", kind="episodic")
        core.build_index()
        hits = core.recall("alpha fact statement", kinds=("episodic",), mode="semantic")
        self.assertEqual([hit.id for hit in hits], [episodic.id])

    def test_semantic_excludes_superseded_memories(self):
        core = self.open_core()
        original = core.remember("theme is dark")
        replacement = core.supersede_memory(original.id, "theme is light")
        core.build_index()
        hits = core.recall("theme is dark", mode="semantic")
        self.assertNotIn(original.id, [hit.id for hit in hits])
        self.assertIn(replacement.id, [hit.id for hit in hits])

    def test_semantic_excludes_forgotten_memories(self):
        core = self.open_core()
        memory = core.remember("secret preference about tea")
        core.build_index()
        self.assertTrue(core.recall("tea", mode="semantic"))
        core.forget(memory.id)
        self.assertEqual(core.recall("tea", mode="semantic"), [])

    def test_a_failing_embedding_provider_degrades_to_no_hits(self):
        core = self.open_core()
        core.remember("用户偏好本地优先")
        core.build_index()
        core.embedding_provider.fail_with = "embedding service down"
        self.assertEqual(core.recall("本地优先", mode="semantic"), [])

    def test_semantic_limit_is_respected(self):
        core = self.open_core()
        for index in range(6):
            core.remember("shared statement number {0}".format(index))
        core.build_index()
        self.assertEqual(len(core.recall("shared statement", mode="semantic", limit=2)), 2)


class HybridModeTests(RetrievalTestCase):
    def test_hybrid_without_an_index_equals_keyword_exactly(self):
        core = self.open_core(embedding=False)
        for text in ("用户偏好本地优先", "本地部署优先", "unrelated note"):
            core.remember(text)
        keyword = self.ids(core, "本地", mode="keyword")
        hybrid = self.ids(core, "本地", mode="hybrid")
        self.assertEqual(hybrid, keyword)

    def test_hybrid_with_an_empty_index_equals_keyword_exactly(self):
        core = self.open_core()
        for text in ("用户偏好本地优先", "本地部署优先", "unrelated note"):
            core.remember(text)
        keyword = self.ids(core, "本地", mode="keyword")
        hybrid = self.ids(core, "本地", mode="hybrid")
        self.assertEqual(hybrid, keyword)

    def test_hybrid_marks_fused_hits(self):
        core = self.open_core()
        core.remember("用户偏好本地优先，隐私敏感")
        core.remember("Deployment runs on Friday")
        core.build_index()
        hits = core.recall("本地优先", mode="hybrid")
        self.assertTrue(hits)
        self.assertEqual(hits[0].matched_on, "hybrid")

    def test_hybrid_finds_what_either_source_finds(self):
        core = self.open_core()
        lexical = core.remember("the exact keyword zebra appears here")
        core.remember("用户偏好本地优先，隐私敏感")
        core.build_index()

        hybrid_ids = self.ids(core, "zebra", mode="hybrid")
        self.assertIn(lexical.id, hybrid_ids)

    def test_hybrid_default_mode_is_hybrid(self):
        core = self.open_core()
        core.remember("用户偏好本地优先")
        core.build_index()
        self.assertEqual(
            self.ids(core, "本地优先"),
            self.ids(core, "本地优先", mode="hybrid"),
        )

    def test_hybrid_includes_preferences_from_the_keyword_side(self):
        core = self.open_core()
        preference = core.set_preference("theme", "dark", statement="theme dark")
        core.remember("dark")
        core.build_index()
        hits = core.recall("dark", mode="hybrid")
        self.assertIn(preference.id, [hit.id for hit in hits])


class DeterminismTests(RetrievalTestCase):
    def test_every_mode_is_repeatable(self):
        core = self.open_core()
        for text in ("用户偏好本地优先", "本地部署优先", "Deployment runs on Friday"):
            core.remember(text)
        core.build_index()
        for mode in ("keyword", "semantic", "hybrid"):
            first = [(hit.id, round(hit.score, 9), hit.matched_on) for hit in core.recall("本地", mode=mode)]
            second = [(hit.id, round(hit.score, 9), hit.matched_on) for hit in core.recall("本地", mode=mode)]
            self.assertEqual(first, second, msg=mode)

    def test_rebuild_does_not_change_semantic_results(self):
        core = self.open_core()
        for text in ("用户偏好本地优先", "本地部署优先", "unrelated note"):
            core.remember(text)
        core.build_index()
        before = [(hit.id, round(hit.score, 9)) for hit in core.recall("本地", mode="semantic")]
        core.rebuild_index()
        after = [(hit.id, round(hit.score, 9)) for hit in core.recall("本地", mode="semantic")]
        self.assertEqual(before, after)


class ModeValidationTests(RetrievalTestCase):
    def test_unknown_mode_is_rejected(self):
        core = self.open_core()
        with self.assertRaises(ValueError):
            core.recall("anything", mode="magic")

    def test_available_modes(self):
        self.assertEqual(list(self.open_core().recall_modes()), ["keyword", "semantic", "hybrid"])

    def test_unsearchable_query_returns_nothing_in_every_mode(self):
        core = self.open_core()
        core.remember("用户偏好本地优先")
        core.build_index()
        for mode in ("keyword", "semantic", "hybrid"):
            self.assertEqual(core.recall("!!!", mode=mode), [])


if __name__ == "__main__":
    unittest.main()
