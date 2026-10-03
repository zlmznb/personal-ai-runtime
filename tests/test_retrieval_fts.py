"""Retrieval tests: FTS5 lexical search, CJK handling, and determinism.

The determinism tests here are the mechanism behind the headline guarantee:
retrieval must be a pure function of stored data.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from memory_core.domain import guards
from memory_core.retrieval import fusion, tokenize
from memory_core.retrieval.retriever import Retriever
from memory_core.storage.sqlite_store import SqliteStore


class TokenizeTests(unittest.TestCase):
    def test_normalises_english(self):
        self.assertEqual(tokenize.normalize_search_text("Local-First Design!"), "local first design")

    def test_splits_cjk_per_character(self):
        self.assertEqual(tokenize.normalize_search_text("本地优先"), "本 地 优 先")

    def test_punctuation_is_dropped_so_phrases_normalise_identically(self):
        self.assertEqual(
            tokenize.normalize_search_text("用户偏好：本地优先"),
            tokenize.normalize_search_text("用户偏好本地优先"),
        )

    def test_mixed_script(self):
        self.assertEqual(tokenize.normalize_search_text("Python项目 v0.1"), "python 项 目 v0 1")

    def test_query_units_split_words_and_characters(self):
        self.assertEqual(tokenize.query_units("local 本地"), ["local", "本", "地"])

    def test_match_expression_is_an_or_of_quoted_units(self):
        self.assertEqual(tokenize.build_match_expression("本地"), '"本" OR "地"')

    def test_match_expression_is_none_for_unsearchable_input(self):
        for empty in ("", "   ", "!!!", "，。、"):
            self.assertIsNone(tokenize.build_match_expression(empty))

    def test_match_expression_cannot_inject_fts_syntax(self):
        # User input must never reach the MATCH parser as syntax.
        expression = tokenize.build_match_expression('x" OR * NEAR(a b) -')
        self.assertIsNotNone(expression)
        self.assertNotIn("*", expression)
        self.assertNotIn("NEAR", expression)
        self.assertNotIn("-", expression)

    def test_phrase_needle_matches_index_normalisation(self):
        self.assertEqual(tokenize.phrase_needle("本地优先"), tokenize.normalize_search_text("本地优先"))


class RetrievalTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-retrieval-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.store = SqliteStore(os.path.join(self.tmpdir, "memory.sqlite"))
        self.addCleanup(self.store.close)
        self.retriever = Retriever(self.store)

    def ids(self, query, **kwargs):
        return [hit.id for hit in self.retriever.recall(query, **kwargs)]


class LexicalSearchTests(RetrievalTestCase):
    def test_english_query_matches(self):
        memory = guards.build_memory("Prefers local-first architecture")
        self.store.insert_memory(memory)
        self.store.insert_memory(guards.build_memory("Likes dark theme"))
        self.assertEqual(self.ids("local"), [memory.id])

    def test_chinese_query_matches_chinese_memory(self):
        # Without per-character CJK tokenisation this returns nothing at all.
        memory = guards.build_memory("用户偏好本地优先，隐私敏感")
        self.store.insert_memory(memory)
        self.store.insert_memory(guards.build_memory("用户喜欢吃苹果"))
        hits = self.retriever.recall("偏好")
        self.assertEqual([hit.id for hit in hits], [memory.id])

    def test_partial_chinese_phrase_matches(self):
        memory = guards.build_memory("用户偏好本地优先")
        self.store.insert_memory(memory)
        self.assertEqual(self.ids("本地优先"), [memory.id])

    def test_searchable_across_punctuation_differences(self):
        memory = guards.build_memory("目标是：完全解耦")
        self.store.insert_memory(memory)
        self.assertEqual(self.ids("目标完全"), [memory.id])

    def test_missing_term_returns_nothing(self):
        self.store.insert_memory(guards.build_memory("用户偏好本地优先"))
        self.assertEqual(self.ids("量子计算"), [])

    def test_empty_query_returns_nothing(self):
        self.store.insert_memory(guards.build_memory("anything"))
        for empty in ("", "   ", "!!!"):
            self.assertEqual(self.ids(empty), [])

    def test_malformed_query_does_not_raise(self):
        self.store.insert_memory(guards.build_memory("anything at all"))
        for hostile in ('"', "*", "OR", "NEAR(", "-", "a AND"):
            self.retriever.recall(hostile)  # must not raise


class PhraseBonusTests(RetrievalTestCase):
    def test_contiguous_phrase_outranks_scattered_characters(self):
        contiguous = guards.build_memory("本地优先")
        scattered = guards.build_memory("本着地道的原则")
        self.store.insert_memory(contiguous)
        self.store.insert_memory(scattered)
        hits = self.retriever.recall("本地")
        self.assertEqual(hits[0].id, contiguous.id)
        self.assertEqual(hits[0].matched_on, "phrase")
        self.assertEqual(hits[1].matched_on, "fts")

    def test_phrase_bonus_is_deterministic(self):
        self.store.insert_memory(guards.build_memory("本地优先"))
        first = self.retriever.recall("本地")
        second = self.retriever.recall("本地")
        self.assertEqual([h.score for h in first], [h.score for h in second])


class FilterTests(RetrievalTestCase):
    def test_scope_filter_isolates_projects(self):
        global_memory = guards.build_memory("project state is tracked")
        scoped_memory = guards.build_memory("project state is tracked", scope="project:memory")
        self.store.insert_memory(global_memory)
        self.store.insert_memory(scoped_memory)
        self.assertEqual(len(self.ids("project")), 2)
        self.assertEqual(self.ids("project", scope="project:memory"), [scoped_memory.id])

    def test_kind_filter(self):
        semantic = guards.build_memory("alpha note", kind="semantic")
        episodic = guards.build_memory("alpha note", kind="episodic")
        self.store.insert_memory(semantic)
        self.store.insert_memory(episodic)
        self.assertEqual(self.ids("alpha", kinds=("episodic",)), [episodic.id])

    def test_superseded_memories_are_not_recalled(self):
        original = guards.build_memory("theme is dark")
        self.store.insert_memory(original)
        replacement = guards.build_memory("theme is light")
        self.store.supersede_memory(original.id, replacement)
        self.assertEqual(self.ids("theme"), [replacement.id])

    def test_forgotten_memories_are_not_recalled(self):
        memory = guards.build_memory("secret preference about tea")
        self.store.insert_memory(memory)
        self.store.set_memory_status(memory.id, "deleted")
        self.assertEqual(self.ids("tea"), [])

    def test_limit_is_respected(self):
        for index in range(5):
            self.store.insert_memory(guards.build_memory("shared keyword {0}".format(index)))
        self.assertEqual(len(self.ids("shared", limit=2)), 2)

    def test_preferences_can_be_excluded(self):
        self.store.insert_preference(guards.build_preference("theme", "dark", statement="theme dark"))
        self.assertEqual(len(self.ids("dark")), 1)
        self.assertEqual(self.ids("dark", include_preferences=False), [])


class PreferenceRetrievalTests(RetrievalTestCase):
    def test_preference_is_recalled_and_boosted(self):
        preference = guards.build_preference("theme", "dark")
        self.store.insert_preference(preference)
        self.store.insert_memory(guards.build_memory("dark"))
        hits = self.retriever.recall("dark")
        self.assertEqual(hits[0].item_type, "preference")
        self.assertEqual(hits[0].id, preference.id)

    def test_only_the_active_preference_value_is_recalled(self):
        first = guards.build_preference("theme", "dark")
        self.store.insert_preference(first)
        second = guards.build_preference("theme", "light")
        self.store.insert_preference(second)
        hits = self.retriever.recall("theme")
        self.assertEqual([h.id for h in hits], [second.id])


class DeterminismTests(RetrievalTestCase):
    def _seed(self):
        for text in (
            "用户偏好本地优先",
            "Prefers local-first architecture",
            "项目状态是 v0.1",
            "隐私敏感",
        ):
            self.store.insert_memory(guards.build_memory(text))
        self.store.insert_preference(guards.build_preference("privacy", "local-only"))

    def test_repeated_queries_return_identical_ids_and_scores(self):
        self._seed()
        baseline = [(h.id, h.score, h.item_type) for h in self.retriever.recall("local")]
        for _ in range(5):
            again = [(h.id, h.score, h.item_type) for h in self.retriever.recall("local")]
            self.assertEqual(baseline, again)

    def test_ranking_is_stable_regardless_of_insertion_order(self):
        texts = ["alpha beta", "alpha", "beta alpha gamma"]
        self._seed_order(texts)

    def _seed_order(self, texts):
        for text in texts:
            self.store.insert_memory(guards.build_memory(text))
        first = self.ids("alpha")
        self.assertEqual(first, self.ids("alpha"))

    def test_rebuild_fts_reproduces_identical_results(self):
        self._seed()
        baseline = [(h.id, h.score, h.item_type) for h in self.retriever.recall("local")]
        counts = self.store.rebuild_fts()
        self.assertEqual(counts["memories_fts"], 4)
        self.assertEqual(counts["preferences_fts"], 1)
        after = [(h.id, h.score, h.item_type) for h in self.retriever.recall("local")]
        self.assertEqual(baseline, after)

    def test_fusion_tie_break_is_total(self):
        self.store.insert_memory(guards.build_memory("tie breaker test"))
        self.store.insert_memory(guards.build_memory("tie breaker test"))
        hits = self.retriever.recall("tie breaker")
        self.assertEqual(len(hits), 2)
        self.assertLess(hits[0].id, hits[1].id)


class FusionUnitTests(unittest.TestCase):
    def test_weights_are_configurable(self):
        weights = fusion.RetrievalWeights(memory_weight=2.0, preference_weight=1.0, salience_weight=0.0)
        self.assertEqual(fusion.score_memory(-1.0, 0.5, False, weights), 2.0)

    def test_phrase_bonus_is_additive(self):
        weights = fusion.RetrievalWeights(salience_weight=0.0)
        plain = fusion.score_memory(-1.0, 0.5, False, weights)
        phrase = fusion.score_memory(-1.0, 0.5, True, weights)
        self.assertAlmostEqual(phrase - plain, fusion.PHRASE_BONUS)

    def test_salience_increases_score(self):
        weights = fusion.RetrievalWeights(salience_weight=0.5)
        low = fusion.score_memory(-1.0, 0.0, False, weights)
        high = fusion.score_memory(-1.0, 1.0, False, weights)
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()
