"""Formation rules: deterministic extraction from explicit instructions."""

from __future__ import annotations

import unittest

from memory_core.formation import base
from memory_core.formation.rules import RuleExtractor, parse_explicit


class NoTriggerTests(unittest.TestCase):
    def test_plain_sentence_forms_nothing(self):
        # v0.1 must not guess. Nothing explicit, nothing extracted.
        result = parse_explicit("今天天气不错")
        self.assertTrue(result.empty)
        self.assertIsNone(result.trigger)
        self.assertIn("no memory verb", result.reason)

    def test_empty_and_whitespace(self):
        for bad in ("", "   ", "\n"):
            self.assertTrue(parse_explicit(bad).empty)

    def test_non_string(self):
        self.assertTrue(RuleExtractor().extract(None).empty)

    def test_verb_without_payload(self):
        result = parse_explicit("记住：")
        self.assertTrue(result.empty)
        self.assertIn("nothing to remember", result.reason)


class MemoryExtractionTests(unittest.TestCase):
    def test_chinese_explicit_memory(self):
        result = parse_explicit("记住：用户偏好本地优先")
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.target, base.MEMORY)
        self.assertEqual(candidate.payload["content"], "用户偏好本地优先")
        self.assertEqual(candidate.payload["kind"], "semantic")
        self.assertEqual(candidate.confidence, 1.0)

    def test_english_explicit_memory(self):
        result = parse_explicit("remember: prefers local-first")
        self.assertEqual(result.candidates[0].payload["content"], "prefers local-first")

    def test_cjk_verb_needs_no_separator(self):
        result = parse_explicit("记住我偏好深色主题")
        self.assertEqual(result.candidates[0].payload["content"], "我偏好深色主题")

    def test_sentence_containing_a_type_word_is_still_semantic(self):
        # The type trigger must START the payload, so this must not be
        # misclassified as a preference.
        result = parse_explicit("记住：我偏好深色主题")
        self.assertEqual(result.candidates[0].target, base.MEMORY)
        self.assertEqual(result.candidates[0].payload["kind"], "semantic")

    def test_identity_trigger(self):
        result = parse_explicit("identity: 我是一个本地优先的个人 AI")
        self.assertEqual(result.candidates[0].target, base.MEMORY)
        self.assertEqual(result.candidates[0].payload["kind"], "identity")

    def test_episodic_trigger(self):
        result = parse_explicit("事件：2026-10-02 决定不使用 Chroma")
        self.assertEqual(result.candidates[0].payload["kind"], "episodic")

    def test_ascii_verb_requires_a_separator(self):
        # "notebook" must not be read as the verb "note".
        self.assertTrue(parse_explicit("notebook: something").empty)

    def test_scope_is_propagated(self):
        result = parse_explicit("记住：项目内偏好", scope="project:memory")
        self.assertEqual(result.candidates[0].payload["scope"], "project:memory")

    def test_source_is_recorded_for_traceability(self):
        source = parse_explicit("记住：可追溯").candidates[0].payload["source"]
        self.assertEqual(source["origin"], "rule")
        self.assertEqual(source["text"], "记住：可追溯")


class PreferenceExtractionTests(unittest.TestCase):
    def test_explicit_preference(self):
        result = parse_explicit("偏好：theme = dark")
        candidate = result.candidates[0]
        self.assertEqual(candidate.target, base.PREFERENCE)
        self.assertEqual(candidate.payload["key"], "theme")
        self.assertEqual(candidate.payload["value"], "dark")

    def test_verb_then_type_trigger(self):
        result = parse_explicit("记住偏好：theme = dark")
        self.assertEqual(result.candidates[0].target, base.PREFERENCE)
        self.assertEqual(result.candidates[0].payload["key"], "theme")

    def test_english_preference(self):
        result = parse_explicit("preference: editor = vscode")
        self.assertEqual(result.candidates[0].payload["key"], "editor")

    def test_value_parsing_prefers_strict_json(self):
        cases = {
            "3": 3,
            "true": True,
            "null": None,
            '"dark"': "dark",
            "dark": "dark",
            '[1, 2]': [1, 2],
            '{"a": 1}': {"a": 1},
        }
        for raw, expected in cases.items():
            result = parse_explicit("偏好：k = {0}".format(raw))
            self.assertEqual(
                result.candidates[0].payload["value"], expected, "for {0!r}".format(raw)
            )

    def test_preference_without_assignment_is_skipped_with_a_reason(self):
        result = parse_explicit("偏好：深色主题")
        self.assertTrue(result.empty)
        self.assertIn("key = value", result.reason)

    def test_full_width_equals_is_accepted(self):
        result = parse_explicit("偏好：theme ＝ dark")
        self.assertEqual(result.candidates[0].payload["key"], "theme")

    def test_statement_keeps_the_human_readable_form(self):
        candidate = parse_explicit("偏好：theme = dark").candidates[0]
        self.assertEqual(candidate.payload["statement"], "theme = dark")


class ProjectStateExtractionTests(unittest.TestCase):
    def test_json_object_state(self):
        result = parse_explicit('project state: memory = {"phase": "v0.1"}')
        candidate = result.candidates[0]
        self.assertEqual(candidate.target, base.PROJECT_STATE)
        self.assertEqual(candidate.payload["project_id"], "memory")
        self.assertEqual(candidate.payload["state"], {"phase": "v0.1"})

    def test_chinese_trigger(self):
        result = parse_explicit('项目状态：memory = {"phase": "v0.1"}')
        self.assertEqual(result.candidates[0].payload["project_id"], "memory")

    def test_non_json_state_is_skipped(self):
        result = parse_explicit("项目状态：memory = v0.1")
        self.assertTrue(result.empty)
        self.assertIn("JSON object", result.reason)

    def test_json_array_state_is_skipped(self):
        result = parse_explicit("项目状态：memory = [1, 2]")
        self.assertTrue(result.empty)
        self.assertIn("JSON object", result.reason)

    def test_missing_assignment_is_skipped(self):
        result = parse_explicit("项目状态：memory")
        self.assertTrue(result.empty)
        self.assertIn("project_id", result.reason)


class PurityTests(unittest.TestCase):
    def test_extraction_has_no_side_effects(self):
        # Formation produces candidates and nothing else: no store, no network,
        # no writes. This is what makes replay possible.
        extractor = RuleExtractor()
        result = extractor.extract("记住：pure function")
        self.assertIsInstance(result, base.Extraction)
        self.assertIsInstance(result.candidates[0], base.Candidate)
        self.assertEqual(extractor.name, "rules")

    def test_extraction_is_repeatable(self):
        first = parse_explicit("记住：stable").as_dict()
        second = parse_explicit("记住：stable").as_dict()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
