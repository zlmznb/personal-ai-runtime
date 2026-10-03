"""LlmExtractor tests: candidates in, nothing written, failures contained."""

from __future__ import annotations

import json
import unittest

from memory_core.formation.base import ADD, MEMORY, NOOP, SUPERSEDE
from memory_core.formation.llm_extractor import SYSTEM_PROMPT, LlmExtractor
from memory_core.providers.fake import FakeProvider


class StubMemory(object):
    def __init__(self, memory_id, content, kind="semantic"):
        self.id = memory_id
        self.content = content
        self.kind = kind


def response(memories):
    return json.dumps({"memories": memories}, ensure_ascii=False)


class ExtractorBasicsTests(unittest.TestCase):
    def test_requires_a_provider(self):
        with self.assertRaises(Exception):
            LlmExtractor(None)

    def test_name_and_description(self):
        extractor = LlmExtractor(FakeProvider(reply="{}"))
        self.assertEqual(extractor.name, "llm")
        described = extractor.describe()
        self.assertEqual(described["name"], "llm")
        self.assertEqual(described["provider"], "fake")

    def test_has_no_storage_handle(self):
        # The model side simply has no way to reach the database: its only
        # collaborator is a ChatProvider.
        extractor = LlmExtractor(FakeProvider(reply="{}"))
        self.assertFalse(hasattr(extractor, "store"))
        self.assertFalse(hasattr(extractor.provider, "store"))
        self.assertFalse(hasattr(extractor.provider, "execute"))

    def test_empty_utterance_is_not_sent_to_the_model(self):
        provider = FakeProvider(reply="{}")
        extraction = LlmExtractor(provider).extract("   ")
        self.assertTrue(extraction.empty)
        self.assertEqual(provider.call_count, 0)


class ValidResponseTests(unittest.TestCase):
    def test_add_candidate(self):
        provider = FakeProvider(reply=response([
            {"operation": "ADD", "content": "用户偏好本地优先", "kind": "semantic", "confidence": 0.8}
        ]))
        extraction = LlmExtractor(provider).extract("我偏好本地优先")
        self.assertEqual(len(extraction.candidates), 1)
        candidate = extraction.candidates[0]
        self.assertEqual(candidate.target, MEMORY)
        self.assertEqual(candidate.operation, ADD)
        self.assertEqual(candidate.origin, "llm")
        self.assertEqual(candidate.payload["content"], "用户偏好本地优先")

    def test_provenance_is_stamped_with_the_model(self):
        provider = FakeProvider(model="fake-A", reply=response([
            {"operation": "ADD", "content": "a durable fact"}
        ]))
        candidate = LlmExtractor(provider).extract("something").candidates[0]
        self.assertEqual(candidate.payload["source"]["model"], "fake:fake-A")
        self.assertEqual(candidate.payload["source"]["origin"], "llm")

    def test_supersede_candidate_with_a_visible_target(self):
        memory = StubMemory("mem_1", "old fact")
        provider = FakeProvider(reply=response([
            {"operation": "SUPERSEDE", "content": "new fact", "supersedes_id": "mem_1"}
        ]))
        candidate = LlmExtractor(provider).extract("changed my mind", context=[memory]).candidates[0]
        self.assertEqual(candidate.operation, SUPERSEDE)
        self.assertEqual(candidate.supersedes, "mem_1")

    def test_noop_candidate(self):
        provider = FakeProvider(reply=response([{"operation": "NOOP", "reason": "small talk"}]))
        candidate = LlmExtractor(provider).extract("nice weather").candidates[0]
        self.assertEqual(candidate.operation, NOOP)

    def test_empty_list_yields_no_candidates(self):
        extraction = LlmExtractor(FakeProvider(reply=response([]))).extract("nothing here")
        self.assertTrue(extraction.empty)
        self.assertIsNone(extraction.reason)

    def test_markdown_fenced_response_is_accepted(self):
        provider = FakeProvider(reply="```json\n" + response([{"operation": "ADD", "content": "a fact"}]) + "\n```")
        self.assertEqual(len(LlmExtractor(provider).extract("x").candidates), 1)


class PromptTests(unittest.TestCase):
    def test_context_is_rendered_with_ids(self):
        memories = [StubMemory("mem_1", "first fact"), StubMemory("mem_2", "second fact")]
        messages = LlmExtractor(FakeProvider()).build_messages("a turn", memories)
        self.assertEqual(len(messages), 2)
        self.assertIn("mem_1", messages[0].content)
        self.assertIn("first fact", messages[0].content)
        self.assertIn("mem_2", messages[0].content)
        self.assertEqual(messages[1].content, "a turn")

    def test_no_context_is_stated_explicitly(self):
        messages = LlmExtractor(FakeProvider()).build_messages("a turn", [])
        self.assertIn("(none)", messages[0].content)

    def test_context_is_bounded(self):
        memories = [StubMemory("mem_{0}".format(i), "fact {0}".format(i)) for i in range(20)]
        extractor = LlmExtractor(FakeProvider(), max_context=3)
        messages = extractor.build_messages("turn", memories)
        self.assertIn("mem_2", messages[0].content)
        self.assertNotIn("mem_3", messages[0].content)

    def test_system_prompt_forbids_extra_keys(self):
        self.assertIn("Do not add extra keys", SYSTEM_PROMPT)

    def test_system_prompt_is_injected(self):
        extractor = LlmExtractor(FakeProvider(), system_prompt="CUSTOM PROMPT")
        self.assertIn("CUSTOM PROMPT", extractor.build_messages("x", [])[0].content)


class FailureContainmentTests(unittest.TestCase):
    """Every failure mode must yield zero candidates, never an exception."""

    def test_provider_error_yields_no_candidates(self):
        provider = FakeProvider(fail_with="connection refused")
        extraction = LlmExtractor(provider).extract("anything")
        self.assertTrue(extraction.empty)
        self.assertIn("provider unavailable", extraction.reason)
        self.assertIn("connection refused", extraction.reason)

    def test_malformed_json_yields_no_candidates(self):
        extraction = LlmExtractor(FakeProvider(reply="I'm sorry, I can't help with that.")).extract("x")
        self.assertTrue(extraction.empty)
        self.assertIn("schema validation", extraction.reason)

    def test_schema_violation_yields_no_candidates(self):
        provider = FakeProvider(reply=json.dumps({"memories": [{"operation": "ADD"}]}))
        extraction = LlmExtractor(provider).extract("x")
        self.assertTrue(extraction.empty)
        self.assertIn("schema validation", extraction.reason)

    def test_hallucinated_supersede_target_yields_no_candidates(self):
        provider = FakeProvider(reply=response([
            {"operation": "SUPERSEDE", "content": "new", "supersedes_id": "mem_invented"}
        ]))
        extraction = LlmExtractor(provider).extract("x", context=[StubMemory("mem_real", "old")])
        self.assertTrue(extraction.empty)
        self.assertIn("schema validation", extraction.reason)

    def test_sql_injection_attempt_is_rejected(self):
        provider = FakeProvider(reply=json.dumps({
            "memories": [{
                "operation": "ADD",
                "content": "a fact",
                "sql": "DROP TABLE memories;",
                "table": "memories",
            }]
        }))
        extraction = LlmExtractor(provider).extract("x")
        self.assertTrue(extraction.empty)
        self.assertIn("unknown key", extraction.reason)

    def test_model_cannot_choose_its_own_id_or_status(self):
        for extra in ({"id": "mem_chosen"}, {"status": "superseded"}, {"created_at": "1999-01-01"}):
            item = {"operation": "ADD", "content": "a fact"}
            item.update(extra)
            extraction = LlmExtractor(FakeProvider(reply=json.dumps({"memories": [item]}))).extract("x")
            self.assertTrue(extraction.empty, "should reject {0}".format(extra))

    def test_unexpected_provider_exception_is_contained(self):
        class ExplodingProvider(FakeProvider):
            def chat(self, messages, **options):
                raise RuntimeError("boom")

        extraction = LlmExtractor(ExplodingProvider()).extract("x")
        self.assertTrue(extraction.empty)
        self.assertIn("unexpected provider failure", extraction.reason)


if __name__ == "__main__":
    unittest.main()
