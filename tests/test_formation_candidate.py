"""Candidate schema tests: strict validation of LLM output.

This is the security boundary. A hallucinating or malformed model response must
never become a memory, and must never be able to name a target it was not shown.
"""

from __future__ import annotations

import json
import unittest

from memory_core.formation.base import ADD, MEMORY, NOOP, SUPERSEDE
from memory_core.formation.candidate import (
    ALLOWED_ITEM_KEYS,
    ALLOWED_TOP_LEVEL_KEYS,
    DEFAULT_LLM_CONFIDENCE,
    CandidateSchemaError,
    extract_json_object,
    parse_llm_candidates,
)


def payload(memories):
    # type: (list) -> str
    return json.dumps({"memories": memories}, ensure_ascii=False)


def item(**overrides):
    # type: (object) -> dict
    base = {
        "operation": "ADD",
        "content": "用户偏好本地优先",
        "kind": "semantic",
        "confidence": 0.8,
        "reason": "explicit statement",
    }
    base.update(overrides)
    return base


class JsonExtractionTests(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json_object('{"memories": []}'), {"memories": []})

    def test_markdown_fence_is_stripped(self):
        text = '```json\n{"memories": []}\n```'
        self.assertEqual(extract_json_object(text), {"memories": []})

    def test_fence_without_language(self):
        self.assertEqual(extract_json_object('```\n{"memories": []}\n```'), {"memories": []})

    def test_surrounding_prose_is_tolerated(self):
        text = 'Sure! Here is the result:\n{"memories": []}\nLet me know if you need more.'
        self.assertEqual(extract_json_object(text), {"memories": []})

    def test_empty_response_is_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(CandidateSchemaError):
                extract_json_object(bad)

    def test_non_json_text_is_rejected(self):
        with self.assertRaises(CandidateSchemaError):
            extract_json_object("I could not find anything to remember.")

    def test_malformed_json_is_rejected(self):
        with self.assertRaises(CandidateSchemaError):
            extract_json_object('{"memories": [}')

    def test_json_array_is_rejected(self):
        with self.assertRaises(CandidateSchemaError):
            extract_json_object("[1, 2, 3]")


class TopLevelSchemaTests(unittest.TestCase):
    def test_minimal_valid_payload(self):
        candidates = parse_llm_candidates(payload([item()]))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].target, MEMORY)
        self.assertEqual(candidates[0].operation, ADD)
        self.assertEqual(candidates[0].payload["content"], "用户偏好本地优先")

    def test_empty_memories_list_is_valid(self):
        self.assertEqual(parse_llm_candidates(payload([])), [])

    def test_missing_memories_key_is_rejected(self):
        with self.assertRaises(CandidateSchemaError) as caught:
            parse_llm_candidates("{}")
        self.assertIn("memories", str(caught.exception))

    def test_unknown_top_level_key_is_rejected(self):
        with self.assertRaises(CandidateSchemaError) as caught:
            parse_llm_candidates('{"memories": [], "notes": "hi"}')
        self.assertIn("unknown top-level key", str(caught.exception))

    def test_memories_must_be_a_list(self):
        with self.assertRaises(CandidateSchemaError):
            parse_llm_candidates('{"memories": {"content": "x"}}')

    def test_item_must_be_an_object(self):
        with self.assertRaises(CandidateSchemaError):
            parse_llm_candidates(payload(["just a string"]))

    def test_allowed_key_sets_are_closed(self):
        self.assertEqual(ALLOWED_TOP_LEVEL_KEYS, frozenset({"memories"}))
        self.assertIn("operation", ALLOWED_ITEM_KEYS)
        self.assertNotIn("sql", ALLOWED_ITEM_KEYS)
        self.assertNotIn("id", ALLOWED_ITEM_KEYS)

    def test_too_many_candidates_is_rejected(self):
        with self.assertRaises(CandidateSchemaError) as caught:
            parse_llm_candidates(payload([item() for _ in range(6)]), max_candidates=5)
        self.assertIn("limit", str(caught.exception))

    def test_max_candidates_is_configurable(self):
        self.assertEqual(len(parse_llm_candidates(payload([item() for _ in range(3)]), max_candidates=3)), 3)


class OperationTests(unittest.TestCase):
    def test_add(self):
        candidate = parse_llm_candidates(payload([item(operation="ADD")]))[0]
        self.assertEqual(candidate.operation, ADD)

    def test_supersede(self):
        candidate = parse_llm_candidates(
            payload([item(operation="SUPERSEDE", supersedes_id="mem_1")]), context_ids=["mem_1"]
        )[0]
        self.assertEqual(candidate.operation, SUPERSEDE)
        self.assertEqual(candidate.supersedes, "mem_1")

    def test_update_is_normalised_to_supersede(self):
        # The spec names both UPDATE and SUPERSEDE for the same concept.
        candidate = parse_llm_candidates(
            payload([item(operation="UPDATE", supersedes_id="mem_1")]), context_ids=["mem_1"]
        )[0]
        self.assertEqual(candidate.operation, SUPERSEDE)

    def test_operation_is_case_insensitive(self):
        self.assertEqual(parse_llm_candidates(payload([item(operation="add")]))[0].operation, ADD)

    def test_noop(self):
        candidate = parse_llm_candidates(payload([{"operation": "NOOP", "reason": "chit-chat"}]))[0]
        self.assertEqual(candidate.operation, NOOP)
        self.assertNotIn("content", candidate.payload)

    def test_unknown_operation_is_rejected(self):
        for bad in ("DELETE", "CREATE_TABLE", "", None, 7):
            with self.assertRaises(CandidateSchemaError):
                parse_llm_candidates(payload([item(operation=bad)]))

    def test_operation_is_required(self):
        with self.assertRaises(CandidateSchemaError):
            parse_llm_candidates(payload([{"content": "x"}]))


class SupersedeSafetyTests(unittest.TestCase):
    """The model may only ever name a target it was actually shown."""

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(CandidateSchemaError) as caught:
            parse_llm_candidates(
                payload([item(operation="SUPERSEDE", supersedes_id="mem_hallucinated")]),
                context_ids=["mem_real"],
            )
        self.assertIn("was not among the memories shown", str(caught.exception))

    def test_target_is_rejected_when_no_context_was_shown(self):
        with self.assertRaises(CandidateSchemaError):
            parse_llm_candidates(payload([item(operation="SUPERSEDE", supersedes_id="mem_1")]))

    def test_target_is_required_for_supersede(self):
        with self.assertRaises(CandidateSchemaError) as caught:
            parse_llm_candidates(payload([item(operation="SUPERSEDE")]), context_ids=["mem_1"])
        self.assertIn("required for SUPERSEDE", str(caught.exception))

    def test_target_is_forbidden_for_add(self):
        with self.assertRaises(CandidateSchemaError) as caught:
            parse_llm_candidates(
                payload([item(operation="ADD", supersedes_id="mem_1")]), context_ids=["mem_1"]
            )
        self.assertIn("only valid for SUPERSEDE", str(caught.exception))

    def test_target_is_forbidden_for_noop(self):
        with self.assertRaises(CandidateSchemaError):
            parse_llm_candidates(
                payload([{"operation": "NOOP", "supersedes_id": "mem_1"}]), context_ids=["mem_1"]
            )


class ItemFieldTests(unittest.TestCase):
    def test_unknown_item_key_is_rejected(self):
        with self.assertRaises(CandidateSchemaError) as caught:
            parse_llm_candidates(payload([item(table="memories", sql="DROP TABLE memories")]))
        self.assertIn("unknown key", str(caught.exception))

    def test_content_is_required_for_add(self):
        with self.assertRaises(CandidateSchemaError) as caught:
            parse_llm_candidates(payload([{"operation": "ADD"}]))
        self.assertIn("content", str(caught.exception))

    def test_empty_content_is_rejected(self):
        with self.assertRaises(CandidateSchemaError):
            parse_llm_candidates(payload([item(content="   ")]))

    def test_content_must_be_a_string(self):
        with self.assertRaises(CandidateSchemaError):
            parse_llm_candidates(payload([item(content=["not", "a", "string"])]))

    def test_oversized_content_is_rejected(self):
        with self.assertRaises(CandidateSchemaError):
            parse_llm_candidates(payload([item(content="x" * 20001)]))

    def test_unknown_kind_is_rejected(self):
        for bad in ("fact", "preference", "PREFERENCE", 7):
            with self.assertRaises(CandidateSchemaError):
                parse_llm_candidates(payload([item(kind=bad)]))

    def test_kind_defaults_to_semantic(self):
        candidate = parse_llm_candidates(payload([{"operation": "ADD", "content": "a fact"}]))[0]
        self.assertEqual(candidate.payload["kind"], "semantic")

    def test_every_valid_kind_is_accepted(self):
        for kind in ("semantic", "episodic", "identity"):
            self.assertEqual(
                parse_llm_candidates(payload([item(kind=kind)]))[0].payload["kind"], kind
            )

    def test_confidence_out_of_range_is_rejected(self):
        for bad in (-0.1, 1.5, "high", True):
            with self.assertRaises(CandidateSchemaError):
                parse_llm_candidates(payload([item(confidence=bad)]))

    def test_salience_out_of_range_is_rejected(self):
        with self.assertRaises(CandidateSchemaError):
            parse_llm_candidates(payload([item(salience=2)]))

    def test_default_confidence_is_below_one(self):
        # A model's proposal is a hypothesis; only a user statement is a fact.
        candidate = parse_llm_candidates(payload([{"operation": "ADD", "content": "a fact"}]))[0]
        self.assertEqual(candidate.confidence, DEFAULT_LLM_CONFIDENCE)
        self.assertLess(candidate.confidence, 1.0)

    def test_tags_are_validated(self):
        candidate = parse_llm_candidates(payload([item(tags=["a", "b"])]))[0]
        self.assertEqual(candidate.payload["tags"], ("a", "b"))
        with self.assertRaises(CandidateSchemaError):
            parse_llm_candidates(payload([item(tags="not-a-list")]))
        with self.assertRaises(CandidateSchemaError):
            parse_llm_candidates(payload([item(tags=[1, 2])]))

    def test_subject_defaults(self):
        candidate = parse_llm_candidates(payload([item()]))[0]
        self.assertEqual(candidate.payload["subject"], "user:self")

    def test_scope_is_propagated(self):
        candidate = parse_llm_candidates(payload([item()]), scope="project:memory")[0]
        self.assertEqual(candidate.payload["scope"], "project:memory")

    def test_origin_is_marked_as_llm(self):
        candidate = parse_llm_candidates(payload([item()]))[0]
        self.assertEqual(candidate.origin, "llm")
        self.assertEqual(candidate.payload["source"]["origin"], "llm")

    def test_reason_is_preserved(self):
        candidate = parse_llm_candidates(payload([item(reason="stated directly")]))[0]
        self.assertEqual(candidate.reason, "stated directly")


if __name__ == "__main__":
    unittest.main()
