"""Acceptance policy tests: the deterministic gate before a write."""

from __future__ import annotations

import unittest

from memory_core.formation.base import ADD, MEMORY, NOOP, PREFERENCE, PROJECT_STATE, SUPERSEDE, Candidate
from memory_core.formation.policy import (
    ACCEPT,
    AcceptancePolicy,
    Decision,
    normalize_text,
)


def memory_candidate(content="用户偏好本地优先", confidence=0.9, operation=ADD, supersedes=None):
    # type: (str, float, str, object) -> Candidate
    return Candidate(
        target=MEMORY,
        payload={"content": content, "kind": "semantic"},
        confidence=confidence,
        operation=operation,
        supersedes=supersedes,
        origin="llm",
    )


class NormalizeTests(unittest.TestCase):
    def test_case_and_whitespace_folding(self):
        self.assertEqual(normalize_text("  Hello   World "), "hello world")

    def test_non_string_is_empty(self):
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text(42), "")


class NoopTests(unittest.TestCase):
    def test_noop_is_rejected_by_default(self):
        decision = AcceptancePolicy().evaluate(
            Candidate(target=MEMORY, operation=NOOP, reason="chit-chat")
        )
        self.assertTrue(decision.rejected)
        self.assertIn("chit-chat", decision.reason)

    def test_noop_can_be_accepted_explicitly(self):
        self.assertTrue(AcceptancePolicy(accept_noop=True).evaluate(
            Candidate(target=MEMORY, operation=NOOP)
        ).accepted)


class ConfidenceTests(unittest.TestCase):
    def test_low_confidence_is_rejected(self):
        decision = AcceptancePolicy(min_confidence=0.5).evaluate(memory_candidate(confidence=0.4))
        self.assertTrue(decision.rejected)
        self.assertIn("below", decision.reason)

    def test_threshold_is_inclusive(self):
        self.assertTrue(AcceptancePolicy(min_confidence=0.5).evaluate(memory_candidate(confidence=0.5)).accepted)

    def test_high_confidence_is_accepted(self):
        self.assertTrue(AcceptancePolicy().evaluate(memory_candidate(confidence=1.0)).accepted)


class ContentTests(unittest.TestCase):
    def test_short_content_is_rejected(self):
        decision = AcceptancePolicy(min_content_chars=10).evaluate(memory_candidate(content="ok"))
        self.assertTrue(decision.rejected)
        self.assertIn("shorter than", decision.reason)

    def test_content_is_compared_after_normalisation(self):
        policy = AcceptancePolicy(min_content_chars=10)
        self.assertTrue(policy.evaluate(memory_candidate(content="  a long enough statement  ")).accepted)


class DuplicateTests(unittest.TestCase):
    def test_duplicate_is_rejected(self):
        decision = AcceptancePolicy().evaluate(
            memory_candidate(content="用户偏好本地优先"),
            existing_contents=["用户偏好本地优先"],
        )
        self.assertTrue(decision.rejected)
        self.assertIn("duplicate", decision.reason)

    def test_duplicate_detection_ignores_case_and_spacing(self):
        decision = AcceptancePolicy().evaluate(
            memory_candidate(content="Retrieval  Must Never Call A Model"),
            existing_contents=["retrieval must never call a model"],
        )
        self.assertTrue(decision.rejected)

    def test_dedupe_can_be_disabled(self):
        self.assertTrue(AcceptancePolicy(dedupe=False).evaluate(
            memory_candidate(content="用户偏好本地优先"),
            existing_contents=["用户偏好本地优先"],
        ).accepted)

    def test_different_content_is_not_a_duplicate(self):
        self.assertTrue(AcceptancePolicy().evaluate(
            memory_candidate(content="用户偏好云端优先"),
            existing_contents=["用户偏好本地优先"],
        ).accepted)


class SupersedeTests(unittest.TestCase):
    def test_existing_target_is_accepted(self):
        candidate = memory_candidate(operation=SUPERSEDE, supersedes="mem_1")
        self.assertTrue(AcceptancePolicy().evaluate(candidate, known_ids=["mem_1"]).accepted)

    def test_missing_target_is_rejected(self):
        candidate = memory_candidate(operation=SUPERSEDE, supersedes="mem_gone")
        decision = AcceptancePolicy().evaluate(candidate, known_ids=["mem_1"])
        self.assertTrue(decision.rejected)
        self.assertIn("does not exist", decision.reason)

    def test_unknown_target_with_no_known_ids_is_left_to_the_write_gate(self):
        # Second line of defence: the storage layer raises NotFoundError and the
        # API records the rejection. Covered end-to-end in test_learn_lifecycle.
        candidate = memory_candidate(operation=SUPERSEDE, supersedes="mem_gone")
        self.assertTrue(AcceptancePolicy().evaluate(candidate, known_ids=[]).accepted)


class OtherTargetTests(unittest.TestCase):
    def test_preference_requires_a_key(self):
        good = Candidate(target=PREFERENCE, payload={"key": "theme", "value": "dark"}, confidence=0.9)
        bad = Candidate(target=PREFERENCE, payload={"value": "dark"}, confidence=0.9)
        policy = AcceptancePolicy()
        self.assertTrue(policy.evaluate(good).accepted)
        self.assertTrue(policy.evaluate(bad).rejected)

    def test_project_state_requires_a_project_id(self):
        good = Candidate(target=PROJECT_STATE, payload={"project_id": "memory", "state": {}}, confidence=0.9)
        bad = Candidate(target=PROJECT_STATE, payload={"state": {}}, confidence=0.9)
        policy = AcceptancePolicy()
        self.assertTrue(policy.evaluate(good).accepted)
        self.assertTrue(policy.evaluate(bad).rejected)


class PolicyShapeTests(unittest.TestCase):
    def test_decision_helpers(self):
        self.assertTrue(ACCEPT.accepted)
        self.assertFalse(ACCEPT.rejected)
        rejected = Decision(False, "nope")
        self.assertTrue(rejected.rejected)
        self.assertEqual(rejected.as_dict(), {"accepted": False, "reason": "nope"})

    def test_policy_serialises(self):
        described = AcceptancePolicy().as_dict()
        self.assertEqual(described["min_content_chars"], 4)
        self.assertTrue(described["dedupe"])

    def test_policy_is_pure_and_repeatable(self):
        policy = AcceptancePolicy()
        candidate = memory_candidate()
        self.assertEqual(policy.evaluate(candidate), policy.evaluate(candidate))


if __name__ == "__main__":
    unittest.main()
