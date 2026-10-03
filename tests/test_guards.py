"""Record-level guards: the gate between untrusted input and Storage."""

from __future__ import annotations

import unittest

from memory_core.domain import guards
from memory_core.domain.errors import ValidationError
from memory_core.domain.records import Memory


class EventGuardTests(unittest.TestCase):
    def test_builds_valid_event(self):
        event = guards.build_event("message", "hello", role="user", session_id="s1")
        self.assertTrue(event.id.startswith("evt_"))
        self.assertTrue(event.created_at.endswith("Z"))
        self.assertEqual(event.scope, "global")

    def test_rejects_unknown_kind_and_role(self):
        with self.assertRaises(ValidationError):
            guards.build_event("chat", "hello")
        with self.assertRaises(ValidationError):
            guards.build_event("message", "hello", role="robot")

    def test_rejects_empty_content(self):
        with self.assertRaises(ValidationError):
            guards.build_event("message", "   ")

    def test_validate_rejects_tampered_record(self):
        event = guards.build_event("message", "hello")
        tampered = event.__class__(**dict(event.as_dict(), kind="chat"))
        with self.assertRaises(ValidationError):
            guards.validate_event(tampered)


class MemoryGuardTests(unittest.TestCase):
    def test_builds_valid_memory_with_defaults(self):
        memory = guards.build_memory("Prefers local-first")
        self.assertTrue(memory.id.startswith("mem_"))
        self.assertEqual(memory.kind, "semantic")
        self.assertEqual(memory.status, "active")
        self.assertEqual(memory.confidence, 1.0)
        self.assertEqual(memory.revision, 1)
        self.assertEqual(memory.tags, ())

    def test_accepts_every_kind(self):
        for kind in ("semantic", "episodic", "identity"):
            self.assertEqual(guards.build_memory("x", kind=kind).kind, kind)

    def test_rejects_invented_kind(self):
        # This is the exact failure mode a hallucinating extractor would produce.
        for bad in ("preference", "fact", "SEMANTIC", None, 7):
            with self.assertRaises(ValidationError):
                guards.build_memory("x", kind=bad)

    def test_rejects_out_of_range_confidence(self):
        for bad in (-0.1, 1.5, "high", None):
            with self.assertRaises(ValidationError):
                guards.build_memory("x", confidence=bad)

    def test_normalises_tags_and_deduplicates(self):
        memory = guards.build_memory("x", tags=[" a ", "a", "b"])
        self.assertEqual(memory.tags, ("a", "b"))

    def test_rejects_bad_scope(self):
        with self.assertRaises(ValidationError):
            guards.build_memory("x", scope="unknown:thing")

    def test_rejects_non_serialisable_structured(self):
        with self.assertRaises(ValidationError):
            guards.build_memory("x", structured={"bad": object()})

    def test_rejects_bad_temporal_validity(self):
        with self.assertRaises(ValidationError):
            guards.build_memory("x", valid_from="2024-01-01")

    def test_validate_rejects_tampered_memory(self):
        memory = guards.build_memory("x")
        tampered = Memory(**dict(memory.as_dict(), status="exploded"))
        with self.assertRaises(ValidationError):
            guards.validate_memory(tampered)

    def test_validate_rejects_zero_revision(self):
        memory = guards.build_memory("x")
        tampered = Memory(**dict(memory.as_dict(), revision=0))
        with self.assertRaises(ValidationError):
            guards.validate_memory(tampered)


class PreferenceGuardTests(unittest.TestCase):
    def test_builds_valid_preference(self):
        pref = guards.build_preference("theme", "dark")
        self.assertTrue(pref.id.startswith("prf_"))
        self.assertEqual(pref.value, "dark")
        self.assertIn("theme", pref.statement)

    def test_derives_human_readable_statement(self):
        pref = guards.build_preference("editor", {"name": "vscode"})
        self.assertEqual(pref.statement, 'editor = {\'name\': \'vscode\'}')

    def test_keeps_structured_value_types(self):
        payload = {"a": [1, 2], "b": True, "c": None}
        self.assertEqual(guards.build_preference("k", payload).value, payload)

    def test_rejects_empty_key(self):
        with self.assertRaises(ValidationError):
            guards.build_preference("  ", "dark")

    def test_rejects_non_serialisable_value(self):
        with self.assertRaises(ValidationError):
            guards.build_preference("k", object())


class ProjectStateGuardTests(unittest.TestCase):
    def test_defaults_scope_to_project_namespace(self):
        record = guards.build_project_state("memory", {"phase": "v0.1"})
        self.assertEqual(record.scope, "project:memory")
        self.assertEqual(record.state, {"phase": "v0.1"})

    def test_rejects_non_object_state(self):
        with self.assertRaises(ValidationError):
            guards.build_project_state("memory", ["not", "an", "object"])

    def test_rejects_empty_project_id(self):
        with self.assertRaises(ValidationError):
            guards.build_project_state("", {})


if __name__ == "__main__":
    unittest.main()
