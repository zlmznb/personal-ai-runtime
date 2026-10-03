"""Domain layer: validation gates, identifiers, record decoding."""

from __future__ import annotations

import time
import unittest

from memory_core.domain import ids, validation
from memory_core.domain.enums import MemoryKind, MemoryStatus
from memory_core.domain.errors import ValidationError
from memory_core.domain.records import Event, Memory, Preference, ProjectState


class IdentifierTests(unittest.TestCase):
    def test_ids_are_time_sortable_across_milliseconds(self):
        first = ids.new_id(ids.MEMORY_PREFIX)
        time.sleep(0.005)
        second = ids.new_id(ids.MEMORY_PREFIX)
        self.assertTrue(first.startswith("mem_"))
        self.assertLess(first, second)

    def test_id_timestamp_prefix_is_non_decreasing(self):
        # Documents the known limit: id order is not insertion order within a
        # single clock tick, but the timestamp prefix never goes backwards.
        # Determinism comes from stored data, not from id order.
        prefixes = [ids.new_id(ids.MEMORY_PREFIX)[4:14] for _ in range(200)]
        self.assertEqual(prefixes, sorted(prefixes))

    def test_all_prefixes_are_applied(self):
        self.assertTrue(ids.new_id(ids.EVENT_PREFIX).startswith("evt_"))
        self.assertTrue(ids.new_id(ids.PREFERENCE_PREFIX).startswith("prf_"))
        self.assertTrue(ids.new_id(ids.PROJECT_PREFIX).startswith("prj_"))

    def test_utc_now_iso_is_sortable_and_zulu(self):
        stamp = ids.utc_now_iso()
        self.assertTrue(stamp.endswith("Z"))
        self.assertTrue(ids.is_iso8601(stamp))

    def test_ids_are_unique(self):
        seen = {ids.new_id(ids.EVENT_PREFIX) for _ in range(500)}
        self.assertEqual(len(seen), 500)


class TextValidationTests(unittest.TestCase):
    def test_strips_and_accepts(self):
        self.assertEqual(validation.require_text("  hi  ", "content"), "hi")

    def test_rejects_empty_and_whitespace(self):
        for bad in ("", "   ", "\n\t"):
            with self.assertRaises(ValidationError):
                validation.require_text(bad, "content")

    def test_rejects_non_string(self):
        for bad in (None, 1, [], {}):
            with self.assertRaises(ValidationError):
                validation.require_text(bad, "content")

    def test_rejects_control_characters(self):
        with self.assertRaises(ValidationError):
            validation.require_text("bad\x00text", "content")

    def test_rejects_overlong_text(self):
        with self.assertRaises(ValidationError):
            validation.require_text("x" * 11, "content", max_chars=10)


class EnumValidationTests(unittest.TestCase):
    def test_accepts_member_and_value(self):
        self.assertEqual(validation.require_enum(MemoryKind, MemoryKind.SEMANTIC, "kind"), "semantic")
        self.assertEqual(validation.require_enum(MemoryKind, "episodic", "kind"), "episodic")

    def test_rejects_unknown_values(self):
        # A model inventing a memory kind must not reach storage.
        for bad in ("preference", "SEMANTIC", "", None, 3):
            with self.assertRaises(ValidationError):
                validation.require_enum(MemoryKind, bad, "kind")

    def test_rejects_unknown_status(self):
        with self.assertRaises(ValidationError):
            validation.require_enum(MemoryStatus, "gone", "status")


class ScopeValidationTests(unittest.TestCase):
    def test_accepts_global_and_prefixes(self):
        for good in ("global", "project:memory", "session:abc-123", "device:phone_1"):
            self.assertEqual(validation.require_scope(good), good)

    def test_rejects_malformed_scope(self):
        for bad in ("", "project", ":memory", "unknown:thing", "project:", None, 5):
            with self.assertRaises(ValidationError):
                validation.require_scope(bad)


class NumericValidationTests(unittest.TestCase):
    def test_accepts_unit_interval(self):
        self.assertEqual(validation.require_unit_float(0, "confidence"), 0.0)
        self.assertEqual(validation.require_unit_float(1, "confidence"), 1.0)
        self.assertEqual(validation.require_unit_float(0.5, "confidence"), 0.5)

    def test_rejects_out_of_range_and_bad_types(self):
        for bad in (-0.1, 1.1, "0.5", None, True):
            with self.assertRaises(ValidationError):
                validation.require_unit_float(bad, "confidence")

    def test_default_applies_only_when_none(self):
        self.assertEqual(validation.require_unit_float(None, "salience", default=0.5), 0.5)
        with self.assertRaises(ValidationError):
            validation.require_unit_float(None, "salience")


class JsonValidationTests(unittest.TestCase):
    def test_accepts_serialisable(self):
        payload = {"a": [1, 2, {"b": None}], "c": "text"}
        self.assertEqual(validation.require_json_value(payload, "structured"), payload)

    def test_rejects_non_serialisable(self):
        with self.assertRaises(ValidationError):
            validation.require_json_value({"x": object()}, "structured")

    def test_rejects_nan_and_infinity(self):
        # Valid to json.dumps by default, but not valid JSON - would corrupt export.
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValidationError):
                validation.require_json_value({"x": bad}, "structured")

    def test_require_json_object(self):
        self.assertEqual(validation.require_json_object(None, "structured"), {})
        with self.assertRaises(ValidationError):
            validation.require_json_object([1, 2], "structured")


class TagValidationTests(unittest.TestCase):
    def test_normalises_and_deduplicates(self):
        self.assertEqual(validation.normalize_tags([" a ", "b", "a"]), ("a", "b"))
        self.assertEqual(validation.normalize_tags(None), ())
        self.assertEqual(validation.normalize_tags([]), ())

    def test_rejects_bare_string(self):
        with self.assertRaises(ValidationError):
            validation.normalize_tags("privacy")


class RecordDecodingTests(unittest.TestCase):
    def test_memory_from_row_decodes_json_columns(self):
        row = {
            "id": "mem_1",
            "kind": "semantic",
            "scope": "global",
            "subject": "user:self",
            "content": "Prefers local-first",
            "structured": '{"k": "v"}',
            "tags": '["privacy", "local"]',
            "source": '{"event_ids": ["evt_1"]}',
            "confidence": 1.0,
            "salience": 0.7,
            "status": "active",
            "valid_from": None,
            "valid_to": None,
            "superseded_by": None,
            "generated_by": None,
            "created_at": "2026-10-02T00:00:00.000Z",
            "updated_at": "2026-10-02T00:00:00.000Z",
            "revision": 1,
        }
        memory = Memory.from_row(row)
        self.assertEqual(memory.structured, {"k": "v"})
        self.assertEqual(memory.tags, ("privacy", "local"))
        self.assertEqual(memory.source["event_ids"], ["evt_1"])
        self.assertEqual(memory.as_dict()["content"], "Prefers local-first")

    def test_preference_from_row_preserves_value_type(self):
        row = {
            "id": "prf_1",
            "scope": "global",
            "key": "theme",
            "value": '"dark"',
            "statement": "theme = dark",
            "confidence": 1.0,
            "source": "{}",
            "created_at": "2026-10-02T00:00:00.000Z",
            "superseded_by": None,
        }
        self.assertEqual(Preference.from_row(row).value, "dark")

    def test_project_state_from_row(self):
        row = {
            "id": "prj_1",
            "project_id": "memory",
            "scope": "project:memory",
            "state": '{"phase": "v0.1"}',
            "note": None,
            "source": "{}",
            "created_at": "2026-10-02T00:00:00.000Z",
            "superseded_by": None,
        }
        self.assertEqual(ProjectState.from_row(row).state, {"phase": "v0.1"})

    def test_event_from_row(self):
        row = {
            "id": "evt_1",
            "kind": "message",
            "role": "user",
            "content": "hello",
            "session_id": "s1",
            "scope": "global",
            "metadata": "{}",
            "created_at": "2026-10-02T00:00:00.000Z",
            "source": "cli",
        }
        event = Event.from_row(row)
        self.assertEqual(event.as_dict()["session_id"], "s1")
        self.assertEqual(event.metadata, {})


if __name__ == "__main__":
    unittest.main()
