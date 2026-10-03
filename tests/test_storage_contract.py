"""Storage contract tests.

These run with no network, no model, and no provider configured - that is the
point. If Storage needs a model to work, the architecture is broken.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import unittest

from memory_core.domain import guards
from memory_core.domain.errors import NotFoundError, StorageError, ValidationError
from memory_core.domain.records import Memory
from memory_core.storage.sqlite_store import SCHEMA_VERSION, SqliteStore


class StoreTestCase(unittest.TestCase):
    """Base class giving each test a private on-disk database."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-test-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")
        self.store = SqliteStore(self.db_path)
        self.addCleanup(self.store.close)


class SchemaTests(StoreTestCase):
    def test_creates_single_file_database(self):
        self.assertTrue(os.path.exists(self.db_path))
        self.assertEqual(self.store.schema_version, SCHEMA_VERSION)

    def test_no_stray_sidecar_files_after_commit(self):
        # DELETE journal mode is deliberate: one file is the source of truth.
        self.store.insert_event(guards.build_event("message", "hello"))
        leftovers = [
            name
            for name in os.listdir(self.tmpdir)
            if name != "memory.sqlite"
        ]
        self.assertEqual(leftovers, [])

    def test_refuses_a_newer_schema(self):
        self.store.set_meta("schema_version", str(SCHEMA_VERSION + 1))
        with self.assertRaises(StorageError):
            SqliteStore(self.db_path)

    def test_reopening_preserves_data(self):
        self.store.insert_memory(guards.build_memory("Persisted across restart"))
        self.store.close()
        reopened = SqliteStore(self.db_path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.count_memories(), 1)

    def test_meta_round_trip(self):
        self.store.set_meta("config_fingerprint", "abc123")
        self.assertEqual(self.store.get_meta("config_fingerprint"), "abc123")
        self.assertIsNone(self.store.get_meta("missing"))


class EventTests(StoreTestCase):
    def test_insert_and_read_back(self):
        event = guards.build_event("message", "hello", role="user", session_id="s1")
        self.store.insert_event(event)
        loaded = self.store.get_event(event.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.content, "hello")
        self.assertEqual(loaded.session_id, "s1")

    def test_list_is_ordered_and_filterable(self):
        for index in range(3):
            self.store.insert_event(
                guards.build_event("message", "m{0}".format(index), session_id="s1")
            )
        self.store.insert_event(
            guards.build_event("message", "other", session_id="s2", scope="project:x")
        )
        self.assertEqual(self.store.count_events(), 4)
        self.assertEqual(len(self.store.list_events(session_id="s1")), 3)
        self.assertEqual(len(self.store.list_events(scope="project:x")), 1)

    def test_storage_rejects_invalid_event(self):
        event = guards.build_event("message", "hello")
        tampered = event.__class__(**dict(event.as_dict(), role="robot"))
        with self.assertRaises(ValidationError):
            self.store.insert_event(tampered)


class MemoryTests(StoreTestCase):
    def test_insert_and_get(self):
        memory = guards.build_memory("Prefers local-first", tags=["privacy"])
        self.store.insert_memory(memory)
        loaded = self.store.get_memory(memory.id)
        self.assertEqual(loaded.tags, ("privacy",))
        self.assertEqual(loaded.status, "active")

    def test_missing_memory_raises(self):
        self.assertIsNone(self.store.get_memory("mem_missing"))
        with self.assertRaises(NotFoundError):
            self.store.require_memory("mem_missing")

    def test_storage_rejects_model_invented_kind(self):
        # Red line 4: a hallucinated field must not be persistable.
        good = guards.build_memory("x")
        for field, bad in (("kind", "preference"), ("status", "exploded"), ("confidence", 3.0)):
            tampered = Memory(**dict(good.as_dict(), **{field: bad}))
            with self.assertRaises(ValidationError):
                self.store.insert_memory(tampered)

    def test_supersede_keeps_history(self):
        original = guards.build_memory("Prefers dark theme")
        self.store.insert_memory(original)
        replacement = guards.build_memory("Prefers light theme")
        self.store.supersede_memory(original.id, replacement)

        old = self.store.get_memory(original.id)
        self.assertEqual(old.status, "superseded")
        self.assertEqual(old.superseded_by, replacement.id)
        self.assertEqual(old.revision, 2)

        new = self.store.get_memory(replacement.id)
        self.assertEqual(new.status, "active")
        self.assertEqual(new.source.get("supersedes"), original.id)

        active = self.store.list_memories(statuses=("active",))
        self.assertEqual([m.id for m in active], [replacement.id])
        self.assertEqual(self.store.count_memories("superseded"), 1)

    def test_supersede_missing_raises(self):
        with self.assertRaises(NotFoundError):
            self.store.supersede_memory("mem_missing", guards.build_memory("x"))

    def test_status_change_and_hard_delete(self):
        memory = guards.build_memory("Temporary")
        self.store.insert_memory(memory)
        self.assertEqual(self.store.set_memory_status(memory.id, "archived").status, "archived")
        self.assertTrue(self.store.delete_memory(memory.id))
        self.assertIsNone(self.store.get_memory(memory.id))
        self.assertFalse(self.store.delete_memory(memory.id))

    def test_list_filters_scope_and_kind(self):
        self.store.insert_memory(guards.build_memory("a", kind="semantic"))
        self.store.insert_memory(guards.build_memory("b", kind="episodic"))
        self.store.insert_memory(guards.build_memory("c", scope="project:memory"))
        self.assertEqual(len(self.store.list_memories(kinds=("episodic",))), 1)
        self.assertEqual(len(self.store.list_memories(scope="project:memory")), 1)
        self.assertEqual(len(self.store.list_memories()), 3)

    def test_bulk_insert_writes_everything(self):
        memories = [guards.build_memory("bulk {0}".format(index)) for index in range(25)]
        self.assertEqual(self.store.insert_memories(memories), 25)
        self.assertEqual(self.store.count_memories(), 25)
        for memory in memories:
            self.assertIsNotNone(self.store.get_memory(memory.id))
        self.assertEqual(self.store.insert_memories([]), 0)

    def test_bulk_insert_is_all_or_nothing(self):
        good = guards.build_memory("a valid memory")
        bad = Memory(**dict(guards.build_memory("x").as_dict(), kind="invented"))
        with self.assertRaises(ValidationError):
            self.store.insert_memories([good, bad])
        # Validation happens before any write, so nothing was persisted.
        self.assertEqual(self.store.count_memories(), 0)


class PreferenceTests(StoreTestCase):
    def test_single_active_value_per_key(self):
        self.store.insert_preference(guards.build_preference("theme", "dark"))
        self.store.insert_preference(guards.build_preference("theme", "light"))
        current = self.store.get_preference("theme")
        self.assertEqual(current.value, "light")
        self.assertEqual(self.store.count_preferences(True), 1)
        self.assertEqual(self.store.count_preferences(False), 2)

    def test_history_is_preserved(self):
        self.store.insert_preference(guards.build_preference("theme", "dark"))
        self.store.insert_preference(guards.build_preference("theme", "light"))
        history = self.store.preference_history("theme")
        self.assertEqual([p.value for p in history], ["dark", "light"])
        self.assertEqual(history[0].superseded_by, history[1].id)

    def test_keys_are_independent(self):
        self.store.insert_preference(guards.build_preference("theme", "dark"))
        self.store.insert_preference(guards.build_preference("editor", "vscode"))
        self.assertEqual(self.store.count_preferences(True), 2)

    def test_same_key_in_two_scopes_is_independent(self):
        self.store.insert_preference(guards.build_preference("theme", "dark"))
        self.store.insert_preference(
            guards.build_preference("theme", "light", scope="project:memory")
        )
        self.assertEqual(self.store.get_preference("theme").value, "dark")
        self.assertEqual(
            self.store.get_preference("theme", scope="project:memory").value, "light"
        )

    def test_structured_values_round_trip(self):
        payload = {"level": 3, "flags": ["a", "b"], "note": None}
        self.store.insert_preference(guards.build_preference("settings", payload))
        self.assertEqual(self.store.get_preference("settings").value, payload)

    def test_list_is_ordered_deterministically(self):
        for key in ("zebra", "alpha", "middle"):
            self.store.insert_preference(guards.build_preference(key, 1))
        self.assertEqual(
            [p.key for p in self.store.list_preferences()],
            ["alpha", "middle", "zebra"],
        )


class ProjectStateTests(StoreTestCase):
    def test_current_value_is_single(self):
        self.store.set_project_state(
            guards.build_project_state("memory", {"phase": "design"})
        )
        self.store.set_project_state(
            guards.build_project_state("memory", {"phase": "v0.1"})
        )
        current = self.store.get_project_state("memory")
        self.assertEqual(current.state, {"phase": "v0.1"})
        self.assertEqual(self.store.count_project_states(True), 1)
        self.assertEqual(self.store.count_project_states(False), 2)

    def test_history_is_ordered(self):
        for phase in ("design", "v0.1", "v0.2"):
            self.store.set_project_state(
                guards.build_project_state("memory", {"phase": phase})
            )
        history = self.store.project_state_history("memory")
        self.assertEqual([h.state["phase"] for h in history], ["design", "v0.1", "v0.2"])

    def test_projects_are_isolated(self):
        self.store.set_project_state(guards.build_project_state("a", {"n": 1}))
        self.store.set_project_state(guards.build_project_state("b", {"n": 2}))
        self.assertEqual(self.store.get_project_state("a").state, {"n": 1})
        self.assertEqual(self.store.get_project_state("b").state, {"n": 2})
        self.assertIsNone(self.store.get_project_state("c"))
        self.assertEqual(len(self.store.list_project_states()), 2)

    def test_scope_defaults_to_project_namespace(self):
        record = guards.build_project_state("memory", {"phase": "v0.1"})
        self.store.set_project_state(record)
        self.assertEqual(self.store.get_project_state("memory").scope, "project:memory")


class ExportSnapshotStatsTests(StoreTestCase):
    def _seed(self):
        self.store.insert_event(guards.build_event("message", "hello"))
        self.store.insert_memory(guards.build_memory("A memory", tags=["t1"]))
        self.store.insert_memory(guards.build_memory("Another", kind="episodic"))
        self.store.insert_preference(guards.build_preference("theme", "dark"))
        self.store.set_project_state(guards.build_project_state("memory", {"phase": "v0.1"}))

    def test_export_is_deterministic(self):
        self._seed()
        self.assertEqual(self.store.export_jsonl(), self.store.export_jsonl())

    def test_export_contains_every_source_table(self):
        self._seed()
        text = self.store.export_jsonl()
        for table in ("meta", "events", "memories", "preferences", "project_state"):
            self.assertIn('"table": "{0}"'.format(table), text)

    def test_export_contains_no_derived_tables(self):
        self._seed()
        self.assertNotIn("memories_fts", self.store.export_jsonl())

    def test_snapshot_is_a_usable_database(self):
        self._seed()
        target = os.path.join(self.tmpdir, "backup.sqlite")
        self.store.snapshot(target)
        self.assertTrue(os.path.exists(target))
        restored = SqliteStore(target)
        self.addCleanup(restored.close)
        self.assertEqual(restored.count_memories(), 2)
        self.assertEqual(restored.count_events(), 1)
        self.assertEqual(restored.get_preference("theme").value, "dark")

    def test_stats(self):
        self._seed()
        stats = self.store.stats()
        self.assertEqual(stats["events"], 1)
        self.assertEqual(stats["memories"]["active"], 2)
        self.assertEqual(stats["preferences"]["active"], 1)
        self.assertEqual(stats["project_state"]["current"], 1)
        self.assertEqual(stats["schema_version"], SCHEMA_VERSION)
        self.assertGreater(stats["file_bytes"], 0)

    def test_iter_rows_refuses_derived_or_unknown_tables(self):
        with self.assertRaises(StorageError):
            list(self.store.iter_rows("memories_fts"))
        with self.assertRaises(StorageError):
            list(self.store.iter_rows("sqlite_master"))


if __name__ == "__main__":
    unittest.main()
