"""Import: idempotency, provenance integrity, and the two hard guarantees.

The two guarantees this file exists to defend:

* ``--dry-run`` leaves the database **byte-identical**.
* importing **never writes to the vault**.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from memory_bridge.config import BridgeConfig
from memory_bridge.obsidian.importer import ObsidianImporter
from memory_core.api import MemoryCore
from memory_core.providers.fake import FakeProvider

FIXTURE_VAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures",
    "obsidian_vault",
)

NO_MEMORIES = json.dumps({"memories": []})


def digest(path):
    # type: (str) -> str
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def tree_digest(root):
    # type: (str) -> str
    parts = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(dirnames)
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace("\\", "/")
            parts.append("{0}:{1}".format(rel, hashlib.sha256(open(path, "rb").read()).hexdigest()))
    return hashlib.sha256("\n".join(sorted(parts)).encode("utf-8")).hexdigest()


class BridgeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="bridge-import-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.vault = os.path.join(self.tmpdir, "vault")
        shutil.copytree(FIXTURE_VAULT, self.vault)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")
        self.config = BridgeConfig.load(self.vault)

    def open_core(self, provider=None, use_llm=False, persona=None):
        core = MemoryCore.open(
            db_path=self.db_path,
            provider=provider,
            use_llm_formation=use_llm,
            persona=persona,
        )
        self.addCleanup(core.close)
        return core

    def importer(self, core):
        return ObsidianImporter(core, self.config)


class DryRunTests(BridgeTestCase):
    def test_dry_run_plan_matches_what_apply_writes(self):
        """The plan must not over-report: it uses the same route as the commit."""
        provider = FakeProvider(reply=json.dumps({
            "memories": [{
                "operation": "ADD",
                "content": "用户更喜欢先看到一个小而完整的闭环，再考虑扩展能力。",
                "confidence": 0.9,
            }]
        }))
        core = self.open_core(provider=provider, use_llm=True)
        report = self.importer(core).run(apply=False)

        declaration_accepted = sum(
            action.accepted
            for action in report.actions
            if action.route == "declaration"
        )
        # Three declarations, one accepted candidate each - not six.
        self.assertEqual(declaration_accepted, 3)

    def test_dry_run_leaves_the_database_byte_identical(self):
        core = self.open_core()
        core.remember("pre-existing memory")
        core.close()

        before = digest(self.db_path)
        core = self.open_core()
        report = self.importer(core).run(apply=False)
        core.close()

        self.assertTrue(report.dry_run)
        self.assertEqual(digest(self.db_path), before, "dry run modified the database")

    def test_dry_run_writes_no_events_or_memories(self):
        core = self.open_core()
        report = self.importer(core).run(apply=False)
        self.assertEqual(core.store.count_events(), 0)
        self.assertEqual(core.store.count_memories(), 0)
        self.assertEqual(report.counts["events_created"], 0)

    def test_dry_run_never_touches_the_vault(self):
        core = self.open_core()
        before = tree_digest(self.vault)
        self.importer(core).run(apply=False)
        self.assertEqual(tree_digest(self.vault), before)

    def test_dry_run_plans_routes_and_scopes(self):
        core = self.open_core()
        report = self.importer(core).run(apply=False)
        routes = {action.route for action in report.actions}
        self.assertEqual(
            routes, {"knowledge", "declaration", "preference", "project_state"}
        )
        scopes = {action.scope for action in report.actions}
        self.assertIn("global", scopes)
        self.assertIn("project:persona-edgeaiot", scopes)
        self.assertIn("persona:aria", scopes)
        self.assertIn("persona:bruno", scopes)


class ApplyTests(BridgeTestCase):
    def test_apply_creates_events_and_memories(self):
        core = self.open_core()
        report = self.importer(core).run(apply=True)

        self.assertFalse(report.dry_run)
        self.assertGreater(report.counts["events_created"], 0)
        # Three explicit declarations, one preference, one project state.
        self.assertGreaterEqual(report.counts["memories_created"], 3)
        self.assertEqual(report.counts["preferences_set"], 1)
        self.assertEqual(report.counts["project_states_set"], 1)
        self.assertEqual(core.store.count_events(), report.counts["events_created"])

    def test_apply_never_touches_the_vault(self):
        core = self.open_core()
        before = tree_digest(self.vault)
        self.importer(core).run(apply=True)
        self.assertEqual(tree_digest(self.vault), before)

    def test_excluded_files_are_never_ingested(self):
        core = self.open_core()
        self.importer(core).run(apply=True)
        exported = core.export_jsonl()
        self.assertNotIn("Private/scratch.md", exported)
        self.assertNotIn("private scratchpad", exported)
        self.assertNotIn(".obsidian", exported)

    def test_every_bridge_memory_carries_full_provenance(self):
        core = self.open_core()
        self.importer(core).run(apply=True)

        memories = core.list_memories(include_personas=True, limit=1000)
        self.assertTrue(memories)
        for memory in memories:
            source = memory.source
            self.assertEqual(source.get("origin"), "obsidian")
            self.assertTrue(source.get("external_ref"), "missing external_ref")
            self.assertTrue(source.get("document_title"), "missing document_title")
            self.assertTrue(source.get("content_hash"), "missing content_hash")
            self.assertTrue(source.get("event_ids"), "missing event_ids")
            self.assertTrue(source.get("vault_id"), "missing vault_id")

    def test_events_carry_the_provenance_metadata(self):
        core = self.open_core()
        self.importer(core).run(apply=True)

        events = core.list_events(limit=1000)
        self.assertTrue(events)
        for event in events:
            self.assertEqual(event.kind, "document")
            self.assertEqual(event.source, "obsidian-bridge")
            self.assertTrue(event.session_id.startswith("obsidian:"))
            for key in (
                "vault_id", "external_ref", "document_title", "document_hash",
                "content_hash", "heading_path", "route", "ingested_at",
            ):
                self.assertIn(key, event.metadata, "event metadata missing {0}".format(key))

    def test_session_id_is_the_external_ref(self):
        core = self.open_core()
        self.importer(core).run(apply=True)
        for event in core.list_events(limit=1000):
            self.assertEqual(
                event.session_id, event.metadata["external_ref"]
            )

    def test_declarations_are_deterministic_not_model_generated(self):
        core = self.open_core()
        self.importer(core).run(apply=True)
        declarations = [
            memory
            for memory in core.list_memories(limit=1000)
            if memory.content.startswith("我更喜欢") or memory.content.startswith("Aria")
        ]
        self.assertTrue(declarations)
        for memory in declarations:
            self.assertEqual(memory.confidence, 1.0)
            # ``origin`` says where it came from; ``formed_by`` says how it was
            # formed. Both are preserved.
            self.assertEqual(memory.source.get("origin"), "obsidian")
            self.assertEqual(memory.source.get("formed_by"), "rule")

    def test_declaration_route_never_calls_the_model(self):
        """An explicit declaration is authoritative, so the model is not asked.

        Running the LLM on top of a rule that already extracted the sentence
        with confidence 1.0 does not add judgement - measured on the demo vault,
        the model reworded ``我更喜欢...`` into ``用户更喜欢...``, and
        exact-content dedupe cannot catch that.
        """
        provider = FakeProvider(reply=NO_MEMORIES)
        core = self.open_core(provider=provider, use_llm=True)
        self.importer(core).run(apply=True)

        transcript = " ".join(
            message.content for call in provider.calls for message in call
        )
        # Knowledge notes are sent to the model...
        self.assertIn("Local-first software", transcript)
        # ...declarations are not.
        self.assertNotIn("记住：", transcript)

        declarations = [
            memory for memory in core.list_memories(limit=1000)
            if "闭环" in memory.content
        ]
        self.assertEqual(len(declarations), 1, "declaration was written twice")
        self.assertEqual(declarations[0].confidence, 1.0)
        self.assertEqual(declarations[0].source.get("formed_by"), "rule")

    def test_declaration_duplicate_from_the_model_is_not_stored(self):
        """Even if a model does propose the same fact, there is only one memory."""
        provider = FakeProvider(reply=json.dumps({
            "memories": [{
                "operation": "ADD",
                "content": "用户更喜欢先看到一个小而完整的闭环，再考虑扩展能力。",
                "confidence": 0.9,
            }]
        }))
        core = self.open_core(provider=provider, use_llm=True)
        self.importer(core).run(apply=True)

        # The knowledge chunks may add their own memory, but the declaration
        # itself is stored exactly once. Personas must be included explicitly -
        # two of the three declarations live in persona scopes, which are
        # excluded by default.
        rule_memories = [
            memory for memory in core.list_memories(include_personas=True, limit=1000)
            if memory.source.get("formed_by") == "rule"
        ]
        self.assertEqual(len(rule_memories), 3, "expected 3 declarations")

    def test_preference_bypasses_the_llm(self):
        provider = FakeProvider(reply=json.dumps({
            "memories": [{"operation": "ADD", "content": "the model should never see this"}]
        }))
        core = self.open_core(provider=provider, use_llm=True)
        self.importer(core).run(apply=True)

        preference = core.get_preference("editor")
        self.assertIsNotNone(preference)
        self.assertEqual(preference.value, "obsidian")
        self.assertEqual(preference.source.get("origin"), "obsidian")
        # The model was never handed the preference document.
        prompts = " ".join(
            message.content for call in provider.calls for message in call
        )
        self.assertNotIn("编辑器偏好", prompts)

    def test_project_state_is_written_through_the_typed_api(self):
        core = self.open_core()
        self.importer(core).run(apply=True)
        state = core.get_project_state("persona-edgeaiot")
        self.assertIsNotNone(state)
        self.assertEqual(state.state["phase"], "architecture")
        self.assertEqual(state.source.get("origin"), "obsidian")


class IdempotencyTests(BridgeTestCase):
    def test_second_import_creates_nothing(self):
        core = self.open_core()
        self.importer(core).run(apply=True)
        events_after_first = core.store.count_events()
        memories_after_first = core.store.count_memories()

        second = self.importer(core).run(apply=True)
        self.assertEqual(second.counts["documents_unchanged"], 6)
        self.assertEqual(second.counts["events_created"], 0)
        self.assertEqual(second.counts["memories_created"], 0)
        self.assertEqual(core.store.count_events(), events_after_first)
        self.assertEqual(core.store.count_memories(), memories_after_first)

    def test_editing_one_section_reingests_only_that_chunk(self):
        core = self.open_core()
        self.importer(core).run(apply=True)
        events_before = core.store.count_events()
        memory_ids_before = {m.id for m in core.list_memories(limit=1000)}

        target = os.path.join(self.vault, "Knowledge", "Local First Architecture.md")
        with open(target, "r", encoding="utf-8") as handle:
            text = handle.read()
        # Rewrite only the first section.
        text = text.replace(
            "Local-first software keeps the authoritative copy",
            "Local-first software KEEPS the authoritative copy",
        )
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

        report = self.importer(core).run(apply=True)
        changed = [d for d in report.documents if d.status == "changed"]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0].rel_path, "Knowledge/Local First Architecture.md")

        actions = changed[0].actions
        self.assertEqual(len(actions), 2, "expected two chunks in the edited document")
        self.assertEqual(
            len([a for a in actions if a.action == "skip-seen"]), 1,
            "the untouched chunk must not be re-ingested",
        )
        self.assertEqual(
            len([a for a in actions if a.action == "ingest"]), 1,
        )
        # One new event for the changed chunk only.
        self.assertEqual(core.store.count_events(), events_before + 1)

    def test_editing_a_declaration_keeps_the_old_memory(self):
        core = self.open_core()
        self.importer(core).run(apply=True)
        before = {m.id: m for m in core.list_memories(limit=1000)}

        target = os.path.join(self.vault, "Inbox", "Remember This.md")
        with open(target, "r", encoding="utf-8") as handle:
            text = handle.read()
        text = text.replace("先看到一个小而完整的闭环", "先把闭环跑通再扩展")
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

        report = self.importer(core).run(apply=True)
        after = {m.id: m for m in core.list_memories(limit=1000)}

        # Nothing was deleted...
        for memory_id, memory in before.items():
            self.assertIn(memory_id, after, "memory {0} disappeared".format(memory_id))
            self.assertEqual(after[memory_id].status, "active")
        # ...a new event was written...
        self.assertGreater(report.counts["events_created"], 0)
        # ...and no memory was duplicated by content.
        contents = [memory.content for memory in after.values()]
        self.assertEqual(len(contents), len(set(contents)))

    def test_unchanged_documents_are_skipped_entirely(self):
        core = self.open_core()
        self.importer(core).run(apply=True)

        target = os.path.join(self.vault, "Inbox", "Remember This.md")
        with open(target, "a", encoding="utf-8") as handle:
            handle.write("\n\n一个全新的段落，长到足以被保留下来作为知识内容。\n")

        report = self.importer(core).run(apply=True)
        unchanged = {d.rel_path for d in report.documents if d.status == "unchanged"}
        self.assertIn("Knowledge/Local First Architecture.md", unchanged)
        self.assertIn("Preferences/Editor.md", unchanged)
        self.assertNotIn("Inbox/Remember This.md", unchanged)


class FailureTests(BridgeTestCase):
    def test_model_failure_still_records_the_event(self):
        provider = FakeProvider(fail_with="model is down")
        core = self.open_core(provider=provider, use_llm=True)
        report = self.importer(core).run(apply=True)

        # Knowledge chunks got an event but no memory.
        self.assertGreater(report.counts["events_created"], 0)
        events = core.list_events(limit=1000)
        self.assertTrue(events)
        # The deterministic declaration path is unaffected by the model outage.
        self.assertGreaterEqual(report.counts["memories_created"], 3)

    def test_bad_model_output_does_not_pollute_the_database(self):
        provider = FakeProvider(reply="not json at all")
        core = self.open_core(provider=provider, use_llm=True)
        self.importer(core).run(apply=True)

        for memory in core.list_memories(include_personas=True, limit=1000):
            self.assertNotIn("not json at all", memory.content)


if __name__ == "__main__":
    unittest.main()
