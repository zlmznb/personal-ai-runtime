"""API facade tests.

These exercise the contract the rest of the world sees, including every
degradation path that must work with no model present.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from memory_core.api import MemoryCore, build_prompt, format_hit
from memory_core.domain.errors import (
    ConfigurationError,
    NotFoundError,
    ValidationError,
)
from memory_core.formation.base import Candidate, Extraction, Extractor
from memory_core.providers.fake import FakeProvider


def file_digest(path):
    # type: (str) -> str
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class CoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-api-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")

    def open_core(self, provider=None, provider_name=None, model=None, **kwargs):
        core = MemoryCore.open(
            db_path=self.db_path, provider=provider, provider_name=provider_name, model=model,
            **kwargs
        )
        self.addCleanup(core.close)
        return core


class WriteGateTests(CoreTestCase):
    def test_remember_then_recall(self):
        core = self.open_core()
        memory = core.remember("用户偏好本地优先", tags=["privacy"])
        hits = core.recall("本地优先")
        self.assertEqual([hit.id for hit in hits], [memory.id])

    def test_remember_rejects_invalid_input(self):
        core = self.open_core()
        for kwargs in (
            {"kind": "preference"},
            {"kind": "invented"},
            {"confidence": 2.0},
            {"scope": "bogus:thing"},
            {"content": "   "},
            {"structured": {"x": object()}},
        ):
            with self.assertRaises(ValidationError):
                core.remember(kwargs.pop("content", "valid content"), **kwargs)

    def test_remember_accepts_every_memory_kind(self):
        core = self.open_core()
        for kind in ("semantic", "episodic", "identity"):
            self.assertEqual(core.remember("note {0}".format(kind), kind=kind).kind, kind)

    def test_remember_records_provenance(self):
        core = self.open_core()
        event = core.record_event("原始对话")
        memory = core.remember("derived", event_ids=[event.id])
        self.assertEqual(memory.source["event_ids"], [event.id])

    def test_supersede_keeps_the_old_row(self):
        core = self.open_core()
        original = core.remember("theme is dark")
        replacement = core.supersede_memory(
            original.id, "theme is light", source={"reason": "changed mind"}
        )
        self.assertEqual(core.get_memory(original.id).status, "superseded")
        self.assertEqual(core.get_memory(original.id).superseded_by, replacement.id)
        self.assertEqual([h.id for h in core.recall("theme")], [replacement.id])


class EventAndObservationTests(CoreTestCase):
    def test_record_event_is_append_only(self):
        core = self.open_core()
        event = core.record_event("hello", role="user", session_id="s1")
        self.assertEqual(core.store.get_event(event.id).content, "hello")

    def test_observe_forms_a_memory_from_an_explicit_instruction(self):
        core = self.open_core()
        observation = core.observe("记住：用户偏好本地优先")
        self.assertEqual(len(observation.memories), 1)
        self.assertEqual(observation.memories[0].content, "用户偏好本地优先")
        self.assertEqual(observation.memories[0].confidence, 1.0)
        self.assertEqual(observation.memories[0].source["event_ids"], [observation.event.id])

    def test_observe_forms_a_preference(self):
        core = self.open_core()
        observation = core.observe("记住偏好：theme = dark")
        self.assertEqual(len(observation.preferences), 1)
        self.assertEqual(core.get_preference("theme").value, "dark")

    def test_observe_forms_project_state(self):
        core = self.open_core()
        observation = core.observe('项目状态：memory = {"phase": "v0.1"}')
        self.assertEqual(len(observation.project_states), 1)
        self.assertEqual(core.get_project_state("memory").state, {"phase": "v0.1"})

    def test_observe_always_writes_the_event_even_when_nothing_is_formed(self):
        core = self.open_core()
        observation = core.observe("今天天气不错")
        self.assertEqual(core.store.count_events(), 1)
        self.assertTrue(observation.skipped_reason)
        self.assertFalse(observation.memories)

    def test_observe_keeps_the_event_when_a_candidate_is_rejected(self):
        # An extractor that emits a malformed candidate must not lose the raw
        # utterance, and must not persist the candidate either.
        class BadExtractor(Extractor):
            name = "bad"

            def extract(self, text, scope="global", context=None):
                return Extraction(
                    candidates=(
                        Candidate(
                            target="memory",
                            payload={"content": "a malformed candidate", "kind": "invented"},
                            confidence=0.9,
                        ),
                    )
                )

        core = self.open_core()
        core.extractor = BadExtractor()
        observation = core.observe("记住：会失败的候选")

        self.assertEqual(core.store.count_events(), 1)
        self.assertEqual(core.store.count_memories(), 0)
        self.assertFalse(observation.memories)
        self.assertEqual(len(observation.rejected), 1)
        self.assertIn("write rejected", observation.rejected[0].reason)
        self.assertEqual(observation.accepted, ())


class PreferenceApiTests(CoreTestCase):
    def test_set_get_and_supersede(self):
        core = self.open_core()
        core.set_preference("theme", "dark")
        core.set_preference("theme", "light")
        self.assertEqual(core.get_preference("theme").value, "light")
        self.assertEqual(len(core.list_preferences()), 1)
        self.assertEqual([p.value for p in core.preference_history("theme")], ["dark", "light"])

    def test_structured_values(self):
        core = self.open_core()
        core.set_preference("layout", {"columns": 3})
        self.assertEqual(core.get_preference("layout").value, {"columns": 3})

    def test_missing_preference_returns_none(self):
        self.assertIsNone(self.open_core().get_preference("nope"))

    def test_invalid_key_rejected(self):
        with self.assertRaises(ValidationError):
            self.open_core().set_preference("  ", "x")


class ProjectStateApiTests(CoreTestCase):
    def test_set_get_history_list(self):
        core = self.open_core()
        core.set_project_state("memory", {"phase": "design"})
        core.set_project_state("memory", {"phase": "v0.1"})
        core.set_project_state("persona", {"status": "planned"})
        self.assertEqual(core.get_project_state("memory").state, {"phase": "v0.1"})
        self.assertEqual(
            [record.state["phase"] for record in core.project_state_history("memory")],
            ["design", "v0.1"],
        )
        self.assertEqual(len(core.list_project_states()), 2)

    def test_state_must_be_an_object(self):
        with self.assertRaises(ValidationError):
            self.open_core().set_project_state("memory", ["not", "an", "object"])


class ForgetTests(CoreTestCase):
    def test_logical_delete_hides_but_keeps_history(self):
        core = self.open_core()
        memory = core.remember("temporary secret")
        self.assertTrue(core.forget(memory.id))
        self.assertEqual(core.get_memory(memory.id).status, "deleted")
        self.assertEqual(core.recall("secret"), [])
        self.assertEqual(core.store.count_memories(), 1)

    def test_hard_delete_removes_the_row(self):
        core = self.open_core()
        memory = core.remember("temporary secret")
        self.assertTrue(core.forget(memory.id, hard=True))
        self.assertIsNone(core.get_memory(memory.id))

    def test_forgetting_a_missing_memory_raises(self):
        with self.assertRaises(NotFoundError):
            self.open_core().forget("mem_missing")


class AskDegradationTests(CoreTestCase):
    def test_ask_without_a_provider_returns_the_recalled_context(self):
        core = self.open_core()
        core.remember("用户偏好本地优先")
        answer = core.ask("本地优先")
        self.assertTrue(answer.degraded)
        self.assertIn("本地优先", answer.text)
        self.assertIn("no model provider configured", answer.reason)
        self.assertEqual(len(answer.context_ids), 1)

    def test_ask_with_a_failing_provider_degrades_with_a_reason(self):
        provider = FakeProvider(fail_with="ollama is not running")
        core = self.open_core(provider=provider)
        core.remember("用户偏好本地优先")
        answer = core.ask("本地优先")
        self.assertTrue(answer.degraded)
        self.assertIn("ollama is not running", answer.reason)
        self.assertIn("用户偏好本地优先", answer.text)

    def test_ask_with_no_memories_still_answers(self):
        answer = self.open_core().ask("never stored anything like this")
        self.assertTrue(answer.degraded)
        self.assertIn("no relevant memories", answer.text)

    def test_ask_with_a_provider_returns_model_output(self):
        provider = FakeProvider(model="fake-A", prefix="[A]")
        core = self.open_core(provider=provider)
        core.remember("用户偏好本地优先")
        answer = core.ask("本地优先")
        self.assertFalse(answer.degraded)
        self.assertEqual(answer.provider, "fake")
        self.assertEqual(answer.model, "fake-A")
        self.assertTrue(answer.text.startswith("[A]"))
        self.assertEqual(provider.call_count, 1)

    def test_ask_sends_the_recalled_context_to_the_model(self):
        provider = FakeProvider(reply="ok")
        core = self.open_core(provider=provider)
        core.remember("用户偏好本地优先")
        core.ask("偏好")
        sent = provider.calls[0]
        self.assertEqual(sent[0].role, "system")
        self.assertIn("用户偏好本地优先", sent[0].content)
        self.assertEqual(sent[1].role, "user")
        self.assertEqual(sent[1].content, "偏好")

    def test_ask_never_writes_to_the_database(self):
        # Read path and write path are strictly separated, which is what lets
        # the model-swap test assert a byte-identical database.
        core = self.open_core(provider=FakeProvider())
        core.remember("用户偏好本地优先")
        core.close()

        before = file_digest(self.db_path)
        core = self.open_core(provider=FakeProvider(prefix="[other]"))
        core.ask("偏好")
        core.close()
        self.assertEqual(file_digest(self.db_path), before)


class ExportStatsTests(CoreTestCase):
    def test_export_to_file_is_deterministic(self):
        core = self.open_core()
        core.remember("a memory")
        core.set_preference("theme", "dark")
        first = os.path.join(self.tmpdir, "one.jsonl")
        second = os.path.join(self.tmpdir, "two.jsonl")
        core.export_to(first)
        core.export_to(second)
        self.assertEqual(file_digest(first), file_digest(second))

    def test_snapshot_is_restorable(self):
        core = self.open_core()
        core.remember("survives a snapshot")
        target = os.path.join(self.tmpdir, "backup.sqlite")
        core.snapshot(target)
        restored = MemoryCore.open(db_path=target)
        self.addCleanup(restored.close)
        self.assertEqual(len(restored.recall("snapshot")), 1)

    def test_stats_reports_extractor_and_provider(self):
        core = self.open_core(provider=FakeProvider())
        stats = core.stats()
        self.assertEqual(stats["extractor"], "rules")
        self.assertEqual(stats["provider"]["name"], "fake")


class OpenAndConfigTests(CoreTestCase):
    def test_open_without_any_configuration_works(self):
        core = self.open_core()
        core.remember("works with no config at all")
        self.assertEqual(len(core.recall("config")), 1)

    def test_provider_can_be_resolved_from_a_models_config(self):
        models_path = os.path.join(self.tmpdir, "models.json")
        with open(models_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "providers": {"local": {"type": "fake", "model": "fake-A", "prefix": "[A]"}},
                    "roles": {"chat": "local"},
                },
                handle,
            )
        core = self.open_core(models_path=models_path)
        self.assertIsNotNone(core.provider)
        self.assertEqual(core.provider.model, "fake-A")

    def test_unbound_chat_role_means_no_provider(self):
        models_path = os.path.join(self.tmpdir, "models.json")
        with open(models_path, "w", encoding="utf-8") as handle:
            json.dump({"providers": {}, "roles": {"chat": None}}, handle)
        core = self.open_core(models_path=models_path)
        self.assertIsNone(core.provider)
        self.assertTrue(core.ask("anything").degraded)

    def test_unknown_configured_provider_raises(self):
        models_path = os.path.join(self.tmpdir, "models.json")
        with open(models_path, "w", encoding="utf-8") as handle:
            json.dump({"providers": {}, "roles": {"chat": "missing"}}, handle)
        with self.assertRaises(ConfigurationError):
            self.open_core(models_path=models_path, provider_name="chat")


class PromptBuildingTests(unittest.TestCase):
    def test_build_prompt_includes_context_and_question(self):
        class Item(object):
            id = "mem_1"
            kind = "semantic"
            content = "Prefers local-first"

        class Hit(object):
            item = Item()
            item_type = "memory"

        messages = build_prompt([Hit()], "what do I prefer?")
        self.assertIn("Prefers local-first", messages[0].content)
        self.assertEqual(messages[1].content, "what do I prefer?")

    def test_build_prompt_handles_empty_context(self):
        messages = build_prompt([], "anything")
        self.assertIn("no relevant memories", messages[0].content)

    def test_format_hit_for_preference(self):
        class Preference(object):
            id = "prf_1"
            key = "theme"
            value = "dark"

        class Hit(object):
            item = Preference()
            item_type = "preference"

        self.assertEqual(format_hit(Hit()), "preference theme = dark")


if __name__ == "__main__":
    unittest.main()
