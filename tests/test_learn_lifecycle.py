"""Candidate lifecycle and LLM-formation integration tests.

Covers: candidate -> validate -> accept / reject -> commit, plus the guarantees
that a failing model can never pollute the database and that ``propose`` writes
nothing at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from memory_core.api import MemoryCore
from memory_core.providers.fake import FakeProvider


def digest(path):
    # type: (str) -> str
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def response(memories):
    return json.dumps({"memories": memories}, ensure_ascii=False)


class LearnTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-learn-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")
        self.reply = response([])

    def open_core(self, reply=None, use_llm_formation=True):
        provider = FakeProvider(
            model="fake-A", reply=reply if reply is not None else self.reply
        )
        core = MemoryCore.open(
            db_path=self.db_path, provider=provider, use_llm_formation=use_llm_formation
        )
        self.addCleanup(core.close)
        return core


class AddLifecycleTests(LearnTestCase):
    def test_add_is_accepted_and_committed(self):
        core = self.open_core(response([
            {"operation": "ADD", "content": "用户正在开发 Persona-EdgeAIoT", "confidence": 0.85}
        ]))
        observation = core.learn("我在做 Persona-EdgeAIoT")

        self.assertEqual(core.store.count_events(), 1)
        self.assertEqual(len(observation.memories), 1)
        self.assertEqual(len(observation.accepted), 1)
        self.assertEqual(observation.rejected, ())
        memory = observation.memories[0]
        self.assertEqual(memory.content, "用户正在开发 Persona-EdgeAIoT")
        self.assertEqual(memory.confidence, 0.85)
        self.assertLess(memory.confidence, 1.0)
        self.assertEqual(memory.source["origin"], "llm")
        self.assertEqual(memory.source["event_ids"], [observation.event.id])

    def test_provenance_records_the_model(self):
        core = self.open_core(response([{"operation": "ADD", "content": "a durable fact"}]))
        core.learn("something")
        memory = core.list_memories()[0]
        self.assertEqual(memory.generated_by, "fake:fake-A")

    def test_noop_writes_nothing_but_keeps_the_event(self):
        core = self.open_core(response([{"operation": "NOOP", "reason": "small talk"}]))
        observation = core.learn("今天天气不错")
        self.assertEqual(core.store.count_events(), 1)
        self.assertEqual(core.store.count_memories(), 0)
        self.assertEqual(observation.memories, ())
        self.assertEqual(len(observation.rejected), 1)
        self.assertIn("decided not to remember", observation.rejected[0].reason)

    def test_empty_candidate_list_writes_only_the_event(self):
        core = self.open_core(response([]))
        observation = core.learn("nothing memorable")
        self.assertEqual(core.store.count_events(), 1)
        self.assertEqual(core.store.count_memories(), 0)
        self.assertEqual(observation.outcomes, ())

    def test_low_confidence_candidate_is_rejected(self):
        core = self.open_core(response([
            {"operation": "ADD", "content": "maybe something", "confidence": 0.1}
        ]))
        observation = core.learn("not sure about this")
        self.assertEqual(core.store.count_memories(), 0)
        self.assertEqual(len(observation.rejected), 1)
        self.assertIn("below", observation.rejected[0].reason)

    def test_short_content_is_rejected(self):
        core = self.open_core(response([{"operation": "ADD", "content": "ok"}]))
        self.assertEqual(core.learn("short").memories, ())

    def test_duplicate_is_rejected_on_the_second_turn(self):
        core = self.open_core(response([{"operation": "ADD", "content": "用户偏好本地优先"}]))
        core.learn("我喜欢本地优先")
        second = core.learn("再说一次，我喜欢本地优先")
        self.assertEqual(core.store.count_memories(), 1)
        self.assertEqual(len(second.rejected), 1)
        self.assertIn("duplicate", second.rejected[0].reason)


class SupersedeLifecycleTests(LearnTestCase):
    def test_supersede_replaces_and_keeps_history(self):
        core = self.open_core(response([{"operation": "ADD", "content": "用户使用 Python"}]))
        first = core.learn("我用 Python").memories[0]

        provider = FakeProvider(model="fake-B", reply=response([
            {"operation": "SUPERSEDE", "content": "用户改用 TypeScript", "supersedes_id": first.id}
        ]))
        replacement_core = MemoryCore(
            core.store, provider=provider, config=core.config
        )
        observation = replacement_core.learn("我改用 TypeScript 了")

        self.assertEqual(len(observation.memories), 1)
        self.assertEqual(observation.memories[0].content, "用户改用 TypeScript")
        self.assertEqual(core.get_memory(first.id).status, "superseded")
        self.assertEqual(core.get_memory(first.id).superseded_by, observation.memories[0].id)
        self.assertEqual(len(core.list_memories(statuses=("active",))), 1)

    def test_supersede_with_a_missing_target_is_rejected_and_changes_nothing(self):
        core = self.open_core(response([
            {"operation": "SUPERSEDE", "content": "new content", "supersedes_id": "mem_gone"}
        ]))
        before = digest(self.db_path)
        observation = core.learn("this names a target that no longer exists")
        # The schema gate rejects it before anything is written.
        self.assertEqual(observation.memories, ())
        self.assertEqual(core.store.count_memories(), 0)
        self.assertEqual(core.store.count_events(), 1)
        # Only the event row was appended.
        self.assertNotEqual(digest(self.db_path), before)

    def test_supersede_rolls_back_when_the_target_is_missing_at_commit_time(self):
        # Force the pseudo-candidate path: the model names a real, visible id,
        # the memory is then deleted before commit.
        core = self.open_core(response([]))
        existing = core.remember("original statement")
        core.forget(existing.id, hard=True)

        from memory_core.formation.base import Candidate, Extraction, Extractor
        from memory_core.formation.base import MEMORY, SUPERSEDE

        class StaleExtractor(Extractor):
            name = "stale"

            def extract(self, text, scope="global", context=None):
                return Extraction(candidates=(
                    Candidate(
                        target=MEMORY,
                        payload={"content": "replacement statement", "kind": "semantic"},
                        confidence=0.9,
                        operation=SUPERSEDE,
                        supersedes=existing.id,
                    ),
                ))

        core.llm_extractor = None
        core.extractor = StaleExtractor()
        observation = core.learn("replace the thing that is gone")

        self.assertEqual(core.store.count_memories(), 0)
        self.assertEqual(len(observation.rejected), 1)
        self.assertIn("write rejected", observation.rejected[0].reason)


class FailureIsolationTests(LearnTestCase):
    def test_provider_failure_writes_only_the_event(self):
        core = self.open_core()
        core.llm_extractor.provider = FakeProvider(fail_with="model is down")
        observation = core.learn("something important")

        self.assertEqual(core.store.count_events(), 1)
        self.assertEqual(core.store.count_memories(), 0)
        self.assertEqual(observation.memories, ())
        self.assertIn("provider unavailable", observation.skipped_reason)

    def test_malformed_model_output_writes_only_the_event(self):
        core = self.open_core("I cannot answer that.")
        observation = core.learn("something important")
        self.assertEqual(core.store.count_events(), 1)
        self.assertEqual(core.store.count_memories(), 0)
        self.assertIn("schema validation", observation.skipped_reason)

    def test_sql_injection_does_not_reach_the_database(self):
        core = self.open_core(json.dumps({
            "memories": [{"operation": "ADD", "content": "x", "sql": "DROP TABLE memories"}]
        }))
        observation = core.learn("injection attempt")

        self.assertEqual(observation.memories, ())
        # The table is not merely intact - it is still queryable.
        self.assertEqual(core.store.count_memories(), 0)
        self.assertEqual(core.recall("anything"), [])

    def test_a_broken_extractor_does_not_lose_the_event(self):
        class Exploding(object):
            name = "exploding"

            def extract(self, text, scope="global", context=None):
                raise RuntimeError("bad extractor")

        core = self.open_core(response([]))
        core.extractor = Exploding()
        observation = core.learn("still recorded")
        self.assertEqual(core.store.count_events(), 1)
        self.assertIn("exploding failed", observation.skipped_reason)


class ProposeTests(LearnTestCase):
    def test_propose_writes_nothing_at_all(self):
        core = self.open_core(response([{"operation": "ADD", "content": "a proposed fact"}]))
        core.remember("an existing memory")
        before = digest(self.db_path)

        proposal = core.propose("我在说一些事情")

        self.assertEqual(digest(self.db_path), before)
        self.assertEqual(core.store.count_events(), 0)
        self.assertEqual(core.store.count_memories(), 1)
        self.assertEqual(len(proposal.candidates), 1)
        self.assertTrue(proposal.outcomes[0].accepted)

    def test_propose_reports_rejections(self):
        core = self.open_core(response([{"operation": "NOOP", "reason": "chit-chat"}]))
        proposal = core.propose("hello there")
        self.assertEqual(len(proposal.outcomes), 1)
        self.assertTrue(proposal.outcomes[0].rejected)

    def test_propose_can_skip_the_llm(self):
        core = self.open_core(response([{"operation": "ADD", "content": "llm only"}]))
        proposal = core.propose("记住：规则应该抓到这条", include_llm=False)
        self.assertEqual(len(proposal.candidates), 1)
        self.assertEqual(proposal.candidates[0].origin, "rule")

    def test_propose_shows_the_context_ids_it_would_offer(self):
        core = self.open_core(response([{"operation": "ADD", "content": "another fact"}]))
        existing = core.remember("用户偏好本地优先，隐私敏感")
        proposal = core.propose("本地优先")
        self.assertIn(existing.id, proposal.context_ids)


class RulesAndLlmTogetherTests(LearnTestCase):
    def test_rules_run_without_any_model(self):
        core = MemoryCore.open(db_path=self.db_path, use_llm_formation=False)
        self.addCleanup(core.close)
        observation = core.observe("记住：用户偏好本地优先")
        self.assertEqual(len(observation.memories), 1)
        self.assertEqual(observation.memories[0].confidence, 1.0)

    def test_rules_and_llm_candidates_are_merged_and_deduped(self):
        # The rule proposes the same content the model does; only one memory
        # should result, and it should be the deterministic rule's.
        core = self.open_core(response([
            {"operation": "ADD", "content": "用户偏好本地优先", "confidence": 0.7}
        ]))
        observation = core.learn("记住：用户偏好本地优先")
        self.assertEqual(core.store.count_memories(), 1)
        self.assertEqual(len(observation.outcomes), 1)
        self.assertEqual(observation.memories[0].confidence, 1.0)

    def test_llm_formation_can_be_disabled(self):
        core = self.open_core(response([{"operation": "ADD", "content": "should not appear"}]), use_llm_formation=False)
        observation = core.learn("a plain sentence with no rule trigger")
        self.assertEqual(observation.memories, ())
        self.assertEqual(core.store.count_memories(), 0)

    def test_formation_info_reports_the_configuration(self):
        core = self.open_core()
        info = core.formation_info()
        self.assertEqual(info["rules"], "rules")
        self.assertTrue(info["llm_formation"])
        self.assertEqual(info["llm"]["model"], "fake-A")

    def test_a_third_of_the_way_failures_do_not_stop_other_candidates(self):
        core = self.open_core(response([
            {"operation": "ADD", "content": "good memory one", "confidence": 0.9},
            {"operation": "ADD", "content": "bad", "confidence": 0.9},
            {"operation": "ADD", "content": "good memory two", "confidence": 0.9},
        ]))
        observation = core.learn("mixed quality turn")
        self.assertEqual(len(observation.accepted), 2)
        self.assertEqual(len(observation.rejected), 1)
        self.assertEqual(core.store.count_memories(), 2)


if __name__ == "__main__":
    unittest.main()
