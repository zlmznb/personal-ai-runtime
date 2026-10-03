"""MSIT-1: Model-Swap Isolation Test.

This is the executable form of the sentence the whole project exists to prove:

    After swapping the model, the AI's long-term identity, memories and
    project state are still there.

The test is deliberately a HARD assertion, not a similarity threshold:

* every recall query must return the **same ids in the same order**;
* the database file must be **byte-identical** after a provider swap.

That strength is only possible because retrieval never consults a model
(ADR 0003). If this module ever starts failing, the architecture has been
compromised - do not weaken the assertion, fix the layering.

No network and no real model is required: Ollama is pointed at a closed local
port and DeepSeek is configured with no key, which is exactly the "model is
gone" scenario the architecture is designed for.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from memory_core.api import MemoryCore
from memory_core.providers.deepseek import DeepSeekProvider
from memory_core.providers.fake import FakeProvider
from memory_core.providers.fake_embedding import FakeEmbeddingProvider
from memory_core.providers.ollama import OllamaProvider

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PATH = os.path.join(PROJECT_ROOT, "fixtures", "scripted_session.json")

#: Environment variable that is guaranteed not to be set, so DeepSeekProvider
#: behaves as "configured but unusable".
ABSENT_KEY_ENV = "MEMORY_CORE_MSIT_ABSENT_KEY"

RECALL_LIMIT = 10


def digest(path):
    # type: (str) -> str
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_fixture():
    # type: () -> dict
    with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def seed(core, fixture):
    # type: (MemoryCore, dict) -> None
    for step in fixture["steps"]:
        operation = step["op"]
        if operation == "observe":
            core.observe(step["text"])
        elif operation == "remember":
            core.remember(
                step["content"],
                kind=step.get("kind", "semantic"),
                tags=step.get("tags", ()),
                salience=step.get("salience", 0.5),
            )
        elif operation == "set_preference":
            core.set_preference(step["key"], step["value"])
        elif operation == "set_project_state":
            core.set_project_state(step["project_id"], step["state"])
        else:  # pragma: no cover - fixture guard
            raise AssertionError("unknown fixture operation {0!r}".format(operation))


def provider_variants():
    # type: () -> list
    """Five ways of having (or not having) a model, none of which may matter."""
    return [
        ("no-provider", None),
        ("fake-A", FakeProvider(model="fake-A", prefix="[A]")),
        ("fake-B", FakeProvider(model="fake-B", prefix="[B]")),
        (
            "ollama-unreachable",
            OllamaProvider(base_url="http://127.0.0.1:1/v1", model="qwen3:8b", timeout=1.0),
        ),
        (
            "deepseek-without-key",
            DeepSeekProvider(model="deepseek-chat", api_key=None, api_key_env=ABSENT_KEY_ENV),
        ),
    ]


class ModelSwapIsolationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-msit-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")
        self.fixture = load_fixture()
        self.queries = self.fixture["queries"]
        self.expected = self.fixture["expected"]
        self.addCleanup(lambda: os.environ.pop(ABSENT_KEY_ENV, None))
        os.environ.pop(ABSENT_KEY_ENV, None)

        core = self._open(None)
        seed(core, self.fixture)
        core.close()

        self.baseline_digest = digest(self.db_path)
        self.baseline_recall = self._recall_map(None)

    def _open(self, provider, path=None):
        # type: (object, object) -> MemoryCore
        return MemoryCore.open(db_path=path or self.db_path, provider=provider)

    def _recall_map(self, provider, path=None):
        # type: (object, object) -> dict
        core = self._open(provider, path)
        try:
            return {
                query: [
                    (hit.id, hit.item_type, hit.matched_on, hit.score)
                    for hit in core.recall(query, limit=RECALL_LIMIT)
                ]
                for query in self.queries
            }
        finally:
            core.close()

    # -- the core guarantee -------------------------------------------------

    def test_recall_is_identical_under_every_provider(self):
        self.assertTrue(self.baseline_recall, "fixture produced no recall baseline")
        for name, provider in provider_variants():
            with self.subTest(provider=name):
                self.assertEqual(
                    self._recall_map(provider),
                    self.baseline_recall,
                    "recall changed when the provider became {0}".format(name),
                )

    def test_database_is_byte_identical_under_every_provider(self):
        for name, provider in provider_variants():
            with self.subTest(provider=name):
                core = self._open(provider)
                try:
                    for query in self.queries[:3]:
                        core.ask(query)
                finally:
                    core.close()
                self.assertEqual(
                    digest(self.db_path),
                    self.baseline_digest,
                    "the database changed when the provider became {0}".format(name),
                )

    def test_swapping_the_model_changes_the_answer_but_not_the_memory(self):
        core_a = self._open(FakeProvider(model="fake-A", prefix="[A]"))
        core_b = self._open(FakeProvider(model="fake-B", prefix="[B]"))
        try:
            answer_a = core_a.ask("本地优先")
            answer_b = core_b.ask("本地优先")
            self.assertNotEqual(answer_a.text, answer_b.text, "the model did not actually change")
            self.assertEqual(answer_a.model, "fake-A")
            self.assertEqual(answer_b.model, "fake-B")
            # ... and yet the memory each model saw is identical.
            self.assertEqual(answer_a.context_ids, answer_b.context_ids)
            self.assertTrue(answer_a.context_ids, "no memory context was supplied")
        finally:
            core_a.close()
            core_b.close()
        self.assertEqual(digest(self.db_path), self.baseline_digest)

    def test_identity_memories_survive_the_swap(self):
        for _, provider in provider_variants():
            core = self._open(provider)
            try:
                identities = core.list_memories(kinds=("identity",))
                self.assertEqual(len(identities), 1)
                self.assertIn("本地优先的个人 AI", identities[0].content)
            finally:
                core.close()

    def test_preferences_survive_the_swap(self):
        expected = {
            "theme": "dark",
            "editor": "vscode",
            "privacy": "local-only",
            "language": "zh",
        }
        for _, provider in provider_variants():
            core = self._open(provider)
            try:
                for key, value in expected.items():
                    self.assertEqual(core.get_preference(key).value, value, msg=key)
            finally:
                core.close()

    def test_project_state_survives_the_swap(self):
        for _, provider in provider_variants():
            core = self._open(provider)
            try:
                memory = core.get_project_state("memory")
                self.assertEqual(memory.state, {"phase": "v0.1", "storage": "sqlite"})
                self.assertEqual(len(core.project_state_history("memory")), 2)
                self.assertEqual(
                    core.get_project_state("persona").state, {"status": "planned"}
                )
            finally:
                core.close()

    def test_a_dead_model_still_answers_from_memory(self):
        for name, provider in provider_variants():
            if provider is None or name.startswith("fake"):
                continue
            with self.subTest(provider=name):
                core = self._open(provider)
                try:
                    answer = core.ask("本地优先")
                    self.assertTrue(answer.degraded)
                    self.assertIn("用户偏好本地优先", answer.text)
                finally:
                    core.close()
                self.assertEqual(digest(self.db_path), self.baseline_digest)

    # -- supporting evidence ------------------------------------------------

    def test_fixture_counts_match_the_database(self):
        core = self._open(None)
        try:
            stats = core.stats()
            self.assertEqual(stats["events"], self.expected["events"])
            self.assertEqual(stats["memories"]["active"], self.expected["memories"])
            self.assertEqual(
                stats["preferences"]["active"], self.expected["active_preferences"]
            )
            self.assertEqual(
                stats["project_state"]["current"], self.expected["current_project_states"]
            )
        finally:
            core.close()

    def test_the_same_script_produces_the_same_logical_memories(self):
        # Ids are random and timestamps differ, but the logical content of an
        # entirely model-free pipeline is reproducible.
        second_dir = tempfile.mkdtemp(prefix="memory-core-msit-2-")
        self.addCleanup(shutil.rmtree, second_dir, True)
        second_db = os.path.join(second_dir, "memory.sqlite")
        core = self._open(None, path=second_db)
        seed(core, self.fixture)
        try:
            first = self._logical_snapshot(self._open(None))
            other = self._logical_snapshot(core)
            self.assertEqual(first, other)
        finally:
            core.close()

    def _logical_snapshot(self, core):
        # type: (MemoryCore) -> dict
        try:
            return {
                "memories": sorted(
                    (memory.kind, memory.content)
                    for memory in core.list_memories(statuses=("active",), limit=1000)
                ),
                "preferences": sorted(
                    (p.key, json.dumps(p.value, sort_keys=True))
                    for p in core.list_preferences()
                ),
                "project_state": sorted(
                    (s.project_id, json.dumps(s.state, sort_keys=True))
                    for s in core.list_project_states()
                ),
            }
        finally:
            core.close()

    def test_export_is_stable_across_provider_swaps(self):
        baseline = None
        for _, provider in provider_variants():
            core = self._open(provider)
            try:
                exported = core.export_jsonl()
            finally:
                core.close()
            if baseline is None:
                baseline = exported
            self.assertEqual(exported, baseline)


class EmbeddingAndIndexSwapIsolationTest(unittest.TestCase):
    """MSIT-2: swap the chat model, swap the embedding model, delete the vector
    index - and no memory may be lost at any point.

    The database hash cannot be used here, because re-embedding legitimately
    rewrites the derived ``embeddings`` table. The correct assertion is on the
    *source* of truth: ``export_jsonl()`` covers exactly the source tables and
    excludes every derived one, so it must be byte-identical throughout.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-msit2-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")

        # Step 1-2: build memories with chat model A and embedding model A.
        core = self._open(chat_model="chat-A", embed_model="embed-A", embed_dim=64)
        for text in (
            "用户正在开发 Persona-EdgeAIoT，这是一个长期项目",
            "用户偏好本地优先，隐私敏感",
            "Retrieval must never call a chat model",
            "模型更换不能导致长期个人数据丢失",
        ):
            core.remember(text)
        core.set_preference("theme", "dark")
        core.set_project_state("memory", {"phase": "v0.2"})
        core.build_index()
        self.embeddings_after_build = core.store.count_embeddings()
        core.close()

        self.baseline_export = None
        core = self._open(chat_model="chat-A", embed_model="embed-A", embed_dim=64)
        self.baseline_export = core.export_jsonl()
        self.baseline_ids = sorted(memory.id for memory in core.list_memories(limit=1000))
        self.baseline_keyword = [hit.id for hit in core.recall("本地优先", mode="keyword")]
        core.close()

    def _open(self, chat_model=None, embed_model=None, embed_dim=64, embedding=True):
        chat = FakeProvider(model=chat_model, prefix="[{0}]".format(chat_model)) if chat_model else None
        embed = (
            FakeEmbeddingProvider(model=embed_model, dim=embed_dim)
            if (embedding and embed_model)
            else None
        )
        return MemoryCore.open(db_path=self.db_path, provider=chat, embedding_provider=embed)

    def _snapshot(self, core):
        return {
            "export": core.export_jsonl(),
            "ids": sorted(memory.id for memory in core.list_memories(limit=1000)),
            "count": core.store.count_memories(),
            "preference": core.get_preference("theme").value,
            "project_state": core.get_project_state("memory").state,
        }

    # -- the guarantee -----------------------------------------------------

    def test_swapping_both_models_does_not_touch_the_source_of_truth(self):
        core = self._open(chat_model="chat-B", embed_model="embed-B", embed_dim=128)
        try:
            snapshot = self._snapshot(core)
        finally:
            core.close()

        self.assertEqual(snapshot["export"], self.baseline_export)
        self.assertEqual(snapshot["ids"], self.baseline_ids)
        self.assertEqual(snapshot["preference"], "dark")
        self.assertEqual(snapshot["project_state"], {"phase": "v0.2"})

    def test_a_new_embedding_model_simply_has_no_vectors_yet(self):
        core = self._open(chat_model="chat-B", embed_model="embed-B", embed_dim=128)
        try:
            # Model A's vectors are still on disk, but they belong to model A and
            # are never compared against model B's query embedding.
            self.assertEqual(core.store.count_embeddings("fake-embedding:embed-A"), self.embeddings_after_build)
            self.assertEqual(core.store.count_embeddings("fake-embedding:embed-B"), 0)
            self.assertEqual(core.recall("本地优先", mode="semantic"), [])
            # Hybrid therefore degrades cleanly to the keyword ranking.
            self.assertEqual(
                [hit.id for hit in core.recall("本地优先", mode="hybrid")],
                self.baseline_keyword,
            )
        finally:
            core.close()

    def test_deleting_the_vector_index_does_not_lose_any_memory(self):
        core = self._open(chat_model="chat-B", embed_model="embed-A", embed_dim=64)
        try:
            self.assertTrue(core.recall("本地优先", mode="semantic"), "index should be usable")

            removed = core.drop_index()
            self.assertEqual(removed, self.embeddings_after_build)
            self.assertEqual(core.store.count_embeddings(), 0)

            # The index is gone; the memories are not.
            self.assertEqual(core.recall("本地优先", mode="semantic"), [])
            self.assertEqual([hit.id for hit in core.recall("本地优先", mode="keyword")], self.baseline_keyword)
            self.assertEqual(self._snapshot(core)["export"], self.baseline_export)
        finally:
            core.close()

    def test_rebuilding_the_index_restores_semantic_search(self):
        core = self._open(chat_model="chat-B", embed_model="embed-B", embed_dim=128)
        try:
            core.drop_index()
            self.assertEqual(core.recall("本地优先", mode="semantic"), [])

            result = core.rebuild_index()
            self.assertEqual(result["embedded"], len(self.baseline_ids))

            hits = core.recall("本地优先", mode="semantic")
            self.assertTrue(hits, "semantic search should work again after a rebuild")
            self.assertIn("本地优先", hits[0].item.content)
            self.assertEqual(self._snapshot(core)["export"], self.baseline_export)
        finally:
            core.close()

    def test_full_recovery_from_sqlite_alone(self):
        # Rebuild the whole semantic stack from the file, with no models at all.
        core = MemoryCore.open(db_path=self.db_path)
        try:
            self.assertEqual(sorted(m.id for m in core.list_memories(limit=1000)), self.baseline_ids)
            self.assertEqual(core.export_jsonl(), self.baseline_export)
            self.assertEqual([hit.id for hit in core.recall("本地优先", mode="keyword")], self.baseline_keyword)
            self.assertEqual(core.get_preference("theme").value, "dark")
            self.assertEqual(core.get_project_state("memory").state, {"phase": "v0.2"})
        finally:
            core.close()

    def test_the_whole_sequence_end_to_end(self):
        """A -> build -> close -> swap chat -> swap embeddings -> drop all -> rebuild -> recall."""
        core = self._open(chat_model="chat-B", embed_model="embed-B", embed_dim=128)
        try:
            self.assertEqual(core.export_jsonl(), self.baseline_export)
            core.drop_index(all_models=True)
            self.assertEqual(core.store.count_embeddings(), 0)
            self.assertEqual(core.export_jsonl(), self.baseline_export)
            core.rebuild_index()
            self.assertEqual(core.export_jsonl(), self.baseline_export)
            hits = core.recall("Persona-EdgeAIoT", mode="hybrid")
            contents = [hit.item.content for hit in hits]
            self.assertTrue(any("Persona-EdgeAIoT" in content for content in contents))
            self.assertEqual(sorted(m.id for m in core.list_memories(limit=1000)), self.baseline_ids)
        finally:
            core.close()

    def test_multiple_embedding_models_can_coexist(self):
        core = self._open(chat_model="chat-B", embed_model="embed-B", embed_dim=128)
        try:
            core.build_index()
            models = set(core.store.embedding_model_ids())
            self.assertIn("fake-embedding:embed-A", models)
            self.assertIn("fake-embedding:embed-B", models)
            # Vectors from different models are never compared with each other.
            self.assertEqual(len(core.recall("本地优先", mode="semantic")), len(self.baseline_ids))
        finally:
            core.close()

        fresh = MemoryCore.open(db_path=self.db_path)
        try:
            self.assertEqual(sorted(m.id for m in fresh.list_memories(limit=1000)), self.baseline_ids)
        finally:
            fresh.close()


if __name__ == "__main__":
    unittest.main()
