"""Opt-in integration test against a real local Ollama.

Skipped by default so the suite never needs a model or a network. Enable with::

    $env:MEMORY_TEST_OLLAMA = "1"
    py -3 -m unittest tests.test_ollama_integration -v

Override the model with ``MEMORY_TEST_OLLAMA_MODEL`` (default ``qwen3:8b``).

The important test here is the last one: it is MSIT-1 with a *real* model. A
real completion changes the answer; it must not change a single byte of
``memory.sqlite``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
import urllib.error
import urllib.request

from memory_core.api import MemoryCore
from memory_core.domain.dto import ChatMessage
from memory_core.domain.errors import ProviderError
from memory_core.providers.ollama import OllamaProvider
from memory_core.providers.ollama_embedding import OllamaEmbeddingProvider

ENABLED = os.environ.get("MEMORY_TEST_OLLAMA") == "1"
MODEL = os.environ.get("MEMORY_TEST_OLLAMA_MODEL", "qwen3:8b")
BASE_URL = os.environ.get("MEMORY_TEST_OLLAMA_URL", "http://127.0.0.1:11434/v1")
OLLAMA_ROOT = BASE_URL.rsplit("/v1", 1)[0]
TIMEOUT = float(os.environ.get("MEMORY_TEST_OLLAMA_TIMEOUT", "300"))

#: Real local embedding models, in preference order (best quality first).
#:
#: Measured on this machine with Chinese content:
#:   nomic-embed-text (768-d) ranked correctly on every probe query
#:   all-minilm      (384-d) is English-only and misranks Chinese text
#: Retrieval quality is therefore model-dependent - durability never is.
EMBED_MODELS = tuple(
    value.strip()
    for value in os.environ.get(
        "MEMORY_TEST_EMBED_MODELS", "nomic-embed-text,all-minilm"
    ).split(",")
    if value.strip()
)


def digest(path):
    # type: (str) -> str
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def ollama_is_reachable():
    # type: () -> bool
    try:
        with urllib.request.urlopen(BASE_URL.rstrip("/") + "/models", timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


@unittest.skipUnless(ENABLED, "set MEMORY_TEST_OLLAMA=1 to run real-Ollama tests")
class RealOllamaIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ollama_is_reachable():
            raise unittest.SkipTest(
                "no Ollama at {0}; start it or set MEMORY_TEST_OLLAMA_URL".format(BASE_URL)
            )

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-ollama-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")

    def _provider(self):
        return OllamaProvider(model=MODEL, base_url=BASE_URL, timeout=TIMEOUT)

    def test_the_local_provider_returns_a_real_completion(self):
        result = self._provider().chat(
            [ChatMessage("system", "Answer with the number only."), ChatMessage("user", "2+2")]
        )
        self.assertEqual(result.provider, "ollama")
        self.assertTrue(result.text.strip())
        self.assertIn("4", result.text)

    def test_ask_with_a_real_local_model_uses_recalled_memory(self):
        core = MemoryCore.open(db_path=self.db_path, provider=self._provider())
        self.addCleanup(core.close)
        core.remember("用户偏好本地优先，隐私敏感")
        core.remember("用户喜欢的编辑器是 vscode")

        answer = core.ask("用户喜欢什么编辑器？请只回答编辑器名称。")
        self.assertFalse(answer.degraded, msg=str(answer.reason))
        self.assertEqual(answer.provider, "ollama")
        self.assertTrue(answer.context_ids, "no memory was recalled for the prompt")
        self.assertIn("vscode", answer.text.lower())

    def test_a_real_model_swap_does_not_change_a_single_byte(self):
        """MSIT-1 against a real model: answer changes, memory does not."""
        core = MemoryCore.open(db_path=self.db_path)
        core.observe("记住：用户偏好本地优先")
        core.remember("长期目标是记忆与模型推理彻底解耦", salience=0.9)
        core.set_preference("theme", "dark")
        core.set_project_state("memory", {"phase": "v0.1"})
        baseline_recall = [
            (hit.id, hit.item_type, hit.matched_on, hit.score)
            for hit in core.recall("本地优先")
        ]
        baseline_export = core.export_jsonl()
        core.close()

        before = digest(self.db_path)

        # Swap from "no model" to a real 8B model running locally.
        core = MemoryCore.open(db_path=self.db_path, provider=self._provider())
        try:
            answer = core.ask("用户偏好什么？")
            self.assertFalse(answer.degraded, msg=str(answer.reason))
            self.assertTrue(answer.text.strip())

            after_recall = [
                (hit.id, hit.item_type, hit.matched_on, hit.score)
                for hit in core.recall("本地优先")
            ]
            self.assertEqual(after_recall, baseline_recall, "recall changed after the model swap")
            self.assertEqual(core.export_jsonl(), baseline_export)
            self.assertEqual(core.get_preference("theme").value, "dark")
            self.assertEqual(core.get_project_state("memory").state, {"phase": "v0.1"})
        finally:
            core.close()

        self.assertEqual(digest(self.db_path), before, "the database changed after a model swap")


@unittest.skipUnless(ENABLED, "set MEMORY_TEST_OLLAMA=1 to run real-model tests")
class RealEmbeddingIntegrationTest(unittest.TestCase):
    """Real local embeddings, and swapping between two real embedding models."""

    @classmethod
    def setUpClass(cls):
        if not ollama_is_reachable():
            raise unittest.SkipTest("no Ollama at {0}".format(BASE_URL))
        cls.available = []
        for model in EMBED_MODELS:
            provider = OllamaEmbeddingProvider(model=model, base_url=OLLAMA_ROOT, timeout=TIMEOUT)
            try:
                provider.embed(["probe"])
                cls.available.append(model)
            except ProviderError:
                continue
        if not cls.available:
            raise unittest.SkipTest(
                "no embedding model available; try: ollama pull {0}".format(EMBED_MODELS[0])
            )

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-embed-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")
        self.model = self.available[0]

    def _provider(self, model=None):
        return OllamaEmbeddingProvider(
            model=model or self.model, base_url=OLLAMA_ROOT, timeout=TIMEOUT
        )

    def test_real_embeddings_are_meaningful(self):
        provider = self._provider()
        vectors = provider.embed(["用户偏好本地优先，隐私敏感", "本地部署优先，注重隐私", "今天天气很好"])
        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(vectors[0]), len(vectors[1]))

        def cosine(left, right):
            numerator = sum(a * b for a, b in zip(left, right))
            left_norm = sum(a * a for a in left) ** 0.5
            right_norm = sum(b * b for b in right) ** 0.5
            return numerator / (left_norm * right_norm)

        self.assertGreater(cosine(vectors[0], vectors[1]), cosine(vectors[0], vectors[2]))

    def test_semantic_retrieval_end_to_end_with_real_embeddings(self):
        core = MemoryCore.open(db_path=self.db_path, embedding_provider=self._provider())
        self.addCleanup(core.close)
        relevant = core.remember("用户偏好本地优先，隐私敏感")
        core.remember("用户喜欢喝咖啡，下午三点左右")
        result = core.build_index()
        self.assertEqual(result["embedded"], 2)

        # A query with enough signal to be fair to any embedding model.
        hits = core.recall("本地优先，隐私", mode="semantic")
        self.assertTrue(hits)
        self.assertEqual(hits[0].id, relevant.id)

        hybrid = core.recall("隐私", mode="hybrid")
        self.assertTrue(hybrid)
        self.assertEqual(hybrid[0].id, relevant.id)

    def test_a_weak_embedding_model_degrades_ranking_but_never_durability(self):
        # whichever model is available last is the weakest one
        weak = self.available[-1]
        core = MemoryCore.open(db_path=self.db_path, embedding_provider=self._provider(weak))
        try:
            memory = core.remember("用户偏好本地优先，隐私敏感")
            core.remember("用户喜欢喝咖啡，下午三点左右")
            core.build_index()

            # The row is durable regardless of how well the model ranks.
            self.assertEqual(core.store.count_memories(), 2)
            self.assertIsNotNone(core.get_memory(memory.id))
            self.assertIn(memory.id, [hit.id for hit in core.recall("本地优先，隐私", mode="semantic")])
            # And the keyword signal is completely model-independent.
            self.assertEqual(
                [hit.id for hit in core.recall("本地优先", mode="keyword")], [memory.id]
            )
        finally:
            core.close()

    def test_swapping_between_two_real_embedding_models(self):
        if len(self.available) < 2:
            self.skipTest("only one embedding model available: {0}".format(self.available))

        first, second = self.available[0], self.available[1]

        core = MemoryCore.open(db_path=self.db_path, embedding_provider=self._provider(first))
        self.addCleanup(core.close)
        memory = core.remember("用户正在开发 Persona-EdgeAIoT，这是一个长期项目")
        core.remember("Retrieval must never call a chat model")
        core.build_index()
        baseline_export = core.export_jsonl()
        baseline_ids = sorted(m.id for m in core.list_memories(limit=100))
        core.close()

        # Swap the embedding model, keep the chat side untouched.
        core = MemoryCore.open(db_path=self.db_path, embedding_provider=self._provider(second))
        try:
            self.assertEqual(sorted(m.id for m in core.list_memories(limit=100)), baseline_ids)
            self.assertEqual(core.export_jsonl(), baseline_export)

            # The new model has no vectors yet, so semantic degrades cleanly.
            self.assertEqual(core.recall("本地优先", mode="semantic"), [])

            core.rebuild_index()
            hits = core.recall("本地优先", mode="semantic")
            self.assertTrue(hits)
            self.assertIn(memory.id, [hit.id for hit in hits])
            self.assertEqual(core.export_jsonl(), baseline_export)
            self.assertEqual(sorted(m.id for m in core.list_memories(limit=100)), baseline_ids)
        finally:
            core.close()

    def test_real_embedding_index_can_be_dropped_and_rebuilt(self):
        core = MemoryCore.open(db_path=self.db_path, embedding_provider=self._provider())
        try:
            core.remember("用户偏好本地优先")
            core.build_index()
            self.assertTrue(core.recall("本地优先", mode="semantic"))

            core.drop_index(all_models=True)
            self.assertEqual(core.store.count_embeddings(), 0)
            self.assertEqual(core.recall("本地优先", mode="semantic"), [])
            self.assertTrue(core.recall("本地优先", mode="keyword"))

            core.rebuild_index()
            self.assertTrue(core.recall("本地优先", mode="semantic"))
        finally:
            core.close()


@unittest.skipUnless(ENABLED, "set MEMORY_TEST_OLLAMA=1 to run real-model tests")
class RealFormationIntegrationTest(unittest.TestCase):
    """Real LLM memory formation.

    The model's output is not deterministic, so these tests assert the
    *architectural* properties rather than exact content: nothing may crash,
    the raw event always survives, and anything that is written must have passed
    validation and carry provenance.
    """

    @classmethod
    def setUpClass(cls):
        if not ollama_is_reachable():
            raise unittest.SkipTest("no Ollama at {0}".format(BASE_URL))

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-learn-ollama-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")

    def _core(self):
        core = MemoryCore.open(
            db_path=self.db_path,
            provider=OllamaProvider(model=MODEL, base_url=BASE_URL, timeout=TIMEOUT),
        )
        self.addCleanup(core.close)
        return core

    def test_learn_produces_validated_memories_with_provenance(self):
        core = self._core()
        observation = core.learn(
            "我叫阿哲，主要用 Python 和 Rust，最近在做一个叫 Memory Core 的本地优先项目。"
        )

        # The raw turn is always kept, whatever the model decided.
        self.assertEqual(core.store.count_events(), 1)

        for memory in observation.memories:
            self.assertLess(memory.confidence, 1.0, "an LLM proposal is never a fact")
            self.assertEqual(memory.source["origin"], "llm")
            self.assertEqual(memory.source["event_ids"], [observation.event.id])
            self.assertEqual(memory.generated_by, "ollama:{0}".format(MODEL))
            self.assertIn(memory.kind, ("semantic", "episodic", "identity"))
            self.assertTrue(memory.content.strip())

        for outcome in observation.rejected:
            self.assertTrue(outcome.reason)

    def test_a_model_failure_never_pollutes_the_database(self):
        core = self._core()
        # Point at a closed port so the call cannot succeed.
        core.llm_extractor.provider = OllamaProvider(
            model=MODEL, base_url="http://127.0.0.1:1/v1", timeout=2.0
        )
        observation = core.learn("这件事很重要，请记住")

        self.assertEqual(core.store.count_events(), 1)
        self.assertEqual(core.store.count_memories(), 0)
        self.assertIn("provider unavailable", observation.skipped_reason)

    def test_propose_with_a_real_model_writes_nothing(self):
        core = self._core()
        core.remember("an existing memory about the project")
        before = digest(self.db_path)

        proposal = core.propose("我在重构 Memory Core 的检索层")

        self.assertEqual(digest(self.db_path), before)
        self.assertEqual(core.store.count_events(), 0)
        self.assertEqual(core.store.count_memories(), 1)
        self.assertTrue(isinstance(proposal.as_dict(), dict))


if __name__ == "__main__":
    unittest.main()
