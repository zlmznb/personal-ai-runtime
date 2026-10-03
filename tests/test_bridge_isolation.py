"""Persona isolation and Bridge purity - the two security-critical properties.

1. **Fail-closed persona isolation.** With no scope arguments, ``persona:*`` must
   never be returned, and one persona must never see another's memories - not in
   retrieval, and not through the formation prompt.
2. **Bridge purity.** ``memory_bridge`` may only use Memory Core's public API.
"""

from __future__ import annotations

import ast
import glob
import os
import shutil
import tempfile
import unittest

from memory_bridge.config import BridgeConfig
from memory_bridge.obsidian.importer import ObsidianImporter
from memory_core.api import MemoryCore
from memory_core.domain.scopes import (
    ALL_VISIBILITY,
    DEFAULT_VISIBILITY,
    Visibility,
    is_persona_scope,
    persona_scope,
    resolve_visibility,
    visibility_for_scope,
)
from memory_core.providers.fake import FakeProvider

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_ROOT = os.path.join(PROJECT_ROOT, "memory_bridge")
FIXTURE_VAULT = os.path.join(PROJECT_ROOT, "fixtures", "obsidian_vault")

#: The only Memory Core modules a bridge may import.
ALLOWED_MEMORY_CORE = (
    "memory_core.api",
    "memory_core.domain.dto",
    "memory_core.domain.errors",
    "memory_core.providers.base",
)

#: Never allowed anywhere in the bridge.
FORBIDDEN_ROOTS = {
    "sqlite3", "urllib", "urllib2", "http", "socket", "ssl", "requests", "httpx",
    "aiohttp", "openai", "anthropic", "ollama", "litellm", "langchain",
    "langgraph", "mem0", "chromadb", "qdrant_client", "numpy",
}


class RecordingProvider(FakeProvider):
    """Records every prompt so leakage can be inspected directly."""

    def __init__(self, **kwargs):
        super(RecordingProvider, self).__init__(**kwargs)
        self.prompts = []

    def chat(self, messages, **options):
        self.prompts.append([message.content for message in messages])
        return super(RecordingProvider, self).chat(messages, **options)

    @property
    def transcript(self):
        # type: () -> str
        return "\n".join(
            content for prompt in self.prompts for content in prompt
        )


class VisibilityUnitTests(unittest.TestCase):
    def test_default_excludes_persona(self):
        visibility = resolve_visibility()
        self.assertIsNone(visibility.scopes)
        self.assertIn("persona", visibility.exclude_prefixes)
        self.assertTrue(visibility.allows("global"))
        self.assertTrue(visibility.allows("project:x"))
        self.assertFalse(visibility.allows("persona:aria"))

    def test_explicit_single_scope_is_exact(self):
        visibility = resolve_visibility(scope="global")
        self.assertEqual(visibility.scopes, ("global",))
        self.assertFalse(visibility.allows("project:x"))
        self.assertFalse(visibility.allows("persona:aria"))

    def test_scope_list_is_exact(self):
        visibility = resolve_visibility(scopes=("global", "project:x"))
        self.assertTrue(visibility.allows("global"))
        self.assertTrue(visibility.allows("project:x"))
        self.assertFalse(visibility.allows("persona:aria"))

    def test_persona_expands_to_global_plus_itself(self):
        visibility = resolve_visibility(persona="aria")
        self.assertEqual(set(visibility.scopes), {"global", "persona:aria"})
        self.assertFalse(visibility.allows("persona:bruno"))

    def test_scopes_plus_persona_are_merged(self):
        visibility = resolve_visibility(scopes=("global", "project:x"), persona="aria")
        self.assertEqual(set(visibility.scopes), {"global", "project:x", "persona:aria"})

    def test_all_visibility_allows_everything(self):
        self.assertTrue(ALL_VISIBILITY.allows("persona:bruno"))

    def test_default_visibility_is_fail_closed(self):
        self.assertFalse(DEFAULT_VISIBILITY.allows("persona:anything"))

    def test_visibility_for_a_persona_write_scope(self):
        visibility = visibility_for_scope("persona:aria")
        self.assertEqual(set(visibility.scopes), {"global", "persona:aria"})

    def test_visibility_for_a_normal_write_scope(self):
        visibility = visibility_for_scope("project:x")
        self.assertEqual(visibility.scopes, ("project:x",))

    def test_helpers(self):
        self.assertEqual(persona_scope("aria"), "persona:aria")
        self.assertTrue(is_persona_scope("persona:aria"))
        self.assertFalse(is_persona_scope("global"))

    def test_allows_rejects_non_strings(self):
        self.assertFalse(DEFAULT_VISIBILITY.allows(None))
        self.assertFalse(Visibility().allows(42))


class IsolationTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="bridge-isolation-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")
        self.vault = os.path.join(self.tmpdir, "vault")
        shutil.copytree(FIXTURE_VAULT, self.vault)
        self.config = BridgeConfig.load(self.vault)

    def open_core(self, persona=None, provider=None, use_llm=False):
        core = MemoryCore.open(
            db_path=self.db_path,
            persona=persona,
            provider=provider,
            use_llm_formation=use_llm,
        )
        self.addCleanup(core.close)
        return core


class RecallIsolationTests(IsolationTestCase):
    def seed_personas(self, core):
        core.learn("记住：Aria 的秘密是先给结论。", scope="persona:aria")
        core.learn("记住：Bruno 的秘密是绝不外传。", scope="persona:bruno")
        core.learn("记住：全局记忆对所有人可见。", scope="global")

    def test_default_recall_never_returns_persona_memories(self):
        core = self.open_core()
        self.seed_personas(core)

        # The global memory is visible...
        global_hits = [hit.item.content for hit in core.recall("全局记忆", limit=50)]
        self.assertTrue(global_hits)

        # ...and a query that matches persona memories exactly returns nothing,
        # because persona scopes are excluded by default.
        secret_hits = [hit.item.content for hit in core.recall("秘密", limit=50)]
        self.assertEqual(secret_hits, [])

    def test_persona_a_cannot_read_persona_b(self):
        core = self.open_core()
        self.seed_personas(core)

        aria = core.recall("秘密", persona="aria", limit=50)
        aria_text = " ".join(hit.item.content for hit in aria)
        self.assertIn("Aria", aria_text)
        self.assertNotIn("Bruno", aria_text)

        bruno = core.recall("秘密", persona="bruno", limit=50)
        bruno_text = " ".join(hit.item.content for hit in bruno)
        self.assertIn("Bruno", bruno_text)
        self.assertNotIn("Aria", bruno_text)

    def test_persona_sees_global_memory(self):
        core = self.open_core()
        self.seed_personas(core)
        hits = core.recall("全局记忆", persona="aria", limit=50)
        self.assertTrue(hits)

    def test_session_persona_applies_without_arguments(self):
        core = self.open_core(persona="aria")
        self.seed_personas(core)
        text = " ".join(hit.item.content for hit in core.recall("秘密", limit=50))
        self.assertIn("Aria", text)
        self.assertNotIn("Bruno", text)

    def test_list_memories_excludes_personas_by_default(self):
        core = self.open_core()
        self.seed_personas(core)
        scopes = {memory.scope for memory in core.list_memories(limit=100)}
        self.assertNotIn("persona:aria", scopes)
        self.assertNotIn("persona:bruno", scopes)

    def test_list_memories_can_include_personas_explicitly(self):
        core = self.open_core()
        self.seed_personas(core)
        scopes = {
            memory.scope for memory in core.list_memories(include_personas=True, limit=100)
        }
        self.assertIn("persona:aria", scopes)
        self.assertIn("persona:bruno", scopes)

    def test_preferences_follow_the_same_rule(self):
        core = self.open_core()
        core.set_preference("theme", "dark", scope="persona:aria")
        core.set_preference("theme", "light", scope="persona:bruno")
        core.set_preference("theme", "system", scope="global")

        visible = {p.scope for p in core.list_preferences()}
        self.assertEqual(visible, {"global"})
        self.assertEqual(len(core.list_preferences(scopes=("persona:aria",))), 1)

    def test_direct_scope_lookup_is_still_exact(self):
        core = self.open_core()
        self.seed_personas(core)
        text = " ".join(
            hit.item.content for hit in core.recall("秘密", scope="persona:aria", limit=50)
        )
        self.assertIn("Aria", text)
        self.assertNotIn("Bruno", text)


class FormationContextIsolationTests(IsolationTestCase):
    """The leak must not simply move from retrieval into the prompt."""

    def test_persona_b_prompt_never_contains_persona_a_memory(self):
        provider = RecordingProvider(reply='{"memories": []}')
        core = self.open_core(provider=provider, use_llm=True)
        core.learn("记住：Aria 的私有偏好是简洁。", scope="persona:aria")

        provider.prompts.clear()
        core.learn("记住：Bruno 的私有偏好是详细。", scope="persona:bruno")

        self.assertTrue(provider.prompts, "the model was never called")
        self.assertIn("Bruno", provider.transcript)
        self.assertNotIn("Aria", provider.transcript)

    def test_global_memory_is_offered_to_a_persona(self):
        provider = RecordingProvider(reply='{"memories": []}')
        core = self.open_core(provider=provider, use_llm=True)
        core.learn("记住：这是一个全局的偏好声明。", scope="global")

        provider.prompts.clear()
        core.learn("记住：Aria 的私有偏好是简洁。", scope="persona:aria")

        self.assertIn("全局的偏好声明", provider.transcript)

    def test_a_persona_write_does_not_see_another_persona_via_retrieval_either(self):
        provider = RecordingProvider(reply='{"memories": []}')
        core = self.open_core(provider=provider, use_llm=True)
        core.learn("记住：Aria 的私有标记词是 zarquon。", scope="persona:aria")

        provider.prompts.clear()
        core.learn("请总结一下我的情况。", scope="persona:bruno")
        self.assertNotIn("zarquon", provider.transcript)


class BridgeEndToEndIsolationTests(IsolationTestCase):
    def test_fixture_vault_personas_do_not_leak(self):
        core = self.open_core()
        ObsidianImporter(core, self.config).run(apply=True)

        # A query that matches BOTH persona declarations.
        default_text = " ".join(hit.item.content for hit in core.recall("结论", limit=50))
        self.assertNotIn("Aria", default_text)
        self.assertNotIn("Bruno", default_text)

        aria_text = " ".join(
            hit.item.content for hit in core.recall("结论", persona="aria", limit=50)
        )
        self.assertIn("Aria", aria_text)
        self.assertNotIn("Bruno", aria_text)

        bruno_text = " ".join(
            hit.item.content for hit in core.recall("私有偏好", persona="bruno", limit=50)
        )
        self.assertIn("Bruno", bruno_text)
        self.assertNotIn("Aria", bruno_text)


class PurityTests(unittest.TestCase):
    """The bridge must use the public API only."""

    def _bridge_files(self):
        return sorted(glob.glob(os.path.join(BRIDGE_ROOT, "**", "*.py"), recursive=True))

    @staticmethod
    def _imports(path):
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(node.module)
        return modules

    def test_bridge_has_files(self):
        self.assertGreater(len(self._bridge_files()), 5)

    def test_no_forbidden_imports(self):
        for path in self._bridge_files():
            for module in self._imports(path):
                root = module.split(".")[0]
                self.assertNotIn(
                    root, FORBIDDEN_ROOTS,
                    "{0} imports {1!r}".format(os.path.relpath(path, PROJECT_ROOT), module),
                )

    def test_only_the_public_core_api_is_imported(self):
        for path in self._bridge_files():
            for module in self._imports(path):
                if module != "memory_core" and not module.startswith("memory_core."):
                    continue
                allowed = module == "memory_core" or any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in ALLOWED_MEMORY_CORE
                )
                self.assertTrue(
                    allowed,
                    "{0} imports {1!r}; only {2} are permitted".format(
                        os.path.relpath(path, PROJECT_ROOT), module, list(ALLOWED_MEMORY_CORE)
                    ),
                )

    def test_bridge_never_imports_storage_directly(self):
        for path in self._bridge_files():
            for module in self._imports(path):
                self.assertFalse(module.startswith("memory_core.storage"))
                self.assertFalse(module.startswith("memory_core.retrieval"))
                self.assertFalse(module.startswith("memory_core.formation"))


if __name__ == "__main__":
    unittest.main()
