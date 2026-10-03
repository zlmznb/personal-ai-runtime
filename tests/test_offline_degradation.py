"""Architectural invariants, enforced as tests.

These are the rules from docs/architecture.md that must not erode as the
project grows. Each one is checked mechanically rather than by convention.
"""

from __future__ import annotations

import ast
import glob
import os
import shutil
import tempfile
import unittest

from memory_core.api import MemoryCore
from memory_core.domain import guards
from memory_core.providers.deepseek import DeepSeekProvider
from memory_core.providers.fake import FakeProvider
from memory_core.providers.ollama import OllamaProvider
from memory_core.storage.sqlite_store import DERIVED_TABLES, SOURCE_TABLES, SqliteStore

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_CORE = os.path.join(PACKAGE_ROOT, "memory_core")

#: Import roots that must never appear below the provider layer.
FORBIDDEN_IMPORTS = {
    "urllib", "urllib2", "http", "socket", "ssl", "requests", "httpx",
    "aiohttp", "openai", "anthropic", "ollama", "litellm", "langchain",
    "langchain_core", "langgraph", "mem0", "chromadb", "qdrant_client",
    "pandas",
}

#: The one optional third-party import permitted in the retrieval layer.
#: It is a math accelerator, never required: the code path is exercised
#: without it in ``tests/test_vector_index.py``.
OPTIONAL_ACCELERATION = {"numpy"}

#: Substrings that must not appear in any column name. The one permitted
#: exception in the schema is `generated_by`, which is free text.
VENDOR_TOKENS = (
    "openai", "deepseek", "ollama", "llama", "gpt", "claude", "anthropic",
    "litellm", "mem0", "langchain", "langgraph", "chroma", "qdrant", "gemini",
)


def python_files(package):
    # type: (str) -> list
    return sorted(glob.glob(os.path.join(MEMORY_CORE, package, "**", "*.py"), recursive=True))


def imported_modules(path):
    # type: (str) -> set
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
            for alias in node.names:
                modules.add(alias.name)
    return modules


class ImportBoundaryTests(unittest.TestCase):
    """Red line 2: the data layers must not know models or HTTP exist."""

    def _assert_clean(self, package, allow_provider_base=False, allow_acceleration=False):
        files = python_files(package)
        self.assertTrue(files, "no files found for package {0}".format(package))
        for path in files:
            for module in imported_modules(path):
                root = module.split(".")[0]
                if root in OPTIONAL_ACCELERATION and allow_acceleration:
                    continue
                self.assertNotIn(
                    root, FORBIDDEN_IMPORTS | OPTIONAL_ACCELERATION,
                    "{0} imports {1!r}".format(os.path.relpath(path, PACKAGE_ROOT), module),
                )
                if module.startswith("memory_core.providers"):
                    if allow_provider_base:
                        self.assertIn(
                            module, ("memory_core.providers.base",),
                            "{0} may only import the provider interface, not {1!r}".format(
                                os.path.relpath(path, PACKAGE_ROOT), module
                            ),
                        )
                    else:
                        self.fail("{0} imports provider module {1!r}".format(
                            os.path.relpath(path, PACKAGE_ROOT), module
                        ))

    def test_domain_is_pure(self):
        self._assert_clean("domain")

    def test_storage_is_pure(self):
        # Storage must not even reach for a maths accelerator: it stores bytes.
        self._assert_clean("storage")

    def test_retrieval_is_pure(self):
        # ADR 0003: retrieval never consults a *chat* model, and never performs
        # HTTP. It may import the provider interface (a pure contract) and numpy
        # as an optional accelerator.
        self._assert_clean("retrieval", allow_provider_base=True, allow_acceleration=True)

    def test_formation_does_not_import_http(self):
        self._assert_clean("formation", allow_provider_base=True)

    def test_only_providers_may_import_http(self):
        # The inverse: HTTP must live somewhere, and it must be providers/.
        http_users = []
        for path in glob.glob(os.path.join(MEMORY_CORE, "**", "*.py"), recursive=True):
            for module in imported_modules(path):
                if module.split(".")[0] in ("urllib", "http", "socket", "ssl"):
                    http_users.append(os.path.relpath(path, PACKAGE_ROOT))
        self.assertTrue(http_users, "expected the provider layer to be the HTTP boundary")
        for path in http_users:
            self.assertTrue(
                path.replace("\\", "/").startswith("memory_core/providers/"),
                "{0} performs HTTP outside the provider layer".format(path),
            )


class SchemaPurityTests(unittest.TestCase):
    """Constraint P0-4: no model concept may leak into the schema."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-offline-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")
        self.store = SqliteStore(self.db_path)
        self.addCleanup(self.store.close)

    def _columns(self):
        connection = self.store.connection
        columns = []
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        for table in tables:
            name = table["name"]
            if name.startswith("sqlite_"):
                continue
            for column in connection.execute("PRAGMA table_info({0})".format(name)).fetchall():
                columns.append((name, column["name"]))
        return columns

    def test_no_vendor_token_in_any_column_name(self):
        for table, column in self._columns():
            lowered = column.lower()
            for token in VENDOR_TOKENS:
                self.assertNotIn(
                    token, lowered,
                    "column {0}.{1} leaks a vendor concept".format(table, column),
                )

    def test_generated_by_is_the_only_model_related_column(self):
        names = [column for _, column in self._columns()]
        self.assertIn("generated_by", names)

    def test_source_tables_are_exactly_the_expected_set(self):
        tables = sorted({
            table for table, _ in self._columns()
            if "_fts" not in table
        })
        self.assertEqual(
            tables,
            sorted(list(SOURCE_TABLES) + ["embeddings"]),
        )

    def test_source_and_derived_tables_are_disjoint(self):
        names = {table for table, _ in self._columns()}
        for table in SOURCE_TABLES:
            self.assertIn(table, names)
        for table in DERIVED_TABLES:
            self.assertIn(table, names)
        self.assertEqual(set(SOURCE_TABLES) & set(DERIVED_TABLES), set())

    def test_derived_tables_are_excluded_from_export(self):
        # Export must round-trip only the truth, never a rebuildable index.
        memory = guards.build_memory("exported memory")
        self.store.insert_memory(memory)
        self.store.upsert_embedding(memory.id, "model-a", 2, b"\x00" * 8)
        exported = self.store.export_jsonl()
        for table in DERIVED_TABLES:
            self.assertNotIn('"table": "{0}"'.format(table), exported)
        # Only tables that actually contain rows appear in the export.
        self.assertIn('"table": "memories"', exported)
        self.assertIn('"table": "meta"', exported)
        self.assertIn(memory.id, exported)

    def test_no_model_column_anywhere_says_which_vendor(self):
        # `generated_by` must be free text: nothing in the schema enumerates vendors.
        schema = self.store.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        text = " ".join(row["sql"] or "" for row in schema).lower()
        for token in VENDOR_TOKENS:
            self.assertNotIn(token, text)

    def test_the_database_is_the_only_artefact(self):
        self.store.insert_memory(guards.build_memory("x"))
        leftovers = [n for n in os.listdir(self.tmpdir) if n != "memory.sqlite"]
        self.assertEqual(leftovers, [])


class OfflineOperationTests(unittest.TestCase):
    """Red line 5: everything except `ask` works with no model at all."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-offline-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")

    def _core(self, provider=None):
        core = MemoryCore.open(db_path=self.db_path, provider=provider)
        self.addCleanup(core.close)
        return core

    def test_full_workflow_with_no_provider_configured(self):
        core = self._core()
        self.assertIsNone(core.provider)

        core.observe("记住：用户偏好本地优先")
        memory = core.remember("model swap must not lose memory", tags=["requirement"])
        core.set_preference("theme", "dark")
        core.set_project_state("memory", {"phase": "v0.1"})

        self.assertEqual(len(core.recall("本地优先")), 1)
        self.assertEqual(core.get_preference("theme").value, "dark")
        self.assertEqual(core.get_project_state("memory").state, {"phase": "v0.1"})
        self.assertTrue(core.forget(memory.id))
        self.assertTrue(core.export_jsonl())
        self.assertTrue(core.stats()["events"] >= 1)

    def test_every_configured_provider_failing_leaves_data_intact(self):
        core = self._core()
        core.remember("survives total model outage")
        core.close()

        before = os.path.getsize(self.db_path)
        for provider in (
            FakeProvider(fail_with="down"),
            OllamaProvider(base_url="http://127.0.0.1:1/v1", timeout=1.0),
            DeepSeekProvider(api_key=None, api_key_env="MEMORY_CORE_ABSENT_KEY"),
        ):
            core = self._core(provider=provider)
            answer = core.ask("survives")
            self.assertTrue(answer.degraded)
            self.assertIn("survives total model outage", answer.text)
            core.close()
        self.assertEqual(os.path.getsize(self.db_path), before)

    def test_derived_index_is_disposable_and_rebuildable(self):
        core = self._core()
        core.remember("用户偏好本地优先")
        baseline = [hit.id for hit in core.recall("本地优先")]

        core.store.rebuild_fts()
        self.assertEqual([hit.id for hit in core.recall("本地优先")], baseline)

    def test_history_is_never_destroyed_by_normal_operation(self):
        core = self._core()
        original = core.remember("theme is dark")
        core.supersede_memory(original.id, "theme is light")
        core.forget(original.id)  # logical delete of an already-superseded row

        exported = core.export_jsonl()
        self.assertIn(original.id, exported)
        self.assertEqual(core.store.count_memories(), 2)


if __name__ == "__main__":
    unittest.main()
