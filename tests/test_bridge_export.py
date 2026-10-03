"""Export: the markdown mirror is a one-way view, never a second database."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest

from memory_bridge.config import BridgeConfig
from memory_bridge.obsidian.exporter import (
    GENERATOR_MARKER,
    README_NAME,
    ObsidianExporter,
    scope_slug,
)
from memory_bridge.obsidian.importer import ObsidianImporter
from memory_core.api import MemoryCore

FIXTURE_VAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures",
    "obsidian_vault",
)

FIXED_NOW = lambda: "2026-10-03T00:00:00.000Z"  # noqa: E731


class ExportTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="bridge-export-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.vault = os.path.join(self.tmpdir, "vault")
        shutil.copytree(FIXTURE_VAULT, self.vault)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")
        self.config = BridgeConfig.load(self.vault)

    def populated_core(self):
        core = MemoryCore.open(db_path=self.db_path)
        self.addCleanup(core.close)
        ObsidianImporter(core, self.config).run(apply=True)
        return core

    def read(self, name):
        # type: (str) -> str
        path = os.path.join(self.config.export_path(), name)
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()


class LayoutTests(ExportTestCase):
    def test_creates_readme_and_one_file_per_scope(self):
        core = self.populated_core()
        report = ObsidianExporter(core, self.config, now=FIXED_NOW).export()

        names = sorted(os.path.basename(path) for path in report.files)
        self.assertIn(README_NAME, names)
        self.assertIn("global.md", names)
        self.assertIn("project-persona-edgeaiot.md", names)
        self.assertIn("persona-aria.md", names)
        self.assertIn("persona-bruno.md", names)
        self.assertEqual(report.scopes, ("global", "persona:aria", "persona:bruno", "project:persona-edgeaiot"))

    def test_slugs(self):
        self.assertEqual(scope_slug("project:persona-edgeaiot"), "project-persona-edgeaiot")
        self.assertEqual(scope_slug("global"), "global")
        self.assertEqual(scope_slug("persona:aria"), "persona-aria")

    def test_readme_explains_that_edits_are_not_remembered(self):
        core = self.populated_core()
        ObsidianExporter(core, self.config, now=FIXED_NOW).export()
        text = self.read(README_NAME)
        self.assertIn("read-only mirror", text)
        self.assertIn("Edits here are not remembered", text)


class MarkerTests(ExportTestCase):
    def test_scope_files_are_marked_as_generated(self):
        core = self.populated_core()
        ObsidianExporter(core, self.config, now=FIXED_NOW).export()
        text = self.read("global.md")
        self.assertIn("generated: true", text)
        self.assertIn(GENERATOR_MARKER, text)
        self.assertIn("source: memory.sqlite", text)

    def test_memory_blocks_carry_id_provenance_and_scope(self):
        core = self.populated_core()
        ObsidianExporter(core, self.config, now=FIXED_NOW).export()
        text = self.read("global.md")

        self.assertIn("<!-- memory ", text)
        self.assertIn("<!-- /memory -->", text)
        for field in ("id=mem_", "kind=", "confidence=", "created=", "scope="):
            self.assertIn(field, text)

    def test_bridge_derived_memories_show_source_ref(self):
        core = self.populated_core()
        ObsidianExporter(core, self.config, now=FIXED_NOW).export()
        text = self.read("global.md")
        self.assertIn("source_ref=obsidian:demo-vault/", text)

    def test_preferences_and_project_states_are_exported(self):
        core = self.populated_core()
        ObsidianExporter(core, self.config, now=FIXED_NOW).export()
        self.assertIn("<!-- preference ", self.read("global.md"))
        self.assertIn("<!-- project-state ", self.read("project-persona-edgeaiot.md"))

    def test_persona_memories_land_in_a_persona_file(self):
        core = self.populated_core()
        ObsidianExporter(core, self.config, now=FIXED_NOW).export()
        aria = self.read("persona-aria.md")
        self.assertIn("scope: persona:aria", aria)
        self.assertIn("Aria 回答时先给结论", aria)
        # And not in the global mirror.
        self.assertNotIn("Aria 回答时先给结论", self.read("global.md"))


class DeterminismTests(ExportTestCase):
    def test_two_exports_are_byte_identical(self):
        core = self.populated_core()
        exporter = ObsidianExporter(core, self.config, now=FIXED_NOW)
        exporter.export()
        first = {
            name: hashlib.sha256(self.read(name).encode("utf-8")).hexdigest()
            for name in os.listdir(self.config.export_path())
        }
        exporter.export()
        second = {
            name: hashlib.sha256(self.read(name).encode("utf-8")).hexdigest()
            for name in os.listdir(self.config.export_path())
        }
        self.assertEqual(first, second)


class OneWayTests(ExportTestCase):
    def test_export_folder_is_excluded_from_import(self):
        core = self.populated_core()
        ObsidianExporter(core, self.config, now=FIXED_NOW).export()
        before_events = core.store.count_events()

        report = ObsidianImporter(core, self.config).run(apply=True)
        self.assertEqual(report.counts["events_created"], 0)
        self.assertEqual(core.store.count_events(), before_events)

    def test_editing_the_mirror_creates_no_memory(self):
        core = self.populated_core()
        ObsidianExporter(core, self.config, now=FIXED_NOW).export()
        memories_before = core.store.count_memories()

        path = os.path.join(self.config.export_path(), "global.md")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n记住：这条人类写的声明不应该被导入。\n")

        ObsidianImporter(core, self.config).run(apply=True)
        self.assertEqual(core.store.count_memories(), memories_before)

        # ...and the next export overwrites the edit.
        ObsidianExporter(core, self.config, now=FIXED_NOW).export()
        self.assertNotIn("这条人类写的声明不应该被导入", self.read("global.md"))

    def test_deleting_the_mirror_loses_nothing(self):
        core = self.populated_core()
        ObsidianExporter(core, self.config, now=FIXED_NOW).export()
        export_dir = self.config.export_path()

        shutil.rmtree(export_dir)
        self.assertFalse(os.path.exists(export_dir))

        self.assertEqual(
            len(core.list_memories(include_personas=True, limit=1000)), 3
        )
        ObsidianExporter(core, self.config, now=FIXED_NOW).export()
        self.assertTrue(os.path.exists(os.path.join(export_dir, "global.md")))


class PruneTests(ExportTestCase):
    def test_stale_generated_files_are_removed(self):
        core = self.populated_core()
        exporter = ObsidianExporter(core, self.config, now=FIXED_NOW)
        exporter.export()

        stale = os.path.join(self.config.export_path(), "persona-gone.md")
        with open(stale, "w", encoding="utf-8") as handle:
            handle.write("---\n{G0} 0.3.0\n---\n\nold\n".format(G0=GENERATOR_MARKER))

        report = exporter.export()
        self.assertIn(stale, report.pruned)
        self.assertFalse(os.path.exists(stale))

    def test_human_files_in_the_export_folder_are_never_pruned(self):
        core = self.populated_core()
        exporter = ObsidianExporter(core, self.config, now=FIXED_NOW)
        exporter.export()

        human = os.path.join(self.config.export_path(), "my-own-note.md")
        with open(human, "w", encoding="utf-8") as handle:
            handle.write("# A note I wrote myself\n\nNot generated by anything.\n")

        report = exporter.export()
        self.assertNotIn(human, report.pruned)
        self.assertTrue(os.path.exists(human))


if __name__ == "__main__":
    unittest.main()
