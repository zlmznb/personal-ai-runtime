"""Vault scanning, glob matching and frontmatter parsing."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from memory_bridge.obsidian.vault import (
    match_glob,
    is_included,
    parse_frontmatter,
    parse_yaml_subset,
    scan_vault,
)

FIXTURE_VAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures",
    "obsidian_vault",
)


class GlobTests(unittest.TestCase):
    def test_double_star_matches_zero_segments(self):
        self.assertTrue(match_glob("a.md", "**/*.md"))
        self.assertTrue(match_glob("notes/deep/a.md", "**/*.md"))

    def test_single_star_does_not_cross_separators(self):
        # A pattern *with* a slash is matched against the whole path, and `*`
        # must not swallow a directory boundary.
        self.assertTrue(match_glob("notes/a.md", "notes/*.md"))
        self.assertFalse(match_glob("notes/deep/a.md", "notes/*.md"))

    def test_basename_pattern_matches_at_any_depth(self):
        self.assertTrue(match_glob("notes/deep/a.md.bak", "*.md.bak"))
        self.assertTrue(match_glob("a.md", "*.md"))

    def test_directory_prefix(self):
        self.assertTrue(match_glob(".obsidian/app.json", ".obsidian/**"))
        self.assertTrue(match_glob("Private/scratch.md", "Private/**"))
        self.assertFalse(match_glob("Public/scratch.md", "Private/**"))

    def test_leading_dot_is_not_stripped_from_a_path(self):
        # Regression: str.lstrip("./") would turn ".obsidian/app.json" into
        # "obsidian/app.json" and silently break hidden-directory exclusion.
        self.assertFalse(match_glob(".obsidian/app.json", "obsidian/**"))
        self.assertTrue(match_glob("./.obsidian/app.json", ".obsidian/**"))

    def test_capture_form(self):
        self.assertTrue(match_glob("Projects/x/State.md", "Projects/*/**"))
        self.assertTrue(match_glob("Projects/x.md", "Projects/*"))
        self.assertFalse(match_glob("Personas/x.md", "Projects/*"))

    def test_case_insensitive(self):
        self.assertTrue(match_glob("private/a.md", "Private/**"))


class InclusionTests(unittest.TestCase):
    def test_markdown_is_included_by_default(self):
        self.assertTrue(is_included("notes/a.md"))

    def test_non_markdown_is_excluded(self):
        self.assertFalse(is_included("notes/a.txt"))

    def test_exclude_wins_over_include(self):
        self.assertFalse(is_included("Private/a.md", exclude=("Private/**",)))
        self.assertTrue(is_included("Public/a.md", exclude=("Private/**",)))


class FrontmatterTests(unittest.TestCase):
    def test_scalars_and_lists(self):
        parsed = parse_yaml_subset(
            "title: Hello\ntags: [a, b]\ncount: 3\nratio: 1.5\nflag: true\nnothing: null"
        )
        self.assertEqual(parsed["title"], "Hello")
        self.assertEqual(parsed["tags"], ["a", "b"])
        self.assertEqual(parsed["count"], 3)
        self.assertEqual(parsed["ratio"], 1.5)
        self.assertTrue(parsed["flag"])
        self.assertIsNone(parsed["nothing"])

    def test_quoted_values_and_comments(self):
        parsed = parse_yaml_subset('title: "a: b" # trailing\nother: plain # comment')
        self.assertEqual(parsed["title"], "a: b")
        self.assertEqual(parsed["other"], "plain")

    def test_block_list(self):
        parsed = parse_yaml_subset("tags:\n  - alpha\n  - beta\n")
        self.assertEqual(parsed["tags"], ["alpha", "beta"])

    def test_document_without_frontmatter(self):
        frontmatter, body = parse_frontmatter("# Just a heading\n\ntext")
        self.assertEqual(frontmatter, {})
        self.assertIn("Just a heading", body)

    def test_document_with_frontmatter(self):
        frontmatter, body = parse_frontmatter("---\ntitle: T\n---\n\nbody here")
        self.assertEqual(frontmatter["title"], "T")
        self.assertIn("body here", body)
        self.assertNotIn("title: T", body)

    def test_unterminated_frontmatter_is_treated_as_body(self):
        frontmatter, body = parse_frontmatter("---\ntitle: T\n\nbody")
        self.assertEqual(frontmatter, {})
        self.assertIn("body", body)

    def test_inline_json_state_survives_as_a_string(self):
        parsed = parse_yaml_subset('state: {"phase": "v0.3"}')
        self.assertEqual(parsed["state"], '{"phase": "v0.3"}')

    def test_parser_never_raises_on_junk(self):
        for junk in ("", ":::", "no colon here", "a:\n  b: c\n", "- orphan"):
            parse_yaml_subset(junk)


class ScanTests(unittest.TestCase):
    def test_scans_the_fixture_vault(self):
        documents = list(scan_vault(FIXTURE_VAULT))
        paths = sorted(document.rel_path for document in documents)
        self.assertIn("Knowledge/Local First Architecture.md", paths)
        self.assertIn("Inbox/Remember This.md", paths)
        self.assertIn("Projects/persona-edgeaiot/State.md", paths)
        self.assertIn("Preferences/Editor.md", paths)
        self.assertIn("Personas/aria/Notes.md", paths)
        self.assertIn("Personas/bruno/Notes.md", paths)

    def test_excluded_paths_are_not_scanned(self):
        # Exclusion is config-driven: scan_vault on its own only applies the
        # defaults, so the vault's memory-bridge.json must be loaded.
        from memory_bridge.config import BridgeConfig

        bridge = BridgeConfig.load(FIXTURE_VAULT)
        paths = [
            document.rel_path
            for document in scan_vault(
                FIXTURE_VAULT, include=bridge.include, exclude=bridge.exclude
            )
        ]
        self.assertNotIn("Private/scratch.md", paths)
        self.assertFalse([path for path in paths if path.startswith(".obsidian/")])
        self.assertEqual(len(paths), 6)

    def test_defaults_alone_do_not_exclude_a_custom_folder(self):
        # Documents the boundary: Private/** is the vault's own rule.
        paths = [document.rel_path for document in scan_vault(FIXTURE_VAULT)]
        self.assertIn("Private/scratch.md", paths)
        # ...but hidden directories are excluded by the built-in defaults.
        self.assertFalse([path for path in paths if path.startswith(".obsidian/")])

    def test_scan_is_deterministically_ordered(self):
        first = [document.rel_path for document in scan_vault(FIXTURE_VAULT)]
        second = [document.rel_path for document in scan_vault(FIXTURE_VAULT)]
        self.assertEqual(first, second)

    def test_frontmatter_metadata_is_parsed(self):
        documents = {d.rel_path: d for d in scan_vault(FIXTURE_VAULT)}
        state = documents["Projects/persona-edgeaiot/State.md"]
        self.assertEqual(state.frontmatter["type"], "project-state")
        self.assertEqual(state.frontmatter["project"], "persona-edgeaiot")
        self.assertTrue(state.mtime.endswith("Z"))

    def test_title_falls_back_to_first_heading(self):
        documents = {d.rel_path: d for d in scan_vault(FIXTURE_VAULT)}
        self.assertEqual(documents["Inbox/Remember This.md"].title, "Remember This")

    def test_vault_is_not_modified_by_scanning(self):
        tmpdir = tempfile.mkdtemp(prefix="bridge-scan-")
        self.addCleanup(shutil.rmtree, tmpdir, True)
        target = os.path.join(tmpdir, "vault")
        shutil.copytree(FIXTURE_VAULT, target)
        before = _tree_state(target)
        list(scan_vault(target))
        self.assertEqual(_tree_state(target), before)


def _tree_state(root):
    # type: (str) -> dict
    state = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            state[os.path.relpath(path, root)] = os.path.getsize(path)
    return state


if __name__ == "__main__":
    unittest.main()
