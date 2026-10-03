"""CLI tests.

The CLI is a deliverable, so it is covered like one: every command is invoked
through ``main(argv)`` and its output and exit code are asserted.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from memory_core.cli import main
from memory_core.storage.sqlite_store import SCHEMA_VERSION


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memory-core-cli-")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.db_path = os.path.join(self.tmpdir, "memory.sqlite")

    def run_cli(self, *argv):
        # type: (str) -> tuple
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(["--db", self.db_path] + list(argv))
            except SystemExit as exc:
                # argparse exits directly on a usage error; that is a real
                # process exit code and the tests should see it.
                code = int(exc.code) if exc.code is not None else 0
        return code, stdout.getvalue(), stderr.getvalue()


class BasicCommandTests(CliTestCase):
    def test_no_command_prints_help_and_succeeds(self):
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("Memory Core", out)

    def test_init(self):
        code, out, _ = self.run_cli("init")
        self.assertEqual(code, 0)
        self.assertIn("schema v{0}".format(SCHEMA_VERSION), out)
        self.assertTrue(os.path.exists(self.db_path))

    def test_remember_then_recall(self):
        self.run_cli("remember", "用户偏好本地优先", "--tag", "privacy")
        code, out, _ = self.run_cli("recall", "本地优先")
        self.assertEqual(code, 0)
        self.assertIn("用户偏好本地优先", out)
        self.assertIn("memory", out)

    def test_recall_with_no_matches(self):
        code, out, _ = self.run_cli("recall", "zzzznothing")
        self.assertEqual(code, 0)
        self.assertIn("no matches", out)

    def test_recall_json_output(self):
        self.run_cli("remember", "a searchable statement")
        code, out, _ = self.run_cli("--json", "recall", "searchable")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["mode"], "hybrid")
        self.assertEqual(len(payload["hits"]), 1)
        self.assertIn("score", payload["hits"][0])
        self.assertEqual(payload["hits"][0]["item_type"], "memory")

    def test_recall_json_reports_the_mode(self):
        self.run_cli("remember", "a searchable statement")
        _, out, _ = self.run_cli("--json", "recall", "searchable", "--search-mode", "keyword")
        self.assertEqual(json.loads(out)["mode"], "keyword")

    def test_recall_search_mode_is_validated(self):
        code, _, err = self.run_cli("recall", "x", "--search-mode", "magic")
        self.assertEqual(code, 2)
        self.assertIn("invalid choice", err)

    def test_ambiguous_abbreviations_are_rejected(self):
        # Guards against the Python 3.8 subparser abbreviation collision.
        code, _, err = self.run_cli("recall", "x", "--mode", "keyword")
        self.assertEqual(code, 2)
        self.assertIn("unrecognized arguments", err)

    def test_remember_json_output(self):
        code, out, _ = self.run_cli("--json", "remember", "structured output")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["content"], "structured output")

    def test_global_options_after_the_subcommand(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["remember", "options can come last", "--db", self.db_path])
        self.assertEqual(code, 0)
        code, out, _ = self.run_cli("recall", "options")
        self.assertIn("options can come last", out)

    def test_forget_logical_then_hidden_from_recall(self):
        self.run_cli("remember", "a temporary detail")
        code, out, _ = self.run_cli("--json", "recall", "temporary")
        memory_id = json.loads(out)["hits"][0]["id"]
        code, _, _ = self.run_cli("forget", memory_id)
        self.assertEqual(code, 0)
        _, out, _ = self.run_cli("recall", "temporary")
        self.assertIn("no matches", out)

    def test_forget_missing_memory_exits_nonzero(self):
        code, _, err = self.run_cli("forget", "mem_missing")
        self.assertEqual(code, 1)
        self.assertIn("not found", err)

    def test_unknown_provider_exits_nonzero(self):
        code, _, err = self.run_cli("--provider", "gpt-5", "stats")
        self.assertEqual(code, 1)
        self.assertIn("unknown", err)
        self.assertIn("provider", err)


class ObservationCommandTests(CliTestCase):
    def test_observe_forms_a_memory(self):
        code, out, _ = self.run_cli("observe", "记住：用户偏好本地优先")
        self.assertEqual(code, 0)
        self.assertIn("+ memory", out)

    def test_observe_reports_when_nothing_is_formed(self):
        code, out, _ = self.run_cli("observe", "今天天气不错")
        self.assertEqual(code, 0)
        self.assertIn("no memory formed", out)

    def test_observe_preference_and_project_state(self):
        _, out, _ = self.run_cli("observe", "记住偏好：theme = dark")
        self.assertIn("+ preference", out)
        _, out, _ = self.run_cli("observe", '项目状态：memory = {"phase": "v0.1"}')
        self.assertIn("+ project", out)

    def test_event_is_appended(self):
        code, out, _ = self.run_cli("event", "raw utterance", "--session", "s1")
        self.assertEqual(code, 0)
        self.assertIn("event evt_", out)


class PreferenceCommandTests(CliTestCase):
    def test_set_get_list_history(self):
        code, out, _ = self.run_cli("pref", "set", "theme", "dark")
        self.assertEqual(code, 0)
        self.assertIn("theme", out)

        code, out, _ = self.run_cli("pref", "get", "theme")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.strip()), "dark")

        self.run_cli("pref", "set", "theme", "light")
        code, out, _ = self.run_cli("pref", "history", "theme")
        self.assertIn("dark", out)
        self.assertIn("light", out)

        code, out, _ = self.run_cli("--json", "pref", "list")
        self.assertEqual(len(json.loads(out)), 1)

    def test_get_missing_preference_exits_nonzero(self):
        code, _, err = self.run_cli("pref", "get", "absent")
        self.assertEqual(code, 1)
        self.assertIn("not set", err)

    def test_json_values_are_parsed(self):
        self.run_cli("pref", "set", "columns", "3")
        code, out, _ = self.run_cli("pref", "get", "columns")
        self.assertEqual(json.loads(out.strip()), 3)

    def test_pref_without_a_subcommand_exits_nonzero(self):
        code, _, err = self.run_cli("pref")
        self.assertEqual(code, 1)
        self.assertIn("subcommand", err)


class ProjectCommandTests(CliTestCase):
    def test_set_get_history_list(self):
        code, out, _ = self.run_cli("project", "set", "memory", '{"phase": "v0.1"}')
        self.assertEqual(code, 0)
        self.assertIn("memory", out)

        code, out, _ = self.run_cli("project", "get", "memory")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.strip()), {"phase": "v0.1"})

        self.run_cli("project", "set", "memory", '{"phase": "v0.2"}')
        _, out, _ = self.run_cli("project", "history", "memory")
        self.assertIn("v0.1", out)
        self.assertIn("v0.2", out)

        _, out, _ = self.run_cli("--json", "project", "list")
        self.assertEqual(len(json.loads(out)), 1)

    def test_invalid_json_state_exits_nonzero(self):
        code, _, err = self.run_cli("project", "set", "memory", "not-json")
        self.assertEqual(code, 1)
        self.assertIn("valid JSON", err)

    def test_non_object_state_exits_nonzero(self):
        code, _, err = self.run_cli("project", "set", "memory", "[1, 2]")
        self.assertEqual(code, 1)
        self.assertIn("JSON object", err)

    def test_get_missing_state_exits_nonzero(self):
        code, _, err = self.run_cli("project", "get", "absent")
        self.assertEqual(code, 1)
        self.assertIn("no state", err)


class AskCommandTests(CliTestCase):
    def test_ask_without_a_model_degrades_but_succeeds(self):
        self.run_cli("remember", "用户偏好本地优先")
        code, out, err = self.run_cli("ask", "本地优先")
        self.assertEqual(code, 0)
        self.assertIn("no model provider configured", err)
        self.assertIn("用户偏好本地优先", out)

    def test_ask_with_the_fake_provider(self):
        self.run_cli("remember", "用户偏好本地优先")
        code, out, err = self.run_cli("--provider", "fake", "--model", "fake-Z", "ask", "偏好")
        self.assertEqual(code, 0)
        self.assertIn("fake/fake-Z", err)
        self.assertTrue(out.strip())

    def test_ask_json_output_reports_the_context(self):
        self.run_cli("remember", "用户偏好本地优先")
        code, out, _ = self.run_cli("--json", "ask", "本地优先")
        payload = json.loads(out)
        self.assertTrue(payload["degraded"])
        self.assertEqual(len(payload["context_ids"]), 1)


class DataCommandTests(CliTestCase):
    def test_export_to_stdout_is_json_lines(self):
        self.run_cli("remember", "exported memory")
        code, out, _ = self.run_cli("export")
        self.assertEqual(code, 0)
        lines = [line for line in out.splitlines() if line.strip()]
        self.assertTrue(lines)
        for line in lines:
            payload = json.loads(line)
            self.assertIn("table", payload)
            self.assertIn("row", payload)

    def test_export_to_a_file(self):
        self.run_cli("remember", "exported memory")
        target = os.path.join(self.tmpdir, "export.jsonl")
        code, out, _ = self.run_cli("export", "--out", target)
        self.assertEqual(code, 0)
        self.assertIn("exported to", out)
        self.assertTrue(os.path.exists(target))

    def test_snapshot(self):
        self.run_cli("remember", "snapshotted memory")
        target = os.path.join(self.tmpdir, "backup.sqlite")
        code, out, _ = self.run_cli("snapshot", "--out", target)
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(target))

    def test_stats(self):
        self.run_cli("remember", "counted memory")
        self.run_cli("pref", "set", "theme", "dark")
        code, out, _ = self.run_cli("stats")
        self.assertEqual(code, 0)
        self.assertIn("schema version: {0}".format(SCHEMA_VERSION), out)
        self.assertIn("provider", out)
        self.assertIn("extractor     : rules", out)

    def test_stats_json(self):
        code, out, _ = self.run_cli("--json", "stats")
        payload = json.loads(out)
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["extractor"], "rules")


if __name__ == "__main__":
    unittest.main()
