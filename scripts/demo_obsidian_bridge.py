"""Runnable acceptance demo for the Obsidian bridge.

Copy of the manual walkthrough, runnable in one command without a model:

    py -3 scripts/demo_obsidian_bridge.py

It verifies, against a throwaway copy of ``fixtures/obsidian_vault``:

1. ``import --dry-run`` leaves the database byte-identical and the vault untouched
2. ``import --apply`` turns notes into events and memories
3. ``export`` produces the mirror, with id + provenance + source_ref + scope
4. re-importing creates nothing new
5. editing a note re-ingests only changed chunks, and deletes no memory
6. persona A cannot see persona B

The real-model walkthrough (with Ollama) is the same sequence of CLI commands;
see ``docs/v0.3_implementation.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_bridge.config import BridgeConfig  # noqa: E402
from memory_bridge.obsidian.exporter import README_NAME, ObsidianExporter  # noqa: E402
from memory_bridge.obsidian.importer import ObsidianImporter  # noqa: E402
from memory_core.api import MemoryCore  # noqa: E402
from memory_core.providers.fake import FakeProvider  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_VAULT = os.path.join(PROJECT_ROOT, "fixtures", "obsidian_vault")

#: Deterministic stand-in for the local LLM: it declines everything, which is
#: what a model should do for most human notes anyway.
DECLINE_ALL = FakeProvider(model="demo", reply=json.dumps({"memories": []}))

CHECKS = []


def check(label, condition, detail=""):
    # type: (str, bool, str) -> None
    CHECKS.append((label, bool(condition), detail))
    print("  [{0}] {1}{2}".format("PASS" if condition else "FAIL", label,
                                  "" if condition else "  <- " + detail))


def file_digest(path):
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
            parts.append("{0}:{1}".format(rel, file_digest(path)))
    return hashlib.sha256("\n".join(sorted(parts)).encode("utf-8")).hexdigest()


def main():
    # type: () -> int
    workdir = tempfile.mkdtemp(prefix="bridge-demo-")
    vault = os.path.join(workdir, "vault")
    shutil.copytree(FIXTURE_VAULT, vault)
    db_path = os.path.join(workdir, "memory.sqlite")
    config = BridgeConfig.load(vault)

    print("Personal Knowledge Bridge - acceptance demo")
    print("=" * 74)
    print("vault    : {0}".format(vault))
    print("database : {0}".format(db_path))
    print("")

    def open_core(**kwargs):
        return MemoryCore.open(db_path=db_path, provider=DECLINE_ALL,
                               use_llm_formation=True, **kwargs)

    core = open_core()
    core.close()

    # -- 1. dry run ----------------------------------------------------
    print("1. import --dry-run")
    before_db = file_digest(db_path)
    before_vault = tree_digest(vault)
    core = open_core()
    report = ObsidianImporter(core, config).run(apply=False)
    core.close()
    check("dry run reports no writes", report.counts["events_created"] == 0)
    check("database byte-identical after dry run", file_digest(db_path) == before_db)
    check("vault untouched by dry run", tree_digest(vault) == before_vault)
    routes = {action.route for action in report.actions}
    check("all four routes exercised",
          routes == {"knowledge", "declaration", "preference", "project_state"},
          str(sorted(routes)))
    check("excluded file not scanned",
          not any("Private" in action.document_rel for action in report.actions))

    # -- 2. apply ------------------------------------------------------
    print("2. import --apply")
    core = open_core()
    applied = ObsidianImporter(core, config).run(apply=True)
    check("events created", applied.counts["events_created"] == 7,
          str(applied.counts["events_created"]))
    check("declarations became memories", applied.counts["memories_created"] >= 3)
    check("preference written through the typed API", applied.counts["preferences_set"] == 1)
    check("project state written through the typed API",
          applied.counts["project_states_set"] == 1)
    check("vault untouched by apply", tree_digest(vault) == before_vault)
    memories = core.list_memories(include_personas=True, limit=1000)
    check("every memory carries external_ref",
          all(m.source.get("external_ref") for m in memories))
    check("every memory carries a source_ref-able event",
          all(m.source.get("event_ids") for m in memories))
    check("declarations are deterministic (formed_by=rule)",
          any(m.source.get("formed_by") == "rule" for m in memories))

    # -- 3. export -----------------------------------------------------
    print("3. export")
    exported = ObsidianExporter(core, config).export()
    names = sorted(os.path.basename(path) for path in exported.files)
    check("_README.md written", README_NAME in names)
    check("global.md written", "global.md" in names)
    check("project-*.md written", "project-persona-edgeaiot.md" in names)
    check("persona-*.md written", "persona-aria.md" in names and "persona-bruno.md" in names)
    with open(os.path.join(config.export_path(), "global.md"), encoding="utf-8") as handle:
        mirror = handle.read()
    for token in ("id=mem_", "scope=global", "origin=obsidian", "source_ref=obsidian:demo-vault/",
                  "content_hash=sha256:", "events=evt_"):
        check("mirror block contains {0}".format(token), token in mirror)

    # -- 4. re-import --------------------------------------------------
    print("4. re-import (idempotency)")
    again = ObsidianImporter(core, config).run(apply=True)
    check("no new events", again.counts["events_created"] == 0)
    check("no new memories", again.counts["memories_created"] == 0)
    check("all documents reported unchanged", again.counts["documents_unchanged"] == 6)

    # -- 5. edit a note ------------------------------------------------
    print("5. edit one note, re-import")
    ids_before = {m.id for m in core.list_memories(include_personas=True, limit=1000)}
    target = os.path.join(vault, "Knowledge", "Local First Architecture.md")
    with open(target, encoding="utf-8") as handle:
        text = handle.read()
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.replace(
            "Storage format stability matters more than raw storage performance.",
            "Storage format stability matters far more than raw storage performance.",
        ))
    edited = ObsidianImporter(core, config).run(apply=True)
    ids_after = {m.id for m in core.list_memories(include_personas=True, limit=1000)}
    check("new events for the edit", edited.counts["events_created"] > 0)
    check("only the edited document changed", edited.counts["documents_changed"] == 1)
    check("untouched chunks were skipped", edited.counts["chunks_skipped_seen"] >= 1)
    check("no memory was deleted", ids_before.issubset(ids_after),
          str(sorted(ids_before - ids_after)))

    # -- 6. persona isolation ------------------------------------------
    print("6. persona isolation")
    core.close()
    fresh = MemoryCore.open(db_path=db_path)
    check("default recall hides persona memories",
          [h for h in fresh.recall("结论", limit=10)] == [])
    aria = fresh.recall("结论", persona="aria", limit=10)
    check("persona aria sees its own memory",
          any("Aria" in hit.item.content for hit in aria))
    check("persona aria cannot see persona bruno",
          not any("Bruno" in hit.item.content for hit in aria))
    bruno = fresh.recall("私有偏好", persona="bruno", limit=10)
    check("persona bruno sees its own memory",
          any("Bruno" in hit.item.content for hit in bruno))
    check("persona bruno cannot see persona aria",
          not any("Aria" in hit.item.content for hit in bruno))
    fresh.close()

    # -- summary -------------------------------------------------------
    failed = [label for label, ok, _ in CHECKS if not ok]
    print("")
    print("=" * 74)
    print("{0} checks, {1} passed, {2} failed".format(
        len(CHECKS), len(CHECKS) - len(failed), len(failed)))
    for label in failed:
        print("  FAILED: {0}".format(label))
    print("DEMO RESULT: {0}".format("PASS" if not failed else "FAIL"))
    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
