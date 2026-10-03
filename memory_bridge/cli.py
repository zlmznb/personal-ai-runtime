"""Command line interface for the Personal Knowledge Bridge.

    python -m memory_bridge import <vault> [--apply]     # dry run by default
    python -m memory_bridge export <vault> [--out DIR]
    python -m memory_bridge status <vault>

``import`` is **dry run unless ``--apply`` is given**. Ingesting a large vault
into a personal memory store is not something that should happen by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional

from memory_bridge import __version__
from memory_bridge.config import BridgeConfig
from memory_bridge.obsidian.exporter import ObsidianExporter
from memory_bridge.obsidian.importer import ObsidianImporter

EXIT_OK = 0
EXIT_ERROR = 1


def _configure_stdout():
    # type: () -> None
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            pass


def _print_json(payload):
    # type: (Any) -> None
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _common(parent):
    # type: (argparse.ArgumentParser) -> None
    parent.add_argument("--db", default=None, help="path to memory.sqlite")
    parent.add_argument("--config", default=None, help="path to memory-bridge.json")
    parent.add_argument("--vault-id", default=None, help="stable id for this vault")
    parent.add_argument("--models", default=None, help="path to memory_core models.json")
    parent.add_argument("--provider", default=None, help="chat provider for LLM formation")
    parent.add_argument("--model", default=None, help="chat model name")
    parent.add_argument(
        "--embedding-provider", default=None, help="embedding provider for the vector index"
    )
    parent.add_argument("--embedding-model", default=None, help="embedding model name")
    parent.add_argument("--no-llm", action="store_true", help="rules-only formation")
    parent.add_argument("--json", action="store_true", help="machine-readable output")


def build_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(
        prog="memory-bridge",
        description="Personal Knowledge Bridge - Obsidian <-> Memory Core",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version="memory-bridge {0}".format(__version__))
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    importer = subparsers.add_parser(
        "import",
        help="ingest vault notes into Memory Core (dry run unless --apply)",
        allow_abbrev=False,
    )
    importer.add_argument("vault")
    importer.add_argument(
        "--apply", action="store_true", help="actually write (default is a dry run)"
    )
    importer.add_argument(
        "--dry-run", action="store_true", help="explicit dry run (this is the default)"
    )
    _common(importer)

    exporter = subparsers.add_parser(
        "export", help="write the Memory Core mirror into the vault", allow_abbrev=False
    )
    exporter.add_argument("vault")
    exporter.add_argument("--out", default=None, help="override the export directory")
    _common(exporter)

    status = subparsers.add_parser(
        "status", help="show what an import would do, without calling a model",
        allow_abbrev=False,
    )
    status.add_argument("vault")
    _common(status)

    return parser


def _build_config(args):
    # type: (argparse.Namespace) -> BridgeConfig
    overrides = {
        "vault_id": args.vault_id,
        "db_path": args.db,
        "use_llm": False if getattr(args, "no_llm", False) else None,
    }
    return BridgeConfig.load(args.vault, config_path=args.config, **overrides)


def _open_core(args, config):
    # type: (argparse.Namespace, BridgeConfig) -> Any
    from memory_core.api import MemoryCore

    return MemoryCore.open(
        db_path=config.db_path,
        models_path=args.models,
        provider_name=args.provider,
        model=args.model,
        embedding_provider_name=args.embedding_provider,
        embedding_model=args.embedding_model,
        use_llm_formation=bool(config.use_llm),
    )


def _cmd_import(core, config, args):
    # type: (Any, BridgeConfig, argparse.Namespace) -> int
    importer = ObsidianImporter(core, config)
    report = importer.run(apply=bool(args.apply))
    if args.json:
        _print_json(report.as_dict())
    else:
        print(report.render())
        if report.dry_run:
            print("")
            print("dry run: nothing was written. Re-run with --apply to commit.")
    return EXIT_OK


def _cmd_export(core, config, args):
    # type: (Any, BridgeConfig, argparse.Namespace) -> int
    exporter = ObsidianExporter(core, config)
    report = exporter.export(out_dir=args.out)
    if args.json:
        _print_json(report.as_dict())
    else:
        print(report.render())
    return EXIT_OK


def _cmd_status(core, config, args):
    # type: (Any, BridgeConfig, argparse.Namespace) -> int
    importer = ObsidianImporter(core, config)
    plans = importer.plan()
    summary = {
        "vault_path": config.vault_path,
        "vault_id": config.vault_id,
        "database": config.db_path,
        "documents_total": len(plans),
        "documents_new": len([p for p in plans if p.status == "new"]),
        "documents_changed": len([p for p in plans if p.status == "changed"]),
        "documents_unchanged": len([p for p in plans if p.status == "unchanged"]),
        "documents_empty": len([p for p in plans if p.status == "empty"]),
        "chunks_to_ingest": len(
            [a for p in plans for a in p.actions if a.action == "ingest"]
        ),
        "chunks_already_seen": len(
            [a for p in plans for a in p.actions if a.action == "skip-seen"]
        ),
        "documents": [
            {"rel_path": p.rel_path, "status": p.status, "chunks": len(p.actions)}
            for p in plans
        ],
    }
    if args.json:
        _print_json(summary)
        return EXIT_OK
    print("Obsidian bridge status")
    print("=" * 78)
    print("vault      : {0}".format(summary["vault_path"]))
    print("vault_id   : {0}".format(summary["vault_id"]))
    print("database   : {0}".format(summary["database"]))
    print("")
    for entry in summary["documents"]:
        print("  [{0:<9}] {1}  ({2} chunks)".format(
            entry["status"], entry["rel_path"], entry["chunks"]
        ))
    print("")
    print("documents     : {0} total, {1} new, {2} changed, {3} unchanged".format(
        summary["documents_total"], summary["documents_new"],
        summary["documents_changed"], summary["documents_unchanged"],
    ))
    print("chunks        : {0} to ingest, {1} already seen".format(
        summary["chunks_to_ingest"], summary["chunks_already_seen"]
    ))
    return EXIT_OK


_HANDLERS = {"import": _cmd_import, "export": _cmd_export, "status": _cmd_status}


def main(argv=None):
    # type: (Optional[List[str]]) -> int
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK

    core = None
    try:
        config = _build_config(args)
        core = _open_core(args, config)
        return _HANDLERS[args.command](core, config, args)
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not trace
        if getattr(args, "json", False):
            _print_json({"error": "{0}: {1}".format(type(exc).__name__, exc)})
        else:
            print("error: {0}: {1}".format(type(exc).__name__, exc), file=sys.stderr)
        return EXIT_ERROR
    finally:
        if core is not None:
            core.close()


__all__ = ["main", "build_parser"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
