"""Command line interface.

Deliberately thin: every command is a few lines of wiring over
:class:`memory_core.api.MemoryCore`. No business logic lives here.

Global options may be given before or after the subcommand::

    python -m memory_core --db my.sqlite recall "偏好"
    python -m memory_core recall "偏好" --db my.sqlite
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from memory_core.api import MemoryCore, format_hit
from memory_core.domain.errors import MemoryCoreError
from memory_core.formation.base import parse_json_value

EXIT_OK = 0
EXIT_ERROR = 1


def _opt(args, name, default=None):
    # type: (argparse.Namespace, str, Any) -> Any
    """Read an option that may have been suppressed by the shared parent parser."""
    value = getattr(args, name, None)
    return default if value is None else value


def _configure_stdout():
    # type: () -> None
    """Make CJK output safe on Windows consoles that default to a legacy code page."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            pass


def _print_json(payload):
    # type: (Any) -> None
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _common_parser():
    # type: () -> argparse.ArgumentParser
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS keeps the main-parser value when a subcommand omits the option.
    common.add_argument("--db", default=argparse.SUPPRESS, help="path to memory.sqlite")
    common.add_argument("--config", default=argparse.SUPPRESS, help="path to config/memory.json")
    common.add_argument("--models", default=argparse.SUPPRESS, help="path to config/models.json")
    common.add_argument("--provider", default=argparse.SUPPRESS, help="provider name or type: fake | ollama | deepseek")
    common.add_argument("--model", default=argparse.SUPPRESS, help="model name override")
    common.add_argument("--embedding-provider", default=argparse.SUPPRESS, help="embedding provider name or type: fake-embedding | ollama-embedding")
    common.add_argument("--embedding-model", default=argparse.SUPPRESS, help="embedding model name override")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable output")
    return common


def build_parser():
    # type: () -> argparse.ArgumentParser
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="memory",
        description="Memory Core - a model-agnostic, local-first personal AI memory.",
        parents=[common],
        # Abbreviations are a footgun with subparsers: on Python 3.8 the main
        # parser validates option strings that actually belong to a subcommand,
        # so "--mode" collides with "--model"/"--models". Full names only.
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name, help_text, **kwargs):
        return subparsers.add_parser(
            name, parents=[common], help=help_text, allow_abbrev=False, **kwargs
        )

    def nested(parent, name, help_text, **kwargs):
        # Sub-subcommands share the same strict parsing: on Python 3.8 an
        # abbreviated option that collides with a parent option produces a
        # confusing "ambiguous option" error instead of a clear one.
        return parent.add_parser(
            name, parents=[common], help=help_text, allow_abbrev=False, **kwargs
        )

    add("init", "create the database and schema")

    remember = add("remember", "write a long-term memory")
    remember.add_argument("content")
    remember.add_argument("--kind", default="semantic", choices=["semantic", "episodic", "identity"])
    remember.add_argument("--scope", default="global")
    remember.add_argument("--subject", default="user:self")
    remember.add_argument("--tag", action="append", default=None)
    remember.add_argument("--confidence", type=float, default=1.0)
    remember.add_argument("--salience", type=float, default=0.5)

    recall = add("recall", "search memories and preferences (chat model never involved)")
    recall.add_argument("query")
    recall.add_argument("--scope", default=None)
    recall.add_argument("--kind", action="append", default=None, choices=["semantic", "episodic", "identity"])
    recall.add_argument("--limit", type=int, default=None)
    recall.add_argument("--no-preferences", action="store_true")
    recall.add_argument(
        "--search-mode",
        default="hybrid",
        choices=["keyword", "semantic", "hybrid"],
        help="keyword (FTS5), semantic (vector index) or hybrid (RRF fusion)",
    )

    forget = add("forget", "remove a memory (logical by default)")
    forget.add_argument("memory_id")
    forget.add_argument("--hard", action="store_true", help="physically delete the row")

    observe = add("observe", "record an utterance and apply the deterministic rules")
    observe.add_argument("text")
    observe.add_argument("--role", default="user")
    observe.add_argument("--session", default=None)
    observe.add_argument("--scope", default="global")

    learn = add("learn", "record an utterance and run rules + LLM memory formation")
    learn.add_argument("text")
    learn.add_argument("--role", default="user")
    learn.add_argument("--session", default=None)
    learn.add_argument("--scope", default="global")
    learn.add_argument("--context-limit", type=int, default=5, help="existing memories shown to the LLM")

    propose = add("propose", "dry run: show validated candidates without writing anything")
    propose.add_argument("text")
    propose.add_argument("--scope", default="global")
    propose.add_argument("--context-limit", type=int, default=5)
    propose.add_argument("--no-llm", action="store_true", help="rules only")

    event = add("event", "append a raw event")
    event.add_argument("content")
    event.add_argument("--role", default="user")
    event.add_argument("--session", default=None)
    event.add_argument("--scope", default="global")

    preference = add("pref", "manage explicit preferences")
    preference_sub = preference.add_subparsers(dest="pref_command", metavar="<set|get|list|history>")
    pref_set = nested(preference_sub, "set", "set a preference value")
    pref_set.add_argument("key")
    pref_set.add_argument("value")
    pref_set.add_argument("--scope", default="global")
    pref_set.add_argument("--statement", default=None)
    pref_get = nested(preference_sub, "get", "read the current value")
    pref_get.add_argument("key")
    pref_get.add_argument("--scope", default="global")
    pref_list = nested(preference_sub, "list", "list current values")
    pref_list.add_argument("--scope", default=None)
    pref_list.add_argument("--all", action="store_true", help="include superseded values")
    pref_history = nested(preference_sub, "history", "show value history")
    pref_history.add_argument("key")
    pref_history.add_argument("--scope", default="global")

    project = add("project", "manage basic project state")
    project_sub = project.add_subparsers(dest="project_command", metavar="<set|get|history|list>")
    project_set = nested(project_sub, "set", "set the current state")
    project_set.add_argument("project_id")
    project_set.add_argument("state", help="a JSON object")
    project_set.add_argument("--scope", default=None)
    project_set.add_argument("--note", default=None)
    project_get = nested(project_sub, "get", "read the current state")
    project_get.add_argument("project_id")
    project_get.add_argument("--scope", default=None)
    project_history = nested(project_sub, "history", "show state history")
    project_history.add_argument("project_id")
    project_history.add_argument("--scope", default=None)
    project_list = nested(project_sub, "list", "list current project states")
    project_list.add_argument("--scope", default=None)

    ask = add("ask", "answer a question using recalled memory as context")
    ask.add_argument("question")
    ask.add_argument("--scope", default=None)
    ask.add_argument("--limit", type=int, default=None)
    ask.add_argument("--system", default=None, help="override the system prompt")

    export = add("export", "export all source tables as deterministic JSON Lines")
    export.add_argument("--out", default=None, help="write to a file instead of stdout")

    snapshot = add("snapshot", "write a consistent copy of the database")
    snapshot.add_argument("--out", required=True)

    index = add("index", "manage the derived vector index (never a source of truth)")
    index_sub = index.add_subparsers(dest="index_command", metavar="<build|rebuild|drop|stats>")
    index_build = nested(index_sub, "build", "embed memories missing from the index")
    index_build.add_argument("--scope", default=None)
    index_build.add_argument("--limit", type=int, default=None)
    index_rebuild = nested(index_sub, "rebuild", "drop this model's vectors and regenerate them")
    index_rebuild.add_argument("--scope", default=None)
    index_rebuild.add_argument("--limit", type=int, default=None)
    index_drop = nested(index_sub, "drop", "delete vectors (memories are untouched)")
    index_drop.add_argument("--all", action="store_true", help="drop every embedding model's vectors, not just the current one")
    nested(index_sub, "stats", "show vector index statistics")

    add("stats", "show database statistics")

    return parser


# -- command handlers ------------------------------------------------------

def _cmd_init(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    stats = core.stats()
    if use_json:
        _print_json(stats)
    else:
        print("initialised {0} (schema v{1})".format(stats["path"], stats["schema_version"]))
    return EXIT_OK


def _cmd_remember(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    memory = core.remember(
        args.content,
        kind=args.kind,
        scope=args.scope,
        subject=args.subject,
        tags=args.tag or (),
        confidence=args.confidence,
        salience=args.salience,
        source={"origin": "cli"},
    )
    if use_json:
        _print_json(memory.as_dict())
    else:
        print("{0}  [{1}] {2}".format(memory.id, memory.kind, memory.content))
    return EXIT_OK


def _cmd_recall(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    hits = core.recall(
        args.query,
        scope=args.scope,
        kinds=tuple(args.kind) if args.kind else None,
        limit=args.limit,
        include_preferences=not args.no_preferences,
        mode=args.search_mode,
    )
    if use_json:
        _print_json({"mode": args.search_mode, "hits": [hit.as_dict() for hit in hits]})
        return EXIT_OK
    if not hits:
        print("(no matches)")
        return EXIT_OK
    for index, hit in enumerate(hits, start=1):
        print("{0}. {1}  {2}  score={3:.4f}  [{4}]".format(
            index, hit.id, hit.item_type, hit.score, hit.matched_on
        ))
        print("   " + format_hit(hit))
    return EXIT_OK


def _cmd_forget(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    core.forget(args.memory_id, hard=args.hard)
    if use_json:
        _print_json({"forgotten": args.memory_id, "hard": bool(args.hard)})
    else:
        print("forgotten {0}{1}".format(args.memory_id, " (hard)" if args.hard else " (logical)"))
    return EXIT_OK


def _print_outcomes(observation):
    # type: (Any) -> None
    for outcome in observation.outcomes:
        marker = "+" if outcome.accepted else "-"
        payload = outcome.candidate.payload or {}
        content = payload.get("content")
        if not content and "key" in payload:
            content = "{0} = {1}".format(payload.get("key"), payload.get("value"))
        print("  {0} {1:<12} {2:<24} {3}".format(
            marker,
            outcome.item_type or outcome.candidate.target,
            outcome.item_id or "",
            content or "(no content)",
        ))
        if outcome.reason:
            print("      {0}".format(outcome.reason))


def _cmd_observe(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    observation = core.observe(args.text, role=args.role, session_id=args.session, scope=args.scope)
    if use_json:
        _print_json(observation.as_dict())
        return EXIT_OK
    print("event {0}".format(observation.event.id))
    _print_outcomes(observation)
    if not observation.outcomes and observation.skipped_reason:
        print("  (no memory formed: {0})".format(observation.skipped_reason))
    return EXIT_OK


def _cmd_learn(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    observation = core.learn(
        args.text,
        role=args.role,
        session_id=args.session,
        scope=args.scope,
        context_limit=args.context_limit,
    )
    if use_json:
        _print_json(observation.as_dict())
        return EXIT_OK
    print("event {0}".format(observation.event.id))
    _print_outcomes(observation)
    if not observation.outcomes and observation.skipped_reason:
        print("  (no candidates: {0})".format(observation.skipped_reason))
    print("  {0} accepted, {1} rejected".format(len(observation.accepted), len(observation.rejected)))
    return EXIT_OK


def _cmd_propose(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    proposal = core.propose(
        args.text,
        scope=args.scope,
        context_limit=args.context_limit,
        include_llm=not args.no_llm,
    )
    if use_json:
        _print_json(proposal.as_dict())
        return EXIT_OK
    print("(dry run: nothing was written)")
    if proposal.context_ids:
        print("  context shown to the model: {0}".format(", ".join(proposal.context_ids)))
    for reason in proposal.reasons:
        print("  note: {0}".format(reason))
    for outcome in proposal.outcomes:
        marker = "accept" if outcome.accepted else "reject"
        content = (outcome.candidate.payload or {}).get("content") or ""
        print("  [{0}] {1} {2}  {3}".format(
            marker,
            outcome.candidate.origin,
            outcome.candidate.operation,
            content or outcome.reason,
        ))
        if outcome.rejected and content:
            print("        reason: {0}".format(outcome.reason))
    if not proposal.outcomes:
        print("  (no candidates)")
    return EXIT_OK


def _cmd_index(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    command = getattr(args, "index_command", None)
    if command == "build":
        result = core.build_index(scope=args.scope, limit=args.limit)
    elif command == "rebuild":
        result = core.rebuild_index(scope=args.scope, limit=args.limit)
    elif command == "drop":
        removed = core.drop_index(all_models=bool(getattr(args, "all", False)))
        result = {"dropped": removed}
    elif command == "stats":
        stats = core.index_stats()
        if stats is None:
            print("no embedding provider configured", file=sys.stderr)
            return EXIT_ERROR
        result = stats
    else:
        print("index requires a subcommand: build | rebuild | drop | stats", file=sys.stderr)
        return EXIT_ERROR

    if use_json:
        _print_json(result)
        return EXIT_OK
    for key in sorted(result):
        print("{0:<16}: {1}".format(key, result[key]))
    return EXIT_OK


def _cmd_event(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    event = core.record_event(
        args.content, role=args.role, session_id=args.session, scope=args.scope
    )
    if use_json:
        _print_json(event.as_dict())
    else:
        print("event {0}".format(event.id))
    return EXIT_OK


def _cmd_pref(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    command = getattr(args, "pref_command", None)
    if command == "set":
        preference = core.set_preference(
            args.key,
            parse_json_value(args.value),
            scope=args.scope,
            statement=args.statement,
            source={"origin": "cli"},
        )
        if use_json:
            _print_json(preference.as_dict())
        else:
            print("{0}  {1} = {2}".format(preference.id, preference.key, preference.value))
        return EXIT_OK
    if command == "get":
        preference = core.get_preference(args.key, scope=args.scope)
        if preference is None:
            print("(not set)", file=sys.stderr)
            return EXIT_ERROR
        if use_json:
            _print_json(preference.as_dict())
        else:
            print(json.dumps(preference.value, ensure_ascii=False))
        return EXIT_OK
    if command == "list":
        preferences = core.list_preferences(scope=args.scope, include_superseded=args.all)
        if use_json:
            _print_json([p.as_dict() for p in preferences])
            return EXIT_OK
        if not preferences:
            print("(none)")
            return EXIT_OK
        for preference in preferences:
            marker = "" if preference.superseded_by is None else "  (superseded)"
            print("{0} = {1}{2}".format(
                preference.key, json.dumps(preference.value, ensure_ascii=False), marker
            ))
        return EXIT_OK
    if command == "history":
        history = core.preference_history(args.key, scope=args.scope)
        if use_json:
            _print_json([p.as_dict() for p in history])
            return EXIT_OK
        if not history:
            print("(none)")
            return EXIT_OK
        for preference in history:
            print("{0}  {1} = {2}".format(
                preference.created_at, preference.key,
                json.dumps(preference.value, ensure_ascii=False),
            ))
        return EXIT_OK
    print("pref requires a subcommand: set | get | list | history", file=sys.stderr)
    return EXIT_ERROR


def _cmd_project(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    command = getattr(args, "project_command", None)
    scope = _opt(args, "scope")
    if command == "set":
        try:
            state = json.loads(args.state)
        except ValueError as exc:
            print("state must be valid JSON: {0}".format(exc), file=sys.stderr)
            return EXIT_ERROR
        if not isinstance(state, dict):
            print("state must be a JSON object", file=sys.stderr)
            return EXIT_ERROR
        record = core.set_project_state(
            args.project_id, state, scope=scope, note=args.note, source={"origin": "cli"}
        )
        if use_json:
            _print_json(record.as_dict())
        else:
            print("{0}  {1} = {2}".format(
                record.id, record.project_id, json.dumps(record.state, ensure_ascii=False)
            ))
        return EXIT_OK
    if command == "get":
        record = core.get_project_state(args.project_id, scope=scope)
        if record is None:
            print("(no state for {0})".format(args.project_id), file=sys.stderr)
            return EXIT_ERROR
        if use_json:
            _print_json(record.as_dict())
        else:
            print(json.dumps(record.state, ensure_ascii=False))
        return EXIT_OK
    if command == "history":
        history = core.project_state_history(args.project_id, scope=scope)
        if use_json:
            _print_json([record.as_dict() for record in history])
            return EXIT_OK
        if not history:
            print("(none)")
            return EXIT_OK
        for record in history:
            print("{0}  {1}".format(record.created_at, json.dumps(record.state, ensure_ascii=False)))
        return EXIT_OK
    if command == "list":
        records = core.list_project_states(scope=scope)
        if use_json:
            _print_json([record.as_dict() for record in records])
            return EXIT_OK
        if not records:
            print("(none)")
            return EXIT_OK
        for record in records:
            print("{0}  {1}".format(record.project_id, json.dumps(record.state, ensure_ascii=False)))
        return EXIT_OK
    print("project requires a subcommand: set | get | history | list", file=sys.stderr)
    return EXIT_ERROR


def _cmd_ask(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    answer = core.ask(
        args.question, scope=args.scope, limit=args.limit, system_prompt=args.system
    )
    if use_json:
        _print_json(answer.as_dict())
        return EXIT_OK
    if answer.degraded:
        print("[degraded: {0}]".format(answer.reason), file=sys.stderr)
    elif answer.provider:
        print("[{0}/{1}]".format(answer.provider, answer.model), file=sys.stderr)
    print(answer.text)
    return EXIT_OK


def _cmd_export(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    if args.out:
        path = core.export_to(args.out)
        if use_json:
            _print_json({"exported_to": path})
        else:
            print("exported to {0}".format(path))
        return EXIT_OK
    sys.stdout.write(core.export_jsonl())
    return EXIT_OK


def _cmd_snapshot(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    path = core.snapshot(args.out)
    if use_json:
        _print_json({"snapshot": path})
    else:
        print("snapshot written to {0}".format(path))
    return EXIT_OK


def _cmd_stats(core, args, use_json):
    # type: (MemoryCore, argparse.Namespace, bool) -> int
    stats = core.stats()
    if use_json:
        _print_json(stats)
        return EXIT_OK
    print("database      : {0}".format(stats["path"]))
    print("schema version: {0}".format(stats["schema_version"]))
    print("events        : {0}".format(stats["events"]))
    memories = stats["memories"]
    print("memories      : {0} active, {1} superseded, {2} total".format(
        memories["active"], memories["superseded"], memories["total"]
    ))
    print("preferences   : {0} active, {1} total".format(
        stats["preferences"]["active"], stats["preferences"]["total"]
    ))
    print("project state : {0} current, {1} total".format(
        stats["project_state"]["current"], stats["project_state"]["total"]
    ))
    provider = stats["provider"]
    print("provider      : {0}".format(
        "none" if provider is None else "{0}/{1}".format(provider["name"], provider["model"])
    ))
    embedding = stats.get("embedding_provider")
    print("embedding     : {0}".format(
        "none" if embedding is None else embedding["identifier"]
    ))
    index_info = stats.get("vector_index") or {}
    print("vector index  : {0} rows, models={1}".format(
        index_info.get("rows", 0), index_info.get("models", [])
    ))
    retrieval = stats.get("retrieval") or {}
    print("retrieval     : modes={0} semantic_available={1}".format(
        retrieval.get("modes"), retrieval.get("semantic_available")
    ))
    print("extractor     : {0}".format(stats["extractor"]))
    formation = stats.get("formation") or {}
    print("llm formation : {0}".format(formation.get("llm_formation")))
    if stats["file_bytes"] is not None:
        print("file size     : {0} bytes".format(stats["file_bytes"]))
    return EXIT_OK


_HANDLERS = {
    "init": _cmd_init,
    "remember": _cmd_remember,
    "recall": _cmd_recall,
    "forget": _cmd_forget,
    "observe": _cmd_observe,
    "learn": _cmd_learn,
    "propose": _cmd_propose,
    "event": _cmd_event,
    "pref": _cmd_pref,
    "project": _cmd_project,
    "index": _cmd_index,
    "ask": _cmd_ask,
    "export": _cmd_export,
    "snapshot": _cmd_snapshot,
    "stats": _cmd_stats,
}


def main(argv=None):
    # type: (Optional[List[str]]) -> int
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK

    handler = _HANDLERS[args.command]
    use_json = bool(getattr(args, "json", False))

    core = None
    try:
        core = MemoryCore.open(
            db_path=_opt(args, "db"),
            config_path=_opt(args, "config"),
            models_path=_opt(args, "models"),
            provider_name=_opt(args, "provider"),
            model=_opt(args, "model"),
            embedding_provider_name=_opt(args, "embedding_provider"),
            embedding_model=_opt(args, "embedding_model"),
        )
        return handler(core, args, use_json)
    except MemoryCoreError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        return EXIT_ERROR
    finally:
        if core is not None:
            core.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
