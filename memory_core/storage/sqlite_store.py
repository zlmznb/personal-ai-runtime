"""SQLite persistence: the single long-term source of truth.

Hard rules enforced here (architecture red lines 1, 2 and 4):

* This package must not import a model SDK, an HTTP client, or any provider.
  Storage is "dumb": it computes no embeddings and makes no similarity
  judgements. Vectors, when they arrive, are passed in.
* Every write re-runs the domain guards, so no code path - including a future
  LLM extractor - can persist a malformed record.
* Removal is logical. History is never destroyed by normal operation.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from memory_core.domain import guards
from memory_core.domain.enums import MemoryStatus
from memory_core.domain.errors import NotFoundError, StorageError
from memory_core.domain.ids import utc_now_iso
from memory_core.domain.records import Event, Memory, Preference, ProjectState
from memory_core.domain.validation import require_enum
from memory_core.retrieval import tokenize

SCHEMA_VERSION = 2

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

#: Source tables, in a fixed order, for deterministic export.
#: Derived tables (``*_fts``, ``embeddings``) are deliberately excluded: they
#: are rebuildable, so they are not part of the truth that must round-trip.
SOURCE_TABLES = ("meta", "events", "memories", "preferences", "project_state")

#: Derived tables. Safe to delete; always rebuildable from SOURCE_TABLES.
DERIVED_TABLES = ("memories_fts", "preferences_fts", "embeddings")

#: Non-additive migration steps, keyed by the version they produce.
#: v1 -> v2 is purely additive (a new table), which ``schema.sql`` already
#: creates via ``CREATE TABLE IF NOT EXISTS``, so no statements are needed -
#: but the mechanism is wired up and the result is verified.
MIGRATIONS = {
    2: (),
}

_ORDER_BY = {
    "meta": "key",
    "events": "id",
    "memories": "id",
    "preferences": "id",
    "project_state": "id",
}

_MEMORY_INSERT = """
INSERT INTO memories (
    id, kind, scope, subject, content, structured, tags, source,
    confidence, salience, status, valid_from, valid_to, superseded_by,
    generated_by, created_at, updated_at, revision
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_MEMORY_COLUMNS = (
    "id, kind, scope, subject, content, structured, tags, source, "
    "confidence, salience, status, valid_from, valid_to, superseded_by, "
    "generated_by, created_at, updated_at, revision"
)

_EVENT_INSERT = """
INSERT INTO events (id, kind, role, content, session_id, scope, metadata, created_at, source)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_PREFERENCE_INSERT = """
INSERT INTO preferences (id, scope, key, value, statement, confidence, source, created_at, superseded_by)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_PROJECT_STATE_INSERT = """
INSERT INTO project_state (id, project_id, scope, state, note, source, created_at, superseded_by)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


def _dump(value):
    # type: (Any) -> str
    """Canonical JSON encoding for text columns.

    ``ensure_ascii=False`` keeps stored content human-readable and diffable;
    ``allow_nan=False`` refuses values that would corrupt an export.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _row_dict(row):
    # type: (sqlite3.Row) -> Dict[str, Any]
    return dict(row)


def _scope_filter_positional(column, scope, scopes, exclude_prefixes):
    # type: (str, Any, Any, Any) -> Tuple[List[str], List[Any]]
    """Build scope-visibility clauses with positional placeholders."""
    clauses = []  # type: List[str]
    values = []  # type: List[Any]
    if scope is not None:
        clauses.append("{0} = ?".format(column))
        values.append(scope)
    elif scopes:
        clauses.append(
            "{0} IN ({1})".format(column, ",".join("?" * len(scopes)))
        )
        values.extend(scopes)
    for prefix in exclude_prefixes or ():
        clauses.append("{0} NOT LIKE ?".format(column))
        values.append("{0}:%".format(prefix))
    return clauses, values


def _scope_filter_named(column, scope, scopes, exclude_prefixes, params):
    # type: (str, Any, Any, Any, Dict[str, Any]) -> List[str]
    """Build scope-visibility clauses with named placeholders."""
    clauses = []  # type: List[str]
    if scope is not None:
        clauses.append("{0} = :scope".format(column))
        params["scope"] = scope
    elif scopes:
        names = []
        for index, value in enumerate(scopes):
            key = "scope{0}".format(index)
            names.append(":" + key)
            params[key] = value
        clauses.append("{0} IN ({1})".format(column, ", ".join(names)))
    for index, prefix in enumerate(exclude_prefixes or ()):
        key = "excl{0}".format(index)
        clauses.append("{0} NOT LIKE :{1}".format(column, key))
        params[key] = "{0}:%".format(prefix)
    return clauses


class SqliteStore(object):
    """A thin, validated, transactional wrapper around one SQLite file."""

    def __init__(self, path, timeout=10.0):
        # type: (str, float) -> None
        self.path = str(path)
        self._is_memory = self.path == ":memory:"
        self._connection = None  # type: Optional[sqlite3.Connection]
        self._connect(timeout)
        self._ensure_schema()

    # -- connection management --------------------------------------------

    def _connect(self, timeout):
        # type: (float) -> None
        if not self._is_memory:
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
        try:
            connection = sqlite3.connect(self.path, timeout=timeout, isolation_level=None)
        except sqlite3.Error as exc:
            raise StorageError("could not open {0}: {1}".format(self.path, exc))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        # DELETE journal mode is deliberate: after every commit the entire
        # database is contained in one file. That is what makes the byte-level
        # model-swap assertion (MSIT-1) meaningful and keeps "one file is the
        # source of truth" literally true.
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        self._connection = connection

    @property
    def connection(self):
        # type: () -> sqlite3.Connection
        if self._connection is None:
            raise StorageError("store is closed")
        return self._connection

    def close(self):
        # type: () -> None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        # type: () -> "SqliteStore"
        return self

    def __exit__(self, exc_type, exc, tb):
        # type: (Any, Any, Any) -> None
        self.close()

    @contextlib.contextmanager
    def transaction(self):
        # type: () -> Iterator[sqlite3.Connection]
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        # Deferred FK checking lets us mark an old row as superseded by a row we
        # are about to insert, without violating the self-referencing FK.
        connection.execute("PRAGMA defer_foreign_keys = ON")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    # -- schema ------------------------------------------------------------

    def _ensure_schema(self):
        # type: () -> None
        try:
            with open(_SCHEMA_PATH, "r", encoding="utf-8") as handle:
                script = handle.read()
        except OSError as exc:
            raise StorageError("could not read schema.sql: {0}".format(exc))
        try:
            self.connection.executescript(script)
        except sqlite3.Error as exc:
            raise StorageError("schema initialisation failed: {0}".format(exc))

        existing = self.get_meta("schema_version")
        if existing is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
        else:
            try:
                version = int(existing)
            except (TypeError, ValueError):
                raise StorageError("invalid schema_version {0!r}".format(existing))
            if version > SCHEMA_VERSION:
                raise StorageError(
                    "database schema version {0} is newer than this code supports "
                    "({1}); refusing to open".format(version, SCHEMA_VERSION)
                )
            if version < SCHEMA_VERSION:
                self._migrate(version)
                self.set_meta("schema_version", str(SCHEMA_VERSION))
        self._verify_schema()

    def _migrate(self, from_version):
        # type: (int) -> None
        """Apply non-additive migration steps up to ``SCHEMA_VERSION``."""
        for version in range(from_version + 1, SCHEMA_VERSION + 1):
            statements = MIGRATIONS.get(version)
            if statements is None:
                raise StorageError(
                    "no migration defined from schema version {0} to {1}".format(
                        version - 1, version
                    )
                )
            for statement in statements:
                try:
                    self.connection.execute(statement)
                except sqlite3.Error as exc:
                    raise StorageError(
                        "migration to version {0} failed: {1}".format(version, exc)
                    )

    def _verify_schema(self):
        # type: () -> None
        """Fail loudly if the on-disk schema is not actually usable.

        A migration that silently did nothing is worse than one that errors.
        """
        expected = set(SOURCE_TABLES) | set(DERIVED_TABLES)
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        present = {str(row["name"]) for row in rows}
        missing = sorted(expected - present)
        if missing:
            raise StorageError(
                "schema is incomplete; missing {0}".format(missing)
            )

    @property
    def schema_version(self):
        # type: () -> int
        value = self.get_meta("schema_version")
        return int(value) if value is not None else SCHEMA_VERSION

    def get_meta(self, key):
        # type: (str) -> Optional[str]
        try:
            row = self.connection.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError("meta read failed: {0}".format(exc))
        return None if row is None else str(row["value"])

    def set_meta(self, key, value):
        # type: (str, str) -> None
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # -- events (append-only) ---------------------------------------------

    def insert_event(self, event):
        # type: (Event) -> Event
        guards.validate_event(event)
        try:
            with self.transaction() as connection:
                connection.execute(
                    _EVENT_INSERT,
                    (
                        event.id,
                        event.kind,
                        event.role,
                        event.content,
                        event.session_id,
                        event.scope,
                        _dump(event.metadata),
                        event.created_at,
                        event.source,
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError("insert_event failed: {0}".format(exc))
        return event

    def get_event(self, event_id):
        # type: (str) -> Optional[Event]
        row = self.connection.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        return None if row is None else Event.from_row(_row_dict(row))

    def list_events(self, scope=None, session_id=None, limit=100, offset=0):
        # type: (Optional[str], Optional[str], int, int) -> List[Event]
        clauses = []
        params = []  # type: List[Any]
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([int(limit), int(offset)])
        rows = self.connection.execute(
            "SELECT * FROM events" + where + " ORDER BY created_at ASC, id ASC LIMIT ? OFFSET ?",
            tuple(params),
        ).fetchall()
        return [Event.from_row(_row_dict(row)) for row in rows]

    def count_events(self):
        # type: () -> int
        return int(self.connection.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"])

    # -- memories ----------------------------------------------------------

    def insert_memory(self, memory):
        # type: (Memory) -> Memory
        guards.validate_memory(memory)
        try:
            with self.transaction() as connection:
                self._insert_memory_row(connection, memory)
        except sqlite3.Error as exc:
            raise StorageError("insert_memory failed: {0}".format(exc))
        return memory

    def _insert_memory_row(self, connection, memory):
        # type: (sqlite3.Connection, Memory) -> None
        connection.execute(
            _MEMORY_INSERT,
            (
                memory.id,
                memory.kind,
                memory.scope,
                memory.subject,
                memory.content,
                _dump(memory.structured),
                _dump(list(memory.tags)),
                _dump(memory.source),
                memory.confidence,
                memory.salience,
                memory.status,
                memory.valid_from,
                memory.valid_to,
                memory.superseded_by,
                memory.generated_by,
                memory.created_at,
                memory.updated_at,
                memory.revision,
            ),
        )
        connection.execute(
            "INSERT INTO memories_fts (memory_id, content, subject, tags) VALUES (?, ?, ?, ?)",
            (
                memory.id,
                tokenize.normalize_search_text(memory.content),
                tokenize.normalize_search_text(memory.subject),
                tokenize.normalize_search_text(" ".join(memory.tags)),
            ),
        )

    def insert_memories(self, memories):
        # type: (Any) -> int
        """Bulk insert in a single transaction.

        Interactive writes stay one-per-transaction (see :meth:`insert_memory`);
        this exists for bulk loading, where per-row fsync would dominate the
        runtime. Every record is validated before anything is written, so a bad
        record in the batch rejects the whole batch rather than half-writing it.
        """
        prepared = list(memories)
        for memory in prepared:
            guards.validate_memory(memory)
        if not prepared:
            return 0
        try:
            with self.transaction() as connection:
                for memory in prepared:
                    self._insert_memory_row(connection, memory)
        except sqlite3.Error as exc:
            raise StorageError("insert_memories failed: {0}".format(exc))
        return len(prepared)

    def get_memory(self, memory_id):
        # type: (str) -> Optional[Memory]
        row = self.connection.execute(
            "SELECT " + _MEMORY_COLUMNS + " FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return None if row is None else Memory.from_row(_row_dict(row))

    def require_memory(self, memory_id):
        # type: (str) -> Memory
        memory = self.get_memory(memory_id)
        if memory is None:
            raise NotFoundError("memory {0} not found".format(memory_id))
        return memory

    def supersede_memory(self, old_id, new_memory, *, reason=None, updated_at=None):
        # type: (str, Memory, Optional[str], Optional[str]) -> Memory
        """Insert ``new_memory`` and mark ``old_id`` superseded. History kept."""
        guards.validate_memory(new_memory)
        try:
            with self.transaction() as connection:
                row = connection.execute(
                    "SELECT id, status, source, revision FROM memories WHERE id = ?",
                    (old_id,),
                ).fetchone()
                if row is None:
                    raise NotFoundError("memory {0} not found".format(old_id))
                source = dict(new_memory.source)
                source.setdefault("supersedes", old_id)
                if reason:
                    source.setdefault("reason", reason)
                new_memory = Memory(**dict(new_memory.as_dict(), source=source))
                guards.validate_memory(new_memory)
                stale = connection.execute(
                    "UPDATE memories SET status = 'superseded', superseded_by = ?, "
                    "updated_at = ?, revision = revision + 1 WHERE id = ?",
                    (
                        new_memory.id,
                        updated_at or new_memory.created_at,
                        old_id,
                    ),
                )
                if stale.rowcount != 1:
                    raise StorageError("supersede_memory: expected to update 1 row")
                self._insert_memory_row(connection, new_memory)
        except sqlite3.Error as exc:
            raise StorageError("supersede_memory failed: {0}".format(exc))
        return new_memory

    def set_memory_status(self, memory_id, status, updated_at=None):
        # type: (str, str, Optional[str]) -> Memory
        clean = require_enum(MemoryStatus, status, "status")
        try:
            with self.transaction() as connection:
                cursor = connection.execute(
                    "UPDATE memories SET status = ?, updated_at = ?, revision = revision + 1 "
                    "WHERE id = ?",
                    (clean, updated_at or utc_now_iso(), memory_id),
                )
                if cursor.rowcount != 1:
                    raise NotFoundError("memory {0} not found".format(memory_id))
        except sqlite3.Error as exc:
            raise StorageError("set_memory_status failed: {0}".format(exc))
        return self.require_memory(memory_id)

    def delete_memory(self, memory_id):
        # type: (str) -> bool
        """Hard delete. Prefer :meth:`set_memory_status` for normal use."""
        try:
            with self.transaction() as connection:
                removed = connection.execute(
                    "DELETE FROM memories WHERE id = ?", (memory_id,)
                ).rowcount
                connection.execute(
                    "DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,)
                )
        except sqlite3.Error as exc:
            raise StorageError("delete_memory failed: {0}".format(exc))
        return removed == 1

    def list_memories(
        self,
        scope=None,
        kinds=None,
        statuses=("active",),
        limit=100,
        offset=0,
        scopes=None,
        exclude_prefixes=(),
    ):
        # type: (Optional[str], Optional[Sequence[str]], Sequence[str], int, int, Any, Any) -> List[Memory]
        clauses = []  # type: List[str]
        params = []  # type: List[Any]
        scope_clauses, scope_values = _scope_filter_positional(
            "scope", scope, scopes, exclude_prefixes
        )
        clauses.extend(scope_clauses)
        params.extend(scope_values)
        if kinds:
            clauses.append("kind IN ({0})".format(",".join("?" * len(kinds))))
            params.extend(kinds)
        if statuses:
            clauses.append("status IN ({0})".format(",".join("?" * len(statuses))))
            params.extend(statuses)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([int(limit), int(offset)])
        rows = self.connection.execute(
            "SELECT " + _MEMORY_COLUMNS + " FROM memories" + where
            + " ORDER BY created_at DESC, id ASC LIMIT ? OFFSET ?",
            tuple(params),
        ).fetchall()
        return [Memory.from_row(_row_dict(row)) for row in rows]

    def list_scopes(self):
        # type: () -> List[str]
        """Every scope that currently holds a row, in any source table."""
        rows = self.connection.execute(
            "SELECT scope FROM memories "
            "UNION SELECT scope FROM preferences "
            "UNION SELECT scope FROM project_state "
            "ORDER BY scope ASC"
        ).fetchall()
        return [str(row["scope"]) for row in rows]

    def count_memories(self, status=None):
        # type: (Optional[str]) -> int
        if status is None:
            sql, params = "SELECT COUNT(*) AS n FROM memories", ()
        else:
            sql, params = "SELECT COUNT(*) AS n FROM memories WHERE status = ?", (status,)
        return int(self.connection.execute(sql, params).fetchone()["n"])

    def search_memories(
        self,
        match_expression,
        needle="",
        scope=None,
        kinds=None,
        statuses=("active",),
        limit=10,
        scopes=None,
        exclude_prefixes=(),
    ):
        # type: (str, str, Optional[str], Optional[Sequence[str]], Sequence[str], int, Any, Any) -> List[Tuple[Memory, float, bool]]
        """BM25 search. Returns ``(memory, rank, phrase_hit)`` triples.

        ``rank`` is raw FTS5 bm25 (more negative is better). Scoring is applied
        by ``retrieval.fusion`` so that all ranking policy lives in one place.

        ``scopes`` / ``exclude_prefixes`` implement scope visibility; see
        ``domain.scopes``. Both default to "no filter", so v0.2 callers are
        unaffected.
        """
        params = {"match": match_expression, "needle": needle, "limit": int(limit)}  # type: Dict[str, Any]
        clauses = ["memories_fts MATCH :match"]
        clauses.extend(
            _scope_filter_named("m.scope", scope, scopes, exclude_prefixes, params)
        )
        if kinds:
            names = []
            for index, kind in enumerate(kinds):
                name = "kind{0}".format(index)
                names.append(":" + name)
                params[name] = kind
            clauses.append("m.kind IN ({0})".format(", ".join(names)))
        if statuses:
            names = []
            for index, status in enumerate(statuses):
                name = "status{0}".format(index)
                names.append(":" + name)
                params[name] = status
            clauses.append("m.status IN ({0})".format(", ".join(names)))
        sql = (
            "SELECT m.*"
            ", bm25(memories_fts, 0.0, 1.0, 0.5, 0.3) AS rank"
            ", instr(memories_fts.content, :needle) AS phrase_pos"
            " FROM memories_fts JOIN memories m ON m.id = memories_fts.memory_id"
            " WHERE " + " AND ".join(clauses)
            + " ORDER BY rank ASC, m.created_at ASC, m.id ASC LIMIT :limit"
        )
        rows = self.connection.execute(sql, params).fetchall()
        results = []
        for row in rows:
            data = _row_dict(row)
            rank = float(data.pop("rank"))
            phrase_pos = int(data.pop("phrase_pos") or 0)
            results.append((Memory.from_row(data), rank, bool(needle) and phrase_pos > 0))
        return results

    # -- preferences -------------------------------------------------------

    def insert_preference(self, preference, supersedes=None):
        # type: (Preference, Optional[str]) -> Preference
        """Insert a preference, superseding the previous active value for its key."""
        guards.validate_preference(preference)
        try:
            with self.transaction() as connection:
                if supersedes is None:
                    current = connection.execute(
                        "SELECT id FROM preferences WHERE scope = ? AND key = ? "
                        "AND superseded_by IS NULL",
                        (preference.scope, preference.key),
                    ).fetchone()
                    supersedes = None if current is None else str(current["id"])
                if supersedes is not None:
                    connection.execute(
                        "UPDATE preferences SET superseded_by = ? WHERE id = ?",
                        (preference.id, supersedes),
                    )
                connection.execute(
                    _PREFERENCE_INSERT,
                    (
                        preference.id,
                        preference.scope,
                        preference.key,
                        _dump(preference.value),
                        preference.statement,
                        preference.confidence,
                        _dump(preference.source),
                        preference.created_at,
                        None,
                    ),
                )
                connection.execute(
                    "INSERT INTO preferences_fts (preference_id, key, statement) VALUES (?, ?, ?)",
                    (
                        preference.id,
                        tokenize.normalize_search_text(preference.key),
                        tokenize.normalize_search_text(preference.statement),
                    ),
                )
                # FK is deferred, so verify the supersede chain ended up sane.
                dangling = connection.execute(
                    "SELECT COUNT(*) AS n FROM preferences p WHERE p.superseded_by IS NOT NULL "
                    "AND NOT EXISTS (SELECT 1 FROM preferences q WHERE q.id = p.superseded_by)"
                ).fetchone()["n"]
                if dangling:
                    raise StorageError("preference supersede chain is inconsistent")
        except sqlite3.Error as exc:
            raise StorageError("insert_preference failed: {0}".format(exc))
        return preference

    def get_preference(self, key, scope="global"):
        # type: (str, str) -> Optional[Preference]
        row = self.connection.execute(
            "SELECT * FROM preferences WHERE scope = ? AND key = ? AND superseded_by IS NULL",
            (scope, key),
        ).fetchone()
        return None if row is None else Preference.from_row(_row_dict(row))

    def get_preference_by_id(self, preference_id):
        # type: (str) -> Optional[Preference]
        row = self.connection.execute(
            "SELECT * FROM preferences WHERE id = ?", (preference_id,)
        ).fetchone()
        return None if row is None else Preference.from_row(_row_dict(row))

    def list_preferences(self, scope=None, include_superseded=False, scopes=None, exclude_prefixes=()):
        # type: (Optional[str], bool, Any, Any) -> List[Preference]
        clauses = []  # type: List[str]
        params = []  # type: List[Any]
        scope_clauses, scope_values = _scope_filter_positional(
            "scope", scope, scopes, exclude_prefixes
        )
        clauses.extend(scope_clauses)
        params.extend(scope_values)
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.connection.execute(
            "SELECT * FROM preferences" + where + " ORDER BY scope ASC, key ASC, created_at ASC, id ASC",
            tuple(params),
        ).fetchall()
        return [Preference.from_row(_row_dict(row)) for row in rows]

    def preference_history(self, key, scope="global"):
        # type: (str, str) -> List[Preference]
        rows = self.connection.execute(
            "SELECT * FROM preferences WHERE scope = ? AND key = ? ORDER BY created_at ASC, id ASC",
            (scope, key),
        ).fetchall()
        return [Preference.from_row(_row_dict(row)) for row in rows]

    def count_preferences(self, active_only=True):
        # type: (bool) -> int
        if active_only:
            sql = "SELECT COUNT(*) AS n FROM preferences WHERE superseded_by IS NULL"
        else:
            sql = "SELECT COUNT(*) AS n FROM preferences"
        return int(self.connection.execute(sql).fetchone()["n"])

    def search_preferences(self, match_expression, needle="", scope=None, limit=10, scopes=None, exclude_prefixes=()):
        # type: (str, str, Optional[str], int, Any, Any) -> List[Tuple[Preference, float, bool]]
        params = {"match": match_expression, "needle": needle, "limit": int(limit)}  # type: Dict[str, Any]
        clauses = ["preferences_fts MATCH :match", "p.superseded_by IS NULL"]
        clauses.extend(
            _scope_filter_named("p.scope", scope, scopes, exclude_prefixes, params)
        )
        sql = (
            "SELECT p.*, bm25(preferences_fts, 0.0, 1.0, 0.5) AS rank"
            ", instr(preferences_fts.key || ' ' || preferences_fts.statement, :needle) AS phrase_pos"
            " FROM preferences_fts JOIN preferences p ON p.id = preferences_fts.preference_id"
            " WHERE " + " AND ".join(clauses)
            + " ORDER BY rank ASC, p.created_at ASC, p.id ASC LIMIT :limit"
        )
        rows = self.connection.execute(sql, params).fetchall()
        results = []
        for row in rows:
            data = _row_dict(row)
            rank = float(data.pop("rank"))
            phrase_pos = int(data.pop("phrase_pos") or 0)
            results.append((Preference.from_row(data), rank, bool(needle) and phrase_pos > 0))
        return results

    # -- project state -----------------------------------------------------

    def set_project_state(self, record):
        # type: (ProjectState) -> ProjectState
        """Write a new current value, superseding the previous one."""
        guards.validate_project_state(record)
        try:
            with self.transaction() as connection:
                current = connection.execute(
                    "SELECT id FROM project_state WHERE project_id = ? AND scope = ? "
                    "AND superseded_by IS NULL",
                    (record.project_id, record.scope),
                ).fetchone()
                if current is not None:
                    connection.execute(
                        "UPDATE project_state SET superseded_by = ? WHERE id = ?",
                        (record.id, str(current["id"])),
                    )
                connection.execute(
                    _PROJECT_STATE_INSERT,
                    (
                        record.id,
                        record.project_id,
                        record.scope,
                        _dump(record.state),
                        record.note,
                        _dump(record.source),
                        record.created_at,
                        None,
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError("set_project_state failed: {0}".format(exc))
        return record

    def get_project_state(self, project_id, scope=None):
        # type: (str, Optional[str]) -> Optional[ProjectState]
        if scope is None:
            scope = "project:{0}".format(project_id)
        row = self.connection.execute(
            "SELECT * FROM project_state WHERE project_id = ? AND scope = ? AND superseded_by IS NULL",
            (project_id, scope),
        ).fetchone()
        return None if row is None else ProjectState.from_row(_row_dict(row))

    def project_state_history(self, project_id, scope=None):
        # type: (str, Optional[str]) -> List[ProjectState]
        if scope is None:
            scope = "project:{0}".format(project_id)
        rows = self.connection.execute(
            "SELECT * FROM project_state WHERE project_id = ? AND scope = ? "
            "ORDER BY created_at ASC, id ASC",
            (project_id, scope),
        ).fetchall()
        return [ProjectState.from_row(_row_dict(row)) for row in rows]

    def list_project_states(self, scope=None, scopes=None, exclude_prefixes=()):
        # type: (Optional[str], Any, Any) -> List[ProjectState]
        scope_clauses, scope_values = _scope_filter_positional(
            "scope", scope, scopes, exclude_prefixes
        )
        clauses = ["superseded_by IS NULL"] + scope_clauses
        where = " WHERE " + " AND ".join(clauses)
        rows = self.connection.execute(
            "SELECT * FROM project_state" + where + " ORDER BY scope ASC, project_id ASC",
            tuple(scope_values),
        ).fetchall()
        return [ProjectState.from_row(_row_dict(row)) for row in rows]

    def count_project_states(self, current_only=True):
        # type: (bool) -> int
        if current_only:
            sql = "SELECT COUNT(*) AS n FROM project_state WHERE superseded_by IS NULL"
        else:
            sql = "SELECT COUNT(*) AS n FROM project_state"
        return int(self.connection.execute(sql).fetchone()["n"])

    # -- derived index maintenance ----------------------------------------

    def rebuild_fts(self):
        # type: () -> Dict[str, int]
        """Rebuild both FTS indexes from the source tables.

        This is what makes the "vector/lexical index is disposable" property
        concrete: the index can be destroyed and regenerated at any time.
        """
        with self.transaction() as connection:
            connection.execute("DELETE FROM memories_fts")
            connection.execute("DELETE FROM preferences_fts")
            for row in connection.execute(
                "SELECT id, content, subject, tags FROM memories"
            ).fetchall():
                tags = json.loads(row["tags"]) if row["tags"] else []
                connection.execute(
                    "INSERT INTO memories_fts (memory_id, content, subject, tags) VALUES (?, ?, ?, ?)",
                    (
                        row["id"],
                        tokenize.normalize_search_text(row["content"]),
                        tokenize.normalize_search_text(row["subject"]),
                        tokenize.normalize_search_text(" ".join(tags)),
                    ),
                )
            for row in connection.execute(
                "SELECT id, key, statement FROM preferences"
            ).fetchall():
                connection.execute(
                    "INSERT INTO preferences_fts (preference_id, key, statement) VALUES (?, ?, ?)",
                    (
                        row["id"],
                        tokenize.normalize_search_text(row["key"]),
                        tokenize.normalize_search_text(row["statement"] or ""),
                    ),
                )
        return {
            "memories_fts": int(
                self.connection.execute("SELECT COUNT(*) AS n FROM memories_fts").fetchone()["n"]
            ),
            "preferences_fts": int(
                self.connection.execute("SELECT COUNT(*) AS n FROM preferences_fts").fetchone()["n"]
            ),
        }

    # -- derived vector index: data access only ---------------------------
    #
    # Storage persists opaque vector bytes and never performs vector maths.
    # Cosine similarity lives in retrieval.vector_index. The only contract
    # enforced here is the encoding length: vectors are little-endian float32,
    # so a blob must be exactly ``dim * 4`` bytes.

    FLOAT32_BYTES = 4

    def upsert_embedding(self, memory_id, model_id, dim, vector_blob, created_at=None):
        # type: (str, str, int, bytes, Optional[str]) -> None
        if not isinstance(memory_id, str) or not memory_id:
            raise StorageError("memory_id must be a non-empty string")
        if not isinstance(model_id, str) or not model_id:
            raise StorageError("model_id must be a non-empty string")
        try:
            dim = int(dim)
        except (TypeError, ValueError):
            raise StorageError("dim must be an integer")
        if dim <= 0:
            raise StorageError("dim must be positive")
        if not isinstance(vector_blob, (bytes, bytearray, memoryview)):
            raise StorageError("vector_blob must be bytes")
        payload = bytes(vector_blob)
        if len(payload) != dim * self.FLOAT32_BYTES:
            raise StorageError(
                "vector length {0} does not match dim {1} at 4 bytes per element".format(
                    len(payload), dim
                )
            )
        try:
            with self.transaction() as connection:
                connection.execute(
                    "INSERT INTO embeddings (memory_id, model_id, dim, vector, created_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(memory_id, model_id) DO UPDATE SET "
                    "dim = excluded.dim, vector = excluded.vector, created_at = excluded.created_at",
                    (memory_id, model_id, dim, payload, created_at or utc_now_iso()),
                )
        except sqlite3.Error as exc:
            raise StorageError("upsert_embedding failed: {0}".format(exc))

    def upsert_embeddings(self, rows, model_id=None):
        # type: (Any, Optional[str]) -> int
        """Bulk upsert of ``(memory_id, dim, vector_blob)`` tuples in one transaction."""
        prepared = []
        for row in rows:
            memory_id, dim, blob = row
            prepared.append((memory_id, model_id, int(dim), bytes(blob), utc_now_iso()))
        if not prepared:
            return 0
        try:
            with self.transaction() as connection:
                connection.executemany(
                    "INSERT INTO embeddings (memory_id, model_id, dim, vector, created_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(memory_id, model_id) DO UPDATE SET "
                    "dim = excluded.dim, vector = excluded.vector, created_at = excluded.created_at",
                    prepared,
                )
        except sqlite3.Error as exc:
            raise StorageError("upsert_embeddings failed: {0}".format(exc))
        return len(prepared)

    def get_embedding(self, memory_id, model_id):
        # type: (str, str) -> Optional[Tuple[int, bytes]]
        row = self.connection.execute(
            "SELECT dim, vector FROM embeddings WHERE memory_id = ? AND model_id = ?",
            (memory_id, model_id),
        ).fetchone()
        if row is None:
            return None
        return int(row["dim"]), bytes(row["vector"])

    def iter_embeddings(self, model_id=None):
        # type: (Optional[str]) -> Iterator[Tuple[str, str, int, bytes]]
        if model_id is None:
            cursor = self.connection.execute(
                "SELECT memory_id, model_id, dim, vector FROM embeddings ORDER BY model_id ASC, memory_id ASC"
            )
        else:
            cursor = self.connection.execute(
                "SELECT memory_id, model_id, dim, vector FROM embeddings WHERE model_id = ? "
                "ORDER BY memory_id ASC",
                (model_id,),
            )
        for row in cursor:
            yield str(row["memory_id"]), str(row["model_id"]), int(row["dim"]), bytes(row["vector"])

    def delete_embeddings(self, model_id=None):
        # type: (Optional[str]) -> int
        try:
            with self.transaction() as connection:
                if model_id is None:
                    removed = connection.execute("DELETE FROM embeddings").rowcount
                else:
                    removed = connection.execute(
                        "DELETE FROM embeddings WHERE model_id = ?", (model_id,)
                    ).rowcount
        except sqlite3.Error as exc:
            raise StorageError("delete_embeddings failed: {0}".format(exc))
        return int(removed)

    def count_embeddings(self, model_id=None):
        # type: (Optional[str]) -> int
        if model_id is None:
            row = self.connection.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) AS n FROM embeddings WHERE model_id = ?", (model_id,)
            ).fetchone()
        return int(row["n"])

    def embedding_model_ids(self):
        # type: () -> List[str]
        rows = self.connection.execute(
            "SELECT DISTINCT model_id FROM embeddings ORDER BY model_id ASC"
        ).fetchall()
        return [str(row["model_id"]) for row in rows]

    def memory_ids_missing_embedding(self, model_id):
        # type: (str) -> List[str]
        rows = self.connection.execute(
            "SELECT m.id AS id FROM memories m WHERE m.status = 'active' "
            "AND NOT EXISTS (SELECT 1 FROM embeddings e "
            "                WHERE e.memory_id = m.id AND e.model_id = ?) "
            "ORDER BY m.id ASC",
            (model_id,),
        ).fetchall()
        return [str(row["id"]) for row in rows]

    def get_memories_by_ids(self, memory_ids):
        # type: (Any) -> List[Memory]
        """Load memories by id, preserving the order of ``memory_ids``."""
        wanted = [str(value) for value in memory_ids]
        if not wanted:
            return []
        placeholders = ",".join("?" * len(wanted))
        rows = self.connection.execute(
            "SELECT " + _MEMORY_COLUMNS + " FROM memories WHERE id IN ({0})".format(placeholders),
            tuple(wanted),
        ).fetchall()
        by_id = {str(row["id"]): Memory.from_row(_row_dict(row)) for row in rows}
        return [by_id[memory_id] for memory_id in wanted if memory_id in by_id]

    def page_size_bytes(self):
        # type: () -> int
        """Total size of the database file in bytes, from SQLite's own accounting."""
        page_size = int(self.connection.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(self.connection.execute("PRAGMA page_count").fetchone()[0])
        return page_size * page_count

    # -- export / snapshot / stats ----------------------------------------

    def iter_rows(self, table):
        # type: (str) -> Iterator[Dict[str, Any]]
        if table not in SOURCE_TABLES:
            raise StorageError("refusing to export unknown table {0!r}".format(table))
        order = _ORDER_BY[table]
        cursor = self.connection.execute(
            "SELECT * FROM {0} ORDER BY {1} ASC".format(table, order)
        )
        for row in cursor:
            yield _row_dict(row)

    def export_jsonl(self):
        # type: () -> str
        """Serialise every source table as deterministic JSON Lines.

        Determinism is the point: an unchanged database must export
        byte-identically, which makes export a usable integrity check.
        """
        lines = []
        for table in SOURCE_TABLES:
            for row in self.iter_rows(table):
                lines.append(
                    json.dumps(
                        {"table": table, "row": row},
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
        return "\n".join(lines) + ("\n" if lines else "")

    def snapshot(self, destination):
        # type: (str) -> str
        """Write a consistent copy of the database using SQLite's backup API."""
        target = str(destination)
        parent = os.path.dirname(os.path.abspath(target))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        try:
            destination_connection = sqlite3.connect(target)
        except sqlite3.Error as exc:
            raise StorageError("could not open snapshot target: {0}".format(exc))
        try:
            self.connection.backup(destination_connection)
            destination_connection.commit()
        except sqlite3.Error as exc:
            raise StorageError("snapshot failed: {0}".format(exc))
        finally:
            destination_connection.close()
        return target

    def stats(self):
        # type: () -> Dict[str, Any]
        result = {
            "path": self.path,
            "schema_version": self.schema_version,
            "events": self.count_events(),
            "memories": {
                "active": self.count_memories("active"),
                "superseded": self.count_memories("superseded"),
                "archived": self.count_memories("archived"),
                "deleted": self.count_memories("deleted"),
                "total": self.count_memories(None),
            },
            "preferences": {
                "active": self.count_preferences(True),
                "total": self.count_preferences(False),
            },
            "project_state": {
                "current": self.count_project_states(True),
                "total": self.count_project_states(False),
            },
            "vector_index": {
                "rows": self.count_embeddings(),
                "models": self.embedding_model_ids(),
            },
        }
        if not self._is_memory and os.path.exists(self.path):
            result["file_bytes"] = os.path.getsize(self.path)
        else:
            result["file_bytes"] = None
        return result


__all__ = ["SqliteStore", "SCHEMA_VERSION", "SOURCE_TABLES"]
