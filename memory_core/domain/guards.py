"""Record-level guards: the only supported way to build or validate a record.

Two functions per record type:

* ``build_*``  - construct a record from untyped input (CLI, config, a model),
  validating every field and minting id/timestamp.
* ``validate_*`` - re-validate a record that already exists.

Storage calls the ``validate_*`` half on every write, so there is no code path -
including a future LLM extractor - that can persist a malformed record
(architecture red line 4).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from memory_core.domain import ids
from memory_core.domain.enums import (
    EventKind,
    MemoryKind,
    MemoryStatus,
    Role,
)
from memory_core.domain.errors import ValidationError
from memory_core.domain.records import Event, Memory, Preference, ProjectState
from memory_core.domain.validation import (
    MAX_SHORT_CHARS,
    normalize_tags,
    require_enum,
    require_identifier,
    require_json_object,
    require_json_value,
    require_scope,
    require_text,
    require_unit_float,
)


def _require_optional_timestamp(value, field):
    # type: (Any, str) -> Optional[str]
    if value is None:
        return None
    if not ids.is_iso8601(value):
        raise ValidationError(
            "{0} must be an ISO-8601 UTC timestamp ending in 'Z'".format(field)
        )
    return value


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

def build_event(
    kind,               # type: Any
    content,            # type: Any
    role="user",        # type: Any
    session_id=None,    # type: Any
    scope="global",     # type: Any
    metadata=None,      # type: Any
    source="cli",       # type: Any
    created_at=None,    # type: Any
):
    # type: (...) -> Event
    """Build a validated :class:`Event`. Raw history is append-only."""
    stamp = created_at or ids.utc_now_iso()
    return validate_event(
        Event(
            id=ids.new_id(ids.EVENT_PREFIX),
            kind=require_enum(EventKind, kind, "event.kind"),
            role=require_enum(Role, role, "event.role"),
            content=require_text(content, "event.content"),
            session_id=(
                require_identifier(session_id, "event.session_id")
                if session_id is not None
                else None
            ),
            scope=require_scope(scope),
            metadata=require_json_object(metadata, "event.metadata"),
            created_at=stamp,
            source=require_text(source, "event.source", max_chars=MAX_SHORT_CHARS),
        )
    )


def validate_event(event):
    # type: (Event) -> Event
    require_enum(EventKind, event.kind, "event.kind")
    require_enum(Role, event.role, "event.role")
    require_text(event.content, "event.content")
    require_identifier(event.id, "event.id")
    require_scope(event.scope)
    require_json_object(event.metadata, "event.metadata")
    _require_optional_timestamp(event.created_at, "event.created_at")
    return event


# --------------------------------------------------------------------------
# Memories
# --------------------------------------------------------------------------

def build_memory(
    content,                  # type: Any
    kind=MemoryKind.SEMANTIC,  # type: Any
    scope="global",           # type: Any
    subject="user:self",      # type: Any
    structured=None,          # type: Any
    tags=(),                  # type: Any
    source=None,              # type: Any
    confidence=1.0,           # type: Any
    salience=0.5,             # type: Any
    status=MemoryStatus.ACTIVE,  # type: Any
    generated_by=None,        # type: Any
    valid_from=None,          # type: Any
    valid_to=None,            # type: Any
    created_at=None,          # type: Any
):
    # type: (...) -> Memory
    """Build a validated :class:`Memory`.

    Memories the user states explicitly use the default ``confidence=1.0``.
    Anything a model produces must be written with a lower value: it is a
    candidate, not a fact (architecture constraint P0-1).
    """
    stamp = created_at or ids.utc_now_iso()
    return validate_memory(
        Memory(
            id=ids.new_id(ids.MEMORY_PREFIX),
            kind=require_enum(MemoryKind, kind, "memory.kind"),
            scope=require_scope(scope),
            subject=require_text(subject, "memory.subject", max_chars=MAX_SHORT_CHARS),
            content=require_text(content, "memory.content"),
            structured=require_json_object(structured, "memory.structured"),
            tags=normalize_tags(tags),
            source=require_json_object(source, "memory.source"),
            confidence=require_unit_float(confidence, "memory.confidence"),
            salience=require_unit_float(salience, "memory.salience"),
            status=require_enum(MemoryStatus, status, "memory.status"),
            valid_from=_require_optional_timestamp(valid_from, "memory.valid_from"),
            valid_to=_require_optional_timestamp(valid_to, "memory.valid_to"),
            superseded_by=None,
            generated_by=(
                require_text(generated_by, "memory.generated_by", max_chars=MAX_SHORT_CHARS)
                if generated_by is not None
                else None
            ),
            created_at=stamp,
            updated_at=stamp,
            revision=1,
        )
    )


def validate_memory(memory):
    # type: (Memory) -> Memory
    """Re-validate a memory immediately before it is persisted."""
    require_identifier(memory.id, "memory.id")
    require_enum(MemoryKind, memory.kind, "memory.kind")
    require_enum(MemoryStatus, memory.status, "memory.status")
    require_scope(memory.scope)
    require_text(memory.subject, "memory.subject", max_chars=MAX_SHORT_CHARS)
    require_text(memory.content, "memory.content")
    require_json_object(memory.structured, "memory.structured")
    require_json_object(memory.source, "memory.source")
    normalize_tags(memory.tags)
    require_unit_float(memory.confidence, "memory.confidence")
    require_unit_float(memory.salience, "memory.salience")
    _require_optional_timestamp(memory.created_at, "memory.created_at")
    _require_optional_timestamp(memory.updated_at, "memory.updated_at")
    _require_optional_timestamp(memory.valid_from, "memory.valid_from")
    _require_optional_timestamp(memory.valid_to, "memory.valid_to")
    if not isinstance(memory.revision, int) or isinstance(memory.revision, bool):
        raise ValidationError("memory.revision must be an integer")
    if memory.revision < 1:
        raise ValidationError("memory.revision must be >= 1")
    return memory


# --------------------------------------------------------------------------
# Preferences
# --------------------------------------------------------------------------

def build_preference(
    key,                # type: Any
    value,              # type: Any
    scope="global",     # type: Any
    statement=None,     # type: Any
    confidence=1.0,     # type: Any
    source=None,        # type: Any
    created_at=None,    # type: Any
):
    # type: (...) -> Preference
    """Build a validated :class:`Preference`."""
    clean_key = require_text(key, "preference.key", max_chars=MAX_SHORT_CHARS)
    clean_value = require_json_value(value, "preference.value")
    if statement is None:
        statement = "{0} = {1}".format(clean_key, clean_value)
    stamp = created_at or ids.utc_now_iso()
    return validate_preference(
        Preference(
            id=ids.new_id(ids.PREFERENCE_PREFIX),
            scope=require_scope(scope),
            key=clean_key,
            value=clean_value,
            statement=require_text(statement, "preference.statement"),
            confidence=require_unit_float(confidence, "preference.confidence"),
            source=require_json_object(source, "preference.source"),
            created_at=stamp,
            superseded_by=None,
        )
    )


def validate_preference(preference):
    # type: (Preference) -> Preference
    require_identifier(preference.id, "preference.id")
    require_scope(preference.scope)
    require_text(preference.key, "preference.key", max_chars=MAX_SHORT_CHARS)
    require_json_value(preference.value, "preference.value")
    require_text(preference.statement, "preference.statement", allow_empty=True)
    require_unit_float(preference.confidence, "preference.confidence")
    require_json_object(preference.source, "preference.source")
    _require_optional_timestamp(preference.created_at, "preference.created_at")
    return preference


# --------------------------------------------------------------------------
# Project state
# --------------------------------------------------------------------------

def build_project_state(
    project_id,         # type: Any
    state,              # type: Any
    scope=None,         # type: Any
    note=None,          # type: Any
    source=None,        # type: Any
    created_at=None,    # type: Any
):
    # type: (...) -> ProjectState
    """Build a validated :class:`ProjectState`.

    Scope defaults to ``project:<project_id>`` so projects are isolated without
    the caller having to think about it.
    """
    clean_id = require_text(project_id, "project_state.project_id", max_chars=MAX_SHORT_CHARS)
    if scope is None:
        scope = "project:{0}".format(clean_id)
    stamp = created_at or ids.utc_now_iso()
    return validate_project_state(
        ProjectState(
            id=ids.new_id(ids.PROJECT_PREFIX),
            project_id=clean_id,
            scope=require_scope(scope),
            state=require_json_object(state, "project_state.state"),
            note=(
                require_text(note, "project_state.note")
                if note is not None
                else None
            ),
            source=require_json_object(source, "project_state.source"),
            created_at=stamp,
            superseded_by=None,
        )
    )


def validate_project_state(record):
    # type: (ProjectState) -> ProjectState
    require_identifier(record.id, "project_state.id")
    require_text(record.project_id, "project_state.project_id", max_chars=MAX_SHORT_CHARS)
    require_scope(record.scope)
    require_json_object(record.state, "project_state.state")
    require_json_object(record.source, "project_state.source")
    _require_optional_timestamp(record.created_at, "project_state.created_at")
    return record


__all__ = [
    "build_event",
    "validate_event",
    "build_memory",
    "validate_memory",
    "build_preference",
    "validate_preference",
    "build_project_state",
    "validate_project_state",
    "Sequence",
    "Dict",
    "Tuple",
]
