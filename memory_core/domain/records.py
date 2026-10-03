"""Immutable record types.

Records are frozen dataclasses: updating a memory means writing a *new* row and
marking the old one superseded, never mutating history in place.

``from_row`` decodes the JSON text columns written by Storage, so the JSON
encoding lives in exactly one place.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


def _loads(text, fallback):
    # type: (Any, Any) -> Any
    if text is None:
        return fallback
    if isinstance(text, (dict, list)):
        return text
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return fallback


@dataclass(frozen=True)
class Event:
    """An immutable raw record: a conversation turn, note, or system message.

    Events are the append-only history from which memories can always be
    regenerated (architecture constraint P0-3).
    """

    id: str
    kind: str
    role: str
    content: str
    session_id: Optional[str] = None
    scope: str = "global"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    source: str = "cli"

    @classmethod
    def from_row(cls, row):
        # type: (Mapping[str, Any]) -> "Event"
        return cls(
            id=row["id"],
            kind=row["kind"],
            role=row["role"],
            content=row["content"],
            session_id=row.get("session_id"),
            scope=row["scope"],
            metadata=_loads(row.get("metadata"), {}),
            created_at=row["created_at"],
            source=row.get("source") or "cli",
        )

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "id": self.id,
            "kind": self.kind,
            "role": self.role,
            "content": self.content,
            "session_id": self.session_id,
            "scope": self.scope,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "source": self.source,
        }


@dataclass(frozen=True)
class Memory:
    """A long-term memory: one atomic, human-readable statement plus provenance.

    ``generated_by`` exists for observability only and is never read by logic
    (architecture constraint P0-4).
    """

    id: str
    kind: str
    scope: str
    subject: str
    content: str
    structured: Dict[str, Any] = field(default_factory=dict)
    tags: Tuple[str, ...] = ()
    source: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    salience: float = 0.5
    status: str = "active"
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    superseded_by: Optional[str] = None
    generated_by: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    revision: int = 1

    @classmethod
    def from_row(cls, row):
        # type: (Mapping[str, Any]) -> "Memory"
        return cls(
            id=row["id"],
            kind=row["kind"],
            scope=row["scope"],
            subject=row["subject"],
            content=row["content"],
            structured=_loads(row.get("structured"), {}),
            tags=tuple(_loads(row.get("tags"), [])),
            source=_loads(row.get("source"), {}),
            confidence=float(row["confidence"]),
            salience=float(row["salience"]),
            status=row["status"],
            valid_from=row.get("valid_from"),
            valid_to=row.get("valid_to"),
            superseded_by=row.get("superseded_by"),
            generated_by=row.get("generated_by"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            revision=int(row["revision"]),
        )

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "id": self.id,
            "kind": self.kind,
            "scope": self.scope,
            "subject": self.subject,
            "content": self.content,
            "structured": self.structured,
            "tags": list(self.tags),
            "source": self.source,
            "confidence": self.confidence,
            "salience": self.salience,
            "status": self.status,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "superseded_by": self.superseded_by,
            "generated_by": self.generated_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class Preference:
    """A binding, explicitly declared constraint with a single current value.

    Read by exact key, never by similarity: a preference must apply
    deterministically every time (see ADR 0003 and architecture.md 5.1).
    """

    id: str
    scope: str
    key: str
    value: Any
    statement: str = ""
    confidence: float = 1.0
    source: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    superseded_by: Optional[str] = None

    @classmethod
    def from_row(cls, row):
        # type: (Mapping[str, Any]) -> "Preference"
        return cls(
            id=row["id"],
            scope=row["scope"],
            key=row["key"],
            value=_loads(row.get("value"), None),
            statement=row.get("statement") or "",
            confidence=float(row["confidence"]),
            source=_loads(row.get("source"), {}),
            created_at=row["created_at"],
            superseded_by=row.get("superseded_by"),
        )

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "id": self.id,
            "scope": self.scope,
            "key": self.key,
            "value": self.value,
            "statement": self.statement,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "superseded_by": self.superseded_by,
        }


@dataclass(frozen=True)
class ProjectState:
    """What is true *now* for a project.

    Not a memory: a memory is a statement about the past, this is a versioned
    current value. The active row is the one with ``superseded_by is None``.
    """

    id: str
    project_id: str
    scope: str
    state: Dict[str, Any] = field(default_factory=dict)
    note: Optional[str] = None
    source: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    superseded_by: Optional[str] = None

    @classmethod
    def from_row(cls, row):
        # type: (Mapping[str, Any]) -> "ProjectState"
        return cls(
            id=row["id"],
            project_id=row["project_id"],
            scope=row["scope"],
            state=_loads(row.get("state"), {}),
            note=row.get("note"),
            source=_loads(row.get("source"), {}),
            created_at=row["created_at"],
            superseded_by=row.get("superseded_by"),
        )

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "id": self.id,
            "project_id": self.project_id,
            "scope": self.scope,
            "state": self.state,
            "note": self.note,
            "source": self.source,
            "created_at": self.created_at,
            "superseded_by": self.superseded_by,
        }


__all__ = ["Event", "Memory", "Preference", "ProjectState", "asdict"]
