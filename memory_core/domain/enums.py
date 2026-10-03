"""Controlled vocabularies.

All values are plain strings at rest. Always use ``.value`` when writing to or
comparing with storage: under Python 3.8-3.10 ``str(SomeEnum.MEMBER)`` renders
as ``"SomeEnum.MEMBER"``, not as the value.
"""

from __future__ import annotations

from enum import Enum


class EventKind(str, Enum):
    """What sort of raw record an event is."""

    MESSAGE = "message"
    NOTE = "note"
    SYSTEM = "system"
    #: An ingested external document (v0.3). Raw history, never a memory.
    DOCUMENT = "document"


class Role(str, Enum):
    """Who produced an event."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MemoryKind(str, Enum):
    """Long-term memory kinds.

    Preferences and project state are NOT here: they have single-current-value
    semantics and live in their own tables (see docs/architecture.md 5.1).
    """

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    IDENTITY = "identity"


class MemoryStatus(str, Enum):
    """Lifecycle status. Removal is logical, never physical."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ItemType(str, Enum):
    """Discriminator for heterogeneous recall results."""

    MEMORY = "memory"
    PREFERENCE = "preference"


#: Scope prefixes allowed. ``global`` is the bare default.
#:
#: ``persona`` was added in v0.3. Persona scopes are **fail-closed**: they are
#: excluded from every read unless explicitly named (see ``domain.scopes``).
SCOPE_PREFIXES = ("project", "session", "device", "user", "org", "persona")
