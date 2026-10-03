"""Provenance for the Obsidian bridge.

Everything here is deterministic and pure: no IO, no model. The values produced
are stored in ``Event.metadata`` and ``Memory.source``, both of which are
free-form JSON - which is why v0.3 needs **no schema migration**.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple

#: Recorded on every artefact the bridge produces.
BRIDGE_VERSION = "0.3.0"

#: ``Event.source`` / ``Memory.source.origin`` for anything the bridge creates.
ORIGIN = "obsidian"

#: ``Event.source`` column value.
EVENT_SOURCE = "obsidian-bridge"

#: The reference scheme: ``obsidian:<vault_id>/<path>``.
REF_SCHEME = "obsidian"


def normalize_text(text):
    # type: (Any) -> str
    """Normalise text before hashing.

    Only whitespace is touched: line endings are unified and trailing spaces are
    stripped per line. Nothing semantic is rewritten, so the same file always
    hashes the same, and reflowing a paragraph *does* count as a change.
    """
    if not isinstance(text, str):
        return ""
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in unified.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def hash_text(text):
    # type: (Any) -> str
    digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
    return "sha256:{0}".format(digest)


def external_ref(vault_id, rel_path):
    # type: (str, str) -> str
    """``("personal", "Projects/X.md")`` -> ``"obsidian:personal/Projects/X.md"``.

    Paths are normalised to POSIX separators so a reference is stable across
    operating systems.
    """
    clean = str(rel_path).replace("\\", "/").lstrip("/")
    return "{0}:{1}/{2}".format(REF_SCHEME, vault_id, clean)


def parse_external_ref(reference):
    # type: (str) -> Tuple[Optional[str], Optional[str]]
    """Inverse of :func:`external_ref`. Returns ``(vault_id, rel_path)``."""
    if not isinstance(reference, str) or not reference.startswith(REF_SCHEME + ":"):
        return None, None
    remainder = reference[len(REF_SCHEME) + 1:]
    if "/" not in remainder:
        return remainder, None
    vault_id, rel_path = remainder.split("/", 1)
    return vault_id, rel_path


def is_bridge_ref(reference):
    # type: (Any) -> bool
    vault_id, rel_path = parse_external_ref(reference)
    return bool(vault_id and rel_path)


def event_metadata(
    vault_id,          # type: str
    rel_path,          # type: str
    title,             # type: str
    document_hash,     # type: str
    content_hash,      # type: str
    heading_path=(),   # type: Any
    line_start=None,   # type: Optional[int]
    line_end=None,     # type: Optional[int]
    mtime=None,        # type: Optional[str]
    ingested_at=None,  # type: Optional[str]
    route="knowledge", # type: str
    extra=None,        # type: Optional[Dict[str, Any]]
):
    # type: (...) -> Dict[str, Any]
    """Build the metadata block recorded on every ingested event."""
    metadata = {
        "origin": ORIGIN,
        "bridge_version": BRIDGE_VERSION,
        "vault_id": str(vault_id),
        "external_ref": external_ref(vault_id, rel_path),
        "document_title": str(title),
        "document_hash": str(document_hash),
        "content_hash": str(content_hash),
        "heading_path": [str(part) for part in heading_path],
        "line_start": line_start,
        "line_end": line_end,
        "mtime": mtime,
        "ingested_at": ingested_at,
        "route": str(route),
    }
    if extra:
        metadata.update(extra)
    return metadata


def memory_provenance(metadata):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Provenance merged into a memory's ``source`` by ``learn()``.

    ``event_ids`` is deliberately absent: Memory Core adds it itself, and the
    bridge cannot know the event id before the event exists.
    """
    return {
        "origin": ORIGIN,
        "bridge_version": BRIDGE_VERSION,
        "vault_id": metadata.get("vault_id"),
        "external_ref": metadata.get("external_ref"),
        "document_title": metadata.get("document_title"),
        "heading_path": metadata.get("heading_path") or [],
        "content_hash": metadata.get("content_hash"),
        "document_hash": metadata.get("document_hash"),
    }


def memory_source(metadata, event_id, model=None, reason=None):
    # type: (Dict[str, Any], str, Optional[str], Optional[str]) -> Dict[str, Any]
    """The provenance block carried onto a memory derived from this event.

    ``event_ids`` is also set by Memory Core itself; including it here keeps the
    bridge's intent explicit and is harmless (Core uses ``setdefault``).
    """
    source = {
        "origin": ORIGIN,
        "bridge_version": BRIDGE_VERSION,
        "vault_id": metadata.get("vault_id"),
        "external_ref": metadata.get("external_ref"),
        "document_title": metadata.get("document_title"),
        "heading_path": metadata.get("heading_path") or [],
        "content_hash": metadata.get("content_hash"),
        "document_hash": metadata.get("document_hash"),
        "event_ids": [event_id],
    }
    if model:
        source["model"] = model
    if reason:
        source["reason"] = reason
    return source


__all__ = [
    "BRIDGE_VERSION",
    "ORIGIN",
    "EVENT_SOURCE",
    "REF_SCHEME",
    "normalize_text",
    "hash_text",
    "external_ref",
    "parse_external_ref",
    "is_bridge_ref",
    "event_metadata",
    "memory_provenance",
    "memory_source",
]
