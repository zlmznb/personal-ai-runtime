"""Route each chunk to the right Memory Core path.

Five kinds of content, decided by **deterministic signals first**:

======================  ============================================  ==========================
kind                    signal                                        Core path
======================  ============================================  ==========================
ordinary knowledge      default                                       ``learn()`` (LLM judges)
explicit declaration    ``记住：`` / ``remember:`` / ``> [!memory]``   ``learn()`` (rule fires, 1.0)
user preference         ``type: preference`` or ``偏好：k = v``        ``set_preference()``
project state           ``type: project-state`` or ``项目状态：p={}``  ``set_project_state()``
excluded                not scanned at all                            -
======================  ============================================  ==========================

Two deliberate stances:

1. **Most human notes should not become memories.** Thousands of notes flowing
   into the memory store would drown it. Unmarked prose goes to the LLM, which
   is expected to answer NOOP most of the time.
2. **Preferences and project state never go through an LLM.** They are exact
   key/value and single-current-value data; Memory Core's own architecture
   forbids a probabilistic model from writing them (ADR 0006). The bridge uses
   the typed API instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from memory_bridge.config import BridgeConfig
from memory_bridge.obsidian.chunker import Chunk, chunk_markdown
from memory_bridge.obsidian.provenance import hash_text, normalize_text

KNOWLEDGE = "knowledge"
DECLARATION = "declaration"
PREFERENCE = "preference"
PROJECT_STATE = "project_state"

#: Frontmatter ``type:`` values that describe a structured declaration.
_TYPED_TYPES = {
    "preference": PREFERENCE,
    "pref": PREFERENCE,
    "project-state": PROJECT_STATE,
    "project_state": PROJECT_STATE,
    "projectstate": PROJECT_STATE,
}

#: Frontmatter keys consumed by routing; never folded into project state.
_RESERVED_KEYS = frozenset({
    "type", "scope", "persona", "project", "project_id", "title",
    "key", "value", "state", "tags",
})

#: Explicit "remember this" markers. Reuses Memory Core's own rule vocabulary.
_DECLARATION_MARKERS = (
    "记住：", "记住:", "记下：", "记下:", "别忘了：", "别忘了:",
    "remember:", "remember that", "> [!memory]",
)

_PREFERENCE_RE = re.compile(
    r"(?:偏好|preference)\s*[:：]\s*([^\s=＝,，]+)\s*[=＝]\s*(\S.*?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_PROJECT_STATE_RE = re.compile(
    r"(?:项目状态|project\s*state)\s*[:：]\s*([^\s=＝,，]+)\s*[=＝]\s*(\{.*\})\s*$",
    re.MULTILINE | re.IGNORECASE,
)

#: Paragraph prefixes that start a new chunk.
#:
#: Memory Core's rule extractor only fires when the marker is at the *start* of
#: the text, and real notes bury ``记住：`` in the middle. Splitting here keeps
#: the text verbatim while letting the deterministic path work.
_BOUNDARY_PREFIXES = _DECLARATION_MARKERS


@dataclass(frozen=True)
class Route(object):
    """What to do with one chunk."""

    route: str
    scope: str
    reason: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "route": self.route,
            "scope": self.scope,
            "reason": self.reason,
            "payload": self.payload,
        }


def parse_value(raw):
    # type: (str) -> Any
    """Strict JSON first, otherwise a plain string (``dark`` -> ``"dark"``)."""
    text = str(raw).strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def declared_type(document):
    # type: (Any) -> Optional[str]
    """The structured route declared by frontmatter, if any."""
    value = document.frontmatter.get("type")
    if isinstance(value, str):
        return _TYPED_TYPES.get(value.strip().lower())
    return None


def resolve_scope(document, config):
    # type: (Any, BridgeConfig) -> Tuple[str, str]
    """Resolve the target scope. Returns ``(scope, reason)``.

    Frontmatter wins, then the folder mapping, then the default.
    """
    frontmatter = document.frontmatter
    declared = frontmatter.get("scope")
    if isinstance(declared, str) and declared.strip():
        return declared.strip(), "frontmatter scope:"

    persona = frontmatter.get("persona")
    if isinstance(persona, str) and persona.strip():
        return "persona:{0}".format(persona.strip()), "frontmatter persona:"

    project = frontmatter.get("project")
    if isinstance(project, str) and project.strip():
        return "project:{0}".format(project.strip()), "frontmatter project:"

    mapped = config.scope_for(document.rel_path)
    if mapped:
        return mapped, "path mapping"

    return config.default_scope, "default"


def chunks_for_document(document, config):
    # type: (Any, BridgeConfig) -> List[Chunk]
    """Chunk a document.

    A document whose frontmatter declares a structured type is **not** split:
    the declaration is a single value, and splitting it would produce several
    competing declarations that supersede each other.
    """
    if declared_type(document):
        text = normalize_text(document.body)
        if len(text) < int(config.min_chunk_chars):
            return []
        return [
            Chunk(
                ordinal=0,
                heading_path=(document.title,),
                text=text,
                line_start=1,
                line_end=len(text.split("\n")),
                content_hash=hash_text(text),
            )
        ]
    return chunk_markdown(
        document.body,
        min_chars=int(config.min_chunk_chars),
        max_chars=int(config.max_chunk_chars),
        include_code_blocks=bool(config.include_code_blocks),
        boundary_prefixes=_BOUNDARY_PREFIXES,
    )


def classify_chunk(chunk, document, config):
    # type: (Chunk, Any, BridgeConfig) -> Route
    """Decide the route for one chunk."""
    scope, scope_reason = resolve_scope(document, config)
    frontmatter = document.frontmatter
    text = chunk.text

    declared = declared_type(document)
    if declared == PROJECT_STATE:
        project_id = frontmatter.get("project") or frontmatter.get("project_id")
        if not isinstance(project_id, str) or not project_id.strip():
            return Route(KNOWLEDGE, scope, "project-state without a project id", {})
        state = _project_state_payload(frontmatter)
        if state is None:
            return Route(KNOWLEDGE, scope, "project-state without usable state", {})
        return Route(
            PROJECT_STATE,
            scope,
            "frontmatter type: project-state ({0})".format(scope_reason),
            {"project_id": project_id.strip(), "state": state},
        )

    if declared == PREFERENCE:
        key = frontmatter.get("key")
        if isinstance(key, str) and key.strip() and "value" in frontmatter:
            return Route(
                PREFERENCE,
                scope,
                "frontmatter type: preference ({0})".format(scope_reason),
                {"key": key.strip(), "value": frontmatter.get("value")},
            )
        return Route(KNOWLEDGE, scope, "preference without key/value", {})

    match = _PROJECT_STATE_RE.search(text)
    if match:
        try:
            state = json.loads(match.group(2))
        except ValueError:
            state = None
        if isinstance(state, dict):
            return Route(
                PROJECT_STATE,
                scope,
                "inline 项目状态 declaration",
                {"project_id": match.group(1).strip(), "state": state},
            )

    match = _PREFERENCE_RE.search(text)
    if match:
        return Route(
            PREFERENCE,
            scope,
            "inline 偏好 declaration",
            {"key": match.group(1).strip(), "value": parse_value(match.group(2))},
        )

    lowered = text.lower()
    for marker in _DECLARATION_MARKERS:
        if marker.lower() in lowered:
            return Route(DECLARATION, scope, "explicit marker {0!r}".format(marker), {})

    return Route(KNOWLEDGE, scope, "unmarked knowledge note", {})


def _project_state_payload(frontmatter):
    # type: (Dict[str, Any]) -> Optional[Dict[str, Any]]
    """Build the state object from frontmatter.

    ``state: {"phase": "v0.3"}`` (inline JSON) is preferred; otherwise every
    non-reserved frontmatter key becomes the state.
    """
    declared = frontmatter.get("state")
    if isinstance(declared, str):
        try:
            parsed = json.loads(declared)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    elif isinstance(declared, dict):
        return dict(declared)

    derived = {
        key: value
        for key, value in frontmatter.items()
        if key not in _RESERVED_KEYS and value is not None
    }
    return derived or None


__all__ = [
    "Route",
    "KNOWLEDGE",
    "DECLARATION",
    "PREFERENCE",
    "PROJECT_STATE",
    "classify_chunk",
    "chunks_for_document",
    "declared_type",
    "resolve_scope",
    "parse_value",
]
