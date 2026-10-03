"""Validation gates.

Everything that enters Storage - whether typed by a human, read from a config
file, or produced by a model - passes through these functions first
(architecture red line 4).

These raise :class:`~memory_core.domain.errors.ValidationError` and never
mutate their input.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Sequence, Tuple

from memory_core.domain.enums import SCOPE_PREFIXES
from memory_core.domain.errors import ValidationError

MAX_CONTENT_CHARS = 20000
MAX_SHORT_CHARS = 512
MAX_TAG_CHARS = 64
MAX_TAGS = 32

_SCOPE_RE = re.compile(r"^[a-z][a-z0-9_]*:[A-Za-z0-9][A-Za-z0-9._:-]*$")


def require_text(value, field, max_chars=MAX_CONTENT_CHARS, allow_empty=False):
    # type: (Any, str, int, bool) -> str
    """Return a validated, stripped string."""
    if not isinstance(value, str):
        raise ValidationError(
            "{0} must be a string, got {1}".format(field, type(value).__name__)
        )
    text = value.strip()
    if not text and not allow_empty:
        raise ValidationError("{0} must not be empty".format(field))
    if len(text) > max_chars:
        raise ValidationError(
            "{0} exceeds the {1} character limit".format(field, max_chars)
        )
    for char in text:
        if ord(char) < 32 and char not in "\n\t\r":
            raise ValidationError("{0} contains control characters".format(field))
    return text


def require_enum(enum_cls, value, field):
    # type: (Any, Any, str) -> str
    """Coerce a value to a member of ``enum_cls`` and return the plain string."""
    if isinstance(value, enum_cls):
        return value.value
    allowed = sorted(member.value for member in enum_cls)
    if not isinstance(value, str) or value not in allowed:
        raise ValidationError(
            "{0} must be one of {1}, got {2!r}".format(field, allowed, value)
        )
    return value


def require_scope(value):
    # type: (Any) -> str
    """Validate a namespace: ``global`` or ``<prefix>:<name>``.

    Scopes are how v0.1 already isolates project/session/device data, so that
    edge-cloud coordination later does not require a schema change.
    """
    if not isinstance(value, str):
        raise ValidationError(
            "scope must be a string, got {0}".format(type(value).__name__)
        )
    scope = value.strip()
    if scope == "global":
        return scope
    if not _SCOPE_RE.match(scope):
        raise ValidationError(
            "scope must be 'global' or '<prefix>:<name>', got {0!r}".format(scope)
        )
    prefix = scope.split(":", 1)[0]
    if prefix not in SCOPE_PREFIXES:
        raise ValidationError(
            "scope prefix must be one of {0}, got {1!r}".format(
                list(SCOPE_PREFIXES), prefix
            )
        )
    return scope


def require_unit_float(value, field, default=None):
    # type: (Any, str, Any) -> float
    """Validate a 0.0-1.0 number (used for confidence and salience)."""
    if value is None:
        if default is None:
            raise ValidationError("{0} is required".format(field))
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            "{0} must be a number between 0.0 and 1.0, got {1!r}".format(field, value)
        )
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValidationError(
            "{0} must be between 0.0 and 1.0, got {1!r}".format(field, number)
        )
    return number


def require_json_value(value, field):
    # type: (Any, str) -> Any
    """Require a value that round-trips through strict JSON.

    ``allow_nan=False`` matters: NaN/Infinity are accepted by ``json.dumps`` by
    default but are not valid JSON, and would corrupt an export.
    """
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError("{0} must be JSON-serialisable: {1}".format(field, exc))
    return value


def require_json_object(value, field, default=None):
    # type: (Any, str, Any) -> Dict[str, Any]
    """Require a JSON object (``dict``)."""
    if value is None:
        if default is None:
            return {}
        value = default
    if not isinstance(value, dict):
        raise ValidationError(
            "{0} must be a JSON object, got {1}".format(field, type(value).__name__)
        )
    require_json_value(value, field)
    return value


def normalize_tags(tags):
    # type: (Any) -> Tuple[str, ...]
    """Validate and de-duplicate a tag collection, preserving order."""
    if tags is None:
        return ()
    if isinstance(tags, str):
        raise ValidationError("tags must be a collection of strings, not a bare string")
    if not isinstance(tags, (list, tuple, set, frozenset)):
        raise ValidationError("tags must be a list or tuple")
    seen = set()
    ordered = []
    for tag in tags:
        text = require_text(tag, "tag", max_chars=MAX_TAG_CHARS)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if len(ordered) > MAX_TAGS:
        raise ValidationError("at most {0} tags are allowed".format(MAX_TAGS))
    return tuple(ordered)


def require_identifier(value, field):
    # type: (Any, str) -> str
    """Validate a record identifier."""
    return require_text(value, field, max_chars=MAX_SHORT_CHARS)
