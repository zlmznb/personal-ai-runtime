"""Strict schema validation for LLM-produced memory candidates.

The contract
------------
The model may return **only** a JSON document of this shape::

    {
      "memories": [
        {
          "operation": "ADD" | "UPDATE" | "SUPERSEDE" | "NOOP",
          "content": "one atomic statement",
          "kind": "semantic" | "episodic" | "identity",
          "subject": "user:self",
          "tags": ["tag"],
          "confidence": 0.0-1.0,
          "salience": 0.0-1.0,
          "reason": "why",
          "supersedes_id": "mem_..." | null
        }
      ]
    }

Everything outside that shape is rejected:

* unknown top-level keys
* unknown keys inside a memory item
* unknown operations
* content that is missing, empty or oversized
* confidence / salience outside ``[0, 1]``
* ``supersedes_id`` that is not one of the ids the model was actually shown

The last rule is the important one. It means a hallucinated id cannot make
Memory Core overwrite an unrelated memory: the model can only ask to supersede
something it can see.

Nothing in this module writes anything. It turns text into untrusted
:class:`~memory_core.formation.base.Candidate` objects and no further.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from memory_core.domain.errors import ValidationError
from memory_core.formation.base import (
    ADD,
    MEMORY,
    NOOP,
    SUPERSEDE,
    Candidate,
)
from memory_core.domain.validation import MAX_CONTENT_CHARS, MAX_TAGS

__all__ = [
    "CandidateSchemaError",
    "parse_llm_candidates",
    "extract_json_object",
    "ALLOWED_TOP_LEVEL_KEYS",
    "ALLOWED_ITEM_KEYS",
    "DEFAULT_LLM_CONFIDENCE",
]

#: Default confidence for an LLM-proposed memory.
#:
#: Deliberately below 1.0: a user's explicit statement is a fact, a model's
#: proposal is a hypothesis (architecture constraint P0-1).
DEFAULT_LLM_CONFIDENCE = 0.6

ALLOWED_TOP_LEVEL_KEYS = frozenset({"memories"})
ALLOWED_ITEM_KEYS = frozenset({
    "operation",
    "content",
    "kind",
    "subject",
    "tags",
    "confidence",
    "salience",
    "reason",
    "supersedes_id",
})
ALLOWED_MEMORY_KINDS = frozenset({"semantic", "episodic", "identity"})

#: The spec names both UPDATE and SUPERSEDE; they mean the same thing here.
_OPERATION_ALIASES = {
    "add": ADD,
    "create": ADD,
    "update": SUPERSEDE,
    "supersede": SUPERSEDE,
    "replace": SUPERSEDE,
    "noop": NOOP,
    "no-op": NOOP,
    "none": NOOP,
    "skip": NOOP,
}

MAX_REASON_CHARS = 500
MAX_SUBJECT_CHARS = 200


class CandidateSchemaError(ValidationError):
    """The model's output did not match the candidate schema."""


def extract_json_object(text):
    # type: (Any) -> Any
    """Pull a JSON object out of a model response.

    Tolerates the two things models do constantly: wrapping the JSON in a
    markdown fence, and surrounding it with a sentence of prose.
    """
    if not isinstance(text, str) or not text.strip():
        raise CandidateSchemaError("model returned no content")
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise CandidateSchemaError("model response contained no JSON object")
        stripped = stripped[start:end + 1]
    try:
        payload = json.loads(stripped)
    except ValueError as exc:
        raise CandidateSchemaError("model response is not valid JSON: {0}".format(exc))
    if not isinstance(payload, dict):
        raise CandidateSchemaError("model response must be a JSON object")
    return payload


def _fail(path, message):
    # type: (str, str) -> None
    raise CandidateSchemaError("{0}: {1}".format(path, message))


def _require_object(value, path):
    # type: (Any, str) -> Dict[str, Any]
    if not isinstance(value, dict):
        _fail(path, "expected a JSON object, got {0}".format(type(value).__name__))
    return value


def _require_optional_string(value, path, max_chars):
    # type: (Any, str, int) -> Optional[str]
    if value is None:
        return None
    if not isinstance(value, str):
        _fail(path, "expected a string, got {0}".format(type(value).__name__))
    text = value.strip()
    if not text:
        return None
    if len(text) > max_chars:
        _fail(path, "longer than {0} characters".format(max_chars))
    return text


def _require_unit_float(value, path, default):
    # type: (Any, str, float) -> float
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected a number between 0 and 1")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        _fail(path, "must be between 0 and 1, got {0!r}".format(value))
    return number


def _require_tags(value, path):
    # type: (Any, str) -> Tuple[str, ...]
    if value is None:
        return ()
    if not isinstance(value, list):
        _fail(path, "expected a list of strings")
    if len(value) > MAX_TAGS:
        _fail(path, "at most {0} tags are allowed".format(MAX_TAGS))
    tags = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            _fail("{0}[{1}]".format(path, index), "expected a non-empty string")
        tags.append(item.strip())
    return tuple(tags)


def _parse_item(item, path, context_ids, scope, default_confidence):
    # type: (Any, str, Set[str], str, float) -> Candidate
    data = _require_object(item, path)

    unknown = sorted(set(data) - ALLOWED_ITEM_KEYS)
    if unknown:
        _fail(path, "unknown key(s) {0}".format(unknown))

    raw_operation = data.get("operation")
    if not isinstance(raw_operation, str):
        _fail("{0}.operation".format(path), "expected a string")
    operation = _OPERATION_ALIASES.get(raw_operation.strip().lower())
    if operation is None:
        _fail(
            "{0}.operation".format(path),
            "unknown operation {0!r}; expected one of {1}".format(
                raw_operation, sorted(_OPERATION_ALIASES)
            ),
        )

    reason = _require_optional_string(data.get("reason"), "{0}.reason".format(path), MAX_REASON_CHARS) or ""
    confidence = _require_unit_float(
        data.get("confidence"), "{0}.confidence".format(path), default_confidence
    )

    if operation == NOOP:
        # A NOOP carries no memory, so it needs no content and must not name a
        # target. It exists so the model can say "nothing worth keeping".
        if data.get("supersedes_id") is not None:
            _fail("{0}.supersedes_id".format(path), "a NOOP must not name a target")
        return Candidate(
            target=MEMORY,
            payload={"scope": scope},
            confidence=confidence,
            rule="llm:noop",
            operation=NOOP,
            reason=reason or "model judged this not worth remembering",
            origin="llm",
        )

    content = _require_optional_string(
        data.get("content"), "{0}.content".format(path), MAX_CONTENT_CHARS
    )
    if content is None:
        _fail("{0}.content".format(path), "required for {0}".format(operation.upper()))

    kind = data.get("kind")
    if kind is None:
        kind = "semantic"
    if not isinstance(kind, str) or kind.strip().lower() not in ALLOWED_MEMORY_KINDS:
        _fail(
            "{0}.kind".format(path),
            "expected one of {0}, got {1!r}".format(sorted(ALLOWED_MEMORY_KINDS), kind),
        )
    kind = kind.strip().lower()

    subject = _require_optional_string(
        data.get("subject"), "{0}.subject".format(path), MAX_SUBJECT_CHARS
    ) or "user:self"

    supersedes = None
    if operation == SUPERSEDE:
        supersedes = _require_optional_string(
            data.get("supersedes_id"), "{0}.supersedes_id".format(path), 200
        )
        if supersedes is None:
            _fail(
                "{0}.supersedes_id".format(path),
                "required for SUPERSEDE; use ADD if this is a new memory",
            )
        if supersedes not in context_ids:
            _fail(
                "{0}.supersedes_id".format(path),
                "{0!r} was not among the memories shown to the model; "
                "refusing to overwrite an unknown memory".format(supersedes),
            )
    elif data.get("supersedes_id") is not None:
        _fail("{0}.supersedes_id".format(path), "only valid for SUPERSEDE")

    return Candidate(
        target=MEMORY,
        payload={
            "content": content,
            "kind": kind,
            "subject": subject,
            "tags": _require_tags(data.get("tags"), "{0}.tags".format(path)),
            "salience": _require_unit_float(data.get("salience"), "{0}.salience".format(path), 0.5),
            "confidence": confidence,
            "scope": scope,
            "source": {"origin": "llm", "reason": reason},
        },
        confidence=confidence,
        rule="llm:{0}".format(operation),
        operation=operation,
        supersedes=supersedes,
        reason=reason,
        origin="llm",
    )


def parse_llm_candidates(
    text,
    context_ids=(),
    scope="global",
    max_candidates=5,
    default_confidence=DEFAULT_LLM_CONFIDENCE,
):
    # type: (Any, Sequence[str], str, int, float) -> List[Candidate]
    """Parse and strictly validate a model response into candidates.

    Raises :class:`CandidateSchemaError` on any schema violation. Callers treat
    that as "no candidates": nothing is written and the raw event survives.
    """
    payload = extract_json_object(text)

    unknown = sorted(set(payload) - ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise CandidateSchemaError("unknown top-level key(s) {0}".format(unknown))

    items = payload.get("memories")
    if items is None:
        raise CandidateSchemaError("missing required key 'memories'")
    if not isinstance(items, list):
        raise CandidateSchemaError("'memories' must be a list")

    limit = int(max_candidates)
    if len(items) > limit:
        raise CandidateSchemaError(
            "model proposed {0} memories but the limit is {1}".format(len(items), limit)
        )

    allowed_ids = {str(value) for value in context_ids}
    candidates = []
    for index, item in enumerate(items):
        candidates.append(
            _parse_item(item, "memories[{0}]".format(index), allowed_ids, scope, float(default_confidence))
        )
    return candidates
