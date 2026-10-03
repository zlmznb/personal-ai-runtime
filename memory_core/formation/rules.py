"""Deterministic, rules-first memory formation.

Design stance: v0.1 extracts only from **explicitly marked** utterances. There
is no natural-language inference, no guessing, and no model involved. A rule
either fires and produces a candidate, or nothing happens.

Grammar
-------
    <memory verb>[<separator>][<type trigger>[<separator>]]<payload>

Memory verbs (optional):

    Chinese : 请记住 / 记住 / 记下 / 别忘了      (no separator required)
    English : please remember / remember / note that / note   (separator required)

Type triggers (optional; default is a semantic memory). A type trigger may stand
alone without a memory verb, because it is already unambiguous:

    偏好 / preference / pref        -> preference      payload: <key> = <value>
    项目状态 / project state / project -> project_state  payload: <project_id> = <json object>
    身份 / identity / persona       -> identity memory
    事件 / episodic                 -> episodic memory
    (none)                          -> semantic memory

At least one of a memory verb or a type trigger is required; an utterance with
neither is ordinary conversation, and v0.1 forms nothing from it.

Examples::

    记住：用户偏好本地优先
    记住偏好：theme = dark
    偏好：theme = "dark"
    project state: memory = {"phase": "v0.1"}
    identity: 我是一个本地优先的个人 AI
    事件：2026-10-02 决定不使用 Chroma

Because the type trigger must start the payload, "记住：我偏好深色主题" stays a
semantic memory rather than being misclassified as a preference.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple

from memory_core.formation.base import (
    MEMORY,
    PREFERENCE,
    PROJECT_STATE,
    Candidate,
    Extraction,
    Extractor,
    parse_json_value,
)

#: Verbs that mark an utterance as intentionally memorable.
#: Longest first so "请记住" wins over "记住".
_MEMORY_VERBS = (
    "请记住",
    "别忘了",
    "记住",
    "记下",
    "please remember",
    "remember that",
    "remember",
    "note that",
    "note",
)

#: Payload type triggers, checked in order. Longest/most specific first.
_TYPE_TRIGGERS = (
    ("项目状态", PROJECT_STATE),
    ("project state", PROJECT_STATE),
    ("project", PROJECT_STATE),
    ("偏好", PREFERENCE),
    ("preference", PREFERENCE),
    ("pref", PREFERENCE),
    ("身份", MEMORY),
    ("identity", MEMORY),
    ("persona", MEMORY),
    ("事件", MEMORY),
    ("episodic", MEMORY),
)

#: Episodic/identity triggers map to a memory kind.
_TRIGGER_MEMORY_KIND = {
    "身份": "identity",
    "identity": "identity",
    "persona": "identity",
    "事件": "episodic",
    "episodic": "episodic",
}

_SEPARATORS = "：:,，、;；= \t\n"

_ASSIGNMENT_SEPARATORS = ("=", "＝")


def _is_cjk(text):
    # type: (str) -> bool
    return any("\u3400" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff" for char in text)


def _strip_prefix(text, prefix):
    # type: (str, str) -> Optional[str]
    """Strip ``prefix`` from the start of ``text``, or return ``None``.

    ASCII prefixes must be followed by a separator, so "note" does not match
    "notebook". CJK prefixes need no separator because Chinese does not require
    one after "记住".
    """
    lowered = text.lower()
    if not lowered.startswith(prefix.lower()):
        return None
    remainder = text[len(prefix):]
    if remainder and remainder[0] not in _SEPARATORS and not _is_cjk(prefix):
        return None
    return remainder.lstrip(_SEPARATORS)


def _strip_any(text, prefixes):
    # type: (str, Tuple[str, ...]) -> Optional[Tuple[str, str]]
    for prefix in prefixes:
        remainder = _strip_prefix(text, prefix)
        if remainder is not None:
            return prefix, remainder
    return None


def _split_assignment(raw):
    # type: (str) -> Tuple[Optional[str], Optional[str]]
    for separator in _ASSIGNMENT_SEPARATORS:
        index = raw.find(separator)
        if index > 0:
            key = raw[:index].strip()
            value = raw[index + len(separator):].strip()
            if key and value:
                return key, value
    return None, None


def _classify_payload(payload):
    # type: (str) -> Tuple[Optional[str], Optional[str], str]
    """Return ``(trigger, kind, remainder)`` for the payload."""
    found = _strip_any(payload, tuple(trigger for trigger, _ in _TYPE_TRIGGERS))
    if found is None:
        return None, "semantic", payload
    trigger, remainder = found
    target = dict(_TYPE_TRIGGERS)[trigger]
    if target == MEMORY:
        return trigger, _TRIGGER_MEMORY_KIND.get(trigger, "semantic"), remainder
    return trigger, target, remainder


class RuleExtractor(Extractor):
    """Explicit-instruction extractor. Deterministic and model-free."""

    name = "rules"

    def extract(self, text, scope="global", context=None):
        # type: (str, str, Any) -> Extraction
        # Rules are deterministic and do not need existing memories to decide;
        # ``context`` exists only to satisfy the Extractor interface.
        del context
        if not isinstance(text, str) or not text.strip():
            return Extraction(reason="empty utterance")

        stripped = text.strip()
        verb_match = _strip_any(stripped, _MEMORY_VERBS)
        verb = None  # type: Optional[str]
        payload = stripped
        if verb_match is not None:
            verb, payload = verb_match
        payload = payload.strip()
        if not payload:
            return Extraction(trigger=verb, reason="nothing to remember after the trigger")

        trigger, target, remainder = _classify_payload(payload)
        remainder = remainder.strip()

        # A memory verb OR an explicit type trigger is required. An utterance
        # with neither is ordinary conversation and forms nothing at all.
        if trigger is None and verb is None:
            return Extraction(
                reason="no memory verb and no explicit type trigger; v0.1 only "
                       "forms memories from explicit instructions such as "
                       "'remember: ...', '记住：...' or '偏好：key = value'"
            )

        if not remainder:
            return Extraction(trigger=trigger or verb, reason="empty payload")

        if target == PREFERENCE:
            key, raw_value = _split_assignment(remainder)
            if key is None:
                return Extraction(
                    trigger=trigger,
                    reason="a preference needs 'key = value', e.g. '偏好：theme = dark'",
                )
            candidate = Candidate(
                target=PREFERENCE,
                payload={
                    "key": key,
                    "value": parse_json_value(raw_value),
                    "scope": scope,
                    "statement": remainder,
                    "confidence": 1.0,
                    "source": {"origin": "rule", "rule": "explicit-preference", "text": stripped},
                },
                confidence=1.0,
                rule="explicit-preference",
            )
            return Extraction(candidates=(candidate,), trigger=trigger)

        if target == PROJECT_STATE:
            project_id, raw_state = _split_assignment(remainder)
            if project_id is None:
                return Extraction(
                    trigger=trigger,
                    reason="project state needs 'project_id = {json object}'",
                )
            try:
                state = json.loads(raw_state)
            except ValueError:
                return Extraction(
                    trigger=trigger,
                    reason="project state value must be a JSON object",
                )
            if not isinstance(state, dict):
                return Extraction(
                    trigger=trigger,
                    reason="project state value must be a JSON object",
                )
            candidate = Candidate(
                target=PROJECT_STATE,
                payload={
                    "project_id": project_id,
                    "state": state,
                    "source": {"origin": "rule", "rule": "explicit-project-state", "text": stripped},
                },
                confidence=1.0,
                rule="explicit-project-state",
            )
            return Extraction(candidates=(candidate,), trigger=trigger)

        # A memory of some kind.
        candidate = Candidate(
            target=MEMORY,
            payload={
                "content": remainder,
                "kind": target,
                "scope": scope,
                "confidence": 1.0,
                "source": {"origin": "rule", "rule": "explicit-memory", "text": stripped},
            },
            confidence=1.0,
            rule="explicit-memory",
        )
        return Extraction(candidates=(candidate,), trigger=trigger or verb)


def parse_explicit(text, scope="global"):
    # type: (str, str) -> Extraction
    """Convenience wrapper around :class:`RuleExtractor`."""
    return RuleExtractor().extract(text, scope=scope)


__all__ = ["RuleExtractor", "parse_explicit"]
