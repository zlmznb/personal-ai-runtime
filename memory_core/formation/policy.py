"""Acceptance policy: the deterministic gate between a candidate and a write.

Pure decision logic. No IO, no model, no storage access - it is handed the facts
it needs and returns a verdict. That makes the rule "Memory Core decides what is
written" (architecture red line 8) both true and unit-testable.

The lifecycle is::

    candidate -> validate (formation.candidate / domain.guards)
              -> accept / reject (this module)
              -> commit (api.MemoryCore)

A rejection is never an error: it is recorded with a reason and the raw event is
kept regardless, so nothing is silently lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, Optional

from memory_core.formation.base import ADD, MEMORY, NOOP, PREFERENCE, PROJECT_STATE

#: Content shorter than this is not worth storing as a long-term memory.
MIN_CONTENT_CHARS = 4

#: Below this confidence a candidate is discarded rather than stored.
MIN_CONFIDENCE = 0.35


def normalize_text(text):
    # type: (Any) -> str
    """Normalisation used for duplicate detection.

    Deliberately simple and locale-agnostic: case-fold and collapse whitespace.
    It is not the tokenizer used for search - duplicate detection must be exact,
    not fuzzy.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class Decision(object):
    """The verdict on one candidate."""

    accepted: bool
    reason: str = ""

    @property
    def rejected(self):
        # type: () -> bool
        return not self.accepted

    def as_dict(self):
        # type: () -> dict
        return {"accepted": self.accepted, "reason": self.reason}


ACCEPT = Decision(True)
DUPLICATE = Decision(False, "duplicate of an existing memory")
NOT_WORTH_REMEMBERING = Decision(False, "not worth remembering long-term")


@dataclass(frozen=True)
class AcceptancePolicy(object):
    """Decides whether a validated candidate becomes a memory."""

    min_confidence: float = MIN_CONFIDENCE
    min_content_chars: int = MIN_CONTENT_CHARS
    accept_noop: bool = False
    dedupe: bool = True

    def evaluate(self, candidate, existing_contents=(), known_ids=()):
        # type: (Any, Iterable[str], Iterable[str]) -> Decision
        """Return a verdict for ``candidate``.

        ``existing_contents`` are already-normalised contents of memories that
        could be duplicates; ``known_ids`` are ids that actually exist, used as a
        second line of defence on SUPERSEDE.
        """
        operation = getattr(candidate, "operation", ADD)

        if operation == NOOP:
            if self.accept_noop:
                return ACCEPT
            reason = getattr(candidate, "reason", "") or NOT_WORTH_REMEMBERING.reason
            return Decision(False, "model decided not to remember: {0}".format(reason))

        confidence = float(getattr(candidate, "confidence", 1.0))
        if confidence < self.min_confidence:
            return Decision(
                False,
                "confidence {0:.2f} is below the {1:.2f} threshold".format(
                    confidence, self.min_confidence
                ),
            )

        target = getattr(candidate, "target", MEMORY)
        payload = getattr(candidate, "payload", {}) or {}

        if target == MEMORY:
            content = normalize_text(payload.get("content"))
            if len(content) < self.min_content_chars:
                return Decision(
                    False,
                    "content is shorter than {0} characters".format(self.min_content_chars),
                )
            if self.dedupe and content in set(existing_contents):
                return DUPLICATE

        if target == PREFERENCE:
            key = payload.get("key")
            if not isinstance(key, str) or not key.strip():
                return Decision(False, "preference candidate has no key")

        if target == PROJECT_STATE:
            project_id = payload.get("project_id")
            if not isinstance(project_id, str) or not project_id.strip():
                return Decision(False, "project state candidate has no project_id")

        # A supersede target must exist. formation.candidate already restricts it
        # to what the model was shown; this catches the rest (rules, stale ids).
        supersedes = getattr(candidate, "supersedes", None)
        if operation == "supersede" and supersedes is not None:
            existing = {str(value) for value in known_ids}
            if existing and str(supersedes) not in existing:
                return Decision(False, "supersede target {0} does not exist".format(supersedes))

        return ACCEPT

    def as_dict(self):
        # type: () -> dict
        return {
            "min_confidence": self.min_confidence,
            "min_content_chars": self.min_content_chars,
            "accept_noop": self.accept_noop,
            "dedupe": self.dedupe,
        }


DEFAULT_POLICY = AcceptancePolicy()

__all__ = [
    "AcceptancePolicy",
    "Decision",
    "ACCEPT",
    "DEFAULT_POLICY",
    "normalize_text",
    "MIN_CONFIDENCE",
    "MIN_CONTENT_CHARS",
]
