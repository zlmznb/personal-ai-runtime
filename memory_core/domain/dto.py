"""Transport-neutral data transfer objects.

These deliberately mirror the OpenAI-compatible *shape* without importing any
vendor code. Keeping our own DTOs means a provider swap touches nothing above
``memory_core.providers`` (architecture red line 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ChatMessage:
    """One message in a prompt."""

    role: str
    content: str


@dataclass(frozen=True)
class Capabilities:
    """What a provider can actually do.

    Callers degrade based on this instead of assuming every model supports
    structured output.
    """

    supports_json_mode: bool = False
    supports_streaming: bool = False
    max_context_tokens: int = 8192

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "supports_json_mode": self.supports_json_mode,
            "supports_streaming": self.supports_streaming,
            "max_context_tokens": self.max_context_tokens,
        }


@dataclass(frozen=True)
class ChatResult:
    """A normalised completion."""

    text: str
    provider: str
    model: str
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecallHit:
    """One ranked retrieval result.

    ``item`` is a ``Memory`` or a ``Preference`` depending on ``item_type``.
    ``score`` is derived only from stored data, never from wall-clock time, so
    identical queries against an unchanged database return identical results.
    """

    item: Any
    item_type: str
    score: float
    matched_on: str

    @property
    def id(self):
        # type: () -> str
        return self.item.id

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "id": self.item.id,
            "item_type": self.item_type,
            "score": self.score,
            "matched_on": self.matched_on,
        }


@dataclass(frozen=True)
class CandidateOutcome:
    """The lifecycle verdict for one candidate.

    ``candidate`` is the proposal as it was evaluated (untrusted, and possibly
    rejected). A rejection always carries a reason and never means data was lost:
    the raw event is still in ``events``.
    """

    candidate: Any
    accepted: bool
    reason: str = ""
    item_type: Optional[str] = None
    item_id: Optional[str] = None

    @property
    def rejected(self):
        # type: () -> bool
        return not self.accepted

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "item_type": self.item_type,
            "item_id": self.item_id,
            "operation": getattr(self.candidate, "operation", None),
            "target": getattr(self.candidate, "target", None),
            "rule": getattr(self.candidate, "rule", None),
            "origin": getattr(self.candidate, "origin", None),
            "confidence": getattr(self.candidate, "confidence", None),
            "supersedes": getattr(self.candidate, "supersedes", None),
            "content": (getattr(self.candidate, "payload", {}) or {}).get("content"),
        }


@dataclass(frozen=True)
class Observation:
    """Result of feeding one utterance into the event log and formation rules."""

    event: Any
    memories: Tuple[Any, ...] = ()
    preferences: Tuple[Any, ...] = ()
    project_states: Tuple[Any, ...] = ()
    skipped_reason: Optional[str] = None
    outcomes: Tuple[Any, ...] = ()

    @property
    def accepted(self):
        # type: () -> Tuple[Any, ...]
        return tuple(outcome for outcome in self.outcomes if outcome.accepted)

    @property
    def rejected(self):
        # type: () -> Tuple[Any, ...]
        return tuple(outcome for outcome in self.outcomes if outcome.rejected)

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "event_id": self.event.id,
            "memory_ids": [m.id for m in self.memories],
            "preference_ids": [p.id for p in self.preferences],
            "project_state_ids": [s.id for s in self.project_states],
            "skipped_reason": self.skipped_reason,
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
        }


__all__ = [
    "ChatMessage",
    "Capabilities",
    "ChatResult",
    "RecallHit",
    "Observation",
    "CandidateOutcome",
    "List",
]
