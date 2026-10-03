"""Memory Formation: turning raw events into *candidates*.

Hard rule (architecture red line 4): Formation never writes to Storage. It
produces :class:`Candidate` objects; the API layer validates them through the
domain guards and only then persists them.

That separation is what makes the event log replayable: the same events can be
re-extracted later with a different (or fixed) strategy without touching history.

v0.1 ships only :class:`memory_core.formation.rules.RuleExtractor`. An LLM
extractor is deferred; when it arrives it implements the same ``Extractor``
interface and returns the same ``Candidate`` type, so it inherits the same
validation gate for free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MEMORY = "memory"
PREFERENCE = "preference"
PROJECT_STATE = "project_state"

#: Candidate operations. ``update`` and ``supersede`` are the same thing; the
#: schema parser normalises both to :data:`SUPERSEDE`.
ADD = "add"
SUPERSEDE = "supersede"
NOOP = "noop"
OPERATIONS = (ADD, SUPERSEDE, NOOP)


@dataclass(frozen=True)
class Candidate(object):
    """A proposed write, still untrusted.

    ``payload`` holds keyword arguments for the matching ``guards.build_*``
    function. Nothing here has been validated yet, and nothing has been written.

    A candidate is a *proposal*. Only Memory Core decides whether it becomes a
    memory (architecture red line 8).
    """

    target: str
    payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    rule: str = ""
    operation: str = ADD
    supersedes: Optional[str] = None
    reason: str = ""
    origin: str = "rule"

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "target": self.target,
            "payload": self.payload,
            "confidence": self.confidence,
            "rule": self.rule,
            "operation": self.operation,
            "supersedes": self.supersedes,
            "reason": self.reason,
            "origin": self.origin,
        }


@dataclass(frozen=True)
class Extraction(object):
    """Result of running an extractor over one utterance."""

    candidates: Tuple[Candidate, ...] = ()
    trigger: Optional[str] = None
    reason: Optional[str] = None

    @property
    def empty(self):
        # type: () -> bool
        return not self.candidates

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "trigger": self.trigger,
            "reason": self.reason,
            "candidates": [c.as_dict() for c in self.candidates],
        }


class Extractor(object):
    """Interface every formation strategy implements.

    ``context`` carries the existing memories the extractor was shown. It is what
    lets an LLM choose SUPERSEDE instead of ADD, and it is also the only set of
    ids a supersede proposal is allowed to reference.
    """

    name = "extractor"

    def extract(self, text, scope="global", context=None):
        # type: (str, str, Any) -> Extraction
        raise NotImplementedError


def parse_json_value(raw):
    # type: (str) -> Any
    """Parse a preference value: strict JSON first, otherwise a plain string.

    ``dark`` becomes the string ``"dark"`` while ``true`` becomes the boolean
    ``True``. This is documented behaviour, not an accident.
    """
    text = raw.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


__all__ = [
    "Candidate",
    "Extraction",
    "Extractor",
    "parse_json_value",
    "MEMORY",
    "PREFERENCE",
    "PROJECT_STATE",
    "ADD",
    "SUPERSEDE",
    "NOOP",
    "OPERATIONS",
    "List",
]
