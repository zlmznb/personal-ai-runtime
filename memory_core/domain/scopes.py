"""Scope visibility rules.

v0.2 let any caller read across every scope by passing ``scope=None``. That is a
*fail-open* default: with more than one persona in one database, one persona's
private memories would silently appear in another's retrieval results and in
another's formation prompt.

v0.3 keeps the same call shape but makes the default **fail-closed**:

======================  ==================================================
call                    visible scopes
======================  ==================================================
``scope="global"``      exactly ``global`` (unchanged from v0.2)
``scopes=[a, b]``       exactly those, plus the persona scope if one is named
``persona="aria"``      ``global`` + ``persona:aria``
*(nothing given)*       every scope **except** ``persona:*``
======================  ==================================================

This module is pure: no IO, no model, no storage. Storage performs the actual
filtering; this decides what the filter should be, in one place, so retrieval and
formation can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

#: Prefix reserved for per-persona namespaces.
PERSONA_PREFIX = "persona"


def persona_scope(persona):
    # type: (str) -> str
    """``"aria"`` -> ``"persona:aria"``."""
    return "{0}:{1}".format(PERSONA_PREFIX, persona)


def is_persona_scope(scope):
    # type: (Any) -> bool
    return isinstance(scope, str) and scope.startswith(PERSONA_PREFIX + ":")


@dataclass(frozen=True)
class Visibility(object):
    """Which scopes a read or a formation context may see."""

    #: ``None`` means "no include-list", i.e. everything not excluded.
    scopes: Optional[Tuple[str, ...]] = None
    #: Scope prefixes that must never be visible (``("persona",)`` by default).
    exclude_prefixes: Tuple[str, ...] = ()

    def allows(self, scope):
        # type: (Any) -> bool
        """True when ``scope`` is readable under this visibility."""
        if not isinstance(scope, str):
            return False
        if self.scopes is not None:
            return scope in self.scopes
        for prefix in self.exclude_prefixes:
            if scope.startswith(prefix + ":"):
                return False
        return True

    def as_dict(self):
        # type: () -> dict
        return {
            "scopes": list(self.scopes) if self.scopes is not None else None,
            "exclude_prefixes": list(self.exclude_prefixes),
        }


#: The default: everything except persona namespaces.
DEFAULT_VISIBILITY = Visibility(scopes=None, exclude_prefixes=(PERSONA_PREFIX,))

#: Everything, including personas. Only for administrative reads (export).
ALL_VISIBILITY = Visibility(scopes=None, exclude_prefixes=())


def resolve_visibility(scope=None, scopes=None, persona=None):
    # type: (Optional[str], Optional[Sequence[str]], Optional[str]) -> Visibility
    """Turn the public call shape into a :class:`Visibility`.

    ``scope`` (singular) wins over ``scopes`` when both are supplied: it is the
    v0.2 shape and must keep behaving exactly as it did.
    """
    if scope is not None:
        return Visibility(scopes=(str(scope),), exclude_prefixes=())

    if scopes is not None:
        values = [str(value) for value in scopes]
        if persona:
            persona_value = persona_scope(persona)
            if persona_value not in values:
                values.append(persona_value)
        return Visibility(scopes=tuple(values), exclude_prefixes=())

    if persona:
        # A persona always sees global memory plus its own namespace, and
        # nothing else unless it asks explicitly.
        return Visibility(scopes=("global", persona_scope(persona)), exclude_prefixes=())

    return DEFAULT_VISIBILITY


def visibility_for_scope(scope):
    # type: (str) -> Visibility
    """The visibility implied by writing *into* ``scope``.

    Used for formation context: when forming a memory inside ``persona:aria``,
    the model may see ``global`` and ``persona:aria`` — never ``persona:bruno``.
    """
    if is_persona_scope(scope):
        return resolve_visibility(persona=scope.split(":", 1)[1])
    return resolve_visibility(scope=scope)


__all__ = [
    "Visibility",
    "DEFAULT_VISIBILITY",
    "ALL_VISIBILITY",
    "PERSONA_PREFIX",
    "persona_scope",
    "is_persona_scope",
    "resolve_visibility",
    "visibility_for_scope",
]
