"""Identifier and timestamp helpers.

Identifiers are ULID-like: a 10-character Crockford base32 millisecond
timestamp followed by 16 random characters.

Property that matters here: identifiers sort lexicographically by creation time
**at millisecond granularity**. Two identifiers minted in the same millisecond
share a timestamp prefix and are ordered by their random suffix, so id order is
not a reliable proxy for insertion order within a tick.

That is fine for retrieval determinism (ADR 0003), because determinism only
requires that the same stored data always produces the same ordering - and the
stored data is fixed at write time. Queries always break ties on stored
``created_at`` and ``id``, never on wall-clock time.
"""

from __future__ import annotations

import secrets
import time
from datetime import datetime, timezone

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

EVENT_PREFIX = "evt_"
MEMORY_PREFIX = "mem_"
PREFERENCE_PREFIX = "prf_"
PROJECT_PREFIX = "prj_"


def _encode(value, length):
    # type: (int, int) -> str
    out = []
    for _ in range(length):
        out.append(_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(out))


def new_id(prefix=""):
    # type: (str) -> str
    """Return a time-sortable identifier with the given prefix."""
    millis = int(time.time() * 1000)
    random_part = "".join(secrets.choice(_ALPHABET) for _ in range(16))
    return prefix + _encode(millis, 10) + random_part


def utc_now_iso():
    # type: () -> str
    """Current UTC time as a lexicographically sortable ISO-8601 string."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def is_iso8601(value):
    # type: (str) -> bool
    """Loose check used by validators; storage always writes ``utc_now_iso()``."""
    if not isinstance(value, str) or len(value) < 20:
        return False
    try:
        datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return False
    return value.endswith("Z")
