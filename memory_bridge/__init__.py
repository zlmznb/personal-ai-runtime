"""Personal Knowledge Bridge.

The first real consumer of Memory Core. It connects a human knowledge layer
(Obsidian) to the AI long-term memory layer (Memory Core) with **two one-way
pipelines**, not a sync:

    vault  -> events -> memories      (truth flows human -> AI)
    memories -> markdown mirror       (a view flows AI -> human)

Purity rule: this package may only use Memory Core's **public API**. It must
never import ``memory_core.storage``, ``sqlite3``, or any provider
implementation. ``tests/test_bridge_isolation.py`` enforces that with a static
import check.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
