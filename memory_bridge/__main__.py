"""Allow ``python -m memory_bridge``."""

from __future__ import annotations

import sys

from memory_bridge.cli import main

if __name__ == "__main__":
    sys.exit(main())
