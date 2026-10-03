"""Allow ``python -m memory_core``."""

from __future__ import annotations

import sys

from memory_core.cli import main

if __name__ == "__main__":
    sys.exit(main())
