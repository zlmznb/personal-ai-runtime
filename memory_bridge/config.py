"""Bridge configuration.

JSON, because the project has zero runtime dependencies (no PyYAML/TOML on
Python 3.8). A config file is optional: with none, sensible defaults are used
and everything is driven by frontmatter and CLI flags.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Default folder -> scope mapping.
#:
#: Matched against the document path; ``{name}`` is the first ``*`` capture.
#: Frontmatter always wins over this.
DEFAULT_SCOPE_MAP = (
    ("Projects/*", "project:{name}"),
    ("Projects/*/**", "project:{name}"),
    ("Personas/*", "persona:{name}"),
    ("Personas/*/**", "persona:{name}"),
)

DEFAULT_EXPORT_DIR = "AI Memory"

_CONFIG_FILENAME = "memory-bridge.json"


def _glob_to_regex(pattern):
    # type: (str) -> Tuple[str, int]
    """Convert a path glob to a regex, capturing the first ``*`` segment."""
    parts = str(pattern).replace("\\", "/").split("/")
    pieces = []
    captures = 0
    for part in parts:
        if part == "**":
            pieces.append(".*")
            continue
        converted = []
        for char in part:
            if char == "*":
                captures += 1
                converted.append("([^/]+)" if captures == 1 else "[^/]*")
            elif char == "?":
                converted.append("[^/]")
            else:
                converted.append(re.escape(char))
        pieces.append("".join(converted))
    return "^" + "/".join(pieces) + "$", captures


@dataclass(frozen=True)
class BridgeConfig(object):
    """Everything the bridge needs to know about one vault."""

    vault_path: str
    vault_id: str = "personal"
    db_path: str = "memory.sqlite"
    include: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()
    export_dir: str = DEFAULT_EXPORT_DIR
    default_scope: str = "global"
    scope_map: Tuple[Tuple[str, str], ...] = DEFAULT_SCOPE_MAP
    min_chunk_chars: int = 40
    max_chunk_chars: int = 2000
    include_code_blocks: bool = False
    use_llm: bool = True
    context_limit: int = 5

    # -- construction --------------------------------------------------

    @classmethod
    def load(cls, vault_path, config_path=None, **overrides):
        # type: (str, Optional[str], Any) -> "BridgeConfig"
        """Load a config file if present, then apply explicit overrides."""
        data = {}  # type: Dict[str, Any]
        resolved = config_path or os.path.join(vault_path, _CONFIG_FILENAME)
        if config_path is not None and not os.path.isfile(resolved):
            raise ValueError("bridge config not found: {0}".format(resolved))
        if os.path.isfile(resolved):
            with open(resolved, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("bridge config must be a JSON object: {0}".format(resolved))

        data.pop("_comment", None)
        data["vault_path"] = os.path.abspath(vault_path)
        data.setdefault("vault_id", os.path.basename(data["vault_path"].rstrip("\\/")) or "vault")

        scope_map = data.get("scope_map")
        if scope_map is None:
            data["scope_map"] = DEFAULT_SCOPE_MAP
        elif isinstance(scope_map, dict):
            data["scope_map"] = tuple(scope_map.items())
        else:
            data["scope_map"] = tuple(tuple(pair) for pair in scope_map)

        for key in ("include", "exclude"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])

        known = set(cls.__dataclass_fields__)
        for key, value in overrides.items():
            if value is not None:
                if key not in known:
                    raise ValueError("unknown bridge config key: {0}".format(key))
                data[key] = value
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError("unknown bridge config key(s): {0}".format(unknown))
        return cls(**data)

    # -- lookup --------------------------------------------------------

    def scope_for(self, rel_path):
        # type: (str) -> Optional[str]
        """Resolve a scope from the folder mapping, or ``None``."""
        clean = str(rel_path).replace("\\", "/")
        for pattern, template in self.scope_map:
            regex, captures = _glob_to_regex(pattern)
            match = re.match(regex, clean, re.IGNORECASE)
            if not match:
                continue
            scope = str(template)
            if captures and "{name}" in scope:
                scope = scope.replace("{name}", match.group(1))
            return scope
        return None

    def export_path(self):
        # type: () -> str
        return os.path.join(self.vault_path, self.export_dir)

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "vault_path": self.vault_path,
            "vault_id": self.vault_id,
            "db_path": self.db_path,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "export_dir": self.export_dir,
            "default_scope": self.default_scope,
            "scope_map": [list(pair) for pair in self.scope_map],
            "min_chunk_chars": self.min_chunk_chars,
            "max_chunk_chars": self.max_chunk_chars,
            "include_code_blocks": self.include_code_blocks,
            "use_llm": self.use_llm,
            "context_limit": self.context_limit,
        }


__all__ = ["BridgeConfig", "DEFAULT_SCOPE_MAP", "DEFAULT_EXPORT_DIR"]
