"""Read-only Obsidian vault scanning.

Two deliberate constraints:

* **Read-only.** Nothing in this module writes to the vault. ``import`` must be
  safe to run against a live vault, so the guarantee is enforced by design and
  asserted in the demo (vault hashes before/after).
* **No third-party YAML.** The project has zero runtime dependencies, so
  frontmatter uses a small, documented subset parser: scalars, inline lists and
  block lists. Nested maps are not supported; ``state: {"a": 1}`` (inline JSON)
  is the supported way to express structure.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

DEFAULT_INCLUDE = ("**/*.md",)
DEFAULT_EXCLUDE = (
    ".obsidian/**",
    ".trash/**",
    ".git/**",
    "AI Memory/**",
    "**/*.md.bak",
    "**/*.tmp",
)


# -- glob matching ---------------------------------------------------------

def _match_segment(segment, pattern):
    # type: (str, str) -> bool
    return fnmatch.fnmatchcase(segment.lower(), pattern.lower())


def _match_parts(parts, patterns):
    # type: (Sequence[str], Sequence[str]) -> bool
    if not patterns:
        return not parts
    head = patterns[0]
    if head == "**":
        # Zero or more path segments.
        for skip in range(len(parts) + 1):
            if _match_parts(parts[skip:], patterns[1:]):
                return True
        return False
    if not parts:
        return False
    if not _match_segment(parts[0], head):
        return False
    return _match_parts(parts[1:], patterns[1:])


def match_glob(path, pattern):
    # type: (str, str) -> bool
    """Match a POSIX relative path against a gitignore-style glob.

    ``**`` matches any number of segments, ``*`` and ``?`` do not cross ``/``.
    A pattern with no ``/`` matches the basename at any depth, so
    ``*.md.bak`` excludes that file anywhere.
    """
    clean = str(path).replace("\\", "/")
    while clean.startswith("./"):
        clean = clean[2:]
    clean = clean.lstrip("/")
    parts = [part for part in clean.split("/") if part]
    patterns = [part for part in str(pattern).replace("\\", "/").split("/") if part]
    if not patterns:
        return False
    if "/" not in str(pattern):
        return _match_parts(parts[-1:], patterns)
    return _match_parts(parts, patterns)


# -- frontmatter -----------------------------------------------------------

def _strip_comment(value):
    # type: (str) -> str
    for quote in ('"', "'"):
        if value.startswith(quote):
            end = value.find(quote, 1)
            return value[: end + 1] if end != -1 else value
    index = value.find(" #")
    return value[:index] if index != -1 else value


def _scalar(value):
    # type: (str) -> Any
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "~", ""):
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_scalar(part) for part in inner.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def parse_yaml_subset(raw):
    # type: (str) -> Dict[str, Any]
    """Parse the frontmatter subset. Never raises; unknown shapes stay strings."""
    result = {}  # type: Dict[str, Any]
    current_list = None  # type: Optional[List[Any]]
    for line in raw.split("\n"):
        if not line.strip() or line.strip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and current_list is not None:
            current_list.append(_scalar(stripped[2:]))
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if not key:
            continue
        value = _strip_comment(value).strip()
        if value == "":
            current_list = []
            result[key] = current_list
            continue
        result[key] = _scalar(value)
        current_list = None
    return result


def parse_frontmatter(text):
    # type: (str) -> Tuple[Dict[str, Any], str]
    """Return ``(frontmatter, body)``. A document without it yields ``({}, text)``."""
    if not isinstance(text, str):
        return {}, ""
    clean = text.replace("\r\n", "\n").replace("\r", "\n")
    if clean.startswith("\ufeff"):
        clean = clean[1:]
    lines = clean.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, clean
    closing = None
    for index in range(1, len(lines)):
        if lines[index].strip() in ("---", "..."):
            closing = index
            break
    if closing is None:
        return {}, clean
    raw = "\n".join(lines[1:closing])
    body = "\n".join(lines[closing + 1:])
    return parse_yaml_subset(raw), body


# -- documents -------------------------------------------------------------

@dataclass(frozen=True)
class VaultDocument(object):
    """One markdown file, read verbatim. Never modified."""

    abs_path: str
    rel_path: str            # POSIX, relative to the vault root
    title: str
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    body: str = ""
    mtime: str = ""
    size: int = 0

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "rel_path": self.rel_path,
            "title": self.title,
            "frontmatter": self.frontmatter,
            "mtime": self.mtime,
            "size": self.size,
        }


def _iso_mtime(path):
    # type: (str) -> str
    stamp = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _derive_title(frontmatter, body, rel_path):
    # type: (Dict[str, Any], str, str) -> str
    declared = frontmatter.get("title")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return os.path.splitext(os.path.basename(rel_path))[0]


def load_document(abs_path, root):
    # type: (str, str) -> VaultDocument
    with open(abs_path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    frontmatter, body = parse_frontmatter(text)
    rel_path = os.path.relpath(abs_path, root).replace("\\", "/")
    return VaultDocument(
        abs_path=abs_path,
        rel_path=rel_path,
        title=_derive_title(frontmatter, body, rel_path),
        frontmatter=frontmatter,
        body=body,
        mtime=_iso_mtime(abs_path),
        size=os.path.getsize(abs_path),
    )


def is_included(rel_path, include=(), exclude=()):
    # type: (str, Sequence[str], Sequence[str]) -> bool
    includes = tuple(include) if include else DEFAULT_INCLUDE
    if not any(match_glob(rel_path, pattern) for pattern in includes):
        return False
    return not any(match_glob(rel_path, pattern) for pattern in exclude)


def scan_vault(root, include=(), exclude=()):
    # type: (str, Sequence[str], Sequence[str]) -> Iterator[VaultDocument]
    """Yield every included markdown document, sorted by relative path."""
    root = os.path.abspath(root)
    excludes = tuple(exclude) if exclude else DEFAULT_EXCLUDE
    found = []  # type: List[str]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames if not name.startswith(".")
            or name in (".obsidian",)
        )
        for filename in sorted(filenames):
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root).replace("\\", "/")
            if is_included(rel_path, include, excludes):
                found.append(abs_path)
    for abs_path in sorted(found, key=lambda value: os.path.relpath(value, root).lower()):
        yield load_document(abs_path, root)


__all__ = [
    "VaultDocument",
    "DEFAULT_INCLUDE",
    "DEFAULT_EXCLUDE",
    "match_glob",
    "is_included",
    "parse_frontmatter",
    "parse_yaml_subset",
    "load_document",
    "scan_vault",
]
