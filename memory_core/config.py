"""Configuration loading.

Precedence, highest first:

1. explicit function arguments (CLI flags)
2. environment variables (``MEMORY_DB``, ``MEMORY_CONFIG``, ``MEMORY_MODELS``)
3. a JSON config file (``config/memory.json`` by default)
4. built-in defaults

JSON is used rather than TOML so the project keeps working on Python 3.8, which
has no ``tomllib`` in the standard library.

No configuration is required: with no config file and no environment, Memory
Core runs against ``./memory.sqlite`` with no model provider at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from memory_core.domain.errors import ConfigurationError
from memory_core.retrieval.fusion import DEFAULT_WEIGHTS, RetrievalWeights

DEFAULT_DB_FILENAME = "memory.sqlite"
DEFAULT_CONFIG_PATH = os.path.join("config", "memory.json")
DEFAULT_MODELS_PATH = os.path.join("config", "models.json")

ENV_DB = "MEMORY_DB"
ENV_CONFIG = "MEMORY_CONFIG"
ENV_MODELS = "MEMORY_MODELS"


def read_json(path):
    # type: (str) -> Dict[str, Any]
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise ConfigurationError("cannot read config {0}: {1}".format(path, exc))
    except ValueError as exc:
        raise ConfigurationError("config {0} is not valid JSON: {1}".format(path, exc))
    if not isinstance(data, dict):
        raise ConfigurationError("config {0} must contain a JSON object".format(path))
    return data


@dataclass(frozen=True)
class RetrievalConfig(object):
    default_limit: int = 10
    max_limit: int = 100
    weights: RetrievalWeights = DEFAULT_WEIGHTS

    def clamp_limit(self, limit):
        # type: (Optional[int]) -> int
        if limit is None:
            return self.default_limit
        try:
            value = int(limit)
        except (TypeError, ValueError):
            return self.default_limit
        if value <= 0:
            return self.default_limit
        return min(value, self.max_limit)


@dataclass(frozen=True)
class Config(object):
    db_path: str
    retrieval: RetrievalConfig
    models: Dict[str, Any]
    config_path: Optional[str] = None
    models_path: Optional[str] = None

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "db_path": self.db_path,
            "retrieval": {
                "default_limit": self.retrieval.default_limit,
                "max_limit": self.retrieval.max_limit,
                "weights": self.retrieval.weights.as_dict(),
            },
            "config_path": self.config_path,
            "models_path": self.models_path,
        }


def _retrieval_config(data):
    # type: (Dict[str, Any]) -> RetrievalConfig
    section = data.get("retrieval") or {}
    if not isinstance(section, dict):
        raise ConfigurationError("'retrieval' must be a JSON object")
    known = set(RetrievalWeights().as_dict())
    weights = DEFAULT_WEIGHTS
    if any(name in section for name in known):
        kwargs = {}
        for name in known:
            if name in section:
                kwargs[name] = float(section[name])
        weights = RetrievalWeights(**kwargs)
    return RetrievalConfig(
        default_limit=int(section.get("default_limit", 10)),
        max_limit=int(section.get("max_limit", 100)),
        weights=weights,
    )


def _first_existing(paths):
    # type: (Any) -> Optional[str]
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


def load_config(config_path=None, db_path=None, models_path=None, cwd=None):
    # type: (Optional[str], Optional[str], Optional[str], Optional[str]) -> Config
    """Load configuration, honouring the documented precedence."""
    base = cwd or os.getcwd()

    resolved_config = _first_existing(
        [config_path, os.environ.get(ENV_CONFIG), os.path.join(base, DEFAULT_CONFIG_PATH)]
    )
    data = read_json(resolved_config) if resolved_config else {}

    resolved_models = _first_existing(
        [models_path, os.environ.get(ENV_MODELS), os.path.join(base, DEFAULT_MODELS_PATH)]
    )
    models = read_json(resolved_models) if resolved_models else {}

    final_db = (
        db_path
        or os.environ.get(ENV_DB)
        or data.get("db_path")
        or DEFAULT_DB_FILENAME
    )
    if not isinstance(final_db, str) or not final_db.strip():
        raise ConfigurationError("db_path must be a non-empty string")

    return Config(
        db_path=final_db,
        retrieval=_retrieval_config(data),
        models=models,
        config_path=resolved_config,
        models_path=resolved_models,
    )


__all__ = [
    "Config",
    "RetrievalConfig",
    "load_config",
    "read_json",
    "DEFAULT_DB_FILENAME",
    "ENV_DB",
    "ENV_CONFIG",
    "ENV_MODELS",
]
