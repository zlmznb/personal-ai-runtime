"""Assemble providers from configuration.

Roles are independent - ``chat``, ``embedding``, and later ``formation`` /
``rerank`` - so a single run may mix vendors freely: a cloud chat model with
local embeddings, for example. That is the normal case, not an edge case.

Unknown keys in a provider block are rejected rather than silently ignored, so a
typo in ``config/models.json`` surfaces immediately.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from memory_core.domain.errors import ConfigurationError
from memory_core.providers.base import ChatProvider, EmbeddingProvider
from memory_core.providers.deepseek import DeepSeekProvider
from memory_core.providers.fake import FakeProvider
from memory_core.providers.fake_embedding import FakeEmbeddingProvider
from memory_core.providers.ollama import OllamaProvider
from memory_core.providers.ollama_embedding import OllamaEmbeddingProvider

#: type -> (factory, allowed keys)
_CHAT_TYPES = {
    "fake": (
        FakeProvider,
        ("model", "prefix", "fail_with", "reply", "supports_json_mode", "max_context_tokens"),
    ),
    "ollama": (OllamaProvider, ("model", "base_url", "timeout", "max_context_tokens")),
    "deepseek": (
        DeepSeekProvider,
        ("model", "base_url", "api_key", "api_key_env", "timeout", "max_context_tokens"),
    ),
}

_EMBEDDING_TYPES = {
    "fake-embedding": (
        FakeEmbeddingProvider,
        ("model", "dim", "fail_with", "salt"),
    ),
    "ollama-embedding": (
        OllamaEmbeddingProvider,
        ("model", "base_url", "timeout", "api_style"),
    ),
}


def provider_types():
    # type: () -> Tuple[str, ...]
    """Chat provider types."""
    return tuple(sorted(_CHAT_TYPES))


def embedding_provider_types():
    # type: () -> Tuple[str, ...]
    """Embedding provider types."""
    return tuple(sorted(_EMBEDDING_TYPES))


def all_provider_types():
    # type: () -> Tuple[str, ...]
    return tuple(sorted(set(_CHAT_TYPES) | set(_EMBEDDING_TYPES)))


def _create(spec, table, kind, base_class):
    # type: (Any, Dict[str, Any], str, Any) -> Any
    if not isinstance(spec, dict):
        raise ConfigurationError("a provider definition must be a JSON object")
    provider_type = str(spec.get("type") or "").strip().lower()
    if provider_type not in table:
        raise ConfigurationError(
            "unknown {0} provider type {1!r}; expected one of {2}".format(
                kind, provider_type, sorted(table)
            )
        )
    factory, allowed = table[provider_type]
    unknown = sorted(set(spec) - set(allowed) - {"type"})
    if unknown:
        raise ConfigurationError(
            "unknown key(s) {0} for {1} provider type {2!r}".format(
                unknown, kind, provider_type
            )
        )
    kwargs = {key: spec[key] for key in allowed if key in spec}
    provider = factory(**kwargs)
    if not isinstance(provider, base_class):
        raise ConfigurationError(
            "{0!r} is not a valid {1} provider".format(provider_type, kind)
        )
    return provider


def create_provider(spec):
    # type: (Dict[str, Any]) -> ChatProvider
    """Build a chat provider from a config block such as ``{"type": "ollama"}``."""
    return _create(spec, _CHAT_TYPES, "chat", ChatProvider)


def create_embedding_provider(spec):
    # type: (Dict[str, Any]) -> EmbeddingProvider
    """Build an embedding provider from a config block."""
    return _create(spec, _EMBEDDING_TYPES, "embedding", EmbeddingProvider)


def create_any_provider(spec):
    # type: (Dict[str, Any]) -> Any
    """Build a provider of whichever kind matches ``spec['type']``."""
    if not isinstance(spec, dict):
        raise ConfigurationError("a provider definition must be a JSON object")
    provider_type = str(spec.get("type") or "").strip().lower()
    if provider_type in _CHAT_TYPES:
        return create_provider(spec)
    if provider_type in _EMBEDDING_TYPES:
        return create_embedding_provider(spec)
    raise ConfigurationError(
        "unknown provider type {0!r}; expected one of {1}".format(
            provider_type, list(all_provider_types())
        )
    )


def _providers_block(models):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    block = models.get("providers") or {}
    if not isinstance(block, dict):
        raise ConfigurationError("'providers' must be a JSON object")
    return block


def provider_from_name(models, name, *, kind="chat"):
    # type: (Dict[str, Any], str, str) -> Any
    """Instantiate the named provider from the config file."""
    block = _providers_block(models)
    spec = block.get(name)
    if spec is None:
        raise ConfigurationError(
            "provider {0!r} is not defined in the models config (known: {1})".format(
                name, sorted(block)
            )
        )
    if kind == "embedding":
        return create_embedding_provider(spec)
    if kind == "any":
        return create_any_provider(spec)
    return create_provider(spec)


def _resolve_role(models, role, kind):
    # type: (Dict[str, Any], str, str) -> Any
    if not models:
        return None
    roles = models.get("roles") or {}
    if not isinstance(roles, dict):
        raise ConfigurationError("'roles' must be a JSON object")
    target = roles.get(role)
    if target is None:
        return None
    if isinstance(target, dict):
        return create_embedding_provider(target) if kind == "embedding" else create_provider(target)
    return provider_from_name(models, str(target), kind=kind)


def resolve_role(models, role="chat"):
    # type: (Dict[str, Any], str) -> Optional[ChatProvider]
    """Build the chat provider bound to ``role``, or ``None`` if unbound."""
    return _resolve_role(models, role, "chat")


def resolve_embedding_role(models, role="embedding"):
    # type: (Dict[str, Any], str) -> Optional[EmbeddingProvider]
    """Build the embedding provider bound to ``role``, or ``None`` if unbound."""
    return _resolve_role(models, role, "embedding")


def _resolve_cli(models, table, kind, name=None, model=None):
    # type: (Dict[str, Any], Dict[str, Any], str, Optional[str], Optional[str]) -> Any
    models = models or {}
    if name is None:
        resolved = _resolve_role(models, kind if kind == "embedding" else "chat", kind)
        if resolved is not None and model:
            resolved.model = model
        return resolved

    clean = name.strip()
    block = _providers_block(models)
    if clean in block:
        spec = dict(block[clean])
        if model:
            spec["model"] = model
        return _create(spec, table, kind, ChatProvider if kind == "chat" else EmbeddingProvider)
    if clean in table:
        spec = {"type": clean}
        if model:
            spec["model"] = model
        return _create(spec, table, kind, ChatProvider if kind == "chat" else EmbeddingProvider)
    raise ConfigurationError(
        "unknown {0} provider {1!r}; use a configured name {2} or a type {3}".format(
            kind, clean, sorted(block), sorted(table)
        )
    )


def resolve_cli_provider(models, provider=None, model=None):
    # type: (Dict[str, Any], Optional[str], Optional[str]) -> Optional[ChatProvider]
    """Resolve ``--provider``/``--model`` from the command line (chat).

    With no model configured at all, returns ``None`` - which is a valid state,
    not an error.
    """
    return _resolve_cli(models, _CHAT_TYPES, "chat", provider, model)


def resolve_cli_embedding(models, provider=None, model=None):
    # type: (Dict[str, Any], Optional[str], Optional[str]) -> Optional[EmbeddingProvider]
    """Resolve ``--embedding-provider``/``--embedding-model`` from the command line."""
    return _resolve_cli(models, _EMBEDDING_TYPES, "embedding", provider, model)


__all__ = [
    "create_provider",
    "create_embedding_provider",
    "create_any_provider",
    "provider_from_name",
    "resolve_role",
    "resolve_embedding_role",
    "resolve_cli_provider",
    "resolve_cli_embedding",
    "provider_types",
    "embedding_provider_types",
    "all_provider_types",
]
