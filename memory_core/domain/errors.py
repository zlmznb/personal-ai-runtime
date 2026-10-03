"""Error types shared across Memory Core.

Note: ``MemoryError`` is a Python builtin and is deliberately not shadowed.
"""

from __future__ import annotations


class MemoryCoreError(Exception):
    """Base class for every error raised by Memory Core."""


class ValidationError(MemoryCoreError):
    """A record failed Memory Core's own validation.

    This is the gate that model output must pass before it can reach Storage
    (architecture red line 4).
    """


class StorageError(MemoryCoreError):
    """The persistence layer failed."""


class NotFoundError(MemoryCoreError):
    """A requested record does not exist."""


class ProviderError(MemoryCoreError):
    """A model provider failed.

    Provider failures are degradations, not crashes: the caller must be able to
    continue without the model (architecture red line 5).
    """

    def __init__(self, message, provider=None, model=None):
        # type: (str, object, object) -> None
        super().__init__(message)
        self.provider = provider
        self.model = model


class ConfigurationError(MemoryCoreError):
    """Configuration is missing or invalid."""
