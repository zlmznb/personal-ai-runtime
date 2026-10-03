"""Model providers.

The only package allowed to import an HTTP client or speak a vendor protocol.
Providers return Memory Core's own DTOs, never vendor objects, and every failure
is normalised to :class:`~memory_core.domain.errors.ProviderError` so callers can
degrade instead of crashing.
"""
