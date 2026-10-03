"""Pure domain layer: data structures, validation, identifiers.

Hard rule (architecture red line 2): nothing in this package may import a model
SDK, an HTTP client, or any third-party library. It must be importable and
testable with no network and no model available at all.

Only ``memory_core.providers`` may touch the network.
"""
