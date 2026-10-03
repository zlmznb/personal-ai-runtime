"""Read-only, deterministic retrieval.

Hard rule (ADR 0003): nothing in this package may call a model. Results must be
a pure function of the stored data, so that swapping a provider provably cannot
change what is recalled.
"""
