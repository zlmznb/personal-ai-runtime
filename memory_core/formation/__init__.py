"""Memory Formation.

Formation turns raw events into validated *candidates*. It never writes to
Storage, and it must degrade to nothing when its model (if any) is unavailable.

v0.1 is rules-only: see :mod:`memory_core.formation.rules`.
"""
