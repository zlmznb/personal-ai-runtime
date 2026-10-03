"""Ranking policy for retrieval.

All scoring lives here so that the policy is in exactly one place and is
obviously model-free (ADR 0003).

Determinism rules:

* Scores are computed only from values stored in the database (BM25 rank,
  salience, confidence) - never from wall-clock time, so no time decay relative
  to "now".
* Ties always break on ``(item_type, created_at, id)``, which are fixed at write
  time. The same query against the same database therefore always produces the
  same ordering, which is what makes the model-swap assertion a hard guarantee
  rather than a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

from memory_core.domain.dto import RecallHit
from memory_core.domain.enums import ItemType

#: Additive bonus for an exact-substring match, on top of BM25.
#: Exact phrase matches always outrank loose token matches.
PHRASE_BONUS = 1.0

#: Flat prior added to preference scores.
#:
#: BM25 is an IDF-weighted measure, so on a small corpus - which a personal
#: memory database always is at the start - every score collapses towards zero
#: and ranking becomes arbitrary. A flat prior makes "an explicit user
#: declaration outranks an equally-matching inference" true regardless of corpus
#: size, without depending on IDF at all.
PREFERENCE_PRIOR = 0.5

#: Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper
#: and is insensitive across corpora, which is exactly why RRF is used to merge
#: two rankings whose raw scores are not comparable (BM25 vs cosine).
RRF_K = 60

#: In *fused* mode, bonuses are multiplicative rather than additive.
#:
#: RRF deliberately flattens score differences into ~1/(k+rank), so an additive
#: bonus of 1.0 would swamp the fusion entirely. Multiplying preserves the
#: fusion ordering while still letting an exact phrase or an explicit
#: preference win a close call.
PHRASE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class RetrievalWeights(object):
    """Tunable weights. Defaults live in config/memory.json."""

    memory_weight: float = 1.0
    preference_weight: float = 1.5
    salience_weight: float = 0.15
    preference_prior: float = PREFERENCE_PRIOR

    def as_dict(self):
        # type: () -> dict
        return {
            "memory_weight": self.memory_weight,
            "preference_weight": self.preference_weight,
            "salience_weight": self.salience_weight,
            "preference_prior": self.preference_prior,
        }


DEFAULT_WEIGHTS = RetrievalWeights()


def score_memory(rank, salience, phrase_hit, weights=DEFAULT_WEIGHTS):
    # type: (float, float, bool, RetrievalWeights) -> float
    """Turn a raw FTS5 bm25 rank (more negative is better) into a score."""
    base = (-float(rank)) * weights.memory_weight
    score = base * (1.0 + weights.salience_weight * float(salience))
    if phrase_hit:
        score += PHRASE_BONUS
    return score


def score_preference(rank, confidence, phrase_hit, weights=DEFAULT_WEIGHTS):
    # type: (float, float, bool, RetrievalWeights) -> float
    """Preferences are boosted because an explicit declaration outranks an inference."""
    base = (-float(rank)) * weights.preference_weight
    score = base * (1.0 + weights.salience_weight * float(confidence))
    score += weights.preference_prior
    if phrase_hit:
        score += PHRASE_BONUS
    return score


def rank_hits(memory_rows, preference_rows, weights=DEFAULT_WEIGHTS, limit=None):
    # type: (Iterable[Tuple[Any, float, bool]], Iterable[Tuple[Any, float, bool]], RetrievalWeights, Any) -> List[RecallHit]
    """Merge memory and preference hits into one deterministically ordered list."""
    hits = []  # type: List[RecallHit]
    for memory, rank, phrase_hit in memory_rows:
        hits.append(
            RecallHit(
                item=memory,
                item_type=ItemType.MEMORY.value,
                score=score_memory(rank, memory.salience, phrase_hit, weights),
                matched_on="phrase" if phrase_hit else "fts",
            )
        )
    for preference, rank, phrase_hit in preference_rows:
        hits.append(
            RecallHit(
                item=preference,
                item_type=ItemType.PREFERENCE.value,
                score=score_preference(rank, preference.confidence, phrase_hit, weights),
                matched_on="phrase" if phrase_hit else "fts",
            )
        )
    hits.sort(
        key=lambda hit: (
            -hit.score,
            hit.item_type,
            getattr(hit.item, "created_at", ""),
            hit.item.id,
        )
    )
    if limit is not None:
        return hits[: int(limit)]
    return hits


def reciprocal_rank_fusion(rankings, k=RRF_K, weights=None):
    # type: (Sequence[Any], int, Optional[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, List[str]]]
    """Fuse several ranked id lists by Reciprocal Rank Fusion.

    ``rankings`` is a sequence of ``(source_name, [(item_id, score), ...])``
    where each list is already ordered best-first. Only *rank* is used, never
    the raw score, which is what makes it safe to merge BM25 with cosine
    similarity - two numbers with no common scale.

    Returns ``(scores, sources)``. Both are plain dicts, and the caller applies a
    total ordering, so the result is deterministic.
    """
    scores = {}  # type: Dict[str, float]
    sources = {}  # type: Dict[str, List[str]]
    for source_name, items in rankings:
        weight = 1.0 if not weights else float(weights.get(source_name, 1.0))
        for position, entry in enumerate(items, start=1):
            item_id = entry[0] if isinstance(entry, (tuple, list)) else entry
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + position)
            sources.setdefault(item_id, []).append(source_name)
    return scores, sources


def fuse_hybrid(keyword_hits, semantic_hits, weights=DEFAULT_WEIGHTS, limit=None, k=RRF_K):
    # type: (Sequence[Any], Sequence[Any], RetrievalWeights, Any, int) -> List[Any]
    """Merge keyword and vector hits into one deterministically ordered list.

    ``keyword_hits`` and ``semantic_hits`` are :class:`RecallHit` sequences.
    Callers must handle the single-source cases themselves (returning the one
    ranking unchanged), so this function is only for genuine fusion.
    """
    from memory_core.domain.dto import RecallHit
    from memory_core.domain.enums import ItemType

    rankings = []
    if keyword_hits:
        rankings.append(("keyword", [(hit.id, hit.score) for hit in keyword_hits]))
    if semantic_hits:
        rankings.append(("vector", [(hit.id, hit.score) for hit in semantic_hits]))
    scores, sources = reciprocal_rank_fusion(rankings, k=k)

    lookup = {}  # type: Dict[str, Any]
    for hit in keyword_hits:
        lookup[hit.id] = hit
    for hit in semantic_hits:
        lookup.setdefault(hit.id, hit)

    fused = []  # type: List[RecallHit]
    for item_id, score in scores.items():
        original = lookup[item_id]
        source_names = sources[item_id]
        if len(source_names) > 1:
            matched_on = "hybrid"
        elif source_names[0] == "keyword":
            matched_on = original.matched_on
        else:
            matched_on = "vector"

        if original.matched_on == "phrase":
            score *= PHRASE_MULTIPLIER
        if original.item_type == ItemType.PREFERENCE.value:
            score *= weights.preference_weight
        fused.append(
            RecallHit(
                item=original.item,
                item_type=original.item_type,
                score=score,
                matched_on=matched_on,
            )
        )

    fused.sort(
        key=lambda hit: (
            -hit.score,
            hit.item_type,
            getattr(hit.item, "created_at", ""),
            hit.item.id,
        )
    )
    if limit is not None:
        return fused[: int(limit)]
    return fused


__all__ = [
    "RetrievalWeights",
    "DEFAULT_WEIGHTS",
    "PHRASE_BONUS",
    "PHRASE_MULTIPLIER",
    "PREFERENCE_PRIOR",
    "RRF_K",
    "score_memory",
    "score_preference",
    "rank_hits",
    "reciprocal_rank_fusion",
    "fuse_hybrid",
]
