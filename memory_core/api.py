"""The public API facade.

``MemoryCore`` is the only supported way to use this project. Everything below
it is an implementation detail.

Rules enforced here and nowhere else:

* **Red line 4 / 9** - every write goes through the domain guards. There is no
  path that lets model output reach Storage unvalidated.
* **Red line 8** - a model can only *propose*. ``LlmExtractor`` returns
  :class:`Candidate` objects; this class decides whether they are written.
* **Red line 5** - the core never requires a model. ``recall``, ``remember``,
  ``forget``, ``export``, ``snapshot``, preferences and project state all work
  with no provider configured. Semantic retrieval needs an *embedding* provider
  and degrades to keyword retrieval without one.

Retrieval is read-only, and index construction is always explicit
(``build_index`` / ``rebuild_index``). ``recall`` never writes, which is what
lets the model-swap tests assert a byte-identical database.

``ask`` is likewise read-only with respect to Storage. Persisting an exchange is
an explicit caller decision via :meth:`record_event` or :meth:`learn`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from memory_core.config import Config, load_config
from memory_core.domain import guards
from memory_core.domain.dto import (
    CandidateOutcome,
    ChatMessage,
    Observation,
    RecallHit,
)
from memory_core.domain.errors import (
    MemoryCoreError,
    NotFoundError,
    ProviderError,
    StorageError,
    ValidationError,
)
from memory_core.domain.records import Event, Memory, Preference, ProjectState
from memory_core.domain.scopes import (
    ALL_VISIBILITY,
    resolve_visibility,
    visibility_for_scope,
)
from memory_core.formation.base import (
    MEMORY,
    PREFERENCE,
    PROJECT_STATE,
    Candidate,
    Extractor,
)
from memory_core.formation.llm_extractor import LlmExtractor
from memory_core.formation.policy import DEFAULT_POLICY, AcceptancePolicy, normalize_text
from memory_core.formation.rules import RuleExtractor
from memory_core.providers.base import ChatProvider, EmbeddingProvider
from memory_core.providers.registry import resolve_cli_embedding, resolve_cli_provider
from memory_core.retrieval import tokenize
from memory_core.retrieval.indexer import SemanticIndexer
from memory_core.retrieval.retriever import MODES, Retriever
from memory_core.retrieval.vector_index import SqliteVectorIndex, VectorIndex
from memory_core.storage.sqlite_store import SqliteStore

DEFAULT_SYSTEM_PROMPT = (
    "You are a personal AI assistant with a long-term memory core. "
    "The memory context below was retrieved from your persistent memory store. "
    "Treat it as your own recollection. If the context does not contain the "
    "answer, say so rather than inventing one."
)


@dataclass(frozen=True)
class Answer(object):
    """Outcome of :meth:`MemoryCore.ask`, including the degraded case."""

    text: str
    question: str
    provider: Optional[str] = None
    model: Optional[str] = None
    degraded: bool = False
    reason: Optional[str] = None
    context_ids: Tuple[str, ...] = ()
    mode: Optional[str] = None

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "text": self.text,
            "question": self.question,
            "provider": self.provider,
            "model": self.model,
            "degraded": self.degraded,
            "reason": self.reason,
            "context_ids": list(self.context_ids),
            "mode": self.mode,
        }


@dataclass(frozen=True)
class Proposal(object):
    """A dry run of memory formation. Nothing has been, or will be, written."""

    text: str
    scope: str
    candidates: Tuple[Candidate, ...] = ()
    outcomes: Tuple[CandidateOutcome, ...] = ()
    context_ids: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "text": self.text,
            "scope": self.scope,
            "context_ids": list(self.context_ids),
            "reasons": list(self.reasons),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
        }


def format_hit(hit):
    # type: (RecallHit) -> str
    """Render one retrieved item for a prompt or for degraded output."""
    item = hit.item
    if hit.item_type == "preference":
        return "preference {0} = {1}".format(item.key, item.value)
    return "[{0}] {1}".format(item.kind, item.content)


def build_prompt(hits, question, system_prompt=None):
    # type: (Sequence[RecallHit], str, Optional[str]) -> List[ChatMessage]
    if hits:
        context = "\n".join("- " + format_hit(hit) for hit in hits)
    else:
        context = "(no relevant memories)"
    system = "{0}\n\nMemory context:\n{1}".format(system_prompt or DEFAULT_SYSTEM_PROMPT, context)
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=question)]


class MemoryCore(object):
    """Model-agnostic personal memory core."""

    def __init__(
        self,
        store,                       # type: SqliteStore
        config=None,                 # type: Optional[Config]
        provider=None,               # type: Optional[ChatProvider]
        extractor=None,              # type: Optional[Extractor]
        embedding_provider=None,     # type: Optional[EmbeddingProvider]
        vector_index=None,           # type: Optional[VectorIndex]
        policy=None,                 # type: Optional[AcceptancePolicy]
        use_llm_formation=True,      # type: bool
        persona=None,                # type: Optional[str]
    ):
        # type: (...) -> None
        self.store = store
        self.config = config
        self.provider = provider
        self.extractor = extractor or RuleExtractor()
        self.embedding_provider = embedding_provider
        self.policy = policy or DEFAULT_POLICY
        self.use_llm_formation = bool(use_llm_formation)
        #: Default persona for reads. ``None`` means "no persona": reads then see
        #: every scope except ``persona:*`` (fail-closed).
        self.persona = persona

        if vector_index is not None and embedding_provider is not None:
            if vector_index.model_id != embedding_provider.identifier():
                raise MemoryCoreError(
                    "vector index is built for {0!r} but the embedding provider is {1!r}".format(
                        vector_index.model_id, embedding_provider.identifier()
                    )
                )
        if vector_index is None and embedding_provider is not None:
            vector_index = SqliteVectorIndex(store, embedding_provider.identifier())
        self.vector_index = vector_index

        weights = config.retrieval.weights if config is not None else None
        self.retriever = Retriever(
            store,
            weights=weights,
            vector_index=vector_index,
            embedding_provider=embedding_provider,
            persona=persona,
        )
        self.llm_extractor = (
            LlmExtractor(provider) if (provider is not None and self.use_llm_formation) else None
        )
        self._indexer = None  # type: Optional[SemanticIndexer]

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(
        cls,
        db_path=None,                    # type: Optional[str]
        config_path=None,                # type: Optional[str]
        models_path=None,                # type: Optional[str]
        provider_name=None,              # type: Optional[str]
        model=None,                      # type: Optional[str]
        provider=None,                   # type: Optional[ChatProvider]
        embedding_provider_name=None,    # type: Optional[str]
        embedding_model=None,            # type: Optional[str]
        embedding_provider=None,         # type: Optional[EmbeddingProvider]
        use_llm_formation=True,          # type: bool
        persona=None,                    # type: Optional[str]
    ):
        # type: (...) -> "MemoryCore"
        """Open (creating if necessary) a memory database.

        Works with no configuration and no model at all.
        """
        config = load_config(config_path=config_path, db_path=None, models_path=models_path)
        if db_path is not None:
            config = Config(
                db_path=db_path,
                retrieval=config.retrieval,
                models=config.models,
                config_path=config.config_path,
                models_path=config.models_path,
            )
        store = SqliteStore(config.db_path)

        resolved_chat = provider
        if resolved_chat is None:
            resolved_chat = resolve_cli_provider(config.models, provider_name, model)
        elif model:
            resolved_chat.model = model

        resolved_embedding = embedding_provider
        if resolved_embedding is None:
            resolved_embedding = resolve_cli_embedding(
                config.models, embedding_provider_name, embedding_model
            )
        elif embedding_model:
            resolved_embedding.model = embedding_model

        return cls(
            store,
            config=config,
            provider=resolved_chat,
            embedding_provider=resolved_embedding,
            use_llm_formation=use_llm_formation,
            persona=persona,
        )

    def close(self):
        # type: () -> None
        self.store.close()

    def __enter__(self):
        # type: () -> "MemoryCore"
        return self

    def __exit__(self, exc_type, exc, tb):
        # type: (Any, Any, Any) -> None
        self.close()

    # -- events ------------------------------------------------------------

    def record_event(
        self,
        content,               # type: str
        role="user",           # type: str
        kind="message",        # type: str
        session_id=None,       # type: Optional[str]
        scope="global",        # type: str
        metadata=None,         # type: Optional[Dict[str, Any]]
        source="cli",          # type: str
    ):
        # type: (...) -> Event
        """Append a raw event. Events are immutable history (constraint P0-3)."""
        event = guards.build_event(
            kind,
            content,
            role=role,
            session_id=session_id,
            scope=scope,
            metadata=metadata,
            source=source,
        )
        return self.store.insert_event(event)

    # -- formation ---------------------------------------------------------

    def observe(
        self,
        text,                  # type: str
        role="user",           # type: str
        session_id=None,       # type: Optional[str]
        scope="global",        # type: str
        kind="message",        # type: str
        metadata=None,         # type: Optional[Dict[str, Any]]
        source="cli",          # type: str
        provenance=None,       # type: Optional[Dict[str, Any]]
    ):
        # type: (...) -> Observation
        """Record an utterance and run the deterministic rules over it.

        No model is involved, by design: explicit instructions such as
        ``记住：...`` must keep working with nothing installed.
        """
        return self._form_and_apply(
            text, [self.extractor], role, session_id, scope, 0,
            kind=kind, metadata=metadata, source=source, provenance=provenance,
        )

    def learn(
        self,
        text,                  # type: str
        role="user",           # type: str
        session_id=None,       # type: Optional[str]
        scope="global",        # type: str
        context_limit=5,       # type: int
        kind="message",        # type: str
        metadata=None,         # type: Optional[Dict[str, Any]]
        source="cli",          # type: str
        provenance=None,       # type: Optional[Dict[str, Any]]
    ):
        # type: (...) -> Observation
        """Record an utterance, then run rules *and* LLM formation over it.

        The event is always written, even when every candidate is rejected: raw
        history is never lost, so the turn can be re-extracted later.

        ``kind`` / ``metadata`` / ``source`` are forwarded to the event, and
        ``provenance`` is merged into every committed record's ``source``. That
        is how an external consumer (a Bridge) attaches where a memory came
        from. All four default to the v0.2 behaviour.
        """
        extractors = [self.extractor]
        if self.llm_extractor is not None:
            extractors.append(self.llm_extractor)
        return self._form_and_apply(
            text, extractors, role, session_id, scope, context_limit,
            kind=kind, metadata=metadata, source=source, provenance=provenance,
        )

    def propose(
        self,
        text,                  # type: str
        scope="global",        # type: str
        context_limit=5,       # type: int
        include_llm=True,      # type: bool
    ):
        # type: (...) -> Proposal
        """Dry run: return validated candidates and verdicts. Writes nothing.

        Neither the event nor any memory is persisted, so this is safe to call
        on arbitrary text.
        """
        extractors = [self.extractor]
        if include_llm and self.llm_extractor is not None:
            extractors.append(self.llm_extractor)
        context, _context_reason = self._formation_context(text, scope, context_limit)
        candidates, reasons = self._collect(text, extractors, scope, context)
        outcomes = []  # type: List[CandidateOutcome]
        for candidate in candidates:
            decision = self.policy.evaluate(candidate, **self._policy_facts(candidate, scope))
            outcomes.append(
                CandidateOutcome(
                    candidate=candidate, accepted=decision.accepted, reason=decision.reason
                )
            )
        return Proposal(
            text=text,
            scope=scope,
            candidates=tuple(candidates),
            outcomes=tuple(outcomes),
            context_ids=tuple(getattr(memory, "id", "") for memory in context),
            reasons=tuple(reasons),
        )

    def _formation_context(self, text, scope, limit):
        # type: (str, str, int) -> Tuple[List[Any], Optional[str]]
        """Existing memories shown to an LLM extractor. Read-only.

        Visibility is derived from the scope being written *into*, so forming a
        memory in ``persona:aria`` can see ``global`` and ``persona:aria``, and
        can never see ``persona:bruno``. This is the second half of the persona
        isolation guarantee - without it the leak would simply move from
        retrieval into the prompt.
        """
        if limit is None or int(limit) <= 0:
            return [], None
        if self.llm_extractor is None or not str(text or "").strip():
            return [], None
        visibility = visibility_for_scope(scope)
        hits = self.retriever.recall(
            text,
            visibility=visibility,
            limit=int(limit),
            include_preferences=False,
            mode="hybrid",
        )
        return [hit.item for hit in hits if hit.item_type == "memory"], None

    def _collect(self, text, extractors, scope, context):
        # type: (str, Sequence[Extractor], str, Sequence[Any]) -> Tuple[List[Candidate], List[str]]
        """Run every extractor and de-duplicate their proposals.

        Rules run first, so when a rule and a model propose the same thing the
        deterministic, higher-confidence proposal wins.
        """
        candidates = []  # type: List[Candidate]
        reasons = []  # type: List[str]
        for extractor in extractors:
            try:
                extraction = extractor.extract(text, scope=scope, context=context)
            except Exception as exc:  # a broken extractor must not lose the event
                reasons.append(
                    "{0} failed: {1}: {2}".format(
                        getattr(extractor, "name", "extractor"), type(exc).__name__, exc
                    )
                )
                continue
            candidates.extend(extraction.candidates)
            if extraction.reason:
                reasons.append(extraction.reason)
        return self._dedupe(candidates), reasons

    @staticmethod
    def _dedupe(candidates):
        # type: (Sequence[Candidate]) -> List[Candidate]
        seen = set()
        unique = []
        for candidate in candidates:
            payload = candidate.payload or {}
            key = (
                candidate.target,
                candidate.operation,
                candidate.supersedes,
                normalize_text(payload.get("content") or payload.get("key") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _policy_facts(self, candidate, scope):
        # type: (Candidate, str) -> Dict[str, Any]
        """The facts the acceptance policy needs. Performs reads only."""
        facts = {"known_ids": ()}  # type: Dict[str, Any]
        payload = candidate.payload or {}
        if candidate.target == MEMORY and payload.get("content"):
            facts["existing_contents"] = self._duplicate_contents(payload["content"], scope)
        if candidate.supersedes:
            exists = self.store.get_memory(candidate.supersedes) is not None
            facts["known_ids"] = (candidate.supersedes,) if exists else ()
        return facts

    def _duplicate_contents(self, content, scope):
        # type: (str, str) -> Tuple[str, ...]
        """Normalised contents of existing memories identical to ``content``."""
        expression = tokenize.build_match_expression(content)
        if expression is None:
            return ()
        target = normalize_text(content)
        rows = self.store.search_memories(
            expression,
            tokenize.phrase_needle(content),
            scope=scope,
            statuses=("active",),
            limit=10,
        )
        return tuple(
            normalize_text(memory.content)
            for memory, _rank, _phrase in rows
            if normalize_text(memory.content) == target
        )

    def _form_and_apply(
        self, text, extractors, role, session_id, scope, context_limit,
        kind="message", metadata=None, source="cli", provenance=None,
    ):
        # type: (str, Sequence[Extractor], str, Optional[str], str, int, str, Any, str, Any) -> Observation
        """The shared formation pipeline: event -> candidates -> verdict -> commit."""
        event = self.record_event(
            text, role=role, kind=kind, session_id=session_id, scope=scope,
            metadata=metadata, source=source,
        )
        needs_context = any(isinstance(extractor, LlmExtractor) for extractor in extractors)
        context = self._formation_context(text, scope, context_limit)[0] if needs_context else []
        candidates, reasons = self._collect(text, extractors, scope, context)

        memories = []  # type: List[Memory]
        preferences = []  # type: List[Preference]
        states = []  # type: List[ProjectState]
        outcomes = []  # type: List[CandidateOutcome]

        for candidate in candidates:
            try:
                decision = self.policy.evaluate(candidate, **self._policy_facts(candidate, scope))
            except Exception as exc:  # a policy bug must not lose the event
                outcomes.append(
                    CandidateOutcome(
                        candidate=candidate,
                        accepted=False,
                        reason="policy error: {0}: {1}".format(type(exc).__name__, exc),
                    )
                )
                continue
            if decision.rejected:
                outcomes.append(
                    CandidateOutcome(candidate=candidate, accepted=False, reason=decision.reason)
                )
                continue

            try:
                item_type, record = self._commit(candidate, event, provenance)
            except (ValidationError, StorageError, NotFoundError) as exc:
                # The write either fully happened or fully rolled back; the
                # candidate is recorded as rejected either way.
                outcomes.append(
                    CandidateOutcome(
                        candidate=candidate,
                        accepted=False,
                        reason="write rejected: {0}".format(exc),
                    )
                )
                continue

            outcomes.append(
                CandidateOutcome(
                    candidate=candidate,
                    accepted=True,
                    reason=candidate.reason,
                    item_type=item_type,
                    item_id=record.id,
                )
            )
            if item_type == MEMORY:
                memories.append(record)
            elif item_type == PREFERENCE:
                preferences.append(record)
            else:
                states.append(record)

        skipped = None
        if not candidates and reasons:
            skipped = "; ".join(reasons)

        return Observation(
            event=event,
            memories=tuple(memories),
            preferences=tuple(preferences),
            project_states=tuple(states),
            skipped_reason=skipped,
            outcomes=tuple(outcomes),
        )

    def _commit(self, candidate, event, provenance=None):
        # type: (Candidate, Event, Any) -> Tuple[str, Any]
        """Validate and write one accepted candidate. Raises on any failure.

        ``guards.build_*`` validates every field, so a malformed candidate
        raises :class:`ValidationError` and is dropped rather than persisted
        (red line 4 / 9).

        ``provenance`` is merged into the record's ``source`` after the
        candidate's own source, so a caller (a Bridge) can attach where the
        content came from without being able to override the formation path
        recorded by the extractor.
        """
        payload = dict(candidate.payload or {})
        source = dict(payload.get("source") or {})
        source.setdefault("event_ids", [event.id])
        if provenance:
            formed_by = source.get("origin")
            source.update(provenance)
            if formed_by and not source.get("formed_by"):
                # Keep the formation path visible even when the caller supplies
                # its own ``origin``.
                source["formed_by"] = formed_by
        payload["source"] = source

        if candidate.target == MEMORY:
            payload.setdefault("confidence", candidate.confidence)
            if candidate.origin == "llm":
                payload.setdefault("generated_by", source.get("model"))
            memory = guards.build_memory(**payload)
            if candidate.supersedes:
                return MEMORY, self.store.supersede_memory(candidate.supersedes, memory)
            return MEMORY, self.store.insert_memory(memory)

        if candidate.target == PREFERENCE:
            payload.setdefault("confidence", candidate.confidence)
            return PREFERENCE, self.store.insert_preference(guards.build_preference(**payload))

        if candidate.target == PROJECT_STATE:
            return PROJECT_STATE, self.store.set_project_state(
                guards.build_project_state(**payload)
            )

        raise ValidationError("unknown candidate target {0!r}".format(candidate.target))

    # -- memories ----------------------------------------------------------

    def remember(
        self,
        content,               # type: str
        kind="semantic",       # type: str
        scope="global",        # type: str
        subject="user:self",   # type: str
        tags=(),               # type: Sequence[str]
        structured=None,       # type: Optional[Dict[str, Any]]
        confidence=1.0,        # type: float
        salience=0.5,          # type: float
        source=None,           # type: Optional[Dict[str, Any]]
        event_ids=None,        # type: Optional[Sequence[str]]
        generated_by=None,     # type: Optional[str]
        valid_from=None,       # type: Optional[str]
        valid_to=None,         # type: Optional[str]
    ):
        # type: (...) -> Memory
        """Write a memory directly. The single validated write gate (red line 4)."""
        source = dict(source or {})
        if event_ids:
            source.setdefault("event_ids", list(event_ids))
        memory = guards.build_memory(
            content,
            kind=kind,
            scope=scope,
            subject=subject,
            tags=tags,
            structured=structured,
            confidence=confidence,
            salience=salience,
            source=source,
            generated_by=generated_by,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        return self.store.insert_memory(memory)

    def supersede_memory(self, memory_id, content, **kwargs):
        # type: (str, str, Any) -> Memory
        """Replace a memory, keeping the old row as history."""
        replacement = guards.build_memory(content, **kwargs)
        return self.store.supersede_memory(memory_id, replacement)

    def get_memory(self, memory_id):
        # type: (str) -> Optional[Memory]
        return self.store.get_memory(memory_id)

    def list_memories(
        self,
        scope=None,
        kinds=None,
        statuses=("active",),
        limit=100,
        offset=0,
        scopes=None,
        persona=None,
        include_personas=False,
    ):
        # type: (Any, Any, Any, int, int, Any, Any, bool) -> List[Memory]
        visibility = self._read_visibility(scope, scopes, persona, include_personas)
        return self.store.list_memories(
            kinds=kinds,
            statuses=statuses,
            limit=limit,
            offset=offset,
            scopes=visibility.scopes,
            exclude_prefixes=visibility.exclude_prefixes,
        )

    def list_events(self, scope=None, session_id=None, limit=100, offset=0):
        # type: (Optional[str], Optional[str], int, int) -> List[Event]
        """Read raw history.

        Exposed so an external consumer can implement ingest idempotency
        (``session_id`` is indexed) without touching storage directly.
        """
        return self.store.list_events(
            scope=scope, session_id=session_id, limit=limit, offset=offset
        )

    def list_scopes(self):
        # type: () -> List[str]
        """Every scope currently holding a row, across all source tables."""
        return self.store.list_scopes()

    def forget(self, memory_id, hard=False):
        # type: (str, bool) -> bool
        """Remove a memory. Logical by default; history survives (ADR 0001)."""
        if self.store.get_memory(memory_id) is None:
            raise NotFoundError("memory {0} not found".format(memory_id))
        if hard:
            return self.store.delete_memory(memory_id)
        self.store.set_memory_status(memory_id, "deleted")
        return True

    # -- retrieval ---------------------------------------------------------

    def recall(
        self,
        query,                  # type: str
        scope=None,             # type: Optional[str]
        kinds=None,             # type: Optional[Sequence[str]]
        limit=None,             # type: Optional[int]
        include_preferences=True,  # type: bool
        mode="hybrid",          # type: str
        scopes=None,            # type: Optional[Sequence[str]]
        persona=None,           # type: Optional[str]
        include_personas=False, # type: bool
    ):
        # type: (...) -> List[RecallHit]
        """Deterministic retrieval: keyword, semantic, or hybrid.

        Model-free on the chat side (ADR 0003). Semantic mode uses the embedding
        provider, which is an independent choice. Hybrid degrades to keyword when
        no embedding provider or no index is available.

        Scope visibility is **fail-closed**: with no scope arguments, ``persona:*``
        is never returned. See ``domain/scopes.py``.
        """
        if self.config is not None:
            resolved_limit = self.config.retrieval.clamp_limit(limit)
        else:
            resolved_limit = 10 if limit is None else int(limit)
        visibility = self._read_visibility(scope, scopes, persona, include_personas)
        return self.retriever.recall(
            query,
            visibility=visibility,
            kinds=kinds,
            limit=resolved_limit,
            include_preferences=include_preferences,
            mode=mode,
        )

    def _read_visibility(self, scope, scopes, persona, include_personas=False):
        # type: (Any, Any, Any, bool) -> Any
        """Resolve read visibility, defaulting to the session persona."""
        if include_personas:
            return ALL_VISIBILITY
        effective = persona if persona is not None else self.persona
        return resolve_visibility(scope=scope, scopes=scopes, persona=effective)

    def recall_modes(self):
        # type: () -> Sequence[str]
        return MODES

    # -- semantic index ----------------------------------------------------

    @property
    def indexer(self):
        # type: () -> SemanticIndexer
        if self.embedding_provider is None:
            raise MemoryCoreError(
                "no embedding provider is configured; pass --embedding-provider "
                "or bind the 'embedding' role in config/models.json"
            )
        if self._indexer is None:
            self._indexer = SemanticIndexer(
                self.store, self.embedding_provider, index=self.vector_index
            )
        return self._indexer

    def build_index(self, scope=None, kinds=None, limit=None):
        # type: (Any, Any, Optional[int]) -> Dict[str, Any]
        """Embed memories that are missing from the vector index (incremental)."""
        return self.indexer.build(scope=scope, kinds=kinds, limit=limit)

    def rebuild_index(self, scope=None, kinds=None, limit=None):
        # type: (Any, Any, Optional[int]) -> Dict[str, Any]
        """Throw this model's vectors away and regenerate them from SQLite."""
        return self.indexer.rebuild(scope=scope, kinds=kinds, limit=limit)

    def drop_index(self, all_models=False):
        # type: (bool) -> int
        """Delete derived vectors. The memories are untouched.

        ``all_models=True`` removes every model's vectors; the default removes
        only the currently configured embedding model's.
        """
        return self.indexer.drop(all_models=all_models)

    def index_stats(self):
        # type: () -> Optional[Dict[str, Any]]
        if self.embedding_provider is None:
            return None
        return self.indexer.stats()

    # -- preferences -------------------------------------------------------

    def set_preference(
        self,
        key,                   # type: str
        value,                 # type: Any
        scope="global",        # type: str
        statement=None,        # type: Optional[str]
        confidence=1.0,        # type: float
        source=None,           # type: Optional[Dict[str, Any]]
    ):
        # type: (...) -> Preference
        """Set a preference, superseding the previous value for that key."""
        preference = guards.build_preference(
            key,
            value,
            scope=scope,
            statement=statement,
            confidence=confidence,
            source=source,
        )
        return self.store.insert_preference(preference)

    def get_preference(self, key, scope="global"):
        # type: (str, str) -> Optional[Preference]
        return self.store.get_preference(key, scope=scope)

    def list_preferences(self, scope=None, include_superseded=False, scopes=None, include_personas=False):
        # type: (Optional[str], bool, Any, bool) -> List[Preference]
        visibility = self._read_visibility(scope, scopes, None, include_personas)
        return self.store.list_preferences(
            include_superseded=include_superseded,
            scopes=visibility.scopes,
            exclude_prefixes=visibility.exclude_prefixes,
        )

    def preference_history(self, key, scope="global"):
        # type: (str, str) -> List[Preference]
        return self.store.preference_history(key, scope=scope)

    # -- project state -----------------------------------------------------

    def set_project_state(self, project_id, state, scope=None, note=None, source=None):
        # type: (str, Any, Optional[str], Optional[str], Any) -> ProjectState
        record = guards.build_project_state(
            project_id, state, scope=scope, note=note, source=source
        )
        return self.store.set_project_state(record)

    def get_project_state(self, project_id, scope=None):
        # type: (str, Optional[str]) -> Optional[ProjectState]
        return self.store.get_project_state(project_id, scope=scope)

    def project_state_history(self, project_id, scope=None):
        # type: (str, Optional[str]) -> List[ProjectState]
        return self.store.project_state_history(project_id, scope=scope)

    def list_project_states(self, scope=None, scopes=None, include_personas=False):
        # type: (Optional[str], Any, bool) -> List[ProjectState]
        visibility = self._read_visibility(scope, scopes, None, include_personas)
        return self.store.list_project_states(
            scopes=visibility.scopes, exclude_prefixes=visibility.exclude_prefixes
        )

    # -- model -------------------------------------------------------------

    def ask(
        self,
        question,               # type: str
        scope=None,             # type: Optional[str]
        kinds=None,             # type: Optional[Sequence[str]]
        limit=None,             # type: Optional[int]
        system_prompt=None,     # type: Optional[str]
        mode="hybrid",          # type: str
    ):
        # type: (...) -> Answer
        """Answer a question using recalled memory as context.

        Read-only with respect to Storage. Degrades to returning the recalled
        context when no provider is configured or the provider fails.
        """
        hits = self.recall(question, scope=scope, kinds=kinds, limit=limit, mode=mode)
        context_ids = tuple(hit.id for hit in hits)
        messages = build_prompt(hits, question, system_prompt=system_prompt)

        if self.provider is None:
            return Answer(
                text=self._degraded_text(hits),
                question=question,
                degraded=True,
                reason="no model provider configured",
                context_ids=context_ids,
                mode=mode,
            )
        try:
            result = self.provider.chat(messages)
        except ProviderError as exc:
            return Answer(
                text=self._degraded_text(hits),
                question=question,
                provider=self.provider.name,
                model=self.provider.model,
                degraded=True,
                reason=str(exc),
                context_ids=context_ids,
                mode=mode,
            )
        return Answer(
            text=result.text,
            question=question,
            provider=result.provider,
            model=result.model,
            degraded=False,
            context_ids=context_ids,
            mode=mode,
        )

    @staticmethod
    def _degraded_text(hits):
        # type: (Sequence[RecallHit]) -> str
        if not hits:
            return "(no model available, and no relevant memories found)"
        lines = ["(no model available; recalled memory context follows)"]
        lines.extend("- " + format_hit(hit) for hit in hits)
        return "\n".join(lines)

    def provider_info(self):
        # type: () -> Optional[Dict[str, Any]]
        return None if self.provider is None else self.provider.describe()

    def embedding_info(self):
        # type: () -> Optional[Dict[str, Any]]
        return None if self.embedding_provider is None else self.embedding_provider.describe()

    def formation_info(self):
        # type: () -> Dict[str, Any]
        described = {
            "rules": self.extractor.name,
            "llm_formation": self.llm_extractor is not None,
            "policy": self.policy.as_dict(),
        }  # type: Dict[str, Any]
        if self.llm_extractor is not None:
            described["llm"] = self.llm_extractor.describe()
        return described

    # -- export / snapshot -------------------------------------------------

    def export_jsonl(self):
        # type: () -> str
        return self.store.export_jsonl()

    def export_to(self, path):
        # type: (str) -> str
        payload = self.export_jsonl()
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        return path

    def snapshot(self, path):
        # type: (str) -> str
        return self.store.snapshot(path)

    def stats(self):
        # type: () -> Dict[str, Any]
        result = self.store.stats()
        result["provider"] = self.provider_info()
        result["embedding_provider"] = self.embedding_info()
        result["extractor"] = self.extractor.name
        result["formation"] = self.formation_info()
        result["retrieval"] = self.retriever.describe()
        result["page_bytes"] = self.store.page_size_bytes()
        return result


__all__ = ["MemoryCore", "Answer", "Proposal", "build_prompt", "format_hit", "DEFAULT_SYSTEM_PROMPT"]
