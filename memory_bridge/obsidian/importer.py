"""Obsidian -> Memory Core import.

The pipeline, per document::

    markdown
      -> document_hash            (skip the whole file if already ingested)
      -> chunks with content_hash (skip only the chunks already ingested)
      -> route per chunk          (classify)
      -> Event                    (verbatim text + full provenance)
      -> formation or typed API
      -> Memory Core / SQLite

Guarantees:

* **Idempotent.** Re-importing an unchanged vault creates zero events and zero
  memories: the document hash short-circuits the whole file.
* **Never writes to the vault.** Read-only scanning.
* **Dry run writes nothing at all.** It uses ``propose()``, which has no side
  effects, so the database file is byte-identical afterwards.
* **The bridge never writes a memory itself.** Only ``learn()`` / ``remember()``
  and the typed setters can, and they validate everything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace as dataclass_replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from memory_bridge.config import BridgeConfig
from memory_bridge.obsidian import classify as classify_module
from memory_bridge.obsidian.provenance import (
    EVENT_SOURCE,
    external_ref,
    hash_text,
    memory_provenance,
    memory_source,
)
from memory_bridge.obsidian.vault import scan_vault

#: Session id == external_ref, which is an indexed column: idempotency queries
#: are therefore cheap and need no schema change.
_SESSION_PREFIX = "obsidian"

#: How many past events to inspect per document when checking idempotency.
_EVENT_SCAN_LIMIT = 100000


def _utc_now():
    # type: () -> str
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass(frozen=True)
class ChunkAction(object):
    """One chunk's plan (dry run) or result (apply)."""

    document_rel: str
    title: str
    heading_path: Tuple[str, ...]
    content_hash: str
    route: str
    scope: str
    reason: str
    action: str                       # "ingest" | "skip-seen"
    preview: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None
    item_ids: Tuple[str, ...] = ()
    accepted: int = 0
    rejected: int = 0
    proposed: Tuple[Dict[str, Any], ...] = ()
    skipped_reason: Optional[str] = None

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "document": self.document_rel,
            "title": self.title,
            "heading_path": list(self.heading_path),
            "content_hash": self.content_hash,
            "route": self.route,
            "scope": self.scope,
            "reason": self.reason,
            "action": self.action,
            "preview": self.preview,
            "payload": self.payload,
            "event_id": self.event_id,
            "item_ids": list(self.item_ids),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "proposed": list(self.proposed),
            "skipped_reason": self.skipped_reason,
        }


@dataclass(frozen=True)
class DocumentPlan(object):
    """One document's plan or result."""

    rel_path: str
    title: str
    document_hash: str
    status: str                       # "new" | "changed" | "unchanged" | "empty"
    actions: Tuple[ChunkAction, ...] = ()

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "rel_path": self.rel_path,
            "title": self.title,
            "document_hash": self.document_hash,
            "status": self.status,
            "actions": [action.as_dict() for action in self.actions],
        }


@dataclass(frozen=True)
class ImportReport(object):
    """Everything one import run did (or would do)."""

    dry_run: bool
    vault_path: str
    vault_id: str
    documents: Tuple[DocumentPlan, ...] = ()
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def actions(self):
        # type: () -> List[ChunkAction]
        return [action for document in self.documents for action in document.actions]

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "dry_run": self.dry_run,
            "vault_path": self.vault_path,
            "vault_id": self.vault_id,
            "counts": dict(self.counts),
            "documents": [document.as_dict() for document in self.documents],
        }

    def render(self):
        # type: () -> str
        lines = []
        mode = "DRY RUN (nothing written)" if self.dry_run else "APPLY"
        lines.append("Obsidian import - {0}".format(mode))
        lines.append("=" * 78)
        lines.append("vault : {0}".format(self.vault_path))
        lines.append("vault_id: {0}".format(self.vault_id))
        lines.append("")
        for document in self.documents:
            lines.append("[{0}] {1}".format(document.status.upper(), document.rel_path))
            if document.status == "unchanged":
                lines.append("    (unchanged since last import - 0 events)")
                continue
            if document.status == "empty":
                lines.append("    (no ingestible content)")
                continue
            for action in document.actions:
                lines.append(
                    "    - {0:<10} scope={1:<26} {2}".format(
                        action.route, action.scope, action.reason
                    )
                )
                lines.append(
                    "      {0:<12} {1}".format(action.action, action.preview)
                )
                if action.skipped_reason:
                    lines.append("      note: {0}".format(action.skipped_reason))
                for proposal in action.proposed:
                    lines.append(
                        "      -> {0} {1}".format(
                            "ACCEPT" if proposal.get("accepted") else "REJECT",
                            proposal.get("content") or proposal.get("reason") or "",
                        )
                    )
        lines.append("")
        lines.append("-" * 78)
        for key in sorted(self.counts):
            lines.append("{0:<28} {1}".format(key, self.counts[key]))
        return "\n".join(lines)


class ObsidianImporter(object):
    """Idempotent, read-only-against-the-vault importer."""

    def __init__(self, core, config, now=None):
        # type: (Any, BridgeConfig, Any) -> None
        self.core = core
        self.config = config
        self._now = now

    def _timestamp(self):
        # type: () -> str
        return self._now() if callable(self._now) else _utc_now()

    # -- idempotency ---------------------------------------------------

    def _seen_hashes(self, reference):
        # type: (str) -> Tuple[set, set]
        events = self.core.list_events(session_id=reference, limit=_EVENT_SCAN_LIMIT)
        documents = set()
        chunks = set()
        for event in events:
            metadata = getattr(event, "metadata", None) or {}
            if metadata.get("document_hash"):
                documents.add(metadata["document_hash"])
            if metadata.get("content_hash"):
                chunks.add(metadata["content_hash"])
        return documents, chunks

    # -- planning ------------------------------------------------------

    def plan(self):
        # type: () -> List[DocumentPlan]
        """Scan the vault and decide what *would* happen. Writes nothing."""
        plans = []
        for document in scan_vault(
            self.config.vault_path, include=self.config.include, exclude=self.config.exclude
        ):
            plans.append(self._plan_document(document))
        return plans

    def _plan_document(self, document):
        # type: (Any) -> DocumentPlan
        document_hash = hash_text(document.body)
        reference = external_ref(self.config.vault_id, document.rel_path)
        seen_documents, seen_chunks = self._seen_hashes(reference)

        if document_hash in seen_documents:
            return DocumentPlan(
                rel_path=document.rel_path,
                title=document.title,
                document_hash=document_hash,
                status="unchanged",
            )

        chunks = classify_module.chunks_for_document(document, self.config)
        if not chunks:
            return DocumentPlan(
                rel_path=document.rel_path,
                title=document.title,
                document_hash=document_hash,
                status="empty",
            )

        status = "changed" if seen_documents else "new"
        actions = []
        for chunk in chunks:
            route = classify_module.classify_chunk(chunk, document, self.config)
            action = "skip-seen" if chunk.content_hash in seen_chunks else "ingest"
            actions.append(
                ChunkAction(
                    document_rel=document.rel_path,
                    title=document.title,
                    heading_path=chunk.heading_path,
                    content_hash=chunk.content_hash,
                    route=route.route,
                    scope=route.scope,
                    reason=route.reason,
                    action=action,
                    preview=_preview(chunk.text),
                    payload=dict(route.payload),
                )
            )
        return DocumentPlan(
            rel_path=document.rel_path,
            title=document.title,
            document_hash=document_hash,
            status=status,
            actions=tuple(actions),
        )

    # -- execution -----------------------------------------------------

    def run(self, apply=False):
        # type: (bool) -> ImportReport
        """Plan, then optionally commit.

        With ``apply=False`` this is a pure read: ``propose()`` is used for the
        LLM routes and the typed routes are only parsed, never written.
        """
        documents = []
        counts = {
            "documents_scanned": 0,
            "documents_unchanged": 0,
            "documents_new": 0,
            "documents_changed": 0,
            "chunks_planned": 0,
            "chunks_skipped_seen": 0,
            "chunks_ingested": 0,
            "events_created": 0,
            "memories_created": 0,
            "preferences_set": 0,
            "project_states_set": 0,
            "candidates_accepted": 0,
            "candidates_rejected": 0,
        }

        for document in scan_vault(
            self.config.vault_path, include=self.config.include, exclude=self.config.exclude
        ):
            counts["documents_scanned"] += 1
            plan = self._plan_document(document)
            if plan.status == "unchanged":
                counts["documents_unchanged"] += 1
                documents.append(plan)
                continue
            if plan.status == "empty":
                documents.append(plan)
                continue
            counts["documents_new" if plan.status == "new" else "documents_changed"] += 1
            counts["chunks_planned"] += len(plan.actions)

            resolved = []
            for action in plan.actions:
                if action.action == "skip-seen":
                    counts["chunks_skipped_seen"] += 1
                    resolved.append(action)
                    continue
                if apply:
                    resolved.append(self._execute(document, action, counts))
                else:
                    resolved.append(self._simulate(document, action, counts))
            documents.append(
                DocumentPlan(
                    rel_path=plan.rel_path,
                    title=plan.title,
                    document_hash=plan.document_hash,
                    status=plan.status,
                    actions=tuple(resolved),
                )
            )

        return ImportReport(
            dry_run=not apply,
            vault_path=self.config.vault_path,
            vault_id=self.config.vault_id,
            documents=tuple(documents),
            counts=counts,
        )

    # -- dry run -------------------------------------------------------

    def _simulate(self, document, action, counts):
        # type: (Any, ChunkAction, Dict[str, int]) -> ChunkAction
        """Preview one chunk without writing anything."""
        if action.route in (classify_module.PREFERENCE, classify_module.PROJECT_STATE):
            # Typed writes have no side-effect-free equivalent, so the dry run
            # reports the parsed payload and stops there.
            return dataclass_replace(action)
        proposal = self.core.propose(
            self._chunk_text(document, action),
            scope=action.scope,
            context_limit=int(self.config.context_limit),
            # Declarations are committed through observe() (rules only), so the
            # preview must use the same path or the plan would over-report.
            include_llm=action.route != classify_module.DECLARATION,
        )
        proposed = []
        for outcome in proposal.outcomes:
            entry = outcome.as_dict()
            if entry.get("accepted"):
                counts["candidates_accepted"] += 1
            else:
                counts["candidates_rejected"] += 1
            proposed.append(entry)
        return dataclass_replace(
            action,
            proposed=tuple(proposed),
            accepted=len([entry for entry in proposed if entry.get("accepted")]),
            rejected=len([entry for entry in proposed if not entry.get("accepted")]),
            skipped_reason="; ".join(proposal.reasons) or None,
        )

    def _chunk_text(self, document, action):
        # type: (Any, ChunkAction) -> str
        """Recover the exact chunk text from a plan entry.

        Chunking is deterministic, so the text is reproduced rather than carried
        around in the plan. That keeps dry-run output small and guarantees the
        bytes written are the bytes that were previewed.
        """
        for chunk in classify_module.chunks_for_document(document, self.config):
            if chunk.content_hash == action.content_hash:
                return chunk.text
        return action.preview

    # -- apply ---------------------------------------------------------

    def _execute(self, document, action, counts):
        # type: (Any, ChunkAction, Dict[str, int]) -> ChunkAction
        text = self._chunk_text(document, action)
        metadata = self._metadata(document, action)
        reference = metadata["external_ref"]

        if action.route == classify_module.PREFERENCE:
            return self._write_preference(action, text, metadata, reference, counts)
        if action.route == classify_module.PROJECT_STATE:
            return self._write_project_state(action, text, metadata, reference, counts)
        return self._write_memory(action, text, metadata, reference, counts)

    def _write_memory(self, action, text, metadata, reference, counts):
        # type: (ChunkAction, str, Dict[str, Any], str, Dict[str, int]) -> ChunkAction
        """Write one memory via the route's Core path.

        An explicitly marked declaration goes through ``observe()`` - the
        deterministic rule path only. Running the LLM on top of a rule that has
        already extracted the sentence with confidence 1.0 does not add
        judgement, it adds a near-duplicate: measured on the demo vault, the
        model reworded ``我更喜欢...`` into ``用户更喜欢...``, which exact-content
        dedupe cannot catch. An explicit declaration is authoritative, so the
        model is not asked to second-guess it.

        Unmarked knowledge goes through ``learn()``, where the model's judgement
        is the entire point - and where NOOP is the expected answer most of the
        time.
        """
        common = {
            "session_id": reference,
            "scope": action.scope,
            "kind": "document",
            "metadata": metadata,
            "source": EVENT_SOURCE,
            "provenance": memory_provenance(metadata),
        }
        if action.route == classify_module.DECLARATION:
            observation = self.core.observe(text, **common)
        else:
            observation = self.core.learn(
                text, context_limit=int(self.config.context_limit), **common
            )
        counts["chunks_ingested"] += 1
        counts["events_created"] += 1
        counts["memories_created"] += len(observation.memories)
        counts["candidates_accepted"] += len(observation.accepted)
        counts["candidates_rejected"] += len(observation.rejected)
        return _replace(
            action,
            event_id=observation.event.id,
            item_ids=tuple(memory.id for memory in observation.memories),
            accepted=len(observation.accepted),
            rejected=len(observation.rejected),
            proposed=tuple(outcome.as_dict() for outcome in observation.outcomes),
            skipped_reason=observation.skipped_reason,
        )

    def _write_preference(self, action, text, metadata, reference, counts):
        # type: (ChunkAction, str, Dict[str, Any], str, Dict[str, int]) -> ChunkAction
        event = self.core.record_event(
            text,
            session_id=reference,
            scope=action.scope,
            kind="document",
            metadata=metadata,
            source=EVENT_SOURCE,
        )
        preference = self.core.set_preference(
            action.payload["key"],
            action.payload["value"],
            scope=action.scope,
            source=memory_source(metadata, event.id),
        )
        counts["chunks_ingested"] += 1
        counts["events_created"] += 1
        counts["preferences_set"] += 1
        return _replace(action, event_id=event.id, item_ids=(preference.id,), accepted=1)

    def _write_project_state(self, action, text, metadata, reference, counts):
        # type: (ChunkAction, str, Dict[str, Any], str, Dict[str, int]) -> ChunkAction
        event = self.core.record_event(
            text,
            session_id=reference,
            scope=action.scope,
            kind="document",
            metadata=metadata,
            source=EVENT_SOURCE,
        )
        record = self.core.set_project_state(
            action.payload["project_id"],
            action.payload["state"],
            scope=action.scope,
            source=memory_source(metadata, event.id),
        )
        counts["chunks_ingested"] += 1
        counts["events_created"] += 1
        counts["project_states_set"] += 1
        return _replace(action, event_id=event.id, item_ids=(record.id,), accepted=1)

    # -- helpers -------------------------------------------------------

    def _metadata(self, document, action):
        # type: (Any, ChunkAction) -> Dict[str, Any]
        from memory_bridge.obsidian.provenance import event_metadata

        return event_metadata(
            self.config.vault_id,
            document.rel_path,
            document.title,
            hash_text(document.body),
            action.content_hash,
            heading_path=action.heading_path,
            line_start=None,
            line_end=None,
            mtime=document.mtime,
            ingested_at=self._timestamp(),
            route=action.route,
        )


def _preview(text, width=120):
    # type: (str, int) -> str
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def _replace(action, **changes):
    # type: (ChunkAction, Any) -> ChunkAction
    """Kept for readability at call sites; delegates to dataclasses.replace."""
    return dataclass_replace(action, **changes)


__all__ = ["ObsidianImporter", "ImportReport", "DocumentPlan", "ChunkAction"]
