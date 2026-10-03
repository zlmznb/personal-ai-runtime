"""LLM-assisted memory formation.

Flow::

    event / utterance
      -> prompt (utterance + the existing memories the model is allowed to
         supersede)
      -> ChatProvider.complete
      -> strict schema validation (formation.candidate)
      -> Candidate objects (still untrusted)

What the model may do
---------------------
* Read the utterance and a bounded list of existing memories.
* Emit a JSON document proposing ADD / SUPERSEDE / NOOP.

What the model may **not** do
-----------------------------
* Touch the database, issue SQL, or hold any storage handle. It only ever
  receives a string and returns a string; the provider interface has no other
  capability.
* Reference an id it was not shown (enforced in :mod:`.candidate`).
* Decide that something is written (enforced in :mod:`.policy` and applied by
  ``api.MemoryCore``).

Failure behaviour: any provider error or schema violation yields an Extraction
with **zero candidates** and a reason. Nothing is written, and the raw event
still exists, so the failure is recoverable and the database is never polluted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from memory_core.domain.dto import ChatMessage
from memory_core.domain.errors import ProviderError
from memory_core.formation.base import Candidate, Extraction, Extractor
from memory_core.formation.candidate import (
    DEFAULT_LLM_CONFIDENCE,
    CandidateSchemaError,
    parse_llm_candidates,
)

SYSTEM_PROMPT = """You maintain the long-term memory of a personal AI assistant.

You receive one conversation turn and a list of existing long-term memories.
Decide what, if anything, should be remembered durably.

Reply with ONLY a JSON object. No prose, no markdown fences, no comments.

Schema:
{
  "memories": [
    {
      "operation": "ADD" | "UPDATE" | "SUPERSEDE" | "NOOP",
      "content": "one atomic statement, in the language of the turn",
      "kind": "semantic" | "episodic" | "identity",
      "subject": "user:self",
      "tags": ["tag"],
      "confidence": 0.0,
      "salience": 0.0,
      "reason": "why this is worth keeping",
      "supersedes_id": null
    }
  ]
}

Rules:
- ADD    a new durable fact about the user, their preferences, projects, tools
         or identity. One atomic statement per entry.
- UPDATE/SUPERSEDE replaces an existing memory when the turn contradicts or
         refines it. supersedes_id MUST be an id copied exactly from EXISTING
         MEMORIES. Never invent an id.
- NOOP   nothing in this turn deserves long-term storage. Emitting an empty
         "memories" list is equally acceptable.
- kind:  "semantic" for stable facts, "episodic" for something that happened,
         "identity" for who the assistant is.
- confidence reflects how certain you are that this is durable and correct.
  Use 0.9+ only for something the user stated explicitly.
- Do not store transient chit-chat, pleasantries, or anything already covered by
  an existing memory.
- Use exact keys from the schema. Do not add extra keys.

Writing "content" correctly is the most important part:
- Write the FACT ITSELF as a standalone statement, in the language of the turn.
  Good: "用户的名字是阿哲"  /  "用户主要使用 Python 和 Rust"
        "用户正在开发名为 Memory Core 的本地优先项目"
- NEVER describe the conversation or the user's action. These are all wrong:
  "用户明确提供了自己的姓名" / "用户说明了主要使用的编程语言"
  "用户主动提及了项目名称"
- Preserve concrete details (names, languages, tools, project names). Losing
  them makes the memory worthless.
- "reason" is a short separate explanation; "content" is never a reason.
"""


class LlmExtractor(Extractor):
    """Turn an utterance into candidate memories using a chat model."""

    name = "llm"

    def __init__(
        self,
        provider,                        # type: Any
        max_candidates=5,                # type: int
        system_prompt=None,              # type: Optional[str]
        temperature=0.0,                 # type: float
        default_confidence=DEFAULT_LLM_CONFIDENCE,  # type: float
        max_context=8,                   # type: int
    ):
        # type: (...) -> None
        if provider is None:
            raise ProviderError("LlmExtractor requires a chat provider")
        self.provider = provider
        self.max_candidates = int(max_candidates)
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.temperature = float(temperature)
        self.default_confidence = float(default_confidence)
        self.max_context = int(max_context)

    # -- prompt ------------------------------------------------------------

    def build_messages(self, text, context=None):
        # type: (str, Optional[Sequence[Any]]) -> List[ChatMessage]
        memories = list(context or [])[: self.max_context]
        if memories:
            lines = []
            for memory in memories:
                lines.append(
                    "- {0} | {1} | {2}".format(
                        getattr(memory, "id", "?"),
                        getattr(memory, "kind", "?"),
                        getattr(memory, "content", ""),
                    )
                )
            block = "EXISTING MEMORIES:\n" + "\n".join(lines)
        else:
            block = "EXISTING MEMORIES:\n(none)"
        return [
            ChatMessage(role="system", content=self.system_prompt + "\n\n" + block),
            ChatMessage(role="user", content=text),
        ]

    # -- extraction --------------------------------------------------------

    def extract(self, text, scope="global", context=None):
        # type: (str, str, Optional[Sequence[Any]]) -> Extraction
        if not isinstance(text, str) or not text.strip():
            return Extraction(reason="llm: empty utterance")

        memories = list(context or [])[: self.max_context]
        context_ids = [getattr(memory, "id", "") for memory in memories]

        options = {"temperature": self.temperature}
        try:
            capabilities = self.provider.capabilities()
            if getattr(capabilities, "supports_json_mode", False):
                options["json_mode"] = True
        except Exception:  # pragma: no cover - capability probing must not fail a run
            pass

        try:
            result = self.provider.chat(self.build_messages(text, memories), **options)
        except ProviderError as exc:
            return Extraction(reason="llm: provider unavailable ({0})".format(exc), trigger="llm")
        except Exception as exc:  # pragma: no cover - defensive: never crash formation
            return Extraction(
                reason="llm: unexpected provider failure ({0}: {1})".format(
                    type(exc).__name__, exc
                ),
                trigger="llm",
            )

        try:
            candidates = parse_llm_candidates(
                result.text,
                context_ids=context_ids,
                scope=scope,
                max_candidates=self.max_candidates,
                default_confidence=self.default_confidence,
            )
        except CandidateSchemaError as exc:
            # Strict validation failed: propose nothing at all rather than
            # guessing at a partially-parsed payload.
            return Extraction(
                reason="llm: response rejected by schema validation ({0})".format(exc),
                trigger="llm",
            )

        stamped = []
        for candidate in candidates:
            payload = dict(candidate.payload)
            source = dict(payload.get("source") or {})
            source["model"] = "{0}:{1}".format(
                getattr(self.provider, "name", "provider"),
                getattr(self.provider, "model", ""),
            )
            payload["source"] = source
            stamped.append(
                Candidate(
                    target=candidate.target,
                    payload=payload,
                    confidence=candidate.confidence,
                    rule=candidate.rule,
                    operation=candidate.operation,
                    supersedes=candidate.supersedes,
                    reason=candidate.reason,
                    origin="llm",
                )
            )
        return Extraction(candidates=tuple(stamped), trigger="llm")

    def describe(self):
        # type: () -> Dict[str, Any]
        return {
            "name": self.name,
            "provider": getattr(self.provider, "name", None),
            "model": getattr(self.provider, "model", None),
            "max_candidates": self.max_candidates,
            "max_context": self.max_context,
            "default_confidence": self.default_confidence,
        }


__all__ = ["LlmExtractor", "SYSTEM_PROMPT"]
