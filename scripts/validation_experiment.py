"""Final core-capability acceptance experiment.

Proves, with captured evidence, that:

    Model A -> Memory Core -> SQLite
    switch
    Model B -> the same Memory Core -> the same SQLite
    still yields the same long-term user information.

It composes the public API only. No core code is modified and nothing here is a
feature of Memory Core: it is an experiment harness, like ``benchmark.py``.

Four experiments:

  1. Real memory lifecycle - Model A writes memories through Learn
     (utterance -> candidate -> validate -> accept -> memory -> SQLite).
  2. Model swap - Model A is closed and a different ChatProvider is used
     against the same database.
  3. Re-read - Model B answers a question, and the exact prompt it received is
     captured, proving the answer came from retrieval rather than from the model.
  4. Vector index destroyed - the derived index is dropped, the memories are
     read again, the index is rebuilt from SQLite, and Model B re-reads them.

Usage::

    py -3 scripts/validation_experiment.py --out docs/model_swap_validation.json
    py -3 scripts/validation_experiment.py --model-a-provider fake   # fully offline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.api import MemoryCore  # noqa: E402
from memory_core.domain.dto import ChatMessage  # noqa: E402
from memory_core.providers.base import ChatProvider  # noqa: E402
from memory_core.providers.fake import FakeProvider  # noqa: E402
from memory_core.providers.ollama import OllamaProvider  # noqa: E402
from memory_core.providers.ollama_embedding import OllamaEmbeddingProvider  # noqa: E402
from memory_core.providers.registry import resolve_cli_provider  # noqa: E402

#: The three facts the user provides, verbatim from the acceptance brief.
USER_TURNS = (
    "我正在开发 Persona-EdgeAIoT，这是一个长期项目。",
    "我的长期研究方向更偏向感知层 AIoT，而不是单纯训练大模型。",
    "我希望 Personal AI 的长期记忆与底层模型解耦。",
)

#: A single onboarding message containing all three, as a user would actually write it.
ONBOARDING_TURN = "".join(USER_TURNS)

QUESTION = "你还记得我正在做什么项目，以及我的长期研究方向吗？"


def digest(path):
    # type: (str) -> str
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class RecordingChatProvider(ChatProvider):
    """Transparent wrapper that records the exact prompts sent to a model.

    This is how experiment 3 proves the answer came from retrieval: the captured
    prompt shows precisely which memories were handed to the model.
    """

    def __init__(self, inner, label):
        # type: (ChatProvider, str) -> None
        self.inner = inner
        self.label = label
        self.calls = []  # type: list

    @property
    def name(self):
        # type: () -> str
        return self.inner.name

    @property
    def model(self):
        # type: () -> str
        return self.inner.model

    def capabilities(self):
        # type: () -> object
        return self.inner.capabilities()

    def chat(self, messages, **options):
        # type: (object, object) -> object
        self.calls.append(
            {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
            }
        )
        return self.inner.chat(messages, **options)


def context_only_reply(messages):
    # type: (object) -> str
    """A deliberately knowledge-free Model B.

    It has no world knowledge and no memory of its own. The only thing it can
    do is repeat what the retrieval layer put into its prompt. That makes it a
    stronger witness than a clever model: if the project name appears in its
    answer, that fact provably travelled SQLite -> retrieval -> prompt.
    """
    system = messages[0].content if messages else ""
    marker = "Memory context:"
    context = system.split(marker, 1)[1].strip() if marker in system else ""
    if not context or context.startswith("(no relevant memories)"):
        return "I have no long-term memory of that."
    facts = [line[2:].strip() for line in context.splitlines() if line.strip().startswith("- ")]
    if not facts:
        return "I have no long-term memory of that."
    return "From my long-term memory I can see:\n" + "\n".join("  * " + f for f in facts)


def build_model_a(args):
    # type: (argparse.Namespace) -> ChatProvider
    if args.model_a_provider == "ollama":
        # NOTE: the chat provider speaks the OpenAI-compatible endpoint, so the
        # base URL must include /v1. The embedding provider uses the native
        # endpoint and must NOT include it. They are separate URLs on purpose.
        return OllamaProvider(
            model=args.model_a_model, base_url=args.ollama_chat_url, timeout=args.timeout
        )
    return resolve_cli_provider({}, args.model_a_provider, args.model_a_model)


def build_model_b(args):
    # type: (argparse.Namespace) -> ChatProvider
    if args.model_b_provider == "context-only-fake":
        return FakeProvider(model="model-b", prefix="[B]", reply=context_only_reply)
    return resolve_cli_provider({}, args.model_b_provider, args.model_b_model)


def independent_read(db_path):
    # type: (str) -> dict
    """Read the database with the sqlite3 module directly.

    No Memory Core code is involved, so this is evidence about the *file*, not
    about the application's view of it.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        tables = sorted(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        )
        memories = [
            dict(row)
            for row in connection.execute(
                "SELECT id, kind, scope, subject, content, confidence, salience, "
                "status, generated_by, source, created_at FROM memories ORDER BY id"
            )
        ]
        events = [
            dict(row)
            for row in connection.execute(
                "SELECT id, role, content, created_at FROM events ORDER BY id"
            )
        ]
        embeddings = {}
        if "embeddings" in tables:
            embeddings = {
                str(row["model_id"]): int(row["n"])
                for row in connection.execute(
                    "SELECT model_id, COUNT(*) AS n FROM embeddings GROUP BY model_id"
                )
            }
        return {
            "tables": tables,
            "memory_count": len(memories),
            "memories": memories,
            "event_count": len(events),
            "events": events,
            "vector_rows_by_model": embeddings,
            "vector_rows_total": sum(embeddings.values()),
        }
    finally:
        connection.close()


def recall_view(core, query):
    # type: (MemoryCore, str) -> dict
    view = {}
    for mode in ("keyword", "semantic", "hybrid"):
        hits = core.recall(query, mode=mode, limit=10)
        view[mode] = [
            {"id": hit.id, "item_type": hit.item_type, "score": round(hit.score, 8),
             "matched_on": hit.matched_on, "content": getattr(hit.item, "content", None)}
            for hit in hits
        ]
    return view


def run(args):
    # type: (argparse.Namespace) -> dict
    workdir = args.workdir or tempfile.mkdtemp(prefix="memory-core-validation-")
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    db_path = args.db or os.path.join(workdir, "source_of_truth.sqlite")

    evidence = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
            "database": db_path,
            "model_a": {
                "provider": args.model_a_provider,
                "model": args.model_a_model,
            },
            "model_b": {
                "provider": args.model_b_provider,
                "model": args.model_b_model,
                "label": args.model_b_label,
            },
            "embedding": {
                "provider": args.embedding_provider,
                "model": args.embedding_model,
            },
        },
        "user_turns": list(USER_TURNS),
        "question": QUESTION,
    }

    # ---------------------------------------------------------------- E1
    embedding = None
    if args.embedding_provider == "ollama-embedding":
        embedding = OllamaEmbeddingProvider(
            model=args.embedding_model, base_url=args.ollama_root, timeout=args.timeout
        )

    model_a = build_model_a(args)
    core = MemoryCore.open(db_path=db_path, provider=model_a, embedding_provider=embedding)
    e1 = evidence["experiment_1_model_a"] = {
        "provider_identity": core.provider_info(),
        "model_a_class": type(model_a).__name__,
        "utterance": ONBOARDING_TURN,
        "attempts": [],
    }

    # A real 8B model is stochastic: a single turn can occasionally come back
    # schema-invalid. Retrying is reported, not hidden, and dedupe means a retry
    # can never double-write the same fact.
    written = []
    for attempt in range(1, int(args.attempts) + 1):
        started = time.time()
        observation = core.learn(ONBOARDING_TURN, role="user", session_id="onboarding")
        elapsed = round(time.time() - started, 2)
        e1["attempts"].append(
            {
                "attempt": attempt,
                "elapsed_seconds": elapsed,
                "event_id": observation.event.id,
                "skipped_reason": observation.skipped_reason,
                "accepted": len(observation.accepted),
                "rejected": len(observation.rejected),
                "outcomes": [
                    {
                        "accepted": outcome.accepted,
                        "operation": outcome.candidate.operation,
                        "target": outcome.candidate.target,
                        "origin": outcome.candidate.origin,
                        "rule": outcome.candidate.rule,
                        "confidence": outcome.candidate.confidence,
                        "content": outcome.candidate.payload.get("content"),
                        "kind": outcome.candidate.payload.get("kind"),
                        "salience": outcome.candidate.payload.get("salience"),
                        "reason": outcome.reason,
                        "item_id": outcome.item_id,
                    }
                    for outcome in observation.outcomes
                ],
            }
        )
        if observation.memories:
            written = list(observation.memories)
            break

    e1["successful_attempt"] = len(e1["attempts"])
    e1["written_memory_ids"] = [memory.id for memory in written]
    e1["written_memories"] = [
        {
            "id": memory.id,
            "kind": memory.kind,
            "content": memory.content,
            "confidence": memory.confidence,
            "salience": memory.salience,
            "generated_by": memory.generated_by,
            "source": memory.source,
            "created_at": memory.created_at,
            "tags": list(memory.tags),
        }
        for memory in written
    ]

    index_result = None
    if args.build_index:
        started = time.time()
        index_result = core.build_index()
        index_result["elapsed_seconds"] = round(time.time() - started, 2)
    e1["index_build"] = index_result

    before_close_digest = digest(db_path)
    core.close()

    evidence["experiment_1_sqlite"] = independent_read(db_path)

    # ---------------------------------------------------------------- E2
    # "Model A is stopped" means: the MemoryCore that used it is closed, and the
    # next reader is a different provider object entirely. There is no model
    # session state anywhere for the data to depend on.
    swap = evidence.setdefault("experiment_2_model_swap", {})
    swap["model_a_class"] = type(model_a).__name__
    swap["model_a_identity"] = "{0}:{1}".format(model_a.name, model_a.model)
    swap["model_a_core_closed"] = True
    swap["database_digest_after_model_a_closed"] = before_close_digest
    swap["database_unchanged_by_closing_model_a"] = digest(db_path) == before_close_digest
    swap["storage_layer_has_no_provider_reference"] = not hasattr(core.store, "provider")

    model_b = build_model_b(args)
    swap["model_b_class"] = type(model_b).__name__
    swap["model_b_identity"] = "{0}:{1}".format(model_b.name, model_b.model)
    swap["different_provider_implementation"] = type(model_a).__name__ != type(model_b).__name__

    core_b = MemoryCore.open(db_path=db_path, provider=model_b, embedding_provider=embedding)
    swap["model_b_is_not_model_a"] = core_b.provider is not model_a
    swap["model_b_provider_info"] = core_b.provider_info()
    swap["memory_count_seen_by_model_b"] = core_b.store.count_memories()
    swap["memory_ids_seen_by_model_b"] = [
        memory.id for memory in core_b.list_memories(limit=1000)
    ]

    # ---------------------------------------------------------------- E3
    recorder = RecordingChatProvider(model_b, args.model_b_label)
    core_b.provider = recorder
    core_b.llm_extractor = None  # retrieval-only question; formation is not involved

    answer = core_b.ask(QUESTION, mode=args.mode)
    evidence["experiment_3_model_b"] = {
        "question": QUESTION,
        "retrieval_mode": args.mode,
        "context_ids": list(answer.context_ids),
        "degraded": answer.degraded,
        "provider": answer.provider,
        "model": answer.model,
        "answer": answer.text,
        "prompt_sent_to_model_b": recorder.calls[-1] if recorder.calls else None,
        "recall_view": recall_view(core_b, QUESTION),
    }

    # Ablation: the same question against a database with no memories.
    ablation_dir = os.path.join(workdir, "ablation-empty")
    os.makedirs(ablation_dir, exist_ok=True)
    empty_core = MemoryCore.open(
        db_path=os.path.join(ablation_dir, "empty.sqlite"), provider=model_b
    )
    try:
        empty_answer = empty_core.ask(QUESTION, mode=args.mode)
        evidence["experiment_3_model_b"]["ablation_without_memory"] = {
            "memory_count": empty_core.store.count_memories(),
            "context_ids": list(empty_answer.context_ids),
            "answer": empty_answer.text,
        }
    finally:
        empty_core.close()

    # ---------------------------------------------------------------- E4
    e4 = evidence.setdefault("experiment_4_vector_index", {})
    e4["before_drop"] = {
        "vector_rows_by_model": independent_read(db_path)["vector_rows_by_model"],
        "semantic_hits": len(recall_view(core_b, QUESTION)["semantic"]),
    }

    dropped = core_b.drop_index(all_models=True)
    after_drop_read = independent_read(db_path)
    e4["dropped_vector_rows"] = dropped
    e4["after_drop"] = {
        "vector_rows_total": after_drop_read["vector_rows_total"],
        "memory_count": after_drop_read["memory_count"],
        "memories_still_present": [
            {"id": m["id"], "content": m["content"]} for m in after_drop_read["memories"]
        ],
        "recall": recall_view(core_b, QUESTION),
    }

    rebuilt = core_b.rebuild_index()
    after_rebuild_read = independent_read(db_path)
    e4["rebuild_result"] = rebuilt
    e4["after_rebuild"] = {
        "vector_rows_by_model": after_rebuild_read["vector_rows_by_model"],
        "memory_count": after_rebuild_read["memory_count"],
        "recall": recall_view(core_b, QUESTION),
    }

    recorder.calls.clear()
    answer_after = core_b.ask(QUESTION, mode=args.mode)
    e4["model_b_after_rebuild"] = {
        "context_ids": list(answer_after.context_ids),
        "degraded": answer_after.degraded,
        "answer": answer_after.text,
        "prompt_sent_to_model_b": recorder.calls[-1] if recorder.calls else None,
    }

    final_read = independent_read(db_path)
    evidence["final_state"] = {
        "database": db_path,
        "database_digest": digest(db_path),
        "memory_count": final_read["memory_count"],
        "memories": [
            {"id": m["id"], "content": m["content"]} for m in final_read["memories"]
        ],
        "vector_rows_by_model": final_read["vector_rows_by_model"],
    }

    core_b.close()

    # ------------------------------------------------------------ verdict
    facts = [m["content"] for m in final_read["memories"]]
    project_memory_present = any("Persona-EdgeAIoT" in fact for fact in facts)
    research_memory_present = any("感知层" in fact for fact in facts)
    decoupling_memory_present = any("解耦" in fact for fact in facts)

    answer_text = evidence["experiment_3_model_b"]["answer"]
    answer_after_text = e4["model_b_after_rebuild"]["answer"]

    evidence["verdict"] = {
        "memories_written_by_model_a": len(evidence["experiment_1_model_a"]["written_memories"]),
        "formation_attempts_used": evidence["experiment_1_model_a"]["successful_attempt"],
        "last_skipped_reason": (
            evidence["experiment_1_model_a"]["attempts"][-1]["skipped_reason"]
            if evidence["experiment_1_model_a"]["attempts"]
            else None
        ),
        "project_memory_present_after_everything": project_memory_present,
        "research_direction_memory_present": research_memory_present,
        "decoupling_memory_present": decoupling_memory_present,
        "model_b_answer_cites_project": "Persona-EdgeAIoT" in answer_text,
        "model_b_answer_cites_research_direction": "感知层" in answer_text,
        "model_b_answer_after_rebuild_cites_project": "Persona-EdgeAIoT" in answer_after_text,
        "model_b_cannot_answer_without_memory": "Persona-EdgeAIoT"
        not in evidence["experiment_3_model_b"]["ablation_without_memory"]["answer"],
        "vector_index_was_destroyed": e4["after_drop"]["vector_rows_total"] == 0,
        "memories_survived_index_destruction": after_drop_read["memory_count"]
        == final_read["memory_count"],
        "vector_index_rebuilt_from_sqlite": after_rebuild_read["vector_rows_total"] > 0,
        "same_memory_ids_before_and_after": sorted(
            m["id"] for m in evidence["experiment_1_sqlite"]["memories"]
        )
        == sorted(m["id"] for m in final_read["memories"]),
    }
    evidence["verdict"]["PASS"] = all(
        [
            evidence["verdict"]["memories_written_by_model_a"] > 0,
            project_memory_present,
            research_memory_present,
            evidence["verdict"]["model_b_answer_cites_project"],
            evidence["verdict"]["model_b_answer_after_rebuild_cites_project"],
            evidence["verdict"]["model_b_cannot_answer_without_memory"],
            evidence["verdict"]["vector_index_was_destroyed"],
            evidence["verdict"]["memories_survived_index_destruction"],
            evidence["verdict"]["vector_index_rebuilt_from_sqlite"],
            evidence["verdict"]["same_memory_ids_before_and_after"],
        ]
    )
    return evidence


def main(argv=None):
    # type: (object) -> int
    parser = argparse.ArgumentParser(description="Memory Core model-swap acceptance experiment")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--model-a-provider", default="ollama")
    parser.add_argument("--model-a-model", default="qwen3:8b")
    parser.add_argument("--model-b-provider", default="context-only-fake")
    parser.add_argument("--model-b-model", default="fake-1")
    parser.add_argument("--model-b-label", default="Model B (knowledge-free adapter)")
    parser.add_argument("--embedding-provider", default="ollama-embedding")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument(
        "--ollama-chat-url",
        default="http://127.0.0.1:11434/v1",
        help="OpenAI-compatible chat base URL (includes /v1)",
    )
    parser.add_argument(
        "--ollama-root",
        default="http://127.0.0.1:11434",
        help="native Ollama base URL for embeddings (no /v1)",
    )
    parser.add_argument("--attempts", type=int, default=3, help="max formation attempts")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--mode", default="hybrid", choices=["keyword", "semantic", "hybrid"])
    parser.add_argument("--no-index", dest="build_index", action="store_false", default=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--keep", action="store_true", help="keep the temporary work directory")
    args = parser.parse_args(argv)

    workdir = args.workdir or tempfile.mkdtemp(prefix="memory-core-validation-")
    args.workdir = workdir
    try:
        evidence = run(args)
    finally:
        if not args.keep and args.workdir == workdir and args.db is None:
            pass  # left in place intentionally: the report cites the database path

    payload = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        sys.stderr.write("evidence written to {0}\n".format(args.out))

    verdict = evidence["verdict"]
    sys.stdout.write(payload + "\n")
    sys.stdout.write("\nPASS: {0}\n".format(verdict["PASS"]))
    return 0 if verdict["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
