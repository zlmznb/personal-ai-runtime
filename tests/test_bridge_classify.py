"""Route classification: five kinds of content, deterministic signals first."""

from __future__ import annotations

import unittest

from memory_bridge.config import BridgeConfig
from memory_bridge.obsidian import classify
from memory_bridge.obsidian.vault import VaultDocument

DOC = VaultDocument(
    abs_path="/vault/Notes/Test.md",
    rel_path="Notes/Test.md",
    title="Test",
    frontmatter={},
    body="",
    mtime="2026-10-03T00:00:00.000Z",
    size=10,
)


def document(rel_path="Notes/Test.md", frontmatter=None, body=""):
    # type: (str, object, str) -> VaultDocument
    return VaultDocument(
        abs_path="/vault/" + rel_path,
        rel_path=rel_path,
        title="Test",
        frontmatter=frontmatter or {},
        body=body,
        mtime="2026-10-03T00:00:00.000Z",
        size=len(body),
    )


def config(**overrides):
    # type: (object) -> BridgeConfig
    data = {"vault_path": "/vault", "vault_id": "v"}
    data.update(overrides)
    return BridgeConfig(**data)


def chunk_of(text):
    # type: (str) -> classify.Chunk
    from memory_bridge.obsidian.chunker import chunk_markdown

    chunks = chunk_markdown(text)
    if chunks:
        return chunks[0]
    from memory_bridge.obsidian.chunker import Chunk
    from memory_bridge.obsidian.provenance import hash_text

    return Chunk(0, (), text, 1, 1, hash_text(text))


class ScopeResolutionTests(unittest.TestCase):
    def test_frontmatter_scope_wins(self):
        scope, reason = classify.resolve_scope(
            document(frontmatter={"scope": "project:explicit"}), config()
        )
        self.assertEqual(scope, "project:explicit")
        self.assertIn("frontmatter", reason)

    def test_frontmatter_persona(self):
        scope, _ = classify.resolve_scope(document(frontmatter={"persona": "aria"}), config())
        self.assertEqual(scope, "persona:aria")

    def test_frontmatter_project(self):
        scope, _ = classify.resolve_scope(document(frontmatter={"project": "memory"}), config())
        self.assertEqual(scope, "project:memory")

    def test_path_mapping(self):
        scope, reason = classify.resolve_scope(
            document(rel_path="Projects/persona-edgeaiot/State.md"), config()
        )
        self.assertEqual(scope, "project:persona-edgeaiot")
        self.assertEqual(reason, "path mapping")

    def test_path_mapping_for_persona_folder(self):
        scope, _ = classify.resolve_scope(document(rel_path="Personas/aria/Notes.md"), config())
        self.assertEqual(scope, "persona:aria")

    def test_default_scope(self):
        scope, reason = classify.resolve_scope(document(rel_path="Inbox/Note.md"), config())
        self.assertEqual(scope, "global")
        self.assertEqual(reason, "default")

    def test_scope_precedence_frontmatter_beats_path(self):
        scope, _ = classify.resolve_scope(
            document(rel_path="Personas/aria/Notes.md", frontmatter={"persona": "bruno"}),
            config(),
        )
        self.assertEqual(scope, "persona:bruno")


class RouteTests(unittest.TestCase):
    def test_plain_note_is_knowledge(self):
        route = classify.classify_chunk(
            chunk_of("Just an ordinary note about architecture and design tradeoffs."),
            document(),
            config(),
        )
        self.assertEqual(route.route, classify.KNOWLEDGE)
        self.assertEqual(route.scope, "global")

    def test_declaration_marker(self):
        route = classify.classify_chunk(
            chunk_of("记住：我更喜欢小而完整的闭环。"), document(), config()
        )
        self.assertEqual(route.route, classify.DECLARATION)

    def test_english_declaration_marker(self):
        route = classify.classify_chunk(
            chunk_of("remember: the user prefers small complete loops."), document(), config()
        )
        self.assertEqual(route.route, classify.DECLARATION)

    def test_obsidian_callout_marker(self):
        route = classify.classify_chunk(
            chunk_of("> [!memory] The user prefers Obsidian for long notes."),
            document(),
            config(),
        )
        self.assertEqual(route.route, classify.DECLARATION)

    def test_inline_preference(self):
        route = classify.classify_chunk(
            chunk_of("偏好：theme = dark"), document(), config()
        )
        self.assertEqual(route.route, classify.PREFERENCE)
        self.assertEqual(route.payload["key"], "theme")
        self.assertEqual(route.payload["value"], "dark")

    def test_inline_preference_with_json_value(self):
        route = classify.classify_chunk(
            chunk_of('偏好：layout = {"columns": 3}'), document(), config()
        )
        self.assertEqual(route.payload["value"], {"columns": 3})

    def test_inline_project_state(self):
        route = classify.classify_chunk(
            chunk_of('项目状态：memory = {"phase": "v0.3"}'), document(), config()
        )
        self.assertEqual(route.route, classify.PROJECT_STATE)
        self.assertEqual(route.payload["project_id"], "memory")
        self.assertEqual(route.payload["state"], {"phase": "v0.3"})

    def test_frontmatter_preference(self):
        doc = document(
            frontmatter={"type": "preference", "key": "editor", "value": "obsidian"}
        )
        route = classify.classify_chunk(chunk_of("body text long enough to exist"), doc, config())
        self.assertEqual(route.route, classify.PREFERENCE)
        self.assertEqual(route.payload, {"key": "editor", "value": "obsidian"})

    def test_frontmatter_preference_without_key_is_knowledge(self):
        doc = document(frontmatter={"type": "preference"})
        route = classify.classify_chunk(chunk_of("body text long enough to exist"), doc, config())
        self.assertEqual(route.route, classify.KNOWLEDGE)

    def test_frontmatter_project_state_uses_remaining_keys(self):
        doc = document(
            frontmatter={
                "type": "project-state",
                "project": "memory",
                "phase": "v0.3",
                "focus": "bridge",
            }
        )
        route = classify.classify_chunk(chunk_of("body text long enough to exist"), doc, config())
        self.assertEqual(route.route, classify.PROJECT_STATE)
        self.assertEqual(route.payload["state"], {"phase": "v0.3", "focus": "bridge"})

    def test_frontmatter_project_state_uses_inline_json_state(self):
        doc = document(
            frontmatter={
                "type": "project-state",
                "project": "memory",
                "state": '{"phase": "v0.3"}',
            }
        )
        route = classify.classify_chunk(chunk_of("body text long enough to exist"), doc, config())
        self.assertEqual(route.payload["state"], {"phase": "v0.3"})

    def test_project_state_without_project_id_falls_back_to_knowledge(self):
        doc = document(frontmatter={"type": "project-state", "phase": "x"})
        route = classify.classify_chunk(chunk_of("body text long enough to exist"), doc, config())
        self.assertEqual(route.route, classify.KNOWLEDGE)

    def test_declared_type_detection(self):
        self.assertEqual(classify.declared_type(document(frontmatter={"type": "preference"})), classify.PREFERENCE)
        self.assertIsNone(classify.declared_type(document()))


class ChunkingPolicyTests(unittest.TestCase):
    def test_typed_document_is_not_split(self):
        body = "# A\n\n" + "Content sentence long enough. " * 20 + "\n\n## B\n\nMore content here."
        doc = document(frontmatter={"type": "preference", "key": "k", "value": "v"}, body=body)
        chunks = classify.chunks_for_document(doc, config())
        self.assertEqual(len(chunks), 1)

    def test_untyped_document_is_chunked(self):
        body = "# A\n\n" + "Content sentence that is long enough to keep. " * 3
        chunks = classify.chunks_for_document(document(body=body), config())
        self.assertGreaterEqual(len(chunks), 1)

    def test_declarations_become_their_own_chunks(self):
        body = (
            "Ordinary context that is long enough to be kept as a note.\n\n"
            "记住：一条明确的声明。\n\n"
            "Another ordinary paragraph that is long enough to be kept.\n"
        )
        chunks = classify.chunks_for_document(document(body=body), config())
        self.assertTrue(any(chunk.text.startswith("记住：") for chunk in chunks))


class ValueParsingTests(unittest.TestCase):
    def test_json_first(self):
        self.assertEqual(classify.parse_value("3"), 3)
        self.assertTrue(classify.parse_value("true"))
        self.assertEqual(classify.parse_value('"dark"'), "dark")

    def test_plain_string_fallback(self):
        self.assertEqual(classify.parse_value("dark"), "dark")


if __name__ == "__main__":
    unittest.main()
