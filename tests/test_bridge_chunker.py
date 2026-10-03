"""Deterministic chunking."""

from __future__ import annotations

import unittest

from memory_bridge.obsidian.chunker import Chunk, chunk_markdown

DECLARATIONS = ("记住：", "remember:")


class BasicChunkingTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(chunk_markdown(""), [])
        self.assertEqual(chunk_markdown("   \n\n  "), [])

    def test_short_text_is_dropped(self):
        self.assertEqual(chunk_markdown("too short"), [])

    def test_a_single_paragraph_becomes_one_chunk(self):
        text = "This is a long enough paragraph to survive the minimum length filter."
        chunks = chunk_markdown(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    def test_headings_split_sections(self):
        body = (
            "# Title\n\n"
            "First section body that is definitely long enough to be kept.\n\n"
            "## Sub\n\n"
            "Second section body that is also long enough to be kept.\n"
        )
        chunks = chunk_markdown(body)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].heading_path, ("Title",))
        self.assertEqual(chunks[1].heading_path, ("Title", "Sub"))

    def test_line_numbers_point_at_the_source(self):
        body = "# H\n\n" + "x" * 60 + "\n"
        chunk = chunk_markdown(body)[0]
        self.assertEqual(chunk.line_start, 3)
        self.assertEqual(chunk.line_end, 3)

    def test_a_hash_inside_a_code_fence_is_not_a_heading(self):
        body = (
            "# Real\n\n"
            "```python\n"
            "# not a heading\n"
            "x = 1\n"
            "```\n\n"
            "Body text that is long enough to be kept for sure.\n"
        )
        chunks = chunk_markdown(body)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading_path, ("Real",))

    def test_code_blocks_are_stripped_by_default(self):
        body = (
            "# H\n\n"
            "Body text that is long enough to be kept here.\n\n"
            "```\n" + "y = 2\n" * 20 + "```\n"
        )
        chunk = chunk_markdown(body)[0]
        self.assertNotIn("y = 2", chunk.text)

    def test_code_blocks_can_be_kept(self):
        body = (
            "# H\n\n"
            "Body text that is long enough to be kept here.\n\n"
            "```\ny = 2\n```\n"
        )
        chunk = chunk_markdown(body, include_code_blocks=True)[0]
        self.assertIn("y = 2", chunk.text)

    def test_long_sections_split_at_paragraphs(self):
        paragraph = "A sentence that is reasonably long. " * 4
        body = "\n\n".join([paragraph] * 6)
        chunks = chunk_markdown(body, max_chars=200)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 220)


class DeclarationBoundaryTests(unittest.TestCase):
    """Explicit declarations must become their own atomic chunk."""

    def test_declaration_starts_a_new_chunk(self):
        body = (
            "Some ordinary context that is long enough to be kept as a note.\n\n"
            "记住：我更喜欢小而完整的闭环。\n\n"
            "More ordinary text that is long enough to be kept as a note.\n"
        )
        chunks = chunk_markdown(body, boundary_prefixes=DECLARATIONS)
        declarations = [c for c in chunks if c.text.startswith("记住：")]
        self.assertEqual(len(declarations), 1)
        self.assertEqual(declarations[0].text, "记住：我更喜欢小而完整的闭环。")

    def test_declaration_is_its_own_paragraph_only(self):
        body = (
            "记住：第一条声明。\n\n"
            "紧接着的普通段落，不应该被并进声明里，因为它讲的是另一件事情。\n"
        )
        chunks = chunk_markdown(body, min_chars=10, boundary_prefixes=DECLARATIONS)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "记住：第一条声明。")
        self.assertIn("紧接着", chunks[1].text)

    def test_short_declaration_bypasses_the_minimum_length(self):
        chunks = chunk_markdown(
            "记住：短句。", min_chars=500, boundary_prefixes=DECLARATIONS
        )
        self.assertEqual(len(chunks), 1)

    def test_without_boundaries_the_declaration_stays_embedded(self):
        body = (
            "Some ordinary context that is long enough to be kept as a note.\n\n"
            "记住：我更喜欢小而完整的闭环。\n"
        )
        chunks = chunk_markdown(body)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].text.startswith("Some ordinary"))


class DeterminismTests(unittest.TestCase):
    def test_hashes_and_boundaries_are_stable(self):
        body = (
            "# H\n\n"
            "First paragraph that is long enough to be kept.\n\n"
            "记住：一条声明。\n\n"
            "Second paragraph that is long enough to be kept.\n"
        )
        first = chunk_markdown(body, boundary_prefixes=DECLARATIONS)
        second = chunk_markdown(body, boundary_prefixes=DECLARATIONS)
        self.assertEqual(
            [(c.ordinal, c.heading_path, c.text, c.content_hash) for c in first],
            [(c.ordinal, c.heading_path, c.text, c.content_hash) for c in second],
        )

    def test_ordinals_are_dense_and_ordered(self):
        body = "\n\n".join(["Paragraph number {0} is long enough to be kept.".format(i) for i in range(6)])
        chunks = chunk_markdown(body, max_chars=120)
        self.assertEqual([c.ordinal for c in chunks], list(range(len(chunks))))

    def test_line_ending_changes_do_not_change_the_hash(self):
        body = "A paragraph that is long enough to be kept for hashing."
        self.assertEqual(
            chunk_markdown(body)[0].content_hash,
            chunk_markdown(body.replace("\n", "\r\n"))[0].content_hash,
        )

    def test_trailing_whitespace_does_not_change_the_hash(self):
        body = "A paragraph that is long enough to be kept.   \n"
        self.assertEqual(
            chunk_markdown(body)[0].content_hash,
            chunk_markdown("A paragraph that is long enough to be kept.\n")[0].content_hash,
        )

    def test_real_edits_do_change_the_hash(self):
        base = "A paragraph that is long enough to be kept for hashing."
        self.assertNotEqual(
            chunk_markdown(base)[0].content_hash,
            chunk_markdown("A paragraph that is long enough to be kept for hashing!")[0].content_hash,
        )


if __name__ == "__main__":
    unittest.main()
