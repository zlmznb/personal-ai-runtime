"""EmbeddingProvider tests.

Chat and embedding are separate protocols on purpose: the DeepSeek API has no
embeddings endpoint, so in practice they come from different vendors. These
tests pin that separation down, and exercise both the offline fake and the real
Ollama provider without needing a network.
"""

from __future__ import annotations

import unittest

from memory_core.domain.errors import ConfigurationError, ProviderError
from memory_core.providers import ollama_embedding, registry
from memory_core.providers.base import (
    EmbeddingCapabilities,
    EmbeddingProvider,
    normalize_embedding_texts,
)
from memory_core.providers.fake import FakeProvider
from memory_core.providers.fake_embedding import FakeEmbeddingProvider, feature_list
from memory_core.providers.ollama_embedding import OllamaEmbeddingProvider
from memory_core.providers.ollama import OllamaProvider


def cosine(left, right):
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class BaseProtocolTests(unittest.TestCase):
    def test_embed_is_abstract(self):
        with self.assertRaises(NotImplementedError):
            EmbeddingProvider().embed(["x"])

    def test_identifier_is_name_and_model(self):
        provider = EmbeddingProvider()
        provider.name = "p"
        provider.model = "m"
        self.assertEqual(provider.identifier(), "p:m")

    def test_capabilities_serialise(self):
        payload = EmbeddingCapabilities(dim=8, max_batch=4).as_dict()
        self.assertEqual(payload, {"dim": 8, "max_batch": 4, "supports_batching": True})

    def test_normalize_rejects_a_bare_string(self):
        with self.assertRaises(ProviderError):
            normalize_embedding_texts("hello")

    def test_normalize_rejects_non_strings(self):
        with self.assertRaises(ProviderError):
            normalize_embedding_texts(["ok", 42])

    def test_normalize_accepts_empty(self):
        self.assertEqual(normalize_embedding_texts([]), [])

    def test_chat_provider_has_no_embed_method(self):
        # The separation is structural, not conventional.
        self.assertFalse(hasattr(FakeProvider(), "embed"))
        self.assertFalse(hasattr(OllamaProvider(), "embed"))


class FakeEmbeddingTests(unittest.TestCase):
    def test_dimension_and_shape(self):
        provider = FakeEmbeddingProvider(dim=32)
        vectors = provider.embed(["a", "b", "c"])
        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(vectors[0]), 32)
        self.assertEqual(provider.capabilities().dim, 32)

    def test_is_deterministic_across_instances(self):
        first = FakeEmbeddingProvider(dim=16).embed(["用户偏好本地优先"])[0]
        second = FakeEmbeddingProvider(dim=16).embed(["用户偏好本地优先"])[0]
        self.assertEqual(first, second)

    def test_vectors_are_unit_length(self):
        vector = FakeEmbeddingProvider(dim=32).embed(["a sentence"])[0]
        self.assertAlmostEqual(sum(v * v for v in vector) ** 0.5, 1.0, places=6)

    def test_related_texts_are_closer_than_unrelated(self):
        provider = FakeEmbeddingProvider(dim=256)
        related = cosine(*provider.embed(["用户偏好本地优先", "本地部署优先，注重隐私"]))
        unrelated = cosine(*provider.embed(["用户偏好本地优先", "今天天气很好适合出门"]))
        self.assertGreater(related, unrelated)

    def test_salt_produces_a_different_model(self):
        base = FakeEmbeddingProvider(dim=32).embed(["same text"])[0]
        salted = FakeEmbeddingProvider(dim=32, salt="v2").embed(["same text"])[0]
        self.assertNotEqual(base, salted)

    def test_empty_text_is_a_zero_vector_not_a_crash(self):
        vector = FakeEmbeddingProvider(dim=8).embed([""])[0]
        self.assertEqual(len(vector), 8)
        self.assertEqual(sum(vector), 0.0)

    def test_embed_one_helper(self):
        self.assertEqual(len(FakeEmbeddingProvider(dim=8).embed_one("x")), 8)

    def test_failure_is_a_provider_error(self):
        provider = FakeEmbeddingProvider(fail_with="embedding service down")
        with self.assertRaises(ProviderError):
            provider.embed(["x"])

    def test_call_counter(self):
        provider = FakeEmbeddingProvider(dim=8)
        provider.embed(["a", "b"])
        self.assertEqual(provider.calls, 2)

    def test_rejects_non_positive_dim(self):
        with self.assertRaises(ProviderError):
            FakeEmbeddingProvider(dim=0)

    def test_identifier_includes_the_model_name(self):
        self.assertEqual(
            FakeEmbeddingProvider(model="fake-embed", dim=64).identifier(),
            "fake-embedding:fake-embed",
        )

    def test_describe(self):
        described = FakeEmbeddingProvider(model="m", dim=16).describe()
        self.assertEqual(described["identifier"], "fake-embedding:m")
        self.assertEqual(described["capabilities"]["dim"], 16)

    def test_features_include_words_and_cjk_characters(self):
        features = feature_list("Python 本地")
        self.assertIn("w:python", features)
        self.assertIn("c:本", features)


class OllamaEmbeddingTests(unittest.TestCase):
    def test_default_endpoint_is_native(self):
        provider = OllamaEmbeddingProvider()
        self.assertEqual(provider.endpoint, "http://127.0.0.1:11434/api/embed")
        self.assertEqual(provider.model, "nomic-embed-text")

    def test_openai_style_endpoint(self):
        provider = OllamaEmbeddingProvider(api_style="openai")
        self.assertEqual(provider.endpoint, "http://127.0.0.1:11434/v1/embeddings")

    def test_invalid_api_style_is_rejected(self):
        with self.assertRaises(ProviderError):
            OllamaEmbeddingProvider(api_style="grpc")

    def test_extract_handles_the_native_shape(self):
        vectors = OllamaEmbeddingProvider._extract({"embeddings": [[1.0, 2.0], [3.0, 4.0]]})
        self.assertEqual(vectors, [[1.0, 2.0], [3.0, 4.0]])

    def test_extract_handles_the_openai_shape(self):
        vectors = OllamaEmbeddingProvider._extract({"data": [{"embedding": [0.5, 0.25]}]})
        self.assertEqual(vectors, [[0.5, 0.25]])

    def test_extract_rejects_empty_payloads(self):
        for bad in ({}, {"embeddings": []}, {"data": []}):
            with self.assertRaises(ProviderError):
                OllamaEmbeddingProvider._extract(bad)

    def test_extract_rejects_non_list_vectors(self):
        with self.assertRaises(ProviderError):
            OllamaEmbeddingProvider._extract({"embeddings": ["nope"]})

    def test_embed_uses_the_injected_transport(self):
        calls = []

        def fake_post(url, payload, headers=None, timeout=120.0):
            calls.append((url, payload))
            return {"embeddings": [[float(len(payload["input"]))] * 3 for _ in payload["input"]]}

        original = ollama_embedding.post_json
        ollama_embedding.post_json = fake_post
        try:
            provider = OllamaEmbeddingProvider(model="m", timeout=5)
            vectors = provider.embed(["a", "b"])
        finally:
            ollama_embedding.post_json = original

        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["model"], "m")
        self.assertEqual(calls[0][1]["input"], ["a", "b"])
        self.assertEqual(provider.capabilities().dim, 3)

    def test_batching_splits_large_requests(self):
        calls = []

        def fake_post(url, payload, headers=None, timeout=120.0):
            calls.append(len(payload["input"]))
            return {"embeddings": [[0.0, 1.0] for _ in payload["input"]]}

        original = ollama_embedding.post_json
        ollama_embedding.post_json = fake_post
        try:
            provider = OllamaEmbeddingProvider()
            vectors = provider.embed(["t{0}".format(i) for i in range(70)])
        finally:
            ollama_embedding.post_json = original

        self.assertEqual(len(vectors), 70)
        self.assertEqual(calls, [32, 32, 6])

    def test_count_mismatch_is_a_provider_error(self):
        def fake_post(url, payload, headers=None, timeout=120.0):
            return {"embeddings": [[0.0, 1.0]]}

        original = ollama_embedding.post_json
        ollama_embedding.post_json = fake_post
        try:
            with self.assertRaises(ProviderError):
                OllamaEmbeddingProvider().embed(["a", "b"])
        finally:
            ollama_embedding.post_json = original

    def test_empty_batch_short_circuits(self):
        self.assertEqual(OllamaEmbeddingProvider().embed([]), [])


class EmbeddingRegistryTests(unittest.TestCase):
    def test_embedding_types(self):
        self.assertEqual(
            list(registry.embedding_provider_types()), ["fake-embedding", "ollama-embedding"]
        )

    def test_chat_types_are_unchanged(self):
        self.assertEqual(list(registry.provider_types()), ["deepseek", "fake", "ollama"])

    def test_create_each_embedding_type(self):
        self.assertIsInstance(
            registry.create_embedding_provider({"type": "fake-embedding"}), FakeEmbeddingProvider
        )
        self.assertIsInstance(
            registry.create_embedding_provider({"type": "ollama-embedding"}),
            OllamaEmbeddingProvider,
        )

    def test_unknown_embedding_type_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            registry.create_embedding_provider({"type": "openai"})

    def test_typo_in_an_embedding_key_is_rejected(self):
        with self.assertRaises(ConfigurationError) as caught:
            registry.create_embedding_provider({"type": "fake-embedding", "dims": 64})
        self.assertIn("dims", str(caught.exception))

    def test_a_chat_type_is_not_a_valid_embedding_provider(self):
        with self.assertRaises(ConfigurationError):
            registry.create_embedding_provider({"type": "ollama"})

    def test_an_embedding_type_is_not_a_valid_chat_provider(self):
        with self.assertRaises(ConfigurationError):
            registry.create_provider({"type": "ollama-embedding"})

    def test_resolve_embedding_role(self):
        models = {
            "providers": {"local-embed": {"type": "fake-embedding", "model": "m", "dim": 8}},
            "roles": {"embedding": "local-embed"},
        }
        provider = registry.resolve_embedding_role(models)
        self.assertEqual(provider.model, "m")

    def test_embedding_role_is_independent_of_chat(self):
        models = {
            "providers": {
                "chat": {"type": "fake", "model": "chat-model"},
                "embed": {"type": "fake-embedding", "model": "embed-model", "dim": 8},
            },
            "roles": {"chat": "chat", "embedding": "embed"},
        }
        self.assertEqual(registry.resolve_role(models, "chat").model, "chat-model")
        self.assertEqual(registry.resolve_embedding_role(models).model, "embed-model")

    def test_unbound_embedding_role_is_none(self):
        self.assertIsNone(registry.resolve_embedding_role({}))
        self.assertIsNone(registry.resolve_embedding_role({"roles": {"embedding": None}}))

    def test_cli_resolution_by_type_and_by_name(self):
        by_type = registry.resolve_cli_embedding({}, "fake-embedding", model="m1")
        self.assertEqual(by_type.model, "m1")
        models = {"providers": {"local": {"type": "fake-embedding", "model": "m2", "dim": 8}}}
        self.assertEqual(registry.resolve_cli_embedding(models, "local").model, "m2")

    def test_cli_resolution_rejects_an_unknown_embedding_provider(self):
        with self.assertRaises(ConfigurationError):
            registry.resolve_cli_embedding({}, "text-embedding-3-large")

    def test_all_provider_types_unions_both_families(self):
        self.assertEqual(
            list(registry.all_provider_types()),
            ["deepseek", "fake", "fake-embedding", "ollama", "ollama-embedding"],
        )


if __name__ == "__main__":
    unittest.main()
