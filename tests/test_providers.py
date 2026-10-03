"""Provider layer tests.

No test in this module requires a real model or an external network: the fake
provider covers behaviour, and transport error paths are exercised against a
closed local port.
"""

from __future__ import annotations

import unittest

from memory_core.domain.dto import Capabilities, ChatMessage
from memory_core.domain.errors import ConfigurationError, ProviderError
from memory_core.providers import openai_compat, registry
from memory_core.providers.base import ChatProvider, to_wire_messages
from memory_core.providers.deepseek import DeepSeekProvider
from memory_core.providers.fake import FakeProvider
from memory_core.providers.ollama import OllamaProvider


class FakeProviderTests(unittest.TestCase):
    def test_reply_is_a_deterministic_function_of_input(self):
        provider = FakeProvider(model="fake-A", prefix="[A]")
        messages = [ChatMessage("user", "hello")]
        first = provider.chat(messages)
        second = provider.chat(messages)
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.text, "[A] hello")
        self.assertEqual(first.provider, "fake")
        self.assertEqual(first.model, "fake-A")

    def test_two_fakes_produce_different_output(self):
        messages = [ChatMessage("user", "same question")]
        self.assertNotEqual(
            FakeProvider(prefix="[A]").chat(messages).text,
            FakeProvider(prefix="[B]").chat(messages).text,
        )

    def test_records_every_call(self):
        provider = FakeProvider()
        self.assertEqual(provider.call_count, 0)
        provider.chat([ChatMessage("user", "one")])
        provider.chat([ChatMessage("user", "two")])
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(provider.describe()["calls"], 2)

    def test_failure_is_a_provider_error(self):
        provider = FakeProvider(fail_with="model unavailable")
        with self.assertRaises(ProviderError) as caught:
            provider.chat([ChatMessage("user", "hi")])
        self.assertIn("model unavailable", str(caught.exception))
        self.assertEqual(caught.exception.provider, "fake")

    def test_capabilities_are_declared(self):
        self.assertTrue(FakeProvider().capabilities().supports_json_mode)
        self.assertFalse(FakeProvider(supports_json_mode=False).capabilities().supports_json_mode)

    def test_reply_can_be_static_text(self):
        self.assertEqual(FakeProvider(reply="fixed").chat([]).text, "fixed")

    def test_no_messages_is_not_a_crash(self):
        self.assertEqual(FakeProvider(prefix="[A]").chat([]).text, "[A] ")


class BaseProviderTests(unittest.TestCase):
    def test_chat_provider_is_abstract(self):
        with self.assertRaises(NotImplementedError):
            ChatProvider().chat([])

    def test_default_capabilities_are_conservative(self):
        capabilities = ChatProvider().capabilities()
        self.assertFalse(capabilities.supports_json_mode)
        self.assertFalse(capabilities.supports_streaming)

    def test_to_wire_messages_accepts_our_dtos(self):
        self.assertEqual(
            to_wire_messages([ChatMessage("system", "s"), ChatMessage("user", "u")]),
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        )

    def test_to_wire_messages_rejects_garbage(self):
        with self.assertRaises(ProviderError):
            to_wire_messages(["not a message"])

    def test_capabilities_serialise(self):
        payload = Capabilities(supports_json_mode=True, max_context_tokens=100).as_dict()
        self.assertEqual(payload["supports_json_mode"], True)
        self.assertEqual(payload["max_context_tokens"], 100)


class WireFormatTests(unittest.TestCase):
    def test_payload_shape(self):
        payload = openai_compat.build_payload(
            "m", [ChatMessage("user", "hi")], temperature=0.2, json_mode=True
        )
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_payload_omits_optional_fields(self):
        payload = openai_compat.build_payload("m", [])
        self.assertNotIn("temperature", payload)
        self.assertNotIn("response_format", payload)

    def test_extract_completion_normalises(self):
        result = openai_compat.extract_completion(
            {
                "model": "reported-model",
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 5},
            },
            provider="p",
            model="fallback",
        )
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.model, "reported-model")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage, {"total_tokens": 5})

    def test_extract_completion_falls_back_to_requested_model(self):
        result = openai_compat.extract_completion(
            {"choices": [{"message": {"content": "x"}}]}, provider="p", model="requested"
        )
        self.assertEqual(result.model, "requested")

    def test_extract_completion_rejects_missing_choices(self):
        for bad in ({}, {"choices": []}, {"choices": "nope"}):
            with self.assertRaises(ProviderError):
                openai_compat.extract_completion(bad, provider="p", model="m")

    def test_transport_failure_becomes_a_provider_error(self):
        # Closed local port: connection refused, no external network required.
        with self.assertRaises(ProviderError):
            openai_compat.post_json(
                "http://127.0.0.1:1/chat/completions", {"model": "m"}, timeout=1.0
            )

    def test_http_error_becomes_a_provider_error(self):
        with self.assertRaises(ProviderError):
            openai_compat.post_json(
                "http://127.0.0.1:1/not-a-json-endpoint", {}, timeout=1.0
            )


class DeepSeekProviderTests(unittest.TestCase):
    def test_missing_key_fails_only_at_call_time(self):
        provider = DeepSeekProvider(api_key=None, api_key_env="MEMORY_CORE_TEST_MISSING_KEY")
        self.assertFalse(provider.has_api_key)
        self.assertEqual(provider.name, "deepseek")
        with self.assertRaises(ProviderError) as caught:
            provider.chat([ChatMessage("user", "hi")])
        self.assertIn("no DeepSeek API key", str(caught.exception))

    def test_has_no_embedding_surface(self):
        # The DeepSeek API has no embeddings endpoint, which is exactly why
        # embeddings will get their own separate protocol.
        provider = DeepSeekProvider(api_key="test")
        self.assertFalse(hasattr(provider, "embed"))

    def test_defaults(self):
        provider = DeepSeekProvider(api_key="test")
        self.assertEqual(provider.model, "deepseek-chat")
        self.assertIn("api.deepseek.com", provider.base_url)

    def test_describe_never_leaks_the_key(self):
        provider = DeepSeekProvider(api_key="super-secret")
        self.assertNotIn("super-secret", str(provider.describe()))


class OllamaProviderTests(unittest.TestCase):
    def test_defaults_target_the_local_openai_compatible_endpoint(self):
        provider = OllamaProvider()
        self.assertEqual(provider.name, "ollama")
        self.assertTrue(provider.base_url.endswith("/v1"))
        self.assertIn("127.0.0.1", provider.base_url)

    def test_unreachable_ollama_raises_a_provider_error(self):
        provider = OllamaProvider(base_url="http://127.0.0.1:1/v1", timeout=1.0)
        with self.assertRaises(ProviderError):
            provider.chat([ChatMessage("user", "hi")])


class RegistryTests(unittest.TestCase):
    def test_provider_types(self):
        self.assertEqual(list(registry.provider_types()), ["deepseek", "fake", "ollama"])

    def test_create_each_type(self):
        self.assertIsInstance(registry.create_provider({"type": "fake"}), FakeProvider)
        self.assertIsInstance(registry.create_provider({"type": "ollama"}), OllamaProvider)
        self.assertIsInstance(
            registry.create_provider({"type": "deepseek", "api_key": "k"}), DeepSeekProvider
        )

    def test_unknown_type_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            registry.create_provider({"type": "litellm"})
        with self.assertRaises(ConfigurationError):
            registry.create_provider({"type": ""})

    def test_typo_in_a_key_is_rejected_not_ignored(self):
        with self.assertRaises(ConfigurationError) as caught:
            registry.create_provider({"type": "ollama", "modle": "qwen3"})
        self.assertIn("modle", str(caught.exception))

    def test_non_object_definition_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            registry.create_provider(["fake"])

    def test_resolve_role_returns_none_when_unbound(self):
        self.assertIsNone(registry.resolve_role({}, "chat"))
        self.assertIsNone(registry.resolve_role({"roles": {}}, "chat"))
        self.assertIsNone(registry.resolve_role({"roles": {"chat": None}}, "chat"))

    def test_resolve_role_builds_the_bound_provider(self):
        models = {
            "providers": {"local": {"type": "fake", "model": "m1"}},
            "roles": {"chat": "local"},
        }
        provider = registry.resolve_role(models, "chat")
        self.assertEqual(provider.model, "m1")

    def test_resolve_role_accepts_an_inline_definition(self):
        models = {"roles": {"chat": {"type": "fake", "model": "inline"}}}
        self.assertEqual(registry.resolve_role(models, "chat").model, "inline")

    def test_unknown_provider_name_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            registry.resolve_role({"providers": {}, "roles": {"chat": "nope"}}, "chat")

    def test_cli_provider_by_configured_name(self):
        models = {"providers": {"local": {"type": "fake", "model": "m1"}}}
        self.assertEqual(registry.resolve_cli_provider(models, "local").model, "m1")

    def test_cli_provider_by_type_with_model_override(self):
        provider = registry.resolve_cli_provider({}, "ollama", model="qwen3:8b")
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.model, "qwen3:8b")

    def test_cli_provider_without_provider_falls_back_to_the_chat_role(self):
        models = {"providers": {"local": {"type": "fake"}}, "roles": {"chat": "local"}}
        self.assertIsNotNone(registry.resolve_cli_provider(models, None))
        self.assertIsNone(registry.resolve_cli_provider({}, None))

    def test_cli_provider_rejects_an_unknown_name(self):
        with self.assertRaises(ConfigurationError):
            registry.resolve_cli_provider({}, "gpt-5")


if __name__ == "__main__":
    unittest.main()
