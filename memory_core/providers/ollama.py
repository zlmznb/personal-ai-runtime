"""Local Ollama provider.

Talks to Ollama's OpenAI-compatible endpoint (``/v1``), which is served on the
same port as the native API. Chat only for now: embeddings arrive with the
vector phase, behind their own separate protocol.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from memory_core.domain.dto import Capabilities, ChatMessage, ChatResult
from memory_core.providers.base import ChatProvider
from memory_core.providers.openai_compat import chat_completion

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3:8b"


class OllamaProvider(ChatProvider):
    """A local model served by Ollama."""

    name = "ollama"

    def __init__(
        self,
        model=DEFAULT_MODEL,           # type: str
        base_url=DEFAULT_BASE_URL,     # type: str
        timeout=120.0,                 # type: float
        max_context_tokens=32768,      # type: int
    ):
        # type: (...) -> None
        self.model = model
        self.base_url = base_url
        self.timeout = float(timeout)
        self._capabilities = Capabilities(
            supports_json_mode=True,
            supports_streaming=True,
            max_context_tokens=int(max_context_tokens),
        )

    def capabilities(self):
        # type: () -> Capabilities
        return self._capabilities

    def chat(self, messages, temperature=None, json_mode=False, extra=None, **ignored):
        # type: (Sequence[ChatMessage], Optional[float], bool, Any, Any) -> ChatResult
        return chat_completion(
            base_url=self.base_url,
            api_key=None,
            model=self.model,
            messages=messages,
            timeout=self.timeout,
            temperature=temperature,
            json_mode=json_mode,
            extra=extra,
            provider=self.name,
        )


__all__ = ["OllamaProvider", "DEFAULT_BASE_URL", "DEFAULT_MODEL"]
