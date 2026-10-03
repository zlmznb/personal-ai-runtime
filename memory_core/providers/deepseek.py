"""DeepSeek cloud provider (chat only).

Important architectural fact
----------------------------
The DeepSeek API exposes chat/reasoner style models only - it has **no
embeddings endpoint**. That is the concrete reason ``ChatProvider`` and a future
``EmbeddingProvider`` must be separate protocols: in practice the chat model and
the embedding model come from different vendors, so a single merged provider
interface would break on the first model swap.

Construction never fails when the API key is missing; ``chat`` raises
:class:`ProviderError` instead, so the rest of the system degrades rather than
crashing at startup.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from memory_core.domain.dto import Capabilities, ChatMessage, ChatResult
from memory_core.domain.errors import ProviderError
from memory_core.providers.base import ChatProvider
from memory_core.providers.openai_compat import chat_completion

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"


class DeepSeekProvider(ChatProvider):
    """A cloud chat model. Chat only - there is no embedding support to expose."""

    name = "deepseek"

    def __init__(
        self,
        model=DEFAULT_MODEL,                      # type: str
        base_url=DEFAULT_BASE_URL,                # type: str
        api_key=None,                             # type: Optional[str]
        api_key_env=DEFAULT_API_KEY_ENV,          # type: str
        timeout=120.0,                            # type: float
        max_context_tokens=65536,                 # type: int
    ):
        # type: (...) -> None
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._api_key = api_key or os.environ.get(api_key_env)
        self.timeout = float(timeout)
        self._capabilities = Capabilities(
            supports_json_mode=True,
            supports_streaming=True,
            max_context_tokens=int(max_context_tokens),
        )

    def capabilities(self):
        # type: () -> Capabilities
        return self._capabilities

    @property
    def has_api_key(self):
        # type: () -> bool
        return bool(self._api_key)

    def chat(self, messages, temperature=None, json_mode=False, extra=None, **ignored):
        # type: (Sequence[ChatMessage], Optional[float], bool, Any, Any) -> ChatResult
        if not self._api_key:
            raise ProviderError(
                "no DeepSeek API key; set {0} or pass api_key".format(self.api_key_env),
                provider=self.name,
                model=self.model,
            )
        return chat_completion(
            base_url=self.base_url,
            api_key=self._api_key,
            model=self.model,
            messages=messages,
            timeout=self.timeout,
            temperature=temperature,
            json_mode=json_mode,
            extra=extra,
            provider=self.name,
        )


__all__ = ["DeepSeekProvider", "DEFAULT_BASE_URL", "DEFAULT_MODEL", "DEFAULT_API_KEY_ENV"]
