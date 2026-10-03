"""Deterministic offline provider.

Two roles:

* It lets the entire system be tested with no network and no model installed.
* Two differently-configured fakes are what make the model-swap test meaningful:
  the model output demonstrably changes, while the memory database provably does
  not.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from memory_core.domain.dto import Capabilities, ChatMessage, ChatResult
from memory_core.domain.errors import ProviderError
from memory_core.providers.base import ChatProvider


class FakeProvider(ChatProvider):
    """A provider whose output is a pure function of its input."""

    name = "fake"

    def __init__(
        self,
        model="fake-1",           # type: str
        prefix="[fake]",          # type: str
        fail_with=None,           # type: Optional[str]
        reply=None,               # type: Optional[Any]
        supports_json_mode=True,  # type: bool
        max_context_tokens=8192,  # type: int
    ):
        # type: (...) -> None
        self.model = model
        self.prefix = prefix
        self.fail_with = fail_with
        self._reply = reply
        self._capabilities = Capabilities(
            supports_json_mode=supports_json_mode,
            supports_streaming=False,
            max_context_tokens=max_context_tokens,
        )
        self.calls = []  # type: List[List[ChatMessage]]

    def capabilities(self):
        # type: () -> Capabilities
        return self._capabilities

    @property
    def call_count(self):
        # type: () -> int
        return len(self.calls)

    def chat(self, messages, **options):
        # type: (Sequence[ChatMessage], Any) -> ChatResult
        recorded = list(messages)
        self.calls.append(recorded)
        if self.fail_with:
            raise ProviderError(self.fail_with, provider=self.name, model=self.model)

        last_user = ""
        for message in recorded:
            if getattr(message, "role", None) == "user":
                last_user = message.content

        if self._reply is not None:
            reply = self._reply
            text = reply(recorded) if isinstance(reply, Callable) else str(reply)
        else:
            text = "{0} {1}".format(self.prefix, last_user)

        return ChatResult(
            text=text,
            provider=self.name,
            model=self.model,
            finish_reason="stop",
            usage={"prompt_messages": len(recorded)},
        )

    def describe(self):
        # type: () -> Dict[str, Any]
        described = super(FakeProvider, self).describe()
        described["calls"] = self.call_count
        return described


__all__ = ["FakeProvider"]
