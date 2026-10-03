"""Model provider layer.

This is the ONLY package permitted to import an HTTP client or speak a vendor
protocol (architecture red line 2). Everything above it deals in our own DTOs.

Two *independent* protocols live here:

* :class:`ChatProvider` - text generation.
* :class:`EmbeddingProvider` - vectorisation.

They are deliberately separate. The concrete reason: the DeepSeek API has no
embeddings endpoint, so in practice the chat model and the embedding model come
from different vendors. A single merged provider interface would break on the
first model swap, and swapping the embedding model would drag the chat model
with it.

Core rule: a provider failure is a degradation, not a crash. Every caller must
be able to continue without the model (red line 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from memory_core.domain.dto import Capabilities, ChatMessage, ChatResult
from memory_core.domain.errors import ProviderError


class ChatProvider(object):
    """Interface every chat model implementation satisfies.

    Note there is deliberately no ``embed`` method here: see the module docstring.
    """

    name = "provider"
    model = ""

    def capabilities(self):
        # type: () -> Capabilities
        return Capabilities()

    def chat(self, messages, **options):
        # type: (Sequence[ChatMessage], Any) -> ChatResult
        raise NotImplementedError

    def describe(self):
        # type: () -> Dict[str, Any]
        return {
            "name": self.name,
            "model": self.model,
            "capabilities": self.capabilities().as_dict(),
        }


@dataclass(frozen=True)
class EmbeddingCapabilities(object):
    """What an embedding provider can do.

    ``dim`` is ``None`` when the dimension is only known after the first call
    (which is normal for HTTP embedding APIs).
    """

    dim: Optional[int] = None
    max_batch: int = 64
    supports_batching: bool = True

    def as_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "dim": self.dim,
            "max_batch": self.max_batch,
            "supports_batching": self.supports_batching,
        }


class EmbeddingProvider(object):
    """Interface every embedding model implementation satisfies."""

    name = "embedding"
    model = ""

    def capabilities(self):
        # type: () -> EmbeddingCapabilities
        return EmbeddingCapabilities()

    def embed(self, texts):
        # type: (Sequence[str]) -> List[List[float]]
        """Embed a batch of texts. Must return exactly ``len(texts)`` vectors."""
        raise NotImplementedError

    def embed_one(self, text):
        # type: (str) -> List[float]
        vectors = self.embed([text])
        if not vectors:
            raise ProviderError("embedding provider returned no vector")
        return vectors[0]

    def identifier(self):
        # type: () -> str
        """Stable identity of this embedding model.

        Stored alongside every vector. Changing the embedding model changes the
        identifier, so vectors produced by a different model are never mixed
        with - or silently compared against - the current ones.
        """
        return "{0}:{1}".format(self.name, self.model)

    def describe(self):
        # type: () -> Dict[str, Any]
        return {
            "name": self.name,
            "model": self.model,
            "identifier": self.identifier(),
            "capabilities": self.capabilities().as_dict(),
        }


def to_wire_messages(messages):
    # type: (Sequence[ChatMessage]) -> List[Dict[str, str]]
    """Convert our DTOs to the OpenAI-compatible wire shape."""
    wire = []
    for message in messages:
        if isinstance(message, ChatMessage):
            wire.append({"role": message.role, "content": message.content})
        elif isinstance(message, dict):
            wire.append({"role": message["role"], "content": message["content"]})
        else:
            raise ProviderError("unsupported message type: {0}".format(type(message).__name__))
    return wire


def normalize_embedding_texts(texts):
    # type: (Any) -> List[str]
    """Validate and normalise an embedding batch before it reaches a provider."""
    if isinstance(texts, str):
        raise ProviderError("embed() expects a sequence of strings, not a bare string")
    try:
        items = list(texts)
    except TypeError:
        raise ProviderError("embed() expects a sequence of strings")
    out = []
    for item in items:
        if not isinstance(item, str):
            raise ProviderError(
                "embed() received a {0}, expected str".format(type(item).__name__)
            )
        out.append(item)
    return out


__all__ = [
    "ChatProvider",
    "EmbeddingProvider",
    "EmbeddingCapabilities",
    "to_wire_messages",
    "normalize_embedding_texts",
    "ProviderError",
    "ChatMessage",
    "ChatResult",
]
