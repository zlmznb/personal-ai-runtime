"""Real local embedding provider backed by Ollama.

Uses Ollama's native ``/api/embed`` endpoint (the current one), with an option
to use the OpenAI-compatible ``/v1/embeddings`` endpoint instead.

This is intentionally a *separate class from* :class:`~memory_core.providers.ollama.OllamaProvider`:
the chat model and the embedding model are independent choices and may even be
served by different vendors.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from memory_core.domain.errors import ProviderError
from memory_core.providers.base import (
    EmbeddingCapabilities,
    EmbeddingProvider,
    normalize_embedding_texts,
)
from memory_core.providers.openai_compat import post_json

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text"


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local embeddings served by Ollama."""

    name = "ollama-embedding"

    def __init__(
        self,
        model=DEFAULT_MODEL,        # type: str
        base_url=DEFAULT_BASE_URL,  # type: str
        timeout=120.0,              # type: float
        api_style="native",         # type: str
    ):
        # type: (...) -> None
        if api_style not in ("native", "openai"):
            raise ProviderError("api_style must be 'native' or 'openai'")
        self.model = model
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        self.api_style = api_style
        self._dim = None  # type: Optional[int]

    def capabilities(self):
        # type: () -> EmbeddingCapabilities
        return EmbeddingCapabilities(dim=self._dim, max_batch=32, supports_batching=True)

    @property
    def endpoint(self):
        # type: () -> str
        if self.api_style == "openai":
            return self.base_url + "/v1/embeddings"
        return self.base_url + "/api/embed"

    def _payload(self, texts):
        # type: (List[str]) -> dict
        if self.api_style == "openai":
            return {"model": self.model, "input": texts}
        return {"model": self.model, "input": texts}

    @staticmethod
    def _extract(data):
        # type: (dict) -> List[List[float]]
        """Accept both the native and the OpenAI-compatible response shapes."""
        vectors = data.get("embeddings")
        if vectors is None:
            items = data.get("data")
            if isinstance(items, list):
                vectors = []
                for item in items:
                    if isinstance(item, dict) and "embedding" in item:
                        vectors.append(item["embedding"])
        if vectors is None and "embedding" in data:
            vectors = [data["embedding"]]
        if not isinstance(vectors, list) or not vectors:
            raise ProviderError("embedding response contained no vectors")
        out = []
        for vector in vectors:
            if not isinstance(vector, list):
                raise ProviderError("embedding response contained a non-list vector")
            out.append([float(value) for value in vector])
        return out

    def embed(self, texts):
        # type: (Sequence[str]) -> List[List[float]]
        items = normalize_embedding_texts(texts)
        if not items:
            return []
        batch_size = self.capabilities().max_batch
        out = []  # type: List[List[float]]
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            data = post_json(
                self.endpoint,
                self._payload(chunk),
                headers=None,
                timeout=self.timeout,
            )
            vectors = self._extract(data)
            if len(vectors) != len(chunk):
                raise ProviderError(
                    "embedding count mismatch: asked for {0}, got {1}".format(
                        len(chunk), len(vectors)
                    )
                )
            out.extend(vectors)
        if out:
            self._dim = len(out[0])
        return out


__all__ = ["OllamaEmbeddingProvider", "DEFAULT_BASE_URL", "DEFAULT_MODEL"]
