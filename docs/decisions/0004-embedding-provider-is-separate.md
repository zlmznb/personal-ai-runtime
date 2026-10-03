# ADR 0004 — EmbeddingProvider is a separate protocol from ChatProvider

Status: accepted (v0.2)

## Context

v0.1 needed only text generation, so it shipped a single `ChatProvider`
protocol. v0.2 adds semantic retrieval, which needs vectors.

The tempting shortcut is to add an `embed()` method to `ChatProvider` and be
done. That would be a mistake with a concrete cause.

## Decision

`EmbeddingProvider` is its own protocol, with its own implementations, its own
registry types and its own configuration role.

The reason is not stylistic. **The DeepSeek API has no embeddings endpoint.** In
practice the chat model and the embedding model come from different vendors, so
a merged interface would break on the first realistic configuration. Worse, it
would couple two independent swap decisions: changing the embedding model would
mean touching the object that also generates answers.

`OllamaProvider` and `OllamaEmbeddingProvider` are separate classes even though
both speak to the same daemon, because that is a coincidence of this particular
deployment, not a property of the design.

Additional consequences:

- Vectors are keyed by `(memory_id, model_id)`, so vectors from different
  embedding models are never compared with each other or silently mixed.
- Swapping the embedding model adds rows under a new `model_id`; the memories
  are untouched.
- `tests/test_embedding_providers.py` asserts that neither family can be
  constructed as the other.

## Consequences

- Configuration has independent roles: `chat` and `embedding` may point at
  different vendors, and `resolve_embedding_role` will refuse a chat provider.
- The registry rejects a chat type used as an embedding provider and vice versa.

## Note on the live assessment

Measured locally: `nomic-embed-text` ranks Chinese content correctly;
`all-minilm` is English-only and misranks it. Quality is model-dependent.
Durability is not, which is exactly what this separation protects.
