# Capability-neutral embedding contracts

Publish capability-neutral embedding contracts (`include/gufo/ext/embeddings.hpp`):
- `EmbeddingBatch`, `EmbeddingInput`, `EmbeddingPart`, `EmbeddingStatus`,
  `EmbeddingVideoOptions`, `EmbeddingRequestOptions`.
- `EmbeddingRunner` abstract execution interface.
- `EmbeddingProvider` registration and discovery (`RegisterEmbeddingProvider`,
  `FindEmbeddingProvider`, `ProbeEmbeddingProvider`, `Registry::AddEmbeddingProvider`).
- Pilot model migration for EmbeddingGemma 2 (Card 13).
