# Rerank capability

Add the `rerank` capability for models that score documents against a query (`include/gufo/ext/rerank.hpp`, `include/gufo/ext/registry.hpp`):
- `rerank.hpp`: `RerankRunner` (`Rerank`, `device_lost`, `ModelId`), `RerankInput`, `RerankScores`, `RerankStatus`, `RerankRequestOptions`, `RerankRunnerOptions`, `RerankProvider`, the aliases `RerankProbe`, `RerankArtifactResolver` and `RerankRunnerOptionsFactory`, and `RegisterRerankProvider`, `FindRerankProvider` and `ProbeRerankProvider`. A request is one query and N documents, handed to the runner whole. The query and each document are a `RerankInput`: ordered `EmbeddingPart`s (text, image or video) from `embeddings.hpp`. A runner returns `RerankStatus::kUnsupportedInput` for a part kind it cannot score. The runner owns the prompt template, tokenization, truncation, packing and scheduling.
- `registry.hpp`: `Capability::kRerank` (`"capability": "rerank"` in metadata) and `Registry::AddRerankProvider`.
- Metadata: `"rerank"` joins the provider `capability` enum in `schemas/extension.schema.json`, and options declared for `serve rerank` reach the runner.

The host serves the capability with `gufo serve rerank` on `/v1/rerank`, `/v1/reranking` and `/rerank`. All additions are optional for existing providers.
