# Embedding video token budget

Embeddings (`embeddings.hpp`): `EmbeddingVideoOptions` gains
`max_soft_tokens`, the per-frame vision budget (default
`kDefaultVideoSoftTokens`, 140), and `kMaxVideoSoftTokens` bounds a request's
video soft tokens. Additive; extensions that aggregate-initialize
`EmbeddingVideoOptions` positionally keep the default.
