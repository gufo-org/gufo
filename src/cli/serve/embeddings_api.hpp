#ifndef GUFO_SERVER_EMBEDDINGS_API_HPP_
#define GUFO_SERVER_EMBEDDINGS_API_HPP_

#include <string_view>

#include "src/cli/serve/embedding_service.hpp"
#include "src/cli/serve/http_server.hpp"

namespace gufo::server {

inline constexpr std::string_view kEmbeddingsApiSchema =
    "gufo.embeddings-api.v1";

[[nodiscard]] bool IsEmbeddingsApiPath(std::string_view path) noexcept;

/// OpenAI-compatible `POST /v1/embeddings`.
[[nodiscard]] HttpResponse HandleEmbeddingsApiRequest(
    const HttpRequest& request, EmbeddingService& service);

}  // namespace gufo::server

#endif  // GUFO_SERVER_EMBEDDINGS_API_HPP_
