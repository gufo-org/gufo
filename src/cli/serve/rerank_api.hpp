#ifndef GUFO_SERVER_RERANK_API_HPP_
#define GUFO_SERVER_RERANK_API_HPP_

#include <string_view>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/rerank_service.hpp"

namespace gufo::server {

inline constexpr std::string_view kRerankApiSchema = "gufo.rerank-api.v1";

[[nodiscard]] bool IsRerankApiPath(std::string_view path) noexcept;

/// `POST /v1/rerank`, `/v1/reranking` and `/rerank`, in llama.cpp's Jina form
/// (`query` and `documents`) and its TEI form (`query` and `texts`).
[[nodiscard]] HttpResponse HandleRerankApiRequest(const HttpRequest& request,
                                                  RerankService& service);

}  // namespace gufo::server

#endif  // GUFO_SERVER_RERANK_API_HPP_
