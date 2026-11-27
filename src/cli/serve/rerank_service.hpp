#ifndef GUFO_SERVER_RERANK_SERVICE_HPP_
#define GUFO_SERVER_RERANK_SERVICE_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/rerank.hpp"

namespace gufo::server {

class RerankRequestScope;

using ext::RerankInput;
using ext::RerankRequestOptions;
using ext::RerankScores;
using ext::RerankStatus;

struct RerankServiceOptions {
  using Runner = std::function<RerankStatus(
      const RerankInput& query, std::span<const RerankInput> documents,
      const RerankRequestOptions& request,
      const std::function<bool()>& is_cancelled, RerankScores* scores,
      std::string* error)>;

  std::filesystem::path model_root;
  /// Values of the provider's declared `serve rerank` options.
  ext::ParsedOptions provider_options;
  bool validate_model{true};
  /// The name clients see. Empty takes the runner's `ModelId()`, then the
  /// model directory's name.
  std::string model_id;
  /// Additional IDs accepted from requests; `model_id` stays primary in
  /// listings and responses.
  std::vector<std::string> model_aliases;
  /// Replaces the provider's runner. The service serializes calls to it.
  Runner runner;
};

/// Serves one reranking model through a registered `RerankProvider`.
class RerankService {
public:
  /// Most documents one request may carry.
  static constexpr std::size_t kMaxDocuments = 2048;

  explicit RerankService(RerankServiceOptions options);
  ~RerankService();

  RerankService(const RerankService&) = delete;
  RerankService& operator=(const RerankService&) = delete;
  RerankService(RerankService&&) = delete;
  RerankService& operator=(RerankService&&) = delete;

  [[nodiscard]] bool ready() const noexcept;
  /// True once any request reported `RerankStatus::kDeviceLost`.
  [[nodiscard]] bool device_lost() const noexcept;
  [[nodiscard]] std::string initialization_error() const;
  [[nodiscard]] std::string model_id() const;
  /// True for the primary ID or a configured alias.
  [[nodiscard]] bool serves_model(std::string_view name) const;
  [[nodiscard]] const std::vector<std::string>& model_aliases() const;
  [[nodiscard]] std::string backend_name() const;
  [[nodiscard]] std::string RenderMetrics() const;

  /// Scores `documents` against `query` as one request. The runner receives
  /// the whole request. On success `scores` holds one score and one token
  /// count per document, all finite.
  [[nodiscard]] RerankStatus Rerank(const RerankInput& query,
                                    std::span<const RerankInput> documents,
                                    const std::function<bool()>& is_cancelled,
                                    RerankScores* scores,
                                    std::string* error = nullptr,
                                    const RerankRequestOptions& request = {});

private:
  struct Impl;
  RerankStatus RerankAdmitted(const RerankInput& query,
                              std::span<const RerankInput> documents,
                              const std::function<bool()>& is_cancelled,
                              RerankScores* scores, std::string* error,
                              const RerankRequestOptions& request,
                              RerankRequestScope& scope);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_RERANK_SERVICE_HPP_
