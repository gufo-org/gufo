#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/embeddings.hpp"
#include "gufo/ext/options.hpp"

namespace gufo::ext {

/// A query or one document: text, image and video parts in order, as
/// embeddings take them (`EmbeddingPart`; an image is a URL or data URL, a
/// video a URL or base64 data). The host sends no audio parts; a document may
/// be one empty text part, a query never is.
struct RerankInput {
  std::vector<EmbeddingPart> content;
};

/// Relevance scores for one request, in document order.
struct RerankScores {
  /// One score per document. A higher score is more relevant. The host sorts
  /// and trims; it applies no normalization, so the scale is the model's own.
  std::vector<float> scores;
  /// Tokens evaluated per document, including the template, after any
  /// truncation. The host sums them for `usage`.
  std::vector<std::size_t> token_counts;
  /// Index of the offending document, set only with
  /// `RerankStatus::kInputTooLong`.
  std::size_t failed_index{0};
};

enum class RerankStatus {
  kOk,
  kCancelled,
  kInputTooLong,
  kInvalidInput,
  /// The input holds a part of a kind the runner cannot score, such as an
  /// image for a text-only model. `error` names the kind.
  kUnsupportedInput,
  kFailed,
  /// The device failed a probe after an error. The service stays lost.
  kDeviceLost,
};

/// Per-request choices that change how a query and its documents are scored.
struct RerankRequestOptions {
  /// Task instruction for models that take one. Empty selects the model's
  /// own default.
  std::string instruction;
  /// Truncate a pair that exceeds the model's limit instead of returning
  /// `RerankStatus::kInputTooLong`.
  bool truncate{false};
};

/// Abstract capability runner for reranking models.
///
/// A request is one query and N documents. The runner owns the template that
/// frames each pair, tokenization, truncation, packing and scheduling; the host
/// owns the HTTP routes, `top_n`, sorting and `usage`. The host passes the
/// whole request in one call, never one document at a time, so the runner can
/// share work across the documents and across concurrent requests. Calls may
/// overlap; the runner serializes the device itself.
class RerankRunner {
public:
  virtual ~RerankRunner() = default;

  /// Scores every document against `query`. On `kOk`, `out->scores` and
  /// `out->token_counts` hold one entry per document in input order.
  /// `is_cancelled` is polled between device batches. A part kind the model
  /// cannot score returns `kUnsupportedInput`.
  [[nodiscard]] virtual RerankStatus Rerank(
      const RerankInput& query, std::span<const RerankInput> documents,
      const std::function<bool()>& is_cancelled, RerankScores* out,
      std::string* error = nullptr,
      const RerankRequestOptions& request = {}) = 0;

  [[nodiscard]] virtual bool device_lost() const noexcept { return false; }

  /// The model identifier a client may send as `model`, when the runner has
  /// one. Empty means the host uses the served model name alone.
  [[nodiscard]] virtual std::string ModelId() const { return {}; }
};

using RerankProbe = bool (*)(std::string_view model_root);
using RerankArtifactResolver = bool (*)(std::string_view model_root,
                                        std::string* error);

/// Model artifacts for a rerank runner.
struct RerankRunnerOptions {
  std::string model_root;
  /// The explicit command-line values of the options the provider declares
  /// for `serve rerank` (see TextRunnerOptions::provider_options).
  ParsedOptions provider_options{};
};
using RerankRunnerOptionsFactory = std::unique_ptr<RerankRunner> (*)(
    const RerankRunnerOptions& options, std::string* error);

struct RerankProvider {
  std::string id;
  RerankProbe probe = nullptr;
  RerankArtifactResolver resolve = nullptr;
  RerankRunnerOptionsFactory create_with_options = nullptr;
};

void RegisterRerankProvider(RerankProvider provider);
const RerankProvider* FindRerankProvider(std::string_view id);
const RerankProvider* ProbeRerankProvider(std::string_view model_root);

}  // namespace gufo::ext
