#ifndef GUFO_SERVER_EMBEDDING_SERVICE_HPP_
#define GUFO_SERVER_EMBEDDING_SERVICE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/models/embedding_gemma2/hip/encoder_runtime.hpp"

namespace gufo::server {

/// Embeddings for one request, in input order. Each vector is the model's
/// full-width, L2-normalized output. The caller applies any Matryoshka
/// truncation.
struct EmbeddingBatch {
  std::vector<std::vector<float>> embeddings;
  /// Tokens per input including the beginning and end markers, after any
  /// truncation.
  std::vector<std::size_t> token_counts;
  /// Index of the offending input, set only with
  /// `EmbeddingStatus::kInputTooLong`.
  std::size_t failed_index{0};
};

struct EmbeddingPart {
  enum class Type { kText, kImage };
  Type type{Type::kText};
  std::string value;
};
struct EmbeddingInput {
  std::vector<EmbeddingPart> content;
};

enum class EmbeddingStatus {
  kOk,
  kCancelled,
  kInputTooLong,
  kInvalidInput,
  kFailed,
  /// The device failed a probe after an error. The service stays lost.
  kDeviceLost,
};

/// Per-request choices that change how inputs are encoded.
struct EmbeddingRequestOptions {
  /// Cut an input longer than `EmbeddingService::kMaxInputTokens` to its
  /// first tokens, keeping the end marker, instead of refusing it. An input
  /// whose images would not fit whole is still refused.
  bool truncate{false};
};

struct EmbeddingServiceOptions {
  using Runner =
      std::function<EmbeddingStatus(const std::vector<EmbeddingInput>& inputs,
                                    const EmbeddingRequestOptions& request,
                                    EmbeddingBatch* batch, std::string* error)>;

  std::filesystem::path model_root;
  bool validate_model{true};
  std::string model_id;
  /// Replaces the HIP encoder. The service serializes calls to it.
  Runner runner;
};

/// Cuts `tokens` (an encoding that ends in one `<eos>`) to `max_tokens` by
/// dropping content tokens before the `<eos>`. Returns false, leaving `tokens`
/// unchanged, when the cut would drop an `image_token` or the end-of-image
/// marker right after the last one.
[[nodiscard]] bool TruncateKeepingImages(std::vector<std::uint32_t>* tokens,
                                         std::size_t max_tokens,
                                         std::uint32_t image_token);

/// Native HIP EmbeddingGemma 2 text and image embedding service.
class EmbeddingService {
public:
  /// Longest accepted input in tokens, including the surrounding markers.
  static constexpr std::size_t kMaxInputTokens =
      models::embedding_gemma2::hip::kMaxSequenceTokens;

  static constexpr std::size_t kMaxImages = 32;

  explicit EmbeddingService(EmbeddingServiceOptions options);
  ~EmbeddingService();

  EmbeddingService(const EmbeddingService&) = delete;
  EmbeddingService& operator=(const EmbeddingService&) = delete;
  EmbeddingService(EmbeddingService&&) = delete;
  EmbeddingService& operator=(EmbeddingService&&) = delete;

  [[nodiscard]] bool ready() const noexcept;
  /// True once any request reported `EmbeddingStatus::kDeviceLost`.
  [[nodiscard]] bool device_lost() const noexcept;
  [[nodiscard]] std::string initialization_error() const;
  [[nodiscard]] std::string model_id() const;
  [[nodiscard]] std::string backend_name() const;

  /// Embeds `texts` as one request. Concurrent requests share device batches
  /// in arrival order, and a long request does not block shorter ones until
  /// it finishes. Returns `kCancelled` for a request cancelled before one of
  /// its batches reaches the device.
  [[nodiscard]] EmbeddingStatus Embed(const std::vector<std::string>& texts,
                                      const std::function<bool()>& is_cancelled,
                                      EmbeddingBatch* batch,
                                      std::string* error = nullptr);

  [[nodiscard]] EmbeddingStatus EmbedInputs(
      const std::vector<EmbeddingInput>& inputs,
      const std::function<bool()>& is_cancelled, EmbeddingBatch* batch,
      std::string* error = nullptr,
      const EmbeddingRequestOptions& request = {});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_EMBEDDING_SERVICE_HPP_
