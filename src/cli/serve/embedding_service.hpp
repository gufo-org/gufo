#ifndef GUFO_SERVER_EMBEDDING_SERVICE_HPP_
#define GUFO_SERVER_EMBEDDING_SERVICE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gufo/ext/embeddings.hpp"

namespace gufo::server {

using ext::EmbeddingBatch;
using ext::EmbeddingInput;
using ext::EmbeddingPart;
using ext::EmbeddingRequestOptions;
using ext::EmbeddingStatus;
using ext::EmbeddingVideoOptions;
using VideoOptions = ext::EmbeddingVideoOptions;
using ext::kDefaultImageSoftTokens;
using ext::kMaxAudio;
using ext::kMaxImages;
using ext::kMaxImageSoftTokens;
using ext::kMaxInputTokens;
using ext::kMaxVideoFrames;
using ext::kMaxVideos;

struct EmbeddingServiceOptions {
  using Runner =
      std::function<EmbeddingStatus(const std::vector<EmbeddingInput>& inputs,
                                    const EmbeddingRequestOptions& request,
                                    EmbeddingBatch* batch, std::string* error)>;

  std::filesystem::path model_root;
  bool validate_model{true};
  std::string model_id;
  /// Named input prefixes, as in a checkpoint's
  /// config_sentence_transformers.json `prompts`.
  std::map<std::string, std::string> prompts;
  /// Prompt applied when a request names none; empty applies none. Must name
  /// an entry of `prompts`.
  std::string default_prompt_name;
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

/// Reads the `prompts` of `model_root`/config_sentence_transformers.json.
/// A missing file yields no prompts; a malformed one fails with `error`.
[[nodiscard]] bool LoadSentenceTransformerPrompts(
    const std::filesystem::path& model_root,
    std::map<std::string, std::string>* prompts, std::string* error);

/// Native HIP text, image and audio embedding service.
class EmbeddingService {
public:
  /// Longest accepted input in tokens, including the surrounding markers.
  static constexpr std::size_t kMaxInputTokens = ext::kMaxInputTokens;

  static constexpr std::size_t kMaxImages = ext::kMaxImages;
  static constexpr std::size_t kMaxAudio = ext::kMaxAudio;
  static constexpr std::size_t kMaxVideos = ext::kMaxVideos;
  static constexpr std::size_t kMaxVideoFrames = ext::kMaxVideoFrames;

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
  /// The prefix registered under `name`, or nullptr.
  [[nodiscard]] const std::string* prompt(const std::string& name) const;
  [[nodiscard]] const std::string& default_prompt_name() const;

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
