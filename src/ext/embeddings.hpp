#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::ext {

/// Embeddings for one request, in input order. Each vector is the model's
/// full-width, L2-normalized output.
struct EmbeddingBatch {
  std::vector<std::vector<float>> embeddings;
  /// Tokens per input including beginning and end markers, after any
  /// truncation.
  std::vector<std::size_t> token_counts;
  /// Index of the offending input, set only with
  /// `EmbeddingStatus::kInputTooLong`.
  std::size_t failed_index{0};
};

struct EmbeddingPart {
  enum class Type { kText, kImage, kAudio, kVideo };
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
  kInvalidAudioInput,
  kInvalidVideoInput,
  kFailed,
  /// The device failed a probe after an error. The service stays lost.
  kDeviceLost,
};

/// Video decoding and sampling options for embedding inputs.
struct EmbeddingVideoOptions {
  double fps{1};
  std::size_t max_frames{32};
  bool timestamps{false};
  bool truncate_overflow{false};
};

inline constexpr std::uint32_t kDefaultImageSoftTokens = 280;
inline constexpr std::uint32_t kMaxImageSoftTokens = 1120;
inline constexpr std::size_t kMaxInputTokens = 8192;
inline constexpr std::size_t kMaxImages = 32;
inline constexpr std::size_t kMaxAudio = 32;
inline constexpr std::size_t kMaxVideos = 32;
inline constexpr std::size_t kMaxVideoFrames = 64;

/// Per-request choices that change how inputs are encoded.
struct EmbeddingRequestOptions {
  bool truncate{false};
  std::uint32_t image_soft_tokens{kDefaultImageSoftTokens};
  EmbeddingVideoOptions video;
};

/// Abstract capability runner for embedding models.
class EmbeddingRunner {
public:
  virtual ~EmbeddingRunner() = default;

  [[nodiscard]] virtual EmbeddingStatus EmbedInputs(
      const std::vector<EmbeddingInput>& inputs,
      const std::function<bool()>& is_cancelled, EmbeddingBatch* batch,
      std::string* error = nullptr,
      const EmbeddingRequestOptions& request = {}) = 0;

  [[nodiscard]] virtual bool device_lost() const noexcept { return false; }
};

using EmbeddingProbe = bool (*)(std::string_view model_root);
using EmbeddingArtifactResolver =
    bool (*)(std::string_view model_root,
             std::map<std::string, std::string>* prompts, std::string* error);
using EmbeddingRunnerFactory = std::unique_ptr<EmbeddingRunner> (*)(
    const std::string& model_root, std::string* error);

struct EmbeddingProvider {
  std::string id;
  EmbeddingProbe probe = nullptr;
  EmbeddingArtifactResolver resolve = nullptr;
  EmbeddingRunnerFactory create = nullptr;
};

void RegisterEmbeddingProvider(EmbeddingProvider provider);
const EmbeddingProvider* FindEmbeddingProvider(std::string_view id);
const EmbeddingProvider* ProbeEmbeddingProvider(std::string_view model_root);

}  // namespace gufo::ext
