#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_HIP_ENCODER_RUNTIME_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_HIP_ENCODER_RUNTIME_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/models/embedding_gemma2/config.hpp"
#include "src/models/embedding_gemma2/image_processor.hpp"

namespace gufo::models::embedding_gemma2::hip {

/// The model's context window. Longer sequences are refused, not truncated.
inline constexpr std::size_t kMaxSequenceTokens = 8192;

struct SequenceImage {
  std::size_t offset{0};
  ImagePatches patches;
};
struct EncoderSequence {
  std::vector<std::uint32_t> tokens;
  std::vector<SequenceImage> images;
};

/// GPU text and image encoder. One instance owns the weights and the scratch
/// buffers and runs one device batch at a time. `Embed` is thread-safe:
/// concurrent calls take turns in arrival order, one device batch per turn, so
/// a long request does not hold the device until it finishes.
class EncoderRuntime {
public:
  ~EncoderRuntime();

  EncoderRuntime(const EncoderRuntime&) = delete;
  EncoderRuntime& operator=(const EncoderRuntime&) = delete;
  EncoderRuntime(EncoderRuntime&&) = delete;
  EncoderRuntime& operator=(EncoderRuntime&&) = delete;

  /// Uploads the text backbone. Vision weights load on the first image.
  [[nodiscard]] static std::unique_ptr<EncoderRuntime> Create(
      const std::string& model_dir, std::string* error = nullptr);

  /// Returns one mean-pooled, L2-normalized embedding of
  /// `config().text.embedding_dim` floats for each sequence. Sequences are
  /// packed into as few device batches of `kMaxSequenceTokens` as fit.
  /// The call polls `is_cancelled` before each device batch. A cancelled call
  /// returns false with `kCancelledMessage` in `error`. After a device
  /// failure that a probe confirms, `device_lost()` turns true and every later
  /// call fails with `kDeviceLostMessage`.
  [[nodiscard]] bool Embed(
      const std::vector<std::vector<std::uint32_t>>& sequences,
      std::vector<std::vector<float>>* embeddings, std::string* error = nullptr,
      const std::function<bool()>& is_cancelled = {});

  [[nodiscard]] bool EmbedMultimodal(
      const std::vector<EncoderSequence>& sequences,
      std::vector<std::vector<float>>* embeddings, std::string* error = nullptr,
      const std::function<bool()>& is_cancelled = {});

  static constexpr const char* kCancelledMessage =
      "embedding_gemma2: request cancelled";
  static constexpr const char* kDeviceLostMessage =
      "embedding_gemma2: GPU device lost";

  [[nodiscard]] const ModelConfig& config() const;
  [[nodiscard]] bool device_lost() const noexcept;

private:
  struct Impl;
  explicit EncoderRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::embedding_gemma2::hip

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_HIP_ENCODER_RUNTIME_HPP_
