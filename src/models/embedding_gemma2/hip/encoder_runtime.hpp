#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_HIP_ENCODER_RUNTIME_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_HIP_ENCODER_RUNTIME_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "batch_scheduler.hpp"
#include "config.hpp"

namespace gufo::models::embedding_gemma2::hip {

/// The model's context window. Longer sequences are refused, not truncated.
inline constexpr std::size_t kMaxSequenceTokens = 8192;

/// GPU text, image and audio encoder. One instance owns the weights and the
/// scratch buffers and runs one device batch at a time. `Embed` is thread-safe:
/// text calls that wait for the device share batches in arrival order, as
/// `BatchScheduler` describes, so a long call does not hold the device until
/// it finishes.
class EncoderRuntime {
public:
  ~EncoderRuntime();

  EncoderRuntime(const EncoderRuntime&) = delete;
  EncoderRuntime& operator=(const EncoderRuntime&) = delete;
  EncoderRuntime(EncoderRuntime&&) = delete;
  EncoderRuntime& operator=(EncoderRuntime&&) = delete;

  /// Uploads the text backbone. Vision and audio weights load on the first
  /// request for each.
  [[nodiscard]] static std::unique_ptr<EncoderRuntime> Create(
      const std::string& model_dir, std::string* error = nullptr);

  /// Returns one mean-pooled, L2-normalized embedding of
  /// `config().text.embedding_dim` floats for each sequence. Sequences are
  /// packed into device batches of up to `kMaxSequenceTokens`, together with
  /// those of concurrent calls. `is_cancelled` is polled before each device
  /// batch that would take this call's sequences. The poll may run on another
  /// call's thread with the scheduler's lock held, so it must not block or
  /// throw. A cancelled call returns false with `kCancelledMessage` in
  /// `error`. A failed batch fails every call with sequences in it. After a
  /// device failure that a probe confirms, `device_lost()` turns true and
  /// every later call fails with `kDeviceLostMessage`.
  [[nodiscard]] bool Embed(
      const std::vector<std::vector<std::uint32_t>>& sequences,
      std::vector<std::vector<float>>* embeddings, std::string* error = nullptr,
      const std::function<bool()>& is_cancelled = {});

  /// Like `Embed`, for sequences that may hold images or audio. A call with an
  /// image or audio runs in a batch of its own, in arrival order. The first
  /// such call loads the vision tower in that batch, so a failed load fails
  /// only that call.
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
