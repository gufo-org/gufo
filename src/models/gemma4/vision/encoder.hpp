#ifndef GUFO_MODELS_GEMMA4_VISION_ENCODER_HPP_
#define GUFO_MODELS_GEMMA4_VISION_ENCODER_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "src/core/image.hpp"

namespace gufo::models::gemma4::vision {

/// Gemma 4 vision tower and embedder on the GPU: BF16 weights and GEMM
/// inputs, FP32 accumulation and residual stream. It follows
/// vision::Reference, whose stage names the observer receives.
class Encoder {
public:
  /// Device rows [rows][width] of FP32 language-model input embeddings.
  class Embedding {
  public:
    ~Embedding();
    Embedding(const Embedding&) = delete;
    Embedding& operator=(const Embedding&) = delete;
    [[nodiscard]] const float* data() const noexcept { return data_; }
    [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }

  private:
    friend class Encoder;
    Embedding() = default;
    float* data_{nullptr};
    std::uint32_t rows_{0};
    std::uint32_t width_{0};
  };

  using Observer =
      std::function<void(std::string_view stage, std::span<const float>)>;
  using CancellationCheck = std::function<bool()>;

  Encoder(const std::filesystem::path& path, std::uint32_t output_width);
  /// An explicit sidecar must exist. Otherwise `mmproj-BF16.gguf` beside the
  /// target (or one directory up) is used when present.
  [[nodiscard]] static std::shared_ptr<Encoder> Open(
      const std::filesystem::path& target, const std::filesystem::path& sidecar,
      std::uint32_t output_width);
  ~Encoder();
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  /// SHA-256 of the whole sidecar, computed once.
  [[nodiscard]] const std::string& identity() const;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// `image` is already resized (sides multiples of 48). Weights upload on
  /// the first call; scratch is sized by the image. Calls are serialized by
  /// the owner.
  [[nodiscard]] std::shared_ptr<const Embedding> Encode(
      const core::Image& image, const Observer& observer = {},
      const CancellationCheck& is_cancelled = {});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_VISION_ENCODER_HPP_
