#ifndef GUFO_MODELS_QWEN_VISION_PROMPT_HPP_
#define GUFO_MODELS_QWEN_VISION_PROMPT_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/vision/admission.hpp"

namespace gufo::models::qwen::vision {

class ImageCache;
class AdmissionBudget;

inline constexpr std::uint32_t kImageToken = 248056;
inline constexpr std::uint32_t kPatchSize = 16;
inline constexpr std::uint32_t kMergeSize = 2;

struct ImageGrid {
  std::uint32_t offset{0};  ///< physical first image-pad token
  std::uint32_t height{0};  ///< merged language-token grid, not ViT patches
  std::uint32_t width{0};
  bool operator==(const ImageGrid&) const = default;
};

/// Compact request-owned position state, sufficient to restore RoPE without
/// retaining pixels or encoder scratch in a continuation snapshot.
struct RopeLayout {
  std::vector<ImageGrid> images;
  [[nodiscard]] std::array<std::int32_t, 3> Position(
      std::uint32_t physical) const;
  [[nodiscard]] std::int32_t Delta() const;
  [[nodiscard]] std::uint32_t PrefixLength() const;
  /// Images contributing at least one embedding to the computed prefix.
  [[nodiscard]] RopeLayout Prefix(std::uint32_t token_count) const;
  void Validate(std::uint32_t max_context) const;
  bool operator==(const RopeLayout&) const = default;
};

struct PreparedImage {
  /// Immutable resized RGB8 shared with CPU preprocessing reuse. Temporal
  /// repetition happens on GPU; continuation snapshots do not retain pixels.
  std::shared_ptr<const core::Image> pixels;
  ImageGrid grid;
  std::array<std::uint8_t, 32> prefix_identity{};
};

struct Prompt {
  std::vector<tokenization::TokenId> tokens;
  /// Frontier unchanged when an interrupted assistant is followed by a user.
  std::size_t stable_prefix_tokens{0};
  RopeLayout rope;
  std::vector<PreparedImage> images;
  /// SHA-256 covers decoded pixels, grid placement, preprocessing version,
  /// and the model-specific encoder identity. Tokens remain a separate key.
  std::vector<std::uint8_t> cache_identity;
  /// Prepared pixel pointers share ownership with their reservation, including
  /// when a prompt or individual image is copied. DeviceInput may omit that
  /// duplicate charge only for this same budget.
  std::weak_ptr<AdmissionBudget> memory_admission;
  /// Attach an existing reservation to every pixel pointer. Copies and value
  /// assignment then preserve accounting independently of the Prompt owner.
  void RetainPixels(std::shared_ptr<const void> reservation,
                    const std::shared_ptr<AdmissionBudget>& budget);
  [[nodiscard]] std::span<const std::uint8_t> IdentityForPrefix(
      std::size_t token_count) const;
};

/// `kLow` limits both sides of the resized image to 512 pixels, which is at
/// most 16 x 16 merged tokens, unless the model's minimum size is larger. It
/// never exceeds the `kAuto` size. `kAuto` uses the model's full range.
[[nodiscard]] core::Image ResizeImage(
    const core::Image& image, tokenization::ChatMessage::ImageDetail detail);
/// Inspect the original image without decoding or allocating image-sized
/// storage. Includes decoder/orientation, resizing and filter-construction
/// peaks, plus retained resized RGB pixels. History owners keep their own
/// admissions alive while the next image is prepared.
[[nodiscard]] PreparationMemory PlanImagePreparation(
    std::span<const std::uint8_t> encoded,
    tokenization::ChatMessage::ImageDetail detail);
[[nodiscard]] Prompt Prepare(
    const tokenization::QwenTokenizer& tokenizer,
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools,
    const tokenization::ChatTemplateOptions& options,
    std::string_view encoder_identity, std::uint32_t max_context,
    ImageCache* image_cache = nullptr,
    const std::shared_ptr<AdmissionBudget>& budget = {});

}  // namespace gufo::models::qwen::vision
#endif
