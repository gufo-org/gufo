#ifndef GUFO_MODELS_QWEN_VISION_IMAGE_CACHE_HPP_
#define GUFO_MODELS_QWEN_VISION_IMAGE_CACHE_HPP_

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"
#include "src/families/qwen/text/chat_template.hpp"

namespace gufo::models::qwen::vision {

inline constexpr std::string_view kImagePreprocessingVersion =
    "qwen38-image-bicubic-rgb8-bf16-v1";

/// Backend-owned CPU preprocessing reuse, independent of continuation state.
/// Only the encoded bytes and the image detail affect the RGB pixels. The
/// encoder identity still participates in each prompt's continuation identity.
/// The budget counts retained RGB and encoded vector capacity; entry metadata
/// is bounded too because every resized image contains at least 65536 pixels.
/// Eviction drops cache ownership; in-flight prompts keep their immutable
/// pixels alive.
class ImageCache {
public:
  static constexpr std::size_t kDefaultCapacityBytes = 128 * 1024 * 1024;
  explicit ImageCache(std::size_t capacity_bytes = kDefaultCapacityBytes)
      : capacity_bytes_(capacity_bytes) {}

  struct Stats {
    std::size_t entries{0};
    std::size_t pixel_bytes{0};
    std::size_t encoded_bytes{0};
    std::size_t hits{0};
    std::size_t misses{0};
    std::size_t evictions{0};
  };

  [[nodiscard]] std::shared_ptr<const core::Image> Get(
      const std::shared_ptr<const std::vector<std::uint8_t>>& encoded,
      tokenization::ChatMessage::ImageDetail detail);
  [[nodiscard]] Stats GetStats() const;

private:
  struct Entry {
    std::size_t key;
    tokenization::ChatMessage::ImageDetail detail;
    // Exact comparison makes lookup-hash collisions harmless. Including these
    // bytes in the budget bounds reuse even for poorly compressible uploads.
    std::shared_ptr<const std::vector<std::uint8_t>> encoded;
    std::shared_ptr<const core::Image> image;
  };

  const std::size_t capacity_bytes_;
  mutable std::mutex mutex_;
  std::list<Entry> entries_;  // Most recently used first.
  Stats stats_;
};

}  // namespace gufo::models::qwen::vision
#endif
