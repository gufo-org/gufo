#include "src/families/qwen/vision/image_cache.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "src/families/qwen/vision/prompt.hpp"

namespace gufo::models::qwen::vision {

std::shared_ptr<const core::Image> ImageCache::Get(
    const std::shared_ptr<const std::vector<std::uint8_t>>& encoded,
    tokenization::ChatMessage::ImageDetail detail) {
  if (!encoded)
    throw std::invalid_argument("image input has no encoded bytes");
  const std::hash<std::string_view> hash;
  const auto key =
      hash({reinterpret_cast<const char*>(encoded->data()), encoded->size()}) ^
      hash(kImagePreprocessingVersion);
  const auto matches = [&](const Entry& entry) {
    return entry.key == key && entry.detail == detail &&
           std::ranges::equal(*entry.encoded, *encoded);
  };
  {
    std::lock_guard lock(mutex_);
    const auto found = std::ranges::find_if(entries_, matches);
    if (found != entries_.end()) {
      ++stats_.hits;
      entries_.splice(entries_.begin(), entries_, found);
      return found->image;
    }
    ++stats_.misses;
  }
  // Decode outside the lock so a new upload cannot block historical hits.
  // A concurrent miss may duplicate work, but only one result is retained.
  auto image = std::make_shared<const core::Image>(
      ResizeImage(core::DecodeImage(*encoded), detail));
  const auto pixels = image->pixels.capacity();
  if (pixels > capacity_bytes_ ||
      encoded->capacity() > capacity_bytes_ - pixels)
    return image;
  const auto bytes = pixels + encoded->capacity();
  std::lock_guard lock(mutex_);
  const auto found = std::ranges::find_if(entries_, matches);
  if (found != entries_.end()) {
    entries_.splice(entries_.begin(), entries_, found);
    return found->image;
  }
  while (stats_.pixel_bytes + stats_.encoded_bytes > capacity_bytes_ - bytes) {
    stats_.pixel_bytes -= entries_.back().image->pixels.capacity();
    stats_.encoded_bytes -= entries_.back().encoded->capacity();
    entries_.pop_back();
    ++stats_.evictions;
  }
  entries_.push_front({key, detail, encoded, image});
  stats_.pixel_bytes += pixels;
  stats_.encoded_bytes += encoded->capacity();
  return image;
}

ImageCache::Stats ImageCache::GetStats() const {
  std::lock_guard lock(mutex_);
  auto stats = stats_;
  stats.entries = entries_.size();
  return stats;
}

}  // namespace gufo::models::qwen::vision
