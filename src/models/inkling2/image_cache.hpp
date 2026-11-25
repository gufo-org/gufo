// Reuse of images across requests: a conversation sends its images again with
// every turn, and a prompt the prefix cache cannot serve encodes them again.
// Both stages are deterministic, so a reused result is the one a new decode
// or encode would produce, byte for byte.
#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string_view>
#include <vector>

#include "src/models/inkling2/engine.hpp"
#include "src/models/inkling2/vision/vision_image.hpp"

namespace gufo::models::inkling2 {

using ImageDigest = std::array<std::uint8_t, 32>;

/// An uploaded image after decoding (and the low-detail shrink): its RGB
/// pixels and the SHA-256 of its size and pixels.
struct DecodedImage {
  vision::ImageU8 pixels;
  ImageDigest digest{};
};

/// The encoder's output for one image: device [count][2304] floats.
struct ImageRows {
  float* device = nullptr;
  std::size_t count = 0;
  ImageRows() = default;
  ImageRows(const ImageRows&) = delete;
  ImageRows& operator=(const ImageRows&) = delete;
  ~ImageRows() {
    if (device != nullptr) {
      // As the other device calls outside the engine mutex: no graph capture
      // may be open.
      const std::shared_lock<std::shared_mutex> gate(CaptureGate());
      (void)hipFree(device);
    }
  }
  [[nodiscard]] std::size_t bytes() const {
    return count * 2304 * sizeof(float);
  }
};

/// Two least-recently-used stores with byte budgets. Eviction drops only the
/// cache's reference: a prompt still using an entry keeps it alive.
class ImageCache {
public:
  /// Decoded pixels (host memory) and encoder rows (device memory).
  static constexpr std::size_t kDecodedBytes = std::size_t{128} << 20;
  static constexpr std::size_t kRowsBytes = std::size_t{256} << 20;

  /// The decoded image of `bytes` at `low` detail, from the cache or from
  /// `decode`. Entries are matched on the exact bytes.
  [[nodiscard]] std::shared_ptr<const DecodedImage> Decoded(
      const std::shared_ptr<const std::vector<std::uint8_t>>& bytes, bool low,
      const std::function<std::shared_ptr<const DecodedImage>()>& decode) {
    const std::size_t key =
        std::hash<std::string_view>{}(std::string_view(
            reinterpret_cast<const char*>(bytes->data()), bytes->size())) ^
        (low ? 0x9e3779b97f4a7c15ULL : 0);
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = decoded_.begin(); it != decoded_.end(); ++it) {
        if (it->key != key || it->low != low ||
            (it->bytes != bytes && *it->bytes != *bytes))
          continue;
        decoded_.splice(decoded_.begin(), decoded_, it);
        return decoded_.front().image;
      }
    }
    auto image = decode();
    const std::size_t cost = bytes->size() + image->pixels.rgb.size();
    const std::lock_guard<std::mutex> lock(mutex_);
    if (cost <= kDecodedBytes) {
      decoded_.push_front({key, low, bytes, image, cost});
      decoded_bytes_ += cost;
      while (decoded_bytes_ > kDecodedBytes) {
        decoded_bytes_ -= decoded_.back().cost;
        decoded_.pop_back();
      }
    }
    return image;
  }

  [[nodiscard]] std::shared_ptr<const ImageRows> FindRows(
      const ImageDigest& digest) {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = rows_.begin(); it != rows_.end(); ++it) {
      if (it->digest != digest)
        continue;
      rows_.splice(rows_.begin(), rows_, it);
      return rows_.front().rows;
    }
    return nullptr;
  }
  void PutRows(const ImageDigest& digest,
               std::shared_ptr<const ImageRows> rows) {
    const std::size_t cost = rows->bytes();
    if (cost > kRowsBytes)
      return;
    std::list<RowsEntry> evicted;  // freed outside the lock
    const std::lock_guard<std::mutex> lock(mutex_);
    rows_.push_front({digest, std::move(rows)});
    rows_bytes_ += cost;
    while (rows_bytes_ > kRowsBytes) {
      rows_bytes_ -= rows_.back().rows->bytes();
      evicted.splice(evicted.begin(), rows_, std::prev(rows_.end()));
    }
  }

private:
  struct DecodedEntry {
    std::size_t key;
    bool low;
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
    std::shared_ptr<const DecodedImage> image;
    std::size_t cost;
  };
  struct RowsEntry {
    ImageDigest digest;
    std::shared_ptr<const ImageRows> rows;
  };
  std::mutex mutex_;
  std::list<DecodedEntry> decoded_;  // most recently used first
  std::list<RowsEntry> rows_;
  std::size_t decoded_bytes_ = 0;
  std::size_t rows_bytes_ = 0;
};

}  // namespace gufo::models::inkling2
