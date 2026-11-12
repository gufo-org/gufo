#ifndef GUFO_SERVER_AUDIO_STREAM_HPP_
#define GUFO_SERVER_AUDIO_STREAM_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "src/core/base64.hpp"

namespace gufo::server {

// An empty result for nonempty input reports an invalid waveform.
inline std::string EncodePcm16(std::span<const float> samples) {
  std::string bytes;
  bytes.reserve(samples.size() * 2);
  for (const float sample : samples) {
    if (!std::isfinite(sample))
      return {};
    const auto value = static_cast<std::uint16_t>(
        static_cast<std::int16_t>(std::clamp(sample, -1.0F, 1.0F) * 32767.0F));
    bytes.push_back(static_cast<char>(value & 255));
    bytes.push_back(static_cast<char>(value >> 8));
  }
  return bytes;
}

inline std::string EncodeAudioBase64(std::string_view bytes) {
  return core::EncodeBase64(bytes);
}

// Strict canonical base64, with an allocation bound supplied by the caller.
inline bool DecodeAudioBase64(std::string_view text, std::size_t limit,
                              std::string* bytes) {
  bytes->clear();
  if (text.size() % 4 != 0 || text.size() / 4 > (limit + 2) / 3)
    return false;
  return core::DecodeBase64(text, {.max_decoded = limit}, *bytes) ==
         core::Base64Status::kOk;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_AUDIO_STREAM_HPP_
