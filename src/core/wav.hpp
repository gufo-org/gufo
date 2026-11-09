#ifndef GUFO_CORE_WAV_HPP_
#define GUFO_CORE_WAV_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace gufo::core {

// What a caller accepts. Integer PCM is decoded from 16 bits up to
// max_pcm_bits (16, 24 or 32); 32-bit float always.
struct WavLimits {
  std::uint32_t max_channels{8};
  std::uint32_t max_sample_rate{192000};
  std::uint64_t max_seconds{30};
  std::uint32_t max_pcm_bits{16};
};

// The message each rejection reports, so every caller keeps its own wording.
struct WavMessages {
  std::string_view not_riff;
  std::string_view truncated;
  std::string_view invalid_fmt;
  std::string_view unsupported_format;
  std::string_view unsupported_layout;
  std::string_view too_long;
  std::string_view non_finite;
};

struct WavAudio {
  std::uint32_t sample_rate{0};
  std::uint32_t channels{0};
  // Interleaved float samples [-1.0, 1.0].
  std::vector<float> samples;
};

// Decodes a RIFF WAV with interleaved integer PCM or 32-bit float samples.
// Integer PCM is divided by 2^(bits - 1); floats must be finite and are
// clamped to [-1, 1]. Throws std::length_error with messages.too_long for
// audio without frames or longer than limits.max_seconds, and
// std::invalid_argument with the matching message for every other rejection.
[[nodiscard]] WavAudio DecodeWav(std::span<const std::byte> bytes,
                                 const WavLimits& limits,
                                 const WavMessages& messages);

}  // namespace gufo::core

#endif  // GUFO_CORE_WAV_HPP_
