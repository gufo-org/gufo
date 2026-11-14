#include "src/cli/serve/music_wav.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace gufo::server {
namespace {

void Append(std::string* output, std::uint32_t value, unsigned bytes) {
  for (unsigned i = 0; i < bytes; ++i)
    output->push_back(static_cast<char>((value >> (8U * i)) & 0xFFU));
}

}  // namespace

std::string EncodeMusicWav(const ext::MusicAudio& audio, bool float32) {
  const std::uint32_t channels = audio.channels;
  const std::uint32_t width = float32 ? 4 : 2;
  if (channels == 0 || channels > 8 || audio.sample_rate == 0 ||
      audio.samples.size() % channels != 0 ||
      audio.samples.size() >
          (std::numeric_limits<std::uint32_t>::max() - 64) / width)
    return {};
  const auto data_bytes =
      static_cast<std::uint32_t>(audio.samples.size() * width);
  std::string output;
  output.reserve(44U + data_bytes);
  output.append("RIFF", 4);
  Append(&output, 36U + data_bytes, 4);
  output.append("WAVEfmt ", 8);
  Append(&output, 16, 4);
  Append(&output, float32 ? 3 : 1, 2);
  Append(&output, channels, 2);
  Append(&output, audio.sample_rate, 4);
  Append(&output, audio.sample_rate * channels * width, 4);
  Append(&output, channels * width, 2);
  Append(&output, width * 8, 2);
  output.append("data", 4);
  Append(&output, data_bytes, 4);
  for (const float sample : audio.samples) {
    if (!std::isfinite(sample))
      return {};
    const float bounded = std::clamp(sample, -1.0F, 1.0F);
    if (float32) {
      Append(&output, std::bit_cast<std::uint32_t>(bounded), 4);
    } else {
      const auto pcm =
          static_cast<std::int16_t>(std::lround(bounded * 32767.0F));
      Append(&output, static_cast<std::uint16_t>(pcm), 2);
    }
  }
  return output;
}

}  // namespace gufo::server
