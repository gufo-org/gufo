#ifndef GUFO_SERVER_AUDIO_STREAM_HPP_
#define GUFO_SERVER_AUDIO_STREAM_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

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
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string encoded;
  encoded.reserve((bytes.size() + 2) / 3 * 4);
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const auto a = static_cast<unsigned char>(bytes[i]);
    const auto b =
        i + 1 < bytes.size() ? static_cast<unsigned char>(bytes[i + 1]) : 0;
    const auto c =
        i + 2 < bytes.size() ? static_cast<unsigned char>(bytes[i + 2]) : 0;
    encoded.push_back(alphabet[a >> 2]);
    encoded.push_back(alphabet[((a & 3) << 4) | (b >> 4)]);
    encoded.push_back(
        i + 1 < bytes.size() ? alphabet[((b & 15) << 2) | (c >> 6)] : '=');
    encoded.push_back(i + 2 < bytes.size() ? alphabet[c & 63] : '=');
  }
  return encoded;
}

// Strict canonical base64, with an allocation bound supplied by the caller.
inline bool DecodeAudioBase64(std::string_view text, std::size_t limit,
                              std::string* bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  bytes->clear();
  if (text.size() % 4 != 0 || text.size() / 4 > (limit + 2) / 3)
    return false;
  for (std::size_t i = 0; i < text.size(); i += 4) {
    std::uint32_t word = 0;
    std::size_t padding = 0;
    for (std::size_t j = 0; j < 4; ++j) {
      const char c = text[i + j];
      if (c == '=') {
        if (j < 2 || i + 4 != text.size())
          return false;
        ++padding;
        word <<= 6;
      } else {
        const auto value = alphabet.find(c);
        if (padding != 0 || value == std::string_view::npos)
          return false;
        word = (word << 6) | static_cast<std::uint32_t>(value);
      }
    }
    if ((padding == 1 && (word & 255) != 0) ||
        (padding == 2 && (word & 65535) != 0))
      return false;
    const std::size_t count = 3 - padding;
    if (count > limit - bytes->size())
      return false;
    for (std::size_t j = 0; j < count; ++j)
      bytes->push_back(static_cast<char>((word >> (16 - j * 8)) & 255));
  }
  return true;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_AUDIO_STREAM_HPP_
