// Frozen copies of the base64 code in 0.33.0, which base64_test compares
// with core::EncodeBase64 and core::DecodeBase64 under each caller's rules.
// Do not change them to match new behavior.
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::base64_reference {
namespace {
constexpr std::size_t kMaxEncodedImageBytes = 20 * 1024 * 1024;
constexpr std::size_t kMaximumEncodedBytes = 12U << 20U;
constexpr std::string_view kBase64 =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// src/core/image.cpp
std::vector<std::uint8_t> DecodeBase64(std::string_view input) {
  if (input.empty() || input.size() % 4 != 0 ||
      input.size() > ((kMaxEncodedImageBytes + 2) / 3) * 4) {
    throw std::invalid_argument("invalid or oversized base64 image");
  }
  std::vector<std::uint8_t> output;
  output.reserve(input.size() / 4 * 3);
  for (std::size_t i = 0; i < input.size(); i += 4) {
    std::uint32_t bits = 0;
    unsigned padding = 0;
    for (unsigned j = 0; j < 4; ++j) {
      const char c = input[i + j];
      if (c == '=') {
        if (j < 2 || i + 4 != input.size()) {
          throw std::invalid_argument("invalid base64 image padding");
        }
        ++padding;
        bits <<= 6;
      } else {
        const auto digit = kBase64.find(c);
        if (digit == std::string_view::npos || padding != 0) {
          throw std::invalid_argument("invalid base64 image character");
        }
        bits = (bits << 6) | static_cast<std::uint32_t>(digit);
      }
    }
    if ((padding == 1 && (bits & 0xff) != 0) ||
        (padding == 2 && (bits & 0xffff) != 0)) {
      throw std::invalid_argument("noncanonical base64 image");
    }
    output.push_back(static_cast<std::uint8_t>(bits >> 16));
    if (padding < 2)
      output.push_back(static_cast<std::uint8_t>(bits >> 8));
    if (padding == 0)
      output.push_back(static_cast<std::uint8_t>(bits));
  }
  if (output.size() > kMaxEncodedImageBytes) {
    throw std::invalid_argument("image exceeds encoded byte limit");
  }
  return output;
}
}  // namespace

// src/cli/serve/embeddings_api.cpp
std::string EncodeEmbeddings(std::span<const std::uint8_t> bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve((bytes.size() + 2U) / 3U * 4U);
  for (std::size_t i = 0; i < bytes.size(); i += 3U) {
    const std::uint32_t word =
        (std::uint32_t{bytes[i]} << 16U) |
        (i + 1U < bytes.size() ? std::uint32_t{bytes[i + 1U]} << 8U : 0U) |
        (i + 2U < bytes.size() ? std::uint32_t{bytes[i + 2U]} : 0U);
    output += alphabet[(word >> 18U) & 63U];
    output += alphabet[(word >> 12U) & 63U];
    output += i + 1U < bytes.size() ? alphabet[(word >> 6U) & 63U] : '=';
    output += i + 2U < bytes.size() ? alphabet[word & 63U] : '=';
  }
  return output;
}

// src/cli/serve/image_api.cpp
std::string EncodeImageApi(std::span<const std::uint8_t> bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve((bytes.size() + 2) / 3 * 4);
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const std::uint32_t word =
        (std::uint32_t(bytes[i]) << 16) |
        (i + 1 < bytes.size() ? std::uint32_t(bytes[i + 1]) << 8 : 0) |
        (i + 2 < bytes.size() ? bytes[i + 2] : 0);
    output += alphabet[(word >> 18) & 63];
    output += alphabet[(word >> 12) & 63];
    output += i + 1 < bytes.size() ? alphabet[(word >> 6) & 63] : '=';
    output += i + 2 < bytes.size() ? alphabet[word & 63] : '=';
  }
  return output;
}

// src/core/image.cpp, ReadMediaUrl: a data URL's payload with
// budget.remaining_bytes left.
std::vector<std::uint8_t> DecodeImageDataUrl(std::string_view encoded,
                                             std::size_t remaining_bytes) {
  const auto padding = encoded.ends_with("==")  ? 2U
                       : encoded.ends_with("=") ? 1U
                                                : 0U;
  if (encoded.size() / 4 * 3 < padding ||
      encoded.size() / 4 * 3 - padding > remaining_bytes)
    throw std::invalid_argument("request image byte budget exceeded");
  return DecodeBase64(encoded);
}

// src/cli/serve/audio_buffer.cpp
int Base64Value(unsigned char character) {
  if (character >= 'A' && character <= 'Z') {
    return character - 'A';
  }
  if (character >= 'a' && character <= 'z') {
    return character - 'a' + 26;
  }
  if (character >= '0' && character <= '9') {
    return character - '0' + 52;
  }
  if (character == '+') {
    return 62;
  }
  if (character == '/') {
    return 63;
  }
  return -1;
}
std::vector<std::byte> DecodeReferenceAudio(std::string_view encoded) {
  constexpr std::string_view prefix = "data:audio/wav;base64,";
  if (encoded.starts_with(prefix)) {
    encoded.remove_prefix(prefix.size());
  }
  if (encoded.empty() || encoded.size() > kMaximumEncodedBytes ||
      encoded.size() % 4 != 0) {
    throw std::invalid_argument("reference audio base64 is invalid");
  }
  std::vector<std::byte> output;
  output.reserve((encoded.size() / 4) * 3);
  for (std::size_t offset = 0; offset < encoded.size(); offset += 4) {
    std::uint32_t block = 0;
    unsigned padding = 0;
    for (unsigned index = 0; index < 4; ++index) {
      const unsigned char character =
          static_cast<unsigned char>(encoded[offset + index]);
      if (character == '=') {
        if (index < 2 || offset + 4 != encoded.size()) {
          throw std::invalid_argument(
              "reference audio base64 padding is invalid");
        }
        ++padding;
        block <<= 6U;
      } else {
        if (padding != 0) {
          throw std::invalid_argument(
              "reference audio base64 padding is invalid");
        }
        const int value = Base64Value(character);
        if (value < 0) {
          throw std::invalid_argument("reference audio base64 is invalid");
        }
        block = (block << 6U) | static_cast<std::uint32_t>(value);
      }
    }
    output.push_back(static_cast<std::byte>((block >> 16U) & 0xFFU));
    if (padding < 2) {
      output.push_back(static_cast<std::byte>((block >> 8U) & 0xFFU));
    }
    if (padding == 0) {
      output.push_back(static_cast<std::byte>(block & 0xFFU));
    }
  }
  return output;
}

// src/cli/serve/audio_stream.hpp
std::string EncodeAudioBase64(std::string_view bytes) {
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
bool DecodeAudioBase64(std::string_view text, std::size_t limit,
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

}  // namespace gufo::base64_reference
