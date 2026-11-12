#ifndef GUFO_CORE_BASE64_HPP_
#define GUFO_CORE_BASE64_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>

namespace gufo::core {

// Standard base64 (RFC 4648, section 4): the alphabet A-Z a-z 0-9 + / with
// "=" padding, encoded and decoded in Rust.

[[nodiscard]] std::string EncodeBase64(std::span<const std::byte> bytes);

[[nodiscard]] inline std::string EncodeBase64(std::string_view bytes) {
  return EncodeBase64(std::as_bytes(std::span(bytes)));
}

[[nodiscard]] inline std::string EncodeBase64(
    std::span<const std::uint8_t> bytes) {
  return EncodeBase64(std::as_bytes(bytes));
}

// Why DecodeBase64 rejected a text. Quads are read in order and each quad's
// characters in order; the first problem is reported.
enum class Base64Status : std::uint8_t {
  kOk,
  // Not a whole number of four-character quads.
  kLength,
  // "=" among a quad's first two characters, or in a quad before the last.
  kPadding,
  // A character after "=".
  kAfterPadding,
  // A byte outside the alphabet.
  kCharacter,
  // canonical only: the padding hides nonzero bits.
  kNonCanonical,
  // More than max_decoded bytes, reported once every quad is read.
  kTooLong,
};

struct Base64Options {
  // Reject a text whose padding hides nonzero bits.
  bool canonical{true};
  std::size_t max_decoded{std::numeric_limits<std::size_t>::max()};
};

// Decodes `text` into the front of `out`, which holds at least
// text.size() / 4 * 3 bytes, and sets `size` to the decoded length (0 on an
// error). An empty text decodes to nothing.
[[nodiscard]] Base64Status DecodeBase64(std::string_view text,
                                        const Base64Options& options,
                                        std::span<std::byte> out,
                                        std::size_t& size);

// Decodes `text` into `bytes` (a std::string or a vector of bytes), replacing
// its contents; on an error, `bytes` is empty.
template<typename Bytes>
[[nodiscard]] Base64Status DecodeBase64(std::string_view text,
                                        const Base64Options& options,
                                        Bytes& bytes) {
  bytes.resize(text.size() / 4 * 3);
  std::size_t size = 0;
  const Base64Status status = DecodeBase64(
      text, options, std::as_writable_bytes(std::span(bytes)), size);
  bytes.resize(size);
  return status;
}

}  // namespace gufo::core

#endif  // GUFO_CORE_BASE64_HPP_
