#include "src/core/base64.hpp"

#include <gufo/base64.h>

#include <limits>
#include <stdexcept>

namespace gufo::core {
namespace {

template<typename String, typename Write>
void WriteString(String& text, std::size_t size, Write write) {
  if constexpr (requires { text.resize_and_overwrite(size, write); }) {
    text.resize_and_overwrite(size, write);
  } else if constexpr (requires { text.__resize_and_overwrite(size, write); }) {
    // libstdc++ exposes the equivalent operation in C++20 mode.
    text.__resize_and_overwrite(size, write);
  } else {
    text.resize(size);
    text.resize(write(text.data(), size));
  }
}

}  // namespace

std::string EncodeBase64(std::span<const std::byte> bytes) {
  const std::size_t quads = bytes.size() / 3 + (bytes.size() % 3 != 0);
  if (quads > std::numeric_limits<std::size_t>::max() / 4)
    throw std::length_error("base64 output length overflow");
  const std::size_t size = quads * 4;
  std::string text;
  bool encoded = false;
  WriteString(text, size, [&](char* out, std::size_t) noexcept {
    encoded =
        gufo_base64_encode(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                           bytes.size(), reinterpret_cast<std::uint8_t*>(out),
                           size) == GUFO_BASE64_OK;
    return encoded ? size : 0;
  });
  if (!encoded) {
    throw std::logic_error("Rust base64 encoder failed internally");
  }
  return text;
}

Base64Status DecodeBase64(std::string_view text, const Base64Options& options,
                          std::span<std::byte> out, std::size_t& size) {
  switch (gufo_base64_decode(
      reinterpret_cast<const std::uint8_t*>(text.data()), text.size(),
      options.canonical, options.max_decoded,
      reinterpret_cast<std::uint8_t*>(out.data()), out.size(), &size)) {
    case GUFO_BASE64_OK:
      return Base64Status::kOk;
    case GUFO_BASE64_LENGTH:
      return Base64Status::kLength;
    case GUFO_BASE64_PADDING:
      return Base64Status::kPadding;
    case GUFO_BASE64_AFTER_PADDING:
      return Base64Status::kAfterPadding;
    case GUFO_BASE64_CHARACTER:
      return Base64Status::kCharacter;
    case GUFO_BASE64_NON_CANONICAL:
      return Base64Status::kNonCanonical;
    case GUFO_BASE64_TOO_LONG:
      return Base64Status::kTooLong;
    case GUFO_BASE64_INTERNAL:
      break;
  }
  throw std::logic_error("Rust base64 decoder failed internally");
}

}  // namespace gufo::core
