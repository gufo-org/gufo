#include "src/core/base64.hpp"

#include <gufo/base64.h>

#include <stdexcept>

namespace gufo::core {

std::string EncodeBase64(std::span<const std::byte> bytes) {
  std::string text((bytes.size() + 2) / 3 * 4, '\0');
  if (gufo_base64_encode(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                         bytes.size(),
                         reinterpret_cast<std::uint8_t*>(text.data()),
                         text.size()) != GUFO_BASE64_OK) {
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
