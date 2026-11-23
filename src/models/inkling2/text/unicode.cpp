#include "src/models/inkling2/text/unicode.hpp"

#include <array>

namespace inkling2::text::unicode {

namespace {
const std::array<std::uint8_t, 256>& Latin1Table() {
  static const auto table = [] {
    std::array<std::uint8_t, 256> t{};
    for (std::size_t i = 0; i < kRangeCount && kRanges[i].first < 256; ++i)
      for (std::uint32_t c = kRanges[i].first; c <= kRanges[i].last && c < 256;
           ++c)
        t[c] = kRanges[i].code;
    return t;
  }();
  return table;
}
}  // namespace

std::uint8_t Lookup(char32_t cp) noexcept {
  if (cp < 256)
    return Latin1Table()[cp];
  if (cp >= 0x110000)
    return 0;
  std::size_t lo = 0, hi = kRangeCount;
  while (lo < hi) {
    const std::size_t mid = (lo + hi) / 2;
    if (cp > kRanges[mid].last) {
      lo = mid + 1;
    } else if (cp < kRanges[mid].first) {
      hi = mid;
    } else {
      return kRanges[mid].code;
    }
  }
  return 0;
}

Decoded DecodeUtf8(std::string_view s, std::size_t pos) noexcept {
  const auto b0 = static_cast<std::uint8_t>(s[pos]);
  if (b0 < 0x80)
    return {b0, 1};
  const auto bad = Decoded{static_cast<char32_t>(0x110000 + b0), 1};
  std::size_t need;
  char32_t cp;
  char32_t min;
  if ((b0 & 0xE0) == 0xC0) {
    need = 1;
    cp = b0 & 0x1F;
    min = 0x80;
  } else if ((b0 & 0xF0) == 0xE0) {
    need = 2;
    cp = b0 & 0x0F;
    min = 0x800;
  } else if ((b0 & 0xF8) == 0xF0) {
    need = 3;
    cp = b0 & 0x07;
    min = 0x10000;
  } else
    return bad;
  for (std::size_t i = 1; i <= need; ++i) {
    if (pos + i >= s.size())
      return bad;
    const auto b = static_cast<std::uint8_t>(s[pos + i]);
    if ((b & 0xC0) != 0x80)
      return bad;
    cp = (cp << 6) | (b & 0x3F);
  }
  if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
    return bad;
  return {cp, static_cast<std::uint8_t>(need + 1)};
}

}  // namespace inkling2::text::unicode
