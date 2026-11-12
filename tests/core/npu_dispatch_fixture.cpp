// A stand-in for an IRON design's dispatch library: two run-time scalars, a
// token id and a position. Word 1 moves with the token, word 3 with the
// position, as address words of a real stream do.
#include <cstdint>

namespace {
std::uint32_t words[4];
}

extern "C" std::int64_t dispatch_generate(std::int32_t token,
                                          std::int32_t position,
                                          std::uint32_t** out) {
  words[0] = 0x06030100u;
  words[1] = 0x1000u + static_cast<std::uint32_t>(token) * 0x200u;
  words[2] = 7u;
  words[3] = 0x80u + static_cast<std::uint32_t>(position) * 0x40u;
  *out = words;
  return 4;
}

extern "C" std::int64_t dispatch_empty(std::uint32_t** out) {
  *out = nullptr;
  return 0;
}
