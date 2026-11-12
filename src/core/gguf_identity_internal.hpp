#ifndef GUFO_CORE_GGUF_IDENTITY_INTERNAL_HPP_
#define GUFO_CORE_GGUF_IDENTITY_INTERNAL_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/core/platform/fs.hpp"

namespace gufo::core::detail {

// Private dependency injection for deterministic tests of the production
// lookup/hash/publication path. Empty callbacks use the real clock and stat.
// File metadata overrides apply only to the open GGUF, never cache security
// checks. Hash reads always read the complete real content; the observer runs
// synchronously after each chunk has been hashed, before the final stat.
struct GgufIdentityHooks {
  std::function<platform::FileInfo(platform::PlatformFile)> file_info;
  std::function<std::int64_t()> now_ns;
  std::function<void(std::size_t)> on_hash_read;
};

[[nodiscard]] std::string GgufIdentityHex(const GgufReader& reader,
                                          const GgufIdentityHooks& hooks);

}  // namespace gufo::core::detail

#endif
