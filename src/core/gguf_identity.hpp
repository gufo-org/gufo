#ifndef GUFO_CORE_GGUF_IDENTITY_HPP_
#define GUFO_CORE_GGUF_IDENTITY_HPP_

#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"

namespace gufo::core {

/// Scheme tag for full-content GGUF identities.
inline constexpr std::string_view kGgufIdentityScheme = "gguf-sha256-v1";

/// Hashes every byte of every shard, including metadata and tensor padding.
/// File digests are cached privately by device, inode, size, mtime and ctime
/// only when timestamps are outside the ambiguity window before scanning.
/// Recent files and memory images are hashed in full; stable files reuse a
/// qualified full digest. Copies have identical identities. Requires coherent
/// change timestamps and clocks; see docs/internal/cache.md#gguf-identity-cache
/// for limits. Throws on observed metadata changes during lookup; identical
/// metadata cannot establish arbitrary concurrent content changes.
[[nodiscard]] std::string GgufIdentityHex(const GgufReader& reader);

}  // namespace gufo::core

#endif  // GUFO_CORE_GGUF_IDENTITY_HPP_
