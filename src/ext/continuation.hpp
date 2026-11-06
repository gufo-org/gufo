#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace gufo::ext {

/// Scheme tag for full-content GGUF identities.
inline constexpr std::string_view kGgufIdentityScheme = "gguf-sha256-v1";

/// Token identifier shared by runners and the continuation cache.
using ContinuationToken = std::uint32_t;

/// Supplemental input identity for prefixes ending at or before token_count.
/// Ordered boundaries allow a checkpoint before a new image to retain its
/// original identity. The complete request identity applies after the last one.
struct ContinuationInputPrefix {
  std::size_t token_count{0};
  std::vector<std::uint8_t> identity;
};

[[nodiscard]] inline std::span<const std::uint8_t> PrefixInputIdentity(
    std::span<const std::uint8_t> complete,
    std::span<const ContinuationInputPrefix> prefixes, std::size_t count) {
  for (const auto& prefix : prefixes) {
    if (count <= prefix.token_count)
      return prefix.identity;
  }
  return complete;
}

/// Model-private continuation state retained by the continuation cache.
///
/// Implementations own all attention, recurrent, position, and graph-bound
/// state. The cache deliberately cannot inspect or copy those bytes.
class ContinuationState {
public:
  ContinuationState() = default;
  virtual ~ContinuationState() = default;

  ContinuationState(const ContinuationState&) = delete;
  ContinuationState& operator=(const ContinuationState&) = delete;
  ContinuationState(ContinuationState&&) = delete;
  ContinuationState& operator=(ContinuationState&&) = delete;

  /// Discards any partially or fully computed continuation.
  virtual void Invalidate() noexcept = 0;
};

}  // namespace gufo::ext
