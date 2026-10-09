#ifndef GUFO_CACHE_IDENTITY_HPP_
#define GUFO_CACHE_IDENTITY_HPP_

#include <cstdint>
#include <span>
#include <vector>

#include "src/cache/types.hpp"

namespace gufo::cache {

// Compatibility identities are canonical opaque bytes supplied by adapters.
// They cover weights, tokenizer/template, state ABI/precision, context policy,
// and all model/speculative configuration that changes restored semantics.
using Identity = std::vector<std::uint8_t>;

// Supplemental input identity for boundaries ending at or before token_count.
// Preserves ContinuationInputPrefix semantics without a serving dependency.
struct InputPrefix {
  Rows token_count{0};
  Identity identity;
};

class InputIdentity {
public:
  // Preserve Acquire's nondecreasing boundaries (first equal boundary wins),
  // and reject boundaries or queries beyond this request's prompt length.
  explicit InputIdentity(Rows token_count = 0, Identity complete = {},
                         std::vector<InputPrefix> prefixes = {});
  [[nodiscard]] std::span<const std::uint8_t> At(Rows count) const;
  [[nodiscard]] Rows TokenCount() const { return token_count_; }
  [[nodiscard]] std::span<const std::uint8_t> Complete() const {
    return complete_;
  }
  // Token prefix matching remains a separate requirement for lookup (card 05).
  [[nodiscard]] bool Matches(const InputIdentity&, Rows count) const;

private:
  Rows token_count_;
  Identity complete_;
  std::vector<InputPrefix> prefixes_;
};

}  // namespace gufo::cache

#endif
