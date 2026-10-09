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

// Prefixes must be in strictly increasing order. Complete applies after the
// last prefix; equality at a boundary uses that prefix's earlier identity.
[[nodiscard]] std::span<const std::uint8_t> PrefixInputIdentity(
    std::span<const std::uint8_t> complete,
    std::span<const InputPrefix> prefixes, Rows count);

class InputIdentity {
public:
  InputIdentity(Identity complete = {}, std::vector<InputPrefix> prefixes = {});
  [[nodiscard]] std::span<const std::uint8_t> At(Rows count) const;
  // Token prefix matching remains a separate requirement for lookup (card 05).
  [[nodiscard]] bool Matches(const InputIdentity&, Rows count) const;

private:
  Identity complete_;
  std::vector<InputPrefix> prefixes_;
};

}  // namespace gufo::cache

#endif
