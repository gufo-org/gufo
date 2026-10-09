#include "src/cache/identity.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace gufo::cache {
namespace {

std::span<const std::uint8_t> PrefixInputIdentity(
    std::span<const std::uint8_t> complete,
    std::span<const InputPrefix> prefixes, Rows count) {
  for (const auto& prefix : prefixes) {
    if (count <= prefix.token_count)
      return prefix.identity;
  }
  return complete;
}
}  // namespace

InputIdentity::InputIdentity(Rows token_count, Identity complete,
                             std::vector<InputPrefix> prefixes)
    : token_count_(token_count),
      complete_(std::move(complete)),
      prefixes_(std::move(prefixes)) {
  Rows previous = 0;
  for (const auto& prefix : prefixes_) {
    if (prefix.token_count < previous || prefix.token_count > token_count_)
      throw std::invalid_argument("input identity boundaries are invalid");
    previous = prefix.token_count;
  }
}

std::span<const std::uint8_t> InputIdentity::At(Rows count) const {
  if (count > token_count_)
    throw std::invalid_argument("input identity query exceeds prompt length");
  return PrefixInputIdentity(complete_, prefixes_, count);
}

bool InputIdentity::Matches(const InputIdentity& other, Rows count) const {
  return std::ranges::equal(At(count), other.At(count));
}

}  // namespace gufo::cache
