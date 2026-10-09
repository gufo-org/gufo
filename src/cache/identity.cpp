#include "src/cache/identity.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace gufo::cache {

std::span<const std::uint8_t> PrefixInputIdentity(
    std::span<const std::uint8_t> complete,
    std::span<const InputPrefix> prefixes, Rows count) {
  for (const auto& prefix : prefixes) {
    if (count <= prefix.token_count)
      return prefix.identity;
  }
  return complete;
}

InputIdentity::InputIdentity(Identity complete,
                             std::vector<InputPrefix> prefixes)
    : complete_(std::move(complete)), prefixes_(std::move(prefixes)) {
  for (std::size_t i = 1; i < prefixes_.size(); ++i) {
    if (prefixes_[i - 1].token_count >= prefixes_[i].token_count)
      throw std::invalid_argument(
          "input boundaries must be strictly increasing");
  }
}

std::span<const std::uint8_t> InputIdentity::At(Rows count) const {
  return PrefixInputIdentity(complete_, prefixes_, count);
}

bool InputIdentity::Matches(const InputIdentity& other, Rows count) const {
  return std::ranges::equal(At(count), other.At(count));
}

}  // namespace gufo::cache
