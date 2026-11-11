#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_NGRAM_REMAINDER_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_NGRAM_REMAINDER_HPP_

#include <cstdint>
#include <limits>

namespace gufo::models::qwen38_flash_next::detail {

// The head vocabulary is a nonzero uint32_t. ceil(2^64 / d) gives a
// quotient at most one too large for every uint64_t dividend. An overshoot
// wraps the remainder above d; adding d then recovers the exact remainder.
class HeadRemainder {
public:
  void SetDivisor(std::uint32_t divisor) noexcept {
    if (divisor_ == divisor)
      return;
    divisor_ = divisor;
#if defined(__SIZEOF_INT128__)
    reciprocal_ = divisor > 1
                      ? std::numeric_limits<std::uint64_t>::max() / divisor + 1
                      : 0;
#endif
  }

  [[nodiscard]] std::uint32_t operator()(std::uint64_t value) const noexcept {
#if defined(__SIZEOF_INT128__)
    if (divisor_ == 1)
      return 0;
    const auto quotient = static_cast<std::uint64_t>(
        (static_cast<__uint128_t>(value) * reciprocal_) >> 64);
    auto remainder = value - quotient * divisor_;
    if (remainder >= divisor_)
      remainder += divisor_;
    return static_cast<std::uint32_t>(remainder);
#else
    return static_cast<std::uint32_t>(value % divisor_);
#endif
  }

private:
  std::uint32_t divisor_{1};
#if defined(__SIZEOF_INT128__)
  std::uint64_t reciprocal_{0};
#endif
};

}  // namespace gufo::models::qwen38_flash_next::detail

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_NGRAM_REMAINDER_HPP_
