#ifndef GUFO_CORE_JSON_NUMBER_LIMITS_HPP_
#define GUFO_CORE_JSON_NUMBER_LIMITS_HPP_

namespace gufo::sampling {

// llama.cpp's integral-part and decimal-part: at most 16 digits each, so a
// number cannot grow until the token limit.
inline constexpr int kMaxJsonNumberDigits = 16;

}  // namespace gufo::sampling

#endif  // GUFO_CORE_JSON_NUMBER_LIMITS_HPP_
