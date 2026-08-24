#ifndef STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_COMPARISONS_HPP_
#define STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_COMPARISONS_HPP_

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>

namespace strix::test {

inline void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Qwen HIP test failure: " << message << '\n';
    std::abort();
  }
}

inline void ExpectNear(float expected, float actual, float tolerance,
                       std::string_view message) {
  Expect(std::isfinite(actual) && std::abs(expected - actual) <= tolerance,
         message);
}

inline void ExpectSpanNear(std::span<const float> expected,
                           std::span<const float> actual, float tolerance,
                           std::string_view message) {
  Expect(expected.size() == actual.size(), message);
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (!std::isfinite(actual[index]) ||
        std::abs(expected[index] - actual[index]) > tolerance) {
      std::cerr << "Qwen HIP span mismatch at " << index << ": expected "
                << expected[index] << ", actual " << actual[index] << '\n';
      Expect(false, message);
    }
  }
}

}  // namespace strix::test

#endif  // STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_COMPARISONS_HPP_
