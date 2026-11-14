#include "src/models/yue2/noise.hpp"

#include <cmath>
#include <numbers>
#include <random>

namespace gufo::models::yue2 {
namespace {

void BoxMuller16(float* data) {
  for (int j = 0; j < 8; ++j) {
    const float u1 = 1.0F - data[j];
    const float u2 = data[j + 8];
    const float radius = std::sqrt(-2.0F * std::log(u1));
    const float theta = 2.0F * std::numbers::pi_v<float> * u2;
    data[j] = radius * std::cos(theta);
    data[j + 8] = radius * std::sin(theta);
  }
}

}  // namespace

std::vector<float> TorchRandn(std::size_t count, std::uint64_t seed) {
  // torch seeds its CPU MT19937 with the low 32 bits.
  std::mt19937 generator(static_cast<std::uint32_t>(seed));
  const auto uniform = [&] {
    return static_cast<float>(generator() & ((1U << 24U) - 1U)) *
           (1.0F / static_cast<float>(1U << 24U));
  };
  std::vector<float> data(count);
  for (float& value : data)
    value = uniform();
  if (count < 16) {
    // Scalar path: one Box-Muller pair per two values, cached second value.
    // Upstream noise always has at least 64 values; keep this simple.
    for (std::size_t i = 0; i + 1 < count; i += 2) {
      const float u1 = 1.0F - data[i];
      const float radius = std::sqrt(-2.0F * std::log(u1));
      const float theta = 2.0F * std::numbers::pi_v<float> * data[i + 1];
      data[i] = radius * std::cos(theta);
      data[i + 1] = radius * std::sin(theta);
    }
    return data;
  }
  for (std::size_t offset = 0; offset + 16 <= count; offset += 16)
    BoxMuller16(data.data() + offset);
  if (count % 16 != 0) {
    for (std::size_t i = count - 16; i < count; ++i)
      data[i] = uniform();
    BoxMuller16(data.data() + count - 16);
  }
  return data;
}

}  // namespace gufo::models::yue2
