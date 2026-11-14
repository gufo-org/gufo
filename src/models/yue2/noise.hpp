#ifndef GUFO_MODELS_YUE2_NOISE_HPP_
#define GUFO_MODELS_YUE2_NOISE_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gufo::models::yue2 {

/// `torch.randn(count, generator=torch.Generator("cpu").manual_seed(seed))`:
/// the same MT19937 uniforms and 16-wide Box-Muller layout, so the acoustic
/// noise matches upstream within an ulp of its transcendental functions.
[[nodiscard]] std::vector<float> TorchRandn(std::size_t count,
                                            std::uint64_t seed);

}  // namespace gufo::models::yue2

#endif  // GUFO_MODELS_YUE2_NOISE_HPP_
