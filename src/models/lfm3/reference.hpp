#ifndef GUFO_MODELS_LFM3_REFERENCE_HPP_
#define GUFO_MODELS_LFM3_REFERENCE_HPP_
#include "src/models/lfm3/weights.hpp"
namespace gufo::models::lfm3 {
// FP32 scalar oracle, built only for explicit qualification targets.
class Reference {
public:
  Reference(const Weights& weights, std::uint32_t context);
  void Reset();
  void Step(std::uint32_t token, std::span<float> logits);
  std::uint32_t Position() const { return position_; }

private:
  const Weights& w_;
  std::uint32_t context_, position_{};
  struct State {
    std::vector<float> conv, k, v;
    std::uint32_t capacity{};
  };
  std::vector<State> states_;
};
}  // namespace gufo::models::lfm3
#endif
