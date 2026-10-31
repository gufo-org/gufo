#ifndef GUFO_MODELS_QWEN_VISION_ROPE_HPP_
#define GUFO_MODELS_QWEN_VISION_ROPE_HPP_
#include <cstdint>
namespace gufo::models::qwen::vision {
struct DeviceRope {
  const std::int32_t* positions{
      nullptr};  ///< interleaved [physical token][T,H,W]
  std::uint32_t prefix_length{0};
  std::int32_t delta{0};
};

#if defined(__HIPCC__)
/// [11,11,10] interleaved mRoPE, shared by both Qwen3.8 language trunks.
/// Physical positions continue to index KV, SSM, PLE and causal masks.
__device__ inline float RopePosition(const DeviceRope* rope,
                                     std::uint32_t physical,
                                     std::uint32_t pair) {
  if (rope == nullptr)
    return static_cast<float>(physical);
  if (physical >= rope->prefix_length) {
    return static_cast<float>(static_cast<std::int64_t>(physical) +
                              rope->delta);
  }
  const auto axis = pair % 3 == 1 && pair < 33   ? 1
                    : pair % 3 == 2 && pair < 30 ? 2
                                                 : 0;
  return static_cast<float>(rope->positions[physical * 3 + axis]);
}
#endif

}  // namespace gufo::models::qwen::vision
#endif
