#ifndef GUFO_MODELS_QWEN_MODULES_MOE_HPP_
#define GUFO_MODELS_QWEN_MODULES_MOE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::models::qwen {

/// Routed-expert MoE layer slice (qwen35moe): softmax router with top-k
/// renormalized weights over routed experts, plus a sigmoid-gated shared
/// expert. The expert tensors keep the full [cols, rows, experts] payload;
/// one expert matrix is `rows * cols` elements.
struct MoeLayerView {
  QwenTensorRef router;          // [hidden x n_experts] F32
  QwenTensorRef shexp_gate_inp;  // [hidden] F32
  QwenTensorRef gate_exps;       // [hidden x expert_ff x n_experts]
  QwenTensorRef up_exps;         // [hidden x expert_ff x n_experts]
  QwenTensorRef down_exps;       // [expert_ff x hidden x n_experts]
  QwenTensorRef shexp_gate;      // [hidden x shared_ff]
  QwenTensorRef shexp_up;        // [hidden x shared_ff]
  QwenTensorRef shexp_down;      // [shared_ff x hidden]
  std::uint32_t n_experts = 0;
  std::uint32_t n_used = 0;
  std::size_t hidden_size = 0;
  std::size_t expert_ff = 0;
  std::size_t shared_ff = 0;
};

inline MoeLayerView MakeMoeView(const QwenLayerWeights& w,
                                const core::ModelConfig& c) {
  return MoeLayerView{w.ffn_gate_inp,
                      w.ffn_gate_inp_shexp,
                      w.ffn_gate_exps,
                      w.ffn_up_exps,
                      w.ffn_down_exps,
                      w.ffn_gate_shexp,
                      w.ffn_up_shexp,
                      w.ffn_down_shexp,
                      c.expert_count,
                      c.expert_used_count,
                      c.hidden_size,
                      c.expert_ff_length,
                      c.expert_shared_ff_length};
}

/// Minimum routed weight sum; matches the HIP top-k kernel so the CPU
/// reference and the GPU pick identical weights.
inline constexpr float kMoeWeightSumFloor = 6.103515625e-5F;

/// Single-token MoE FFN: out = sum_i(w_i * expert_i(x)) +
/// sigmoid(gate_inp_shexp . x) * shexp(x). The scratch spans come from
/// QwenScratchArena::moe_*.
void MoeForward(const CpuModuleContext& ctx, const MoeLayerView& view,
                std::span<const float> x, QwenScratchArena& scratch,
                std::span<float> out) noexcept;

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_MODULES_MOE_HPP_
