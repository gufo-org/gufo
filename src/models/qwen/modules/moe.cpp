#include "src/models/qwen/modules/moe.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "src/models/qwen/forward.hpp"  // TensorGEMV
#include "src/models/qwen/modules/ffn.hpp"

namespace gufo::models::qwen {
namespace {

/// One expert matrix out of a stacked [cols, rows, experts] payload.
QwenTensorRef ExpertSlice(const QwenTensorRef& stacked, std::uint32_t expert,
                          std::size_t rows, std::size_t cols) noexcept {
  const std::size_t per_expert_elements = rows * cols;
  const std::size_t per_expert_bytes =
      gufo::quant::EncodedSizeBytes(stacked.type, per_expert_elements);
  return {.data = static_cast<const std::uint8_t*>(stacked.data) +
                  static_cast<std::size_t>(expert) * per_expert_bytes,
          .type = stacked.type,
          .num_elements = per_expert_elements,
          .available_bytes = stacked.available_bytes};
}

}  // namespace

void MoeForward(const CpuModuleContext& ctx, const MoeLayerView& view,
                std::span<const float> x, QwenScratchArena& scratch,
                std::span<float> out) noexcept {
  (void)ctx;
  const std::size_t hidden = view.hidden_size;
  const std::size_t expert_ff = view.expert_ff;
  const std::uint32_t n_experts = view.n_experts;
  const std::uint32_t n_used = view.n_used;

  // Router logits + softmax over the full expert set.
  TensorGEMV(view.router, x, n_experts, hidden, scratch.moe_router_logits);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (std::uint32_t e = 0; e < n_experts; ++e) {
    max_logit = std::max(max_logit, scratch.moe_router_logits[e]);
  }
  float denom = 0.0F;
  for (std::uint32_t e = 0; e < n_experts; ++e) {
    const float p = std::exp(scratch.moe_router_logits[e] - max_logit);
    scratch.moe_router_logits[e] = p;
    denom += p;
  }
  for (std::uint32_t e = 0; e < n_experts; ++e) {
    scratch.moe_router_logits[e] /= denom;
  }

  // Top-k by probability, lowest expert index wins ties (matches the HIP
  // kernel's shuffle reduce).
  for (std::uint32_t slot = 0; slot < n_used; ++slot) {
    float best = -1.0F;
    std::uint32_t index = n_experts;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      if (scratch.moe_router_logits[e] > best) {
        best = scratch.moe_router_logits[e];
        index = e;
      }
    }
    scratch.moe_expert_ids[slot] = index;
    scratch.moe_expert_weights[slot] = best;
    if (index < n_experts) {
      scratch.moe_router_logits[index] = -1.0F;
    }
  }
  float weight_sum = 0.0F;
  for (std::uint32_t slot = 0; slot < n_used; ++slot) {
    weight_sum += scratch.moe_expert_weights[slot];
  }
  weight_sum = std::max(weight_sum, kMoeWeightSumFloor);
  for (std::uint32_t slot = 0; slot < n_used; ++slot) {
    scratch.moe_expert_weights[slot] /= weight_sum;
  }

  std::fill_n(out.data(), hidden, 0.0F);
  for (std::uint32_t slot = 0; slot < n_used; ++slot) {
    const std::uint32_t expert = scratch.moe_expert_ids[slot];
    if (expert >= n_experts) {
      continue;
    }
    const std::size_t slot_ff = static_cast<std::size_t>(slot) * expert_ff;
    const std::size_t slot_hidden = static_cast<std::size_t>(slot) * hidden;
    auto gate = scratch.moe_expert_gate.subspan(slot_ff, expert_ff);
    auto up = scratch.moe_expert_up.subspan(slot_ff, expert_ff);
    auto act = scratch.moe_expert_act.subspan(slot_ff, expert_ff);
    TensorGEMV(ExpertSlice(view.gate_exps, expert, expert_ff, hidden), x,
               expert_ff, hidden, gate);
    TensorGEMV(ExpertSlice(view.up_exps, expert, expert_ff, hidden), x,
               expert_ff, hidden, up);
    for (std::size_t i = 0; i < expert_ff; ++i) {
      const float g = gate[i];
      act[i] = (g / (1.0F + std::exp(-g))) * up[i];
    }
    TensorGEMV(ExpertSlice(view.down_exps, expert, hidden, expert_ff), act,
               hidden, expert_ff,
               scratch.moe_expert_out.subspan(slot_hidden, hidden));
    const float w = scratch.moe_expert_weights[slot];
    for (std::size_t i = 0; i < hidden; ++i) {
      out[i] += w * scratch.moe_expert_out[slot_hidden + i];
    }
  }

  // Sigmoid-gated shared expert.
  float gate_logit = 0.0F;
  if (view.shexp_gate_inp.type == core::GgmlType::kF32) {
    const auto* gate_inp = static_cast<const float*>(view.shexp_gate_inp.data);
    for (std::size_t i = 0; i < hidden; ++i) {
      gate_logit += gate_inp[i] * x[i];
    }
  } else {
    // The MTP block of Qwen3.6-35B-A3B stores this gate quantized.
    TensorGEMV(view.shexp_gate_inp, x, 1, hidden,
               std::span<float>(&gate_logit, 1));
  }
  const float shexp_gate = 1.0F / (1.0F + std::exp(-gate_logit));
  const FfnLayerView shexp_view{view.shexp_gate, view.shexp_up, view.shexp_down,
                                hidden, view.shared_ff};
  FfnForward(ctx, shexp_view, x, scratch.moe_shexp_gate, scratch.moe_shexp_up,
             scratch.moe_shexp_act, scratch.moe_shexp_out);
  for (std::size_t i = 0; i < hidden; ++i) {
    out[i] += shexp_gate * scratch.moe_shexp_out[i];
  }
}

}  // namespace gufo::models::qwen
