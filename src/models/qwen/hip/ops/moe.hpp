#ifndef GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Softmax over the full router logits, then top-k selection with lowest-index
/// tie-break and renormalized weights (sum floored at 2^-14). logits is
/// [n_tokens, logits_stride] with n_experts valid entries per row; ids and
/// weights are [n_tokens, n_used].
void LaunchMoeRouterTopK(const float* logits, std::uint32_t logits_stride,
                         std::int32_t* ids, float* weights,
                         std::uint32_t n_tokens, std::uint32_t n_experts,
                         std::uint32_t n_used, hipStream_t stream = nullptr);

/// MoE combine: out[t] = sum_s(weights[t,s] * expert_out[(t*k+s), :]) +
/// sigmoid(shared_gate[t]) * shared_out[t]. All buffers fp32.
void LaunchMoeEpilogue(const float* expert_out, const float* weights,
                       const float* shared_out, const float* shared_gate,
                       float* out, std::uint32_t n_tokens, std::uint32_t n_used,
                       std::uint32_t dim, hipStream_t stream = nullptr);

/// Routed per-slot GEMV fallback for formats the MMQ MoE kernels do not cover
/// (BF16/F32/Q6_K and friends, decoded via DecodeQuantSub16). For each slot s
/// the expert index is ids[s]; out[s, :] = W_e * x_row. With x_per_slot the
/// input row is s itself (down projection reading per-slot activations);
/// otherwise it is s / x_token_divisor (gate/up sharing the token's row).
void LaunchMoeSlotGemv(const void* w, core::GgmlType type, const float* x,
                       const std::int32_t* ids, float* out, std::size_t m,
                       std::size_t k, std::uint32_t slots,
                       std::uint32_t x_token_divisor,
                       hipStream_t stream = nullptr);

/// Fused routed SwiGLU fallback: act[s, :] = SiLU(gate_e * x) * (up_e * x)
/// for the slot's expert e = ids[s]; x is the shared token row (batch one) or
/// row s / x_token_divisor.
void LaunchMoeSlotSwigluGemv(const void* gate_w, core::GgmlType gate_type,
                             const void* up_w, core::GgmlType up_type,
                             const float* x, const std::int32_t* ids,
                             float* act, std::size_t expert_ff,
                             std::size_t hidden, std::uint32_t slots,
                             std::uint32_t x_token_divisor,
                             hipStream_t stream = nullptr);

/// Scratch for the grouped BF16 expert path. Every buffer is device memory;
/// sorted_slots and tiles hold at least `slots` entries, expert_bounds at
/// least n_experts + 1.
struct MoeGroupedScratch {
  std::int32_t* sorted_slots;
  std::int32_t* tiles;
  std::int32_t* expert_bounds;
};

/// Whether LaunchMoeGroupedBf16Gemm supports an [m, k] expert matrix with
/// n_experts experts.
[[nodiscard]] bool IsMoeGroupedBf16GemmSupported(
    std::size_t m, std::size_t k, std::uint32_t n_experts) noexcept;

/// Groups the routed slots by expert for LaunchMoeGroupedBf16Gemm. Run once
/// per ids array; the grouping is valid for every projection that shares it.
void LaunchMoeGroupSlots(const std::int32_t* ids, std::uint32_t slots,
                         std::uint32_t n_experts,
                         const MoeGroupedScratch& scratch,
                         hipStream_t stream = nullptr);

/// Batched prefill expert projection for BF16 expert weights stacked
/// [experts, m, k]: out[s, :] = W_{ids[s]} * x_row(s), with the same row
/// selection as LaunchMoeSlotGemv. Each expert's slots share one pass over its
/// weights on the WMMA matrix cores; activations are rounded to BF16 and
/// accumulated in FP32. Requires LaunchMoeGroupSlots on the same ids and
/// n_experts first, and IsMoeGroupedBf16GemmSupported(m, k, n_experts).
void LaunchMoeGroupedBf16Gemm(const void* w, const float* x, float* out,
                              std::size_t m, std::size_t k, std::uint32_t slots,
                              std::uint32_t n_experts,
                              std::uint32_t x_token_divisor,
                              const MoeGroupedScratch& scratch,
                              hipStream_t stream = nullptr);

/// Dense prefill GEMM on the same WMMA kernel: out[b, :] = W * x[b, :] for a
/// BF16 [m, k] matrix, activations rounded to BF16 and accumulated in FP32.
/// Requires IsMoeGroupedBf16GemmSupported(m, k, 1). Unlike the exact BF16
/// prefill route it does not reproduce decode's FP32-activation arithmetic.
void LaunchBf16WmmaGemm(const void* w, const float* x, float* out,
                        std::size_t batch_size, std::size_t m, std::size_t k,
                        hipStream_t stream = nullptr);

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
