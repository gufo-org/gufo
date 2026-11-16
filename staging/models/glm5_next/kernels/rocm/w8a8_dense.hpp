#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_W8A8_DENSE_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_W8A8_DENSE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glm5_next::rocm {

/// Blocked W8A8 (A8 x native-K-quant weights) tensor-core tier for the dense
/// projections of the wide-chunk dispatch, enabled by default and selected in
/// `Executor::Create`. It reuses the shared Qwen prefill kernels: the FP32
/// activation rows are quantized once per chunk into the tiled Q8_1 layout
/// and consumed natively with the kernel's per-K32 INT8 scales, so the
/// weight matrices stay in their GGUF encoding. Accumulation order differs
/// from both the scalar dot and the F16 DenseWmma tier; the quality contract
/// lives in the dense_wmma/w8a8 tests, not bit-equality.

/// True when `type` has a blocked W8A8 tier (Q5_K, Q6_K and Q8_0; the same
/// dense set as `DenseWmmaSupports`).
bool Glm5W8A8Supported(core::GgmlType type);

/// Size in bytes of the activation scratch `Glm5W8A8Dense` needs for
/// `tokens` rows of `k` elements (tiled Q8_1 blocks plus the activation-sum
/// sidecar).
[[nodiscard]] std::size_t Glm5W8A8ScratchBytes(std::size_t tokens,
                                               std::size_t k);

/// out[t*m + r] = sum_k W[r][k] * X[t][k] for the dense W8A8 set. `x` holds
/// `tokens` contiguous rows of stride `k` (the caller's gate); `act_scratch`
/// must provide `Glm5W8A8ScratchBytes(tokens, k)` bytes. Requires
/// k % 32 == 0 and 1 <= tokens.
void Glm5W8A8Dense(const void* w, core::GgmlType type, const float* x,
                   void* act_scratch, float* y, std::size_t tokens,
                   std::size_t m, std::size_t k, hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_W8A8_DENSE_HPP_
