#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::minimax_h3::dit_ops {
// Exact BF16 permutation from [row][K] to [K/16][row][16]. Columns must be
// divisible by 64. The input and equally sized output must not overlap.
void LaunchPackDense(const std::uint16_t* input, std::uint16_t* output,
                     std::uint32_t rows, std::uint32_t columns,
                     hipStream_t stream);
// Restore row-major diagnostic tensors; same extent/non-overlap contract.
void LaunchUnpackDense(const std::uint16_t* input, std::uint16_t* output,
                       std::uint32_t rows, std::uint32_t columns,
                       hipStream_t stream);
// Pack the gate and up halves separately, without changing BF16 weights.
void LaunchPackFfnUpWeights(const std::uint16_t* input, std::uint16_t* output,
                            hipStream_t stream);
// H3 MLP AdaLN (modulation slots 3/4) directly in [H/16][row][16] layout.
void LaunchPackedMlpAdaLn(const std::uint16_t* input,
                          const std::uint16_t* weight,
                          const std::uint16_t* modulation,
                          const std::uint32_t* row_map, const float* inverse,
                          std::uint16_t* output, std::uint32_t rows,
                          hipStream_t stream);
// Attention AdaLN (slots 0/1), using the same packed layout.
void LaunchPackedAttentionAdaLn(const std::uint16_t* input,
                                const std::uint16_t* weight,
                                const std::uint16_t* modulation,
                                const std::uint32_t* row_map,
                                const float* inverse, std::uint16_t* output,
                                std::uint32_t rows, hipStream_t stream);
// Interleaved QKV weights and input are packed. Q/K are head-major after
// RMSNorm/RoPE; V is [head][dimension][roundUp(rows,32)], including zero tails.
// All outputs must be disjoint from one another and the packed input.
void LaunchQkv(const std::uint16_t* packed_weight,
               const std::uint16_t* packed_input,
               const std::uint16_t* query_weight,
               const std::uint16_t* key_weight, const std::uint16_t* rope_cos,
               const std::uint16_t* rope_sin, std::uint16_t* query,
               std::uint16_t* key, std::uint16_t* packed_value,
               std::uint32_t rows, hipStream_t stream);
// Both inputs are packed. The projections round to BF16 before SwiGLU;
// the output is already packed for LaunchFfnDown.
void LaunchFfnUp(const std::uint16_t* packed_weight,
                 const std::uint16_t* packed_input, std::uint16_t* output,
                 const float* silu_lookup, std::uint32_t rows,
                 hipStream_t stream);
// H3's 14336 -> 5376 feed-forward projection. Both inputs are already packed;
// SwiGLU writes packed_input directly into the existing activation buffer.
void LaunchFfnDown(const std::uint16_t* packed_weight,
                   const std::uint16_t* packed_input, std::uint16_t* output,
                   std::uint32_t rows, hipStream_t stream);
// H3's 7168 -> 5376 attention output projection, with packed operands.
void LaunchAttentionOutput(const std::uint16_t* packed_weight,
                           const std::uint16_t* packed_input,
                           std::uint16_t* output, std::uint32_t rows,
                           hipStream_t stream);
}  // namespace gufo::minimax_h3::dit_ops
