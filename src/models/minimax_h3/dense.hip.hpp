#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::minimax_h3::dit_ops {
// Exact BF16 permutation from [row][K] to [K/16][row][16]. Columns must be
// divisible by 64. The input and equally sized output must not overlap.
void LaunchPackDense(const std::uint16_t* input, std::uint16_t* output,
                     std::uint32_t rows, std::uint32_t columns,
                     hipStream_t stream);
// H3's 14336 -> 5376 feed-forward projection. Both inputs are already packed;
// SwiGLU writes packed_input directly into the existing activation buffer.
void LaunchFfnDown(const std::uint16_t* packed_weight,
                   const std::uint16_t* packed_input, std::uint16_t* output,
                   std::uint32_t rows, hipStream_t stream);
}  // namespace gufo::minimax_h3::dit_ops
