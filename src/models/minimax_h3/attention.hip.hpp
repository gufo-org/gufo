#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::minimax_h3::dit_ops {

// Dense, noncausal BF16 attention, head dimension 128. Inputs are head-major;
// output is row-major, ready for the attention output projection. For >4096
// rows, packed_values must hold roundUp(rows, 32) * heads * 128 BF16 values
// disjoint from the inputs/output. Short sequences do not use this scratch.
void LaunchWmmaAttention(const std::uint16_t* query, const std::uint16_t* key,
                         const std::uint16_t* value, std::uint16_t* output,
                         std::uint16_t* packed_values, std::uint32_t rows,
                         std::uint32_t heads, hipStream_t stream);

}  // namespace gufo::minimax_h3::dit_ops
