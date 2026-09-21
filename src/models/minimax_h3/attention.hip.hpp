#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::minimax_h3::dit_ops {

// Dense, noncausal BF16 attention, head dimension 128. Inputs are head-major;
// output is row-major, ready for the attention output projection.
void LaunchWmmaAttention(const std::uint16_t* query, const std::uint16_t* key,
                         const std::uint16_t* value, std::uint16_t* output,
                         std::uint32_t rows, std::uint32_t heads,
                         hipStream_t stream);

}  // namespace gufo::minimax_h3::dit_ops
