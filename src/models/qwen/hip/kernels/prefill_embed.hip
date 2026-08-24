#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace strix::hip {

// Quant block layouts (Q8_0 / Q5_K / Q6_K / Q8_K) and the quant helpers are
// shared via src/models/qwen/hip/quant_ops.hpp.
__global__ void BatchedEmbeddingLookupKernel(
    const void* __restrict__ table, core::GgmlType type,
    const std::uint32_t* __restrict__ token_ids, float* __restrict__ out_hidden,
    std::size_t batch_size, std::size_t hidden_size) {
  const std::size_t b = blockIdx.y;
  if (b >= batch_size)
    return;
  const std::size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= hidden_size)
    return;

  const std::uint32_t tok = token_ids[b];
  float* out_row = out_hidden + (b * hidden_size);
  const std::size_t row_off = static_cast<std::size_t>(tok) * hidden_size;

  if (type == core::GgmlType::kBF16) {
    const auto* table_bf16 = static_cast<const hip_bfloat16*>(table);
    out_row[tid] = static_cast<float>(table_bf16[row_off + tid]);
  } else if (type == core::GgmlType::kF32) {
    const auto* table_f32 = static_cast<const float*>(table);
    out_row[tid] = table_f32[row_off + tid];
  } else {
    const auto* q8 = static_cast<const Q8_0Block*>(table);
    const Q8_0Block& wblk =
        q8[row_off / kQ8_0BlockSize + tid / kQ8_0BlockSize];
    out_row[tid] = __half2float(wblk.d) *
                   static_cast<float>(wblk.qs[tid % kQ8_0BlockSize]);
  }
}

void LaunchBatchedEmbeddingLookup(const void* table, core::GgmlType type,
                                  const std::uint32_t* token_ids,
                                  float* out_hidden, std::size_t batch_size,
                                  std::size_t hidden_size, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_x = (hidden_size + block_size - 1) / block_size;
  const dim3 grid(static_cast<unsigned int>(grid_x),
                  static_cast<unsigned int>(batch_size));
  BatchedEmbeddingLookupKernel<<<grid, static_cast<unsigned int>(block_size), 0,
                                 stream>>>(table, type, token_ids, out_hidden,
                                           batch_size, hidden_size);
}
}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
