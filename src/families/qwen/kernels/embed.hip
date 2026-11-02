#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/kernels/ops/token.hpp"
#include "src/families/qwen/hip_common/quant_ops.hpp"

namespace gufo::hip {

__global__ void EmbeddingLookupKernel(const void* __restrict__ table,
                                      core::GgmlType type,
                                      std::uint32_t token_id,
                                      float* __restrict__ out_hidden,
                                      std::size_t hidden_size) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hidden_size)
    return;

  const std::size_t offset =
      static_cast<std::size_t>(token_id) * hidden_size + idx;
  if (type == core::GgmlType::kBF16) {
    const auto* bf_ptr = static_cast<const hip_bfloat16*>(table);
    out_hidden[idx] = static_cast<float>(bf_ptr[offset]);
  } else if (type == core::GgmlType::kF32) {
    const auto* f_ptr = static_cast<const float*>(table);
    out_hidden[idx] = f_ptr[offset];
  } else if (type != core::GgmlType::kQ8_0) {
    // opt-q4kxl: the UD-Q4_K_XL shard stores token_embd.weight as Q4_K, so the
    // lookup decodes whichever block format the row actually uses instead of
    // reinterpreting it as Q8_0.
    const auto* row =
        static_cast<const std::uint8_t*>(table) +
        (static_cast<std::size_t>(token_id) * QuantRowBytes(type, hidden_size));
    out_hidden[idx] = QuantBlockElement(type, row, idx);
  } else {
    const auto* q8 = static_cast<const Q8_0Block*>(table);
    const Q8_0Block& wblk =
        q8[static_cast<std::size_t>(token_id) * (hidden_size / kQ8_0BlockSize) +
           idx / kQ8_0BlockSize];
    out_hidden[idx] = __half2float(wblk.d) *
                      static_cast<float>(wblk.qs[idx % kQ8_0BlockSize]);
  }
}

__global__ void EmbeddingLookupPtrKernel(
    const void* __restrict__ table, core::GgmlType type,
    const std::uint32_t* __restrict__ d_token_id,
    float* __restrict__ out_hidden, std::size_t hidden_size) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hidden_size)
    return;

  const std::uint32_t token_id = *d_token_id;
  const std::size_t offset =
      static_cast<std::size_t>(token_id) * hidden_size + idx;
  if (type == core::GgmlType::kBF16) {
    const auto* bf_ptr = static_cast<const hip_bfloat16*>(table);
    out_hidden[idx] = static_cast<float>(bf_ptr[offset]);
  } else if (type == core::GgmlType::kF32) {
    const auto* f_ptr = static_cast<const float*>(table);
    out_hidden[idx] = f_ptr[offset];
  } else if (type != core::GgmlType::kQ8_0) {
    // opt-q4kxl: the UD-Q4_K_XL shard stores token_embd.weight as Q4_K, so the
    // lookup decodes whichever block format the row actually uses instead of
    // reinterpreting it as Q8_0.
    const auto* row =
        static_cast<const std::uint8_t*>(table) +
        (static_cast<std::size_t>(token_id) * QuantRowBytes(type, hidden_size));
    out_hidden[idx] = QuantBlockElement(type, row, idx);
  } else {
    const auto* q8 = static_cast<const Q8_0Block*>(table);
    const Q8_0Block& wblk =
        q8[static_cast<std::size_t>(token_id) * (hidden_size / kQ8_0BlockSize) +
           idx / kQ8_0BlockSize];
    out_hidden[idx] = __half2float(wblk.d) *
                      static_cast<float>(wblk.qs[idx % kQ8_0BlockSize]);
  }
}

void LaunchEmbeddingLookup(const void* table, core::GgmlType type,
                           std::uint32_t token_id, float* out_hidden,
                           std::size_t hidden_size, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size = (hidden_size + block_size - 1) / block_size;
  EmbeddingLookupKernel<<<grid_size, block_size, 0, stream>>>(
      table, type, token_id, out_hidden, hidden_size);
}

void LaunchEmbeddingLookup(const void* table, core::GgmlType type,
                           const std::uint32_t* d_token_id, float* out_hidden,
                           std::size_t hidden_size, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size = (hidden_size + block_size - 1) / block_size;
  EmbeddingLookupPtrKernel<<<grid_size, block_size, 0, stream>>>(
      table, type, d_token_id, out_hidden, hidden_size);
}

}  // namespace gufo::hip
#endif
