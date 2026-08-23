#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/detail/gemv_dispatcher.hpp"
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"
#include "src/models/qwen/hip/qwen_gpu_quant_ops.hpp"
#include "src/core/gguf_reader.hpp"

namespace strix::hip {

namespace {

// Quant block layouts (Q8_0 / Q5_K / Q6_K) and quant row-dot helpers are
// shared via src/models/qwen/hip/qwen_gpu_quant_ops.hpp.

// opt-c014-layer-prefetch: touch up to 16 bytes per 4KiB page of a weight
// region so the page mappings are resident before the real projection kernels
// read them. Reads only; the guarded store can never fire (its value depends
// on the loaded data) and exists to defeat dead-code elimination. Reads are
// clamped to `bytes` so the last partial page never runs past the region.
__launch_bounds__(256, 4) __global__
    void LayerWeightPrefetchKernel(const void* __restrict__ data,
                                   std::size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  constexpr std::size_t kPageBytes = 4096U;
  constexpr std::size_t kTouchBytes = 16U;
  const std::size_t num_pages = (bytes + kPageBytes - 1U) / kPageBytes;
  float acc = 0.0F;
  for (std::size_t page = threadIdx.x; page < num_pages; page += blockDim.x) {
    const std::size_t base = page * kPageBytes;
    const std::size_t n = std::min<std::size_t>(kTouchBytes, bytes - base);
    for (std::size_t i = 0; i < n; ++i) {
      acc += static_cast<float>(p[base + i]);
    }
  }
  if (acc == 1.234567e30F) {
    const_cast<unsigned char*>(p)[0] = 0U;
  }
}

}  // namespace

// opt-c014-layer-prefetch: touch one 16B chunk per 4KiB page of a weight
// region on the given stream (async). Reads only; the guarded store inside the
// kernel can never fire.
void LaunchLayerWeightPrefetch(const void* data, std::size_t bytes,
                               hipStream_t stream) {
  if (data == nullptr || bytes == 0) {
    return;
  }
  const std::size_t num_pages = (bytes + 4095U) / 4096U;
  const std::size_t grid = (num_pages + 255U) / 256U;
  LayerWeightPrefetchKernel<<<grid, 256, 0, stream>>>(data, bytes);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)

