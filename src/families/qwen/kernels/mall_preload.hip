#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/hip/hip_utils.hpp"
#include "src/families/qwen/kernels/ops/gemm.hpp"

namespace gufo::hip {

// Reads `n` 16-byte words with ordinary loads, so the lines allocate in the
// MALL. The XOR feeds a store that never happens in practice; it only keeps
// the compiler from dropping the loads. The sink is private to this file.
__global__ void MallPreloadKernel(const uint4* __restrict__ weight,
                                  std::size_t n, unsigned* __restrict__ sink) {
  uint4 acc = make_uint4(0U, 0U, 0U, 0U);
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = weight[i];
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9U)
    *sink = 1U;
}

void LaunchMallPreload(const void* weight, std::size_t bytes,
                       hipStream_t stream) {
  // One work-group per gfx1151 compute unit: enough to keep DRAM busy while
  // the main stream runs its small kernels beside it.
  constexpr unsigned kWorkGroups = 40;
  static unsigned* const sink = [] {
    unsigned* p = nullptr;
    HIP_CHECK(hipMalloc(&p, sizeof(unsigned)));
    return p;
  }();
  MallPreloadKernel<<<kWorkGroups, 256, 0, stream>>>(
      static_cast<const uint4*>(weight), bytes / sizeof(uint4), sink);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
