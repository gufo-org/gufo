#include "src/families/qwen/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

hipError_t PrepareW8A8GemmWave64() {
  hipFuncAttributes attributes{};
  return hipFuncGetAttributes(
      &attributes,
      reinterpret_cast<const void*>(
          gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<
              128, 128, 2, 4, 1, core::GgmlType::kQ8_0, 64, true>));
}

void W8A8GemmWave64(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 127) / 128),
                  static_cast<unsigned>((m + 127) / 128));
  // The blocked W8A8 kernel's pinned scale order: a row must not depend on
  // whether its pass reached 1,024 rows (projection_ops_test).
  gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 128, 2, 4, 1,
                                            core::GgmlType::kQ8_0, 64, true>
      <<<grid, 256, 0, stream>>>(w, x_tiled, out, batch, m, k);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
