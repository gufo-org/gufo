#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

void W8A8GemmWave64(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream) {
#if defined(__GFX12__)
  // No wave64 on RDNA4: W8A8Gemm() uses the wave32 tiles instead.
  (void)w; (void)x_tiled; (void)out;
  (void)batch; (void)m; (void)k; (void)stream;
#else
  const dim3 grid(static_cast<unsigned>((batch + 127) / 128),
                  static_cast<unsigned>((m + 127) / 128));
  // Keep Flash-Next's compilation and scale order. This geometry is distinct
  // from Qwen27B's Q8 instantiation, so each retains its qualified schedule.
  gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 128, 2, 4, 1,
                                            core::GgmlType::kQ8_0, 64>
      <<<grid, 256, 0, stream>>>(w, x_tiled, out, batch, m, k);
#endif
}

}  // namespace gufo::models::qwen38_flash_next::rocm
