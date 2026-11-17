// Wave64 192-row blocked W8A8 GEMM for the GLM-5.3-Flash wide-chunk
// dispatch. Compiled with -mwavefrontsize64; the wave32 defaults stay in
// w8a8_dense.hip.cpp. The 192x128 BK2 4x1 tile beat the shipped 128x128
// 4x2 wave32 schedule by +3.5/+4.0 % at m2048 k4096 batch 2048/8192 on the
// Flint sweep, and every configuration keeps the per-element K-ascending
// int32 accumulation order, so quality is gated by the existing
// glm5_next.rocm_w8a8_dense contract.
#include "staging/models/glm5_next/kernels/rocm/w8a8_wave64.hpp"

#include "src/families/qwen/kernels/prefill_quant_gemm.hpp"

namespace gufo::models::glm5_next::rocm {
namespace {

template <core::GgmlType WType>
void Launch192(const void* w, const void* q8_1_x, float* y,
               std::size_t batch, std::size_t m, std::size_t k,
               hipStream_t stream) {
  constexpr int kBM = 192;
  constexpr int kBN = 128;
  const dim3 grid(static_cast<unsigned>((batch + kBN - 1) / kBN),
                  static_cast<unsigned>((m + kBM - 1) / kBM));
  gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 1, WType, 64>
      <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
}

}  // namespace

void Glm5W8A8DenseWave64(core::GgmlType type, const void* w,
                         const void* q8_1_x, float* y, std::size_t batch,
                         std::size_t m, std::size_t k, hipStream_t stream) {
  if (type == core::GgmlType::kQ8_0) {
    Launch192<core::GgmlType::kQ8_0>(w, q8_1_x, y, batch, m, k, stream);
    return;
  }
  if (type == core::GgmlType::kQ5_K) {
    Launch192<core::GgmlType::kQ5_K>(w, q8_1_x, y, batch, m, k, stream);
    return;
  }
  Launch192<core::GgmlType::kQ6_K>(w, q8_1_x, y, batch, m, k, stream);
}

}  // namespace gufo::models::glm5_next::rocm
