// Dense-tier blocked W8A8 GEMM for the GLM-5.3-Flash wide-chunk dispatch.
// Instantiates the shared Qwen WKQuantA8BlockedWmmaGEMMKernel over the
// artifact's dense trunk types; the activation quantizer and the tiled
// Q8_1 buffer math come from the family launchers (gufo_qwen_kernels).
#include "staging/models/glm5_next/kernels/rocm/w8a8_dense.hpp"

#include "src/families/qwen/kernels/ops/gemm.hpp"
#include "src/families/qwen/kernels/prefill_quant_gemm.hpp"

namespace gufo::models::glm5_next::rocm {
namespace {

template <core::GgmlType WType>
void LaunchTuned(const void* w, const void* q8_1_x, float* y,
                 std::size_t batch, std::size_t m, std::size_t k,
                 hipStream_t stream) {
  if (batch >= 96) {
    const dim3 grid(static_cast<unsigned>((batch + 127) / 128),
                    static_cast<unsigned>((m + 127) / 128));
    gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 128, 2, 4, 2, WType>
        <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    return;
  }
  const dim3 grid(static_cast<unsigned>((batch + 63) / 64),
                  static_cast<unsigned>((m + 127) / 128));
  gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 64, 4, 8, 1, WType>
      <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
}

}  // namespace

bool Glm5W8A8Supported(core::GgmlType type) {
  return type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
         type == core::GgmlType::kQ8_0;
}

std::size_t Glm5W8A8ScratchBytes(std::size_t tokens, std::size_t k) {
  return gufo::hip::QuantizedActivationBytes(tokens, k);
}

void Glm5W8A8Dense(const void* w, core::GgmlType type, const float* x,
                   void* act_scratch, float* y, std::size_t tokens,
                   std::size_t m, std::size_t k, hipStream_t stream) {
  gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(x, act_scratch, tokens, k,
                                                  stream);
  if (type == core::GgmlType::kQ8_0) {
    LaunchTuned<core::GgmlType::kQ8_0>(w, act_scratch, y, tokens, m, k,
                                       stream);
    return;
  }
  if (type == core::GgmlType::kQ5_K) {
    LaunchTuned<core::GgmlType::kQ5_K>(w, act_scratch, y, tokens, m, k,
                                       stream);
    return;
  }
  LaunchTuned<core::GgmlType::kQ6_K>(w, act_scratch, y, tokens, m, k, stream);
}

}  // namespace gufo::models::glm5_next::rocm
