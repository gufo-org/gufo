#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"

namespace gufo::hip::detail {

bool TryLaunchQuantPrefillWave64(core::GgmlType type, const void* w,
                                 const void* x, float* y, std::size_t batch,
                                 std::size_t m, std::size_t k,
                                 hipStream_t stream) {
  if (batch < 96 || m < 1024 || k < 1024 || k % 256 != 0) {
    return false;
  }
  const dim3 grid(static_cast<unsigned>((batch + 127) / 128),
                  static_cast<unsigned>((m + 127) / 128));
  const auto launch = [&]<core::GgmlType Type>() {
    constexpr bool square =
        Type == core::GgmlType::kQ8_0 || Type == core::GgmlType::kIQ4_XS;
    WKQuantA8BlockedWmmaGEMMKernel<128, 128, 2, square ? 2 : 4, square ? 2 : 1,
                                   Type, 64>
        <<<grid, 256, 0, stream>>>(w, x, y, batch, m, k);
  };
  switch (type) {
    case core::GgmlType::kQ4_K:
      launch.template operator()<core::GgmlType::kQ4_K>();
      return true;
    case core::GgmlType::kQ5_K:
      launch.template operator()<core::GgmlType::kQ5_K>();
      return true;
    case core::GgmlType::kQ6_K:
      launch.template operator()<core::GgmlType::kQ6_K>();
      return true;
    case core::GgmlType::kQ8_0:
      launch.template operator()<core::GgmlType::kQ8_0>();
      return true;
    case core::GgmlType::kIQ4_XS:
      if (m > k) {
        return false;
      }
      launch.template operator()<core::GgmlType::kIQ4_XS>();
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::hip::detail
#endif
