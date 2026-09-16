#if defined(ENGINE_ENABLE_HIP)
#include <cassert>

#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/kernels/small_batch_quant16_wave64.hpp"

namespace gufo::hip::detail {
namespace {

template<core::GgmlType Type, std::size_t Groups, bool FusedSwiGLU = false>
void Launch(const void* w, const float* x, float* y, std::size_t m,
            std::size_t k, hipStream_t stream) {
  const auto launch = [&]<bool DistributedOutput>() {
    SmallBatchKQuantExactFp32GEMMKernel<8, 16, 6, Type, 1, 4, Groups, true, 1,
                                        64, 32, DistributedOutput, FusedSwiGLU>
        <<<((m + 47) / 48) * Groups, 256, 0, stream>>>(w, x, y, m, k);
  };
  const bool ffn =
      (k == 5120 && (m == 17408 || m == 34816)) || (m == 5120 && k == 17408);
  const bool q5_ssm_out =
      Type == core::GgmlType::kQ5_K && m == 5120 && k == 6144;
  // Other projection shapes retain their qualified compiler schedule.
  if constexpr (FusedSwiGLU) {
    launch.template operator()<true>();
  } else if (ffn || q5_ssm_out) {
    launch.template operator()<true>();
  } else {
    launch.template operator()<false>();
  }
}

template<core::GgmlType Type, bool FusedSwiGLU = false>
void LaunchGroups(const void* w, const float* x, float* y, std::size_t m,
                  std::size_t k, hipStream_t stream, std::size_t groups) {
  switch (groups) {
    case 1:
      Launch<Type, 1, FusedSwiGLU>(w, x, y, m, k, stream);
      break;
    case 2:
      Launch<Type, 2, FusedSwiGLU>(w, x, y, m, k, stream);
      break;
    case 3:
      Launch<Type, 3, FusedSwiGLU>(w, x, y, m, k, stream);
      break;
    case 4:
      Launch<Type, 4, FusedSwiGLU>(w, x, y, m, k, stream);
      break;
  }
}

}  // namespace

void LaunchKQuantSixteenWave64(core::GgmlType type, const void* w,
                               const float* x, float* y, std::size_t m,
                               std::size_t k, hipStream_t stream,
                               std::size_t groups) {
  assert(groups >= 1 && groups <= 4);
  assert(type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K);
  if (type == core::GgmlType::kQ4_K)
    LaunchGroups<core::GgmlType::kQ4_K>(w, x, y, m, k, stream, groups);
  else
    LaunchGroups<core::GgmlType::kQ5_K>(w, x, y, m, k, stream, groups);
}

void LaunchPackedSwiGLUSixteenWave64(core::GgmlType type, const void* w,
                                     const float* x, float* y,
                                     std::size_t groups, hipStream_t stream) {
  assert(groups >= 1 && groups <= 4);
  assert(type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K);
  if (type == core::GgmlType::kQ4_K)
    LaunchGroups<core::GgmlType::kQ4_K, true>(w, x, y, 34816, 5120, stream,
                                              groups);
  else
    LaunchGroups<core::GgmlType::kQ5_K, true>(w, x, y, 34816, 5120, stream,
                                              groups);
}

}  // namespace gufo::hip::detail
#endif
