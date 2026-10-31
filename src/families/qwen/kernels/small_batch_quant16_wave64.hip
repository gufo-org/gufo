#if defined(ENGINE_ENABLE_HIP)
#include <cassert>

#include "src/families/qwen/kernels/small_batch_gemm.hpp"
#include "src/families/qwen/kernels/small_batch_quant16_wave64.hpp"

namespace gufo::hip::detail {
namespace {

template<core::GgmlType Type, std::size_t Groups, bool FusedSwiGLU = false>
void Launch(const void* w, const float* x, float* y, std::size_t m,
            std::size_t k, hipStream_t stream, std::size_t valid_rows) {
  const auto launch = [&]<bool DistributedOutput>() {
    if constexpr (Groups == 1 && !FusedSwiGLU) {
      if (valid_rows != 0) {
        SmallBatchKQuantExactFp32GEMMKernel<8, 16, 6, Type, 1, 4, 1, true, 1,
                                            64, 32, DistributedOutput, false,
                                            true>
            <<<(m + 47) / 48, 256, 0, stream>>>(w, x, y, m, k, valid_rows);
        return;
      }
    }
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
                  std::size_t k, hipStream_t stream, std::size_t groups,
                  std::size_t valid_rows = 0) {
  switch (groups) {
    case 1:
      Launch<Type, 1, FusedSwiGLU>(w, x, y, m, k, stream, valid_rows);
      break;
    case 2:
      Launch<Type, 2, FusedSwiGLU>(w, x, y, m, k, stream, 0);
      break;
    case 3:
      Launch<Type, 3, FusedSwiGLU>(w, x, y, m, k, stream, 0);
      break;
    case 4:
      Launch<Type, 4, FusedSwiGLU>(w, x, y, m, k, stream, 0);
      break;
  }
}

}  // namespace

void LaunchKQuantSixteenWave64(core::GgmlType type, const void* w,
                               const float* x, float* y, std::size_t m,
                               std::size_t k, hipStream_t stream,
                               std::size_t groups, std::size_t valid_rows) {
  assert(groups >= 1 && groups <= 4);
  assert(valid_rows == 0 || (groups == 1 && valid_rows < 16));
  assert(type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K);
  if (type == core::GgmlType::kQ4_K)
    LaunchGroups<core::GgmlType::kQ4_K>(w, x, y, m, k, stream, groups,
                                        valid_rows);
  else
    LaunchGroups<core::GgmlType::kQ5_K>(w, x, y, m, k, stream, groups,
                                        valid_rows);
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
