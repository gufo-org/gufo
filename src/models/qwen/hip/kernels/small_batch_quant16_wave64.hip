#if defined(ENGINE_ENABLE_HIP)
#include <cassert>

#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/kernels/small_batch_quant16_wave64.hpp"

namespace gufo::hip::detail {
namespace {

template<core::GgmlType Type, std::size_t Groups>
void Launch(const void* w, const float* x, float* y, std::size_t m,
            std::size_t k, hipStream_t stream) {
  SmallBatchKQuantExactFp32GEMMKernel<8, 16, 6, Type, 1, 4, Groups, true, 1, 64>
      <<<((m + 47) / 48) * Groups, 256, 0, stream>>>(w, x, y, m, k);
}

template<core::GgmlType Type>
void LaunchGroups(const void* w, const float* x, float* y, std::size_t m,
                  std::size_t k, hipStream_t stream, std::size_t groups) {
  switch (groups) {
    case 1:
      Launch<Type, 1>(w, x, y, m, k, stream);
      break;
    case 2:
      Launch<Type, 2>(w, x, y, m, k, stream);
      break;
    case 3:
      Launch<Type, 3>(w, x, y, m, k, stream);
      break;
    case 4:
      Launch<Type, 4>(w, x, y, m, k, stream);
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

}  // namespace gufo::hip::detail
#endif
