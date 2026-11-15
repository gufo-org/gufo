#ifndef GUFO_FAMILIES_QWEN_EAGER_BLAS_HPP_
#define GUFO_FAMILIES_QWEN_EAGER_BLAS_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <memory>

namespace gufo::qwen::eager {

// BF16 weights/activations, FP32 accumulation, BF16 or FP32 result. Cache
// library plans by shape; no online tuning or request-dependent arithmetic.
class Gemm {
public:
  Gemm();
  ~Gemm();
  /// `rank` picks that hipBLASLt heuristic candidate instead of the shape
  /// policy's (when the list is long enough).
  bool Run(const void* weight, const void* input, void* output, int rows,
           int channels, int inner, bool float_output, hipStream_t stream,
           int rank = -1);
  bool RunBatched(const void* weight, const void* input, void* output, int rows,
                  int channels, int inner, int batches,
                  std::int64_t weight_stride, std::int64_t input_stride,
                  std::int64_t output_stride, hipStream_t stream);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::qwen::eager
#endif
