#ifndef GUFO_MODELS_QWEN_IMAGE_21_HIP_BLAS_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_HIP_BLAS_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <memory>

namespace gufo::models::qwen_image_21::hip {

// BF16 weights/activations, FP32 accumulation, BF16 or FP32 result. Cache
// library plans by shape; no online tuning or request-dependent arithmetic.
class Gemm {
public:
  Gemm();
  ~Gemm();
  bool Run(const void* weight, const void* input, void* output, int rows,
           int channels, int inner, bool float_output, hipStream_t stream);
  bool RunBatched(const void* weight, const void* input, void* output, int rows,
                  int channels, int inner, int batches,
                  std::int64_t weight_stride, std::int64_t input_stride,
                  std::int64_t output_stride, hipStream_t stream);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen_image_21::hip
#endif
