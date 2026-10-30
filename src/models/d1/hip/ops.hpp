#ifndef GUFO_MODELS_D1_HIP_OPS_HPP_
#define GUFO_MODELS_D1_HIP_OPS_HPP_
#include <hip/hip_runtime.h>
namespace gufo::models::d1::gpu {
void SpatialConv(const float* p, const float* w, float* out, int h, int width,
                 int hidden, hipStream_t stream);
void AxialRope(float* x, int h, int w, int heads, hipStream_t stream);
}  // namespace gufo::models::d1::gpu
#endif
