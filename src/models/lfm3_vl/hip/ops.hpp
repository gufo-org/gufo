#ifndef GUFO_MODELS_LFM3_VL_HIP_OPS_HPP_
#define GUFO_MODELS_LFM3_VL_HIP_OPS_HPP_
#include <hip/hip_runtime.h>
namespace gufo::models::lfm3_vl::gpu {
void SpatialConv(const float* p, const float* w, float* out, int h, int width,
                 int hidden, hipStream_t stream);
void AxialRope(float* x, int h, int w, int heads, hipStream_t stream);
}  // namespace gufo::models::lfm3_vl::gpu
#endif
