#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>

#include "src/models/qwen/hip/kernels/common.hip"
#include "src/models/qwen/hip/ops/swiglu.hpp"

namespace strix::hip {

__launch_bounds__(256, 4) __global__ void FusedRMSNormSwiGLUGEMVKernel(
    const float* __restrict__ x, const float* __restrict__ norm_w, float eps,
    const hip_bfloat16* __restrict__ gate_w,
    const hip_bfloat16* __restrict__ up_w, float* __restrict__ out,
    std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;

  __shared__ float s_sum[256];
  const float inv_rms = FusedRMSNormInvScale(x, norm_w, K, eps, s_sum);

  const std::size_t row_pair = (blockIdx.x * 8U) + warp_id;
  const std::size_t m0 = row_pair * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  const auto* g_row0 = gate_w + (m0 * K);
  const auto* u_row0 = up_w + (m0 * K);
  const auto* g_row1 = (m1 < M) ? gate_w + (m1 * K) : nullptr;
  const auto* u_row1 = (m1 < M) ? up_w + (m1 * K) : nullptr;

  const auto* g_vec0 = reinterpret_cast<const uint4*>(g_row0);
  const auto* u_vec0 = reinterpret_cast<const uint4*>(u_row0);
  const auto* g_vec1 = reinterpret_cast<const uint4*>(g_row1);
  const auto* u_vec1 = reinterpret_cast<const uint4*>(u_row1);
  const std::size_t K_vec = K / 8;

  const float* nw = (norm_w != nullptr) ? norm_w : nullptr;
  float dot_g0 = 0.0F, dot_u0 = 0.0F;
  float dot_g1 = 0.0F, dot_u1 = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
    const float n0 = (x0.x * inv_rms) * ((nw != nullptr) ? nw[k + 0] : 1.0F);
    const float n1 = (x0.y * inv_rms) * ((nw != nullptr) ? nw[k + 1] : 1.0F);
    const float n2 = (x0.z * inv_rms) * ((nw != nullptr) ? nw[k + 2] : 1.0F);
    const float n3 = (x0.w * inv_rms) * ((nw != nullptr) ? nw[k + 3] : 1.0F);
    const float n4 = (x1.x * inv_rms) * ((nw != nullptr) ? nw[k + 4] : 1.0F);
    const float n5 = (x1.y * inv_rms) * ((nw != nullptr) ? nw[k + 5] : 1.0F);
    const float n6 = (x1.z * inv_rms) * ((nw != nullptr) ? nw[k + 6] : 1.0F);
    const float n7 = (x1.w * inv_rms) * ((nw != nullptr) ? nw[k + 7] : 1.0F);

    const uint4 gv0 = g_vec0[vi];
    const uint4 uv0 = u_vec0[vi];
    const auto* g_vals0 = reinterpret_cast<const hip_bfloat16*>(&gv0);
    const auto* u_vals0 = reinterpret_cast<const hip_bfloat16*>(&uv0);

    dot_g0 += (static_cast<float>(g_vals0[0]) * n0) +
              (static_cast<float>(g_vals0[1]) * n1) +
              (static_cast<float>(g_vals0[2]) * n2) +
              (static_cast<float>(g_vals0[3]) * n3) +
              (static_cast<float>(g_vals0[4]) * n4) +
              (static_cast<float>(g_vals0[5]) * n5) +
              (static_cast<float>(g_vals0[6]) * n6) +
              (static_cast<float>(g_vals0[7]) * n7);

    dot_u0 += (static_cast<float>(u_vals0[0]) * n0) +
              (static_cast<float>(u_vals0[1]) * n1) +
              (static_cast<float>(u_vals0[2]) * n2) +
              (static_cast<float>(u_vals0[3]) * n3) +
              (static_cast<float>(u_vals0[4]) * n4) +
              (static_cast<float>(u_vals0[5]) * n5) +
              (static_cast<float>(u_vals0[6]) * n6) +
              (static_cast<float>(u_vals0[7]) * n7);

    if (g_row1 != nullptr) {
      const uint4 gv1 = g_vec1[vi];
      const uint4 uv1 = u_vec1[vi];
      const auto* g_vals1 = reinterpret_cast<const hip_bfloat16*>(&gv1);
      const auto* u_vals1 = reinterpret_cast<const hip_bfloat16*>(&uv1);

      dot_g1 += (static_cast<float>(g_vals1[0]) * n0) +
                (static_cast<float>(g_vals1[1]) * n1) +
                (static_cast<float>(g_vals1[2]) * n2) +
                (static_cast<float>(g_vals1[3]) * n3) +
                (static_cast<float>(g_vals1[4]) * n4) +
                (static_cast<float>(g_vals1[5]) * n5) +
                (static_cast<float>(g_vals1[6]) * n6) +
                (static_cast<float>(g_vals1[7]) * n7);

      dot_u1 += (static_cast<float>(u_vals1[0]) * n0) +
                (static_cast<float>(u_vals1[1]) * n1) +
                (static_cast<float>(u_vals1[2]) * n2) +
                (static_cast<float>(u_vals1[3]) * n3) +
                (static_cast<float>(u_vals1[4]) * n4) +
                (static_cast<float>(u_vals1[5]) * n5) +
                (static_cast<float>(u_vals1[6]) * n6) +
                (static_cast<float>(u_vals1[7]) * n7);
    }
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float nv = (x[k] * inv_rms) * ((nw != nullptr) ? nw[k] : 1.0F);
    dot_g0 += static_cast<float>(g_row0[k]) * nv;
    dot_u0 += static_cast<float>(u_row0[k]) * nv;
    if (g_row1 != nullptr) {
      dot_g1 += static_cast<float>(g_row1[k]) * nv;
      dot_u1 += static_cast<float>(u_row1[k]) * nv;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_g0 += __shfl_xor(dot_g0, offset);
    dot_u0 += __shfl_xor(dot_u0, offset);
    dot_g1 += __shfl_xor(dot_g1, offset);
    dot_u1 += __shfl_xor(dot_u1, offset);
  }

  if (lane_id == 0) {
    const float sig_g0 = 1.0F / (1.0F + expf(-dot_g0));
    out[m0] = (dot_g0 * sig_g0) * dot_u0;
    if (m1 < M) {
      const float sig_g1 = 1.0F / (1.0F + expf(-dot_g1));
      out[m1] = (dot_g1 * sig_g1) * dot_u1;
    }
  }
}

void LaunchFusedRMSNormSwiGLUGEMV(const float* x, const float* norm_w,
                                  float eps, const void* gate_w,
                                  const void* up_w, float* out,
                                  std::size_t intermediate_size,
                                  std::size_t hidden_size, hipStream_t stream) {
  const std::size_t num_pairs = intermediate_size / 2;
  const std::size_t grid = (num_pairs + 7U) / 8U;
  FusedRMSNormSwiGLUGEMVKernel<<<grid, 256, 0, stream>>>(
      x, norm_w, eps, static_cast<const hip_bfloat16*>(gate_w),
      static_cast<const hip_bfloat16*>(up_w), out, intermediate_size,
      hidden_size);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
