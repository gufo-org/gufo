#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

// DS4 Flash indexer: 64 heads of width 128, one 128-thread block per score.
__global__ static void indexer_score_one_direct_kernel(
    float* scores, const float* q, const float* weights,
    const float* index_comp, uint32_t n_comp, uint32_t pos0, uint32_t ratio,
    float scale, int causal) {
  const uint32_t token = blockIdx.y;
  scores += (uint64_t)token * n_comp;
  q += (uint64_t)token * 64u * 128u;
  weights += (uint64_t)token * 64u;
  pos0 += token;
  const uint32_t c = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5u;
  if (c >= n_comp || tid >= 128u)
    return;
  if (causal) {
    const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
    if (c >= visible) {
      if (tid == 0)
        scores[c] = -INFINITY;
      return;
    }
  }

  __shared__ float krow[128];
  // Each head owns its slot: no inter-warp synchronization inside the loop.
  __shared__ float partial[64];
  if (tid < 128u)
    krow[tid] = index_comp[(uint64_t)c * 128u + tid];
  __syncthreads();

  for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
    const uint32_t h = h0 + warp;
    const float4 qv = ((const float4*)(q + (uint64_t)h * 128u))[lane];
    const float4 kv = ((const float4*)krow)[lane];
    float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
    for (int offset = 16; offset > 0; offset >>= 1)
      dot += __shfl_down(dot, offset, 32);
    if (lane == 0)
      partial[h] = fmaxf(dot, 0.0f) * weights[h] * scale;
  }
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
    // Keep the previous groups of four and their accumulation order.
    for (uint32_t h = 0; h < 64u; h += 4u)
      total += partial[h] + partial[h + 1] + partial[h + 2] + partial[h + 3];
    scores[c] = total;
  }
}
