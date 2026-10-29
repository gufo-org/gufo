// SPDX-License-Identifier: MIT
// Raw-pointer C ABI for Nemotron 3.5 Bolt's HIP quantized matrix kernels,
// adapted from llama.cpp's MMQ/MMVQ (see VENDOR.md). Functions return zero on
// success and nonzero on failure. Device pointers remain caller-owned; every
// operation runs on the supplied stream. Activations are quantized to Q8_1
// internally, as in llama.cpp; NVFP4 per-tensor scales are not applied.

#pragma once

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// GGML type codes accepted as weight_type.
enum {
  NB_MMQ_Q8_0 = 8,
  NB_MMQ_IQ2_XXS = 16,
  NB_MMQ_NVFP4 = 40,
};

// One-time init for the device; safe to call repeatedly.
int nb_mmq_init(int device);
void nb_mmq_cleanup(void);

// Reserves the scratch arena of `stream` before its first product, so the
// first product may be recorded by stream capture. Release frees it; the
// stream must be idle.
int nb_mmq_bind_stream(hipStream_t stream);
void nb_mmq_release_stream(hipStream_t stream);

// Dense matmul:
//   out[col * M + row] = sum_k W[row, k] * X[col * K + k]
// for 0 <= row < M, 0 <= col < N. W holds M rows of K elements in the
// weight type's block format; K must be a multiple of 256. X holds N rows of
// K floats, out N rows of M floats. Small N uses the vector kernels, larger N
// the tiled kernels.
int nb_mmq_dense(int weight_type, const void* W, const float* X, float* out,
                 int M, int N, int K, hipStream_t stream);

// Dense matmul for N <= 8 on activations the caller already quantized:
//   out[col * M + row] = scale * sum_k W[row, k] * Xq[col, k]
//                        (+ bias[col * M + row])
// Xq holds N rows of K / 32 block_q8_1 (half d = max|x| / 127, half sum,
// int8 q[32] = round(x / d)); K must be a multiple of 512. A scale other
// than 1 and a bias need N == 1; bias may equal out, which then accumulates.
int nb_mmq_dense_q8_1(int weight_type, const void* W, const void* Xq,
                      float scale, const float* bias, float* out, int M, int N,
                      int K, hipStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif
