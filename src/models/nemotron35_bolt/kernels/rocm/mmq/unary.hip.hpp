// SPDX-License-Identifier: MIT
// Adapted from llama.cpp ggml/src/ggml-cuda (MIT, The ggml authors); see
// VENDOR.md.
#pragma once
#include "common.hip.hpp"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void nb_ggml_hip_op_abs(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_sgn(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_neg(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_step(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_gelu(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_silu(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_silu_back(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_gelu_erf(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_gelu_quick(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_tanh(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_relu(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_sigmoid(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_hardsigmoid(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_exp(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_hardswish(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_leaky_relu(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_sqr(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_sqrt(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_sin(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_cos(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_log(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_expm1(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_softplus(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_elu(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_floor(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_ceil(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_round(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_trunc(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_reglu(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_geglu(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_swiglu(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_swiglu_oai(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_geglu_erf(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_geglu_quick(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_xielu(nb_ggml_hip_context & ctx, ggml_tensor * dst);

void nb_ggml_hip_op_unary_mul(nb_ggml_hip_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);

void nb_ggml_hip_op_relu_sqr(nb_ggml_hip_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);

__device__ __forceinline__ float nb_ggml_hip_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float nb_ggml_hip_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float nb_ggml_hip_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}
