#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/core/hip/detail/qwen_attention_policy.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/qwen_gpu_ops.hpp"
#include "src/core/quant/ggml_dequant.hpp"

static inline std::uint16_t FloatToBf16Bits(float f) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16);
}

static inline float Bf16BitsToFloat(std::uint16_t bits) {
  std::uint32_t f_bits = static_cast<std::uint32_t>(bits) << 16;
  float f = 0.0F;
  std::memcpy(&f, &f_bits, sizeof(f));
  return f;
}

void TestGpuRMSNorm() {
  const std::size_t dim = 256;
  std::vector<float> h_x(dim, 1.0F);
  std::vector<float> h_w(dim, 2.0F);
  std::vector<float> h_out(dim, 0.0F);

  float *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_x, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, dim * sizeof(float)));

  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_w, h_w.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  strix::hip::LaunchRMSNorm(d_x, d_w, d_out, dim, 1e-6F);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_out.data(), d_out, dim * sizeof(float),
                      hipMemcpyDeviceToHost));

  // mean(x^2) = 1.0, rms = 1.0, out = (1.0 / 1.0) * 2.0 = 2.0
  for (std::size_t i = 0; i < dim; ++i) {
    assert(std::abs(h_out[i] - 2.0F) < 1e-4F);
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));
}

void TestGpuResidualAdd() {
  const std::size_t dim = 128;
  std::vector<float> h_a(dim, 3.5F);
  std::vector<float> h_b(dim, 1.5F);
  std::vector<float> h_out(dim, 0.0F);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_a, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, dim * sizeof(float)));

  HIP_CHECK(
      hipMemcpy(d_a, h_a.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_b, h_b.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  strix::hip::LaunchResidualAdd(d_a, d_b, d_out, dim);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_out.data(), d_out, dim * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < dim; ++i) {
    assert(std::abs(h_out[i] - 5.0F) < 1e-5F);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_out));
}

void TestGpuGEMV() {
  // Test 1: FP32 GEMV baseline fallback
  {
    const std::size_t M = 4;
    const std::size_t K = 8;
    std::vector<float> h_A(M * K, 1.0F);  // all ones
    std::vector<float> h_x(K, 2.0F);      // all twos
    std::vector<float> h_y(M, 0.0F);

    float *d_A = nullptr, *d_x = nullptr, *d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

    strix::hip::LaunchGEMV(d_A, strix::core::GgmlType::kF32, d_x, d_y, M, K);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(
        hipMemcpy(h_y.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

    // Each row has K=8 ones * 2.0 = 16.0
    for (std::size_t m = 0; m < M; ++m) {
      assert(std::abs(h_y[m] - 16.0F) < 1e-4F);
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
  }

  // Test 2: BF16 Wave32 Single-Row, Dual-Row, Quad-Row and Qwen projection
  // shapes
  const std::vector<std::pair<std::size_t, std::size_t>> test_shapes = {
      {1, 256},     // Single row (kWave32SingleRow)
      {2, 512},     // Dual row (kWave32DualRow)
      {3, 256},     // Odd rows (kWave32SingleRow)
      {4, 512},     // Quad row (kWave32QuadRow)
      {64, 1024},   // Medium quad row
      {256, 4096},  // Large Qwen-like projection
  };

  for (const auto& [M, K] : test_shapes) {
    std::vector<std::uint16_t> h_A(M * K);
    std::vector<float> h_A_f32(M * K);
    std::vector<float> h_x(K);
    std::vector<float> h_y_ref(M, 0.0F);
    std::vector<float> h_y(M, 0.0F);

    for (std::size_t i = 0; i < M * K; ++i) {
      const float val =
          0.05F * static_cast<float>(static_cast<int>(i % 13) - 6);
      h_A_f32[i] = val;
      h_A[i] = FloatToBf16Bits(val);
    }
    for (std::size_t k = 0; k < K; ++k) {
      h_x[k] = 0.1F * static_cast<float>(static_cast<int>(k % 17) - 8);
    }

    // CPU reference computation
    for (std::size_t m = 0; m < M; ++m) {
      float dot = 0.0F;
      for (std::size_t k = 0; k < K; ++k) {
        dot += Bf16BitsToFloat(h_A[m * K + k]) * h_x[k];
      }
      h_y_ref[m] = dot;
    }

    void* d_A = nullptr;
    float *d_x = nullptr, *d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

    strix::hip::LaunchGEMV(d_A, strix::core::GgmlType::kBF16, d_x, d_y, M, K);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(
        hipMemcpy(h_y.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "M=" << M << " K=" << K
              << " h_A[0]=" << Bf16BitsToFloat(h_A[0]) << " h_x[0]=" << h_x[0]
              << " ref[0]=" << h_y_ref[0] << " y[0]=" << h_y[0] << "\n";

    float max_diff = 0.0F;
    for (std::size_t m = 0; m < M; ++m) {
      max_diff = std::max(max_diff, std::abs(h_y[m] - h_y_ref[m]));
    }
    std::cout << "Shape M=" << M << " K=" << K << " max diff=" << max_diff
              << " y[0]=" << h_y[0] << " ref[0]=" << h_y_ref[0] << "\n";
    assert(max_diff < 1e-2F);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
  }
}

void TestBatchedGEMM() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t M = 8;
  constexpr std::size_t K = 16;

  std::vector<float> h_A(M * K, 1.5F);
  std::vector<float> h_X(batch * K);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t k = 0; k < K; ++k) {
      h_X[b * K + k] = static_cast<float>(b + 1);
    }
  }
  std::vector<float> h_Y(batch * M, 0.0F);

  float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_X, batch * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_Y, batch * M * sizeof(float)));

  HIP_CHECK(
      hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_X, h_X.data(), batch * K * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchBatchedGEMM(d_A, false, d_X, d_Y, batch, M, K);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_Y.data(), d_Y, batch * M * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t b = 0; b < batch; ++b) {
    const float expected =
        static_cast<float>(b + 1) * 1.5F * static_cast<float>(K);
    for (std::size_t m = 0; m < M; ++m) {
      const float diff = std::abs(h_Y[b * M + m] - expected);
      if (diff > 1e-4F) {
        std::cerr << "BatchedGEMM mismatch at b=" << b << " m=" << m
                  << " got=" << h_Y[b * M + m] << " expected=" << expected
                  << "\n";
        assert(false);
      }
    }
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_X));
  HIP_CHECK(hipFree(d_Y));
}

void TestHipblasGEMM() {
  hipblasHandle_t handle = nullptr;
  HIPBLAS_CHECK(hipblasCreate(&handle));

  constexpr std::size_t batch = 4;
  constexpr std::size_t M = 8;
  constexpr std::size_t K = 16;

  // Test FP32
  {
    std::vector<float> h_A(M * K, 1.5F);
    std::vector<float> h_X(batch * K);
    for (std::size_t b = 0; b < batch; ++b) {
      for (std::size_t k = 0; k < K; ++k) {
        h_X[b * K + k] = static_cast<float>(b + 1);
      }
    }
    std::vector<float> h_Y(batch * M, 0.0F);

    float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_X, batch * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Y, batch * M * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_X, h_X.data(), batch * K * sizeof(float),
                        hipMemcpyHostToDevice));

    strix::hip::LaunchHipblasGEMM(handle, d_A, false, d_X, d_Y, batch, M, K,
                                  nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_Y.data(), d_Y, batch * M * sizeof(float),
                        hipMemcpyDeviceToHost));

    for (std::size_t b = 0; b < batch; ++b) {
      const float expected =
          static_cast<float>(b + 1) * 1.5F * static_cast<float>(K);
      for (std::size_t m = 0; m < M; ++m) {
        const float diff = std::abs(h_Y[b * M + m] - expected);
        if (diff >= 1e-4F) {
          std::cerr << "Mismatch in FP32 HipblasGEMM at b=" << b << " m=" << m
                    << "\n";
          std::abort();
        }
      }
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_Y));
  }

  // Test BF16
  {
    std::vector<std::uint16_t> h_A(M * K);
    for (std::size_t i = 0; i < M * K; ++i) {
      h_A[i] = FloatToBf16Bits(1.5F);
    }
    std::vector<float> h_X(batch * K);
    for (std::size_t b = 0; b < batch; ++b) {
      for (std::size_t k = 0; k < K; ++k) {
        h_X[b * K + k] = static_cast<float>(b + 1);
      }
    }
    std::vector<float> h_Y(batch * M, 0.0F);

    void* d_A = nullptr;
    float *d_X = nullptr, *d_Y = nullptr;
    void* d_x_bf16 = nullptr;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&d_X, batch * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Y, batch * M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x_bf16, batch * K * sizeof(std::uint16_t)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_X, h_X.data(), batch * K * sizeof(float),
                        hipMemcpyHostToDevice));

    strix::hip::LaunchHipblasGEMM(handle, d_A, true, d_X, d_Y, batch, M, K,
                                  d_x_bf16);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_Y.data(), d_Y, batch * M * sizeof(float),
                        hipMemcpyDeviceToHost));

    for (std::size_t b = 0; b < batch; ++b) {
      const float expected =
          static_cast<float>(b + 1) * 1.5F * static_cast<float>(K);
      for (std::size_t m = 0; m < M; ++m) {
        const float diff = std::abs(h_Y[b * M + m] - expected);
        if (diff >= 1e-2F) {
          std::cerr << "Mismatch in BF16 HipblasGEMM at b=" << b << " m=" << m
                    << " got=" << h_Y[b * M + m] << " expected=" << expected
                    << "\n";
          std::abort();
        }
      }
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_Y));
    HIP_CHECK(hipFree(d_x_bf16));
  }

  HIPBLAS_CHECK(hipblasDestroy(handle));
}

void TestHipblasLtGEMM() {
  constexpr std::size_t batch = 32;
  constexpr std::size_t m = 256;
  constexpr std::size_t k = 64;

  std::vector<std::uint16_t> h_a(m * k, FloatToBf16Bits(1.5F));
  std::vector<std::uint16_t> h_x(batch * k);
  for (std::size_t b = 0; b < batch; ++b) {
    const auto value = FloatToBf16Bits(static_cast<float>(b + 1));
    std::fill_n(h_x.begin() + static_cast<std::ptrdiff_t>(b * k), k, value);
  }
  std::vector<float> h_y(batch * m, 0.0F);

  void* d_a = nullptr;
  void* d_x = nullptr;
  float* d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_a, h_a.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_y, h_y.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), h_a.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));

  strix::hip::HipblasLtGemm gemm;
  if (!gemm.RunBf16(d_a, d_x, d_y, batch, m, k)) {
    std::cerr << "hipBLASLt did not return a supported BF16 GEMM plan\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t b = 0; b < batch; ++b) {
    const float expected =
        static_cast<float>(b + 1) * 1.5F * static_cast<float>(k);
    for (std::size_t row = 0; row < m; ++row) {
      if (std::abs(h_y[b * m + row] - expected) >= 1e-2F) {
        std::cerr << "hipBLASLt mismatch at batch=" << b << " row=" << row
                  << " got=" << h_y[b * m + row] << " expected=" << expected
                  << '\n';
        std::abort();
      }
    }
  }

  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_a));
}

void TestBatchedSSMConvEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_key_heads = 16;
  constexpr std::uint32_t num_heads = 48;
  constexpr std::uint32_t key_dim = 128;
  constexpr std::uint32_t val_dim = 128;
  constexpr std::size_t qkv_dim =
      (2 * num_key_heads * key_dim) + (num_heads * val_dim);
  constexpr std::size_t inner_size = num_heads * val_dim;

  std::vector<float> h_qkv(batch * qkv_dim);
  for (std::size_t i = 0; i < h_qkv.size(); ++i) {
    h_qkv[i] = std::sin(static_cast<float>(i) * 0.01F);
  }
  std::vector<float> h_weights(qkv_dim * 4, 0.25F);
  std::vector<float> h_ssm_a(num_heads, -0.05F);
  std::vector<float> h_ssm_dt(num_heads, 0.01F);
  std::vector<float> h_ssm_norm(val_dim, 1.0F);
  std::vector<float> h_gate(batch * inner_size, 0.5F);

  float *d_qkv = nullptr, *d_w = nullptr;
  float *d_state_seq = nullptr, *d_state_batch = nullptr;
  float *d_conv_out_seq = nullptr, *d_conv_out_batch = nullptr;
  float *d_delta_seq = nullptr, *d_delta_batch = nullptr;
  float *d_alpha = nullptr, *d_beta = nullptr;
  float *d_ssm_a = nullptr, *d_ssm_dt = nullptr, *d_ssm_norm = nullptr;
  float* d_gate = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr;

  const std::size_t delta_size = num_heads * key_dim * val_dim;
  HIP_CHECK(hipMalloc(&d_qkv, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_seq, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_batch, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_seq, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_batch, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_seq, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_batch, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_a, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_dt, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_norm, val_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, batch * inner_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(), batch * qkv_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_weights.data(), qkv_dim * 4 * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_state_seq, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_state_batch, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_seq, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_batch, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_alpha, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemset(d_beta, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_ssm_a, h_ssm_a.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_dt, h_ssm_dt.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_norm, h_ssm_norm.data(), val_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), batch * inner_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchSSMConvRecurrence(
        d_qkv + t * qkv_dim, d_w, d_state_seq, d_conv_out_seq + t * qkv_dim,
        d_delta_seq, d_alpha + t * num_heads, d_beta + t * num_heads, d_ssm_a,
        d_ssm_dt, d_ssm_norm, d_gate + t * inner_size,
        d_out_seq + t * inner_size, 0, qkv_dim, num_key_heads, num_heads,
        key_dim, val_dim);
  }

  // Batched
  strix::hip::LaunchBatchedSSMConvRecurrence(
      d_qkv, d_w, d_state_batch, d_conv_out_batch, d_delta_batch, d_alpha,
      d_beta, d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate, d_out_batch, 0, batch,
      qkv_dim, num_key_heads, num_heads, key_dim, val_dim);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(batch * inner_size);
  std::vector<float> res_batch(batch * inner_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "DeltaNet Seq vs Batch max diff: " << max_diff << "\n";
  if (max_diff >= 1e-4F) {
    std::cerr << "DeltaNet batched recurrence mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_state_seq));
  HIP_CHECK(hipFree(d_state_batch));
  HIP_CHECK(hipFree(d_conv_out_seq));
  HIP_CHECK(hipFree(d_conv_out_batch));
  HIP_CHECK(hipFree(d_delta_seq));
  HIP_CHECK(hipFree(d_delta_batch));
  HIP_CHECK(hipFree(d_alpha));
  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_ssm_a));
  HIP_CHECK(hipFree(d_ssm_dt));
  HIP_CHECK(hipFree(d_ssm_norm));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
}

void TestBatchedAttentionEquivalence() {
  constexpr std::size_t batch = 2048;
  constexpr std::uint32_t num_heads = 2;
  constexpr std::uint32_t num_kv_heads = 1;
  constexpr std::uint32_t head_dim = 32;
  constexpr std::uint32_t max_context = 2048;

  const std::size_t q_size = batch * num_heads * head_dim;
  const std::size_t kv_size = batch * num_kv_heads * head_dim;
  const std::size_t cache_size = 8 * num_kv_heads * max_context * head_dim * 2;

  std::vector<float> h_q(q_size);
  std::vector<float> h_k(kv_size);
  std::vector<float> h_v(kv_size);
  std::vector<float> h_gate(q_size);
  for (std::size_t index = 0; index < q_size; ++index) {
    h_q[index] = 0.08F * std::sin(static_cast<float>(index % 257) * 0.07F);
    h_gate[index] = 0.4F * std::cos(static_cast<float>(index % 193) * 0.05F);
  }
  for (std::size_t index = 0; index < kv_size; ++index) {
    h_k[index] = 0.07F * std::cos(static_cast<float>(index % 251) * 0.06F);
    h_v[index] = 0.2F * std::sin(static_cast<float>(index % 239) * 0.04F);
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache_seq = nullptr, *d_cache_batch = nullptr,
        *d_cache_gemm = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr, *d_out_gemm = nullptr;
  float* d_scores = nullptr;
  hipblasHandle_t hipblas_handle = nullptr;

  HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_batch, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_gemm, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_gemm, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_scores, (num_heads / num_kv_heads) * batch *
                                     max_context * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_cache_seq, 0, cache_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_batch, 0, cache_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_gemm, 0, cache_size * sizeof(float)));

  const std::size_t total_k = 8 * num_kv_heads * max_context * head_dim;

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchAttention(
        d_q + t * num_heads * head_dim, d_k + t * num_kv_heads * head_dim,
        d_v + t * num_kv_heads * head_dim, d_gate + t * num_heads * head_dim,
        d_cache_seq, d_cache_seq + total_k, nullptr, nullptr,
        d_out_seq + t * num_heads * head_dim, 0, static_cast<std::uint32_t>(t),
        max_context, num_heads, num_kv_heads, head_dim);
  }

  // Batched
  strix::hip::LaunchBatchedAttention(d_q, d_k, d_v, d_gate, d_cache_batch,
                                     d_cache_batch + total_k, nullptr, nullptr,
                                     d_out_batch, 0, 0, batch, max_context,
                                     num_heads, num_kv_heads, head_dim);
  strix::hip::LaunchBatchedAttentionGemm(
      hipblas_handle, d_q, d_k, d_v, d_gate, d_cache_gemm,
      d_cache_gemm + total_k, nullptr, nullptr, d_scores, d_out_gemm, 0, 0,
      batch, max_context, num_heads, num_kv_heads, head_dim);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(q_size), res_batch(q_size), res_gemm(q_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_gemm.data(), d_out_gemm, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  float max_gemm_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
    const float gemm_diff = std::abs(res_seq[i] - res_gemm[i]);
    if (gemm_diff > max_gemm_diff)
      max_gemm_diff = gemm_diff;
  }
  std::cout << "Attention Seq vs Batch max diff: " << max_diff << "\n";
  std::cout << "Attention Seq vs GEMM max diff: " << max_gemm_diff << "\n";
  if (max_diff >= 1e-4F || max_gemm_diff >= 1e-4F) {
    std::cerr << "Batched attention mismatch\n";
    std::abort();
  }

  HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_cache_batch));
  HIP_CHECK(hipFree(d_cache_gemm));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
  HIP_CHECK(hipFree(d_out_gemm));
  HIP_CHECK(hipFree(d_scores));
}

void TestAttentionBackendEquivalence() {
  constexpr std::size_t batch = 128;
  constexpr std::uint32_t num_heads = 24;
  constexpr std::uint32_t num_kv_heads = 4;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t max_context = 128;
  const std::size_t attention_width = num_heads * head_dim;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t q_size = batch * attention_width;
  const std::size_t kv_size = batch * kv_width;
  const std::size_t cache_elements = num_kv_heads * max_context * head_dim;

  std::vector<float> h_q(q_size);
  std::vector<float> h_k(kv_size);
  std::vector<float> h_v(kv_size);
  std::vector<float> h_gate(q_size);
  for (std::size_t index = 0; index < q_size; ++index) {
    h_q[index] =
        0.15F * std::sin(static_cast<float>((index % 257) + 1) * 0.017F);
    h_gate[index] =
        0.5F * std::cos(static_cast<float>((index % 193) + 1) * 0.013F);
  }
  for (std::size_t index = 0; index < kv_size; ++index) {
    h_k[index] =
        0.2F * std::cos(static_cast<float>((index % 251) + 1) * 0.019F);
    h_v[index] =
        0.25F * std::sin(static_cast<float>((index % 239) + 1) * 0.023F);
  }
  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache_seq = nullptr, *d_cache_tile = nullptr, *d_cache_ck = nullptr;
  float *d_out_seq = nullptr, *d_out_tile = nullptr, *d_out_ck = nullptr;
  void *d_cache_tile_f16 = nullptr, *d_cache_ck_f16 = nullptr,
       *d_scratch_ck_f16 = nullptr;

  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_tile, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_ck, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_cache_tile_f16, 2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(
      hipMalloc(&d_cache_ck_f16, 2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_scratch_ck_f16, 2 * q_size * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_tile, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ck, q_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_cache_seq, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_ck, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(
      hipMemset(d_cache_ck_f16, 0, 2 * cache_elements * sizeof(hip_bfloat16)));

  for (std::size_t token = 0; token < batch; ++token) {
    strix::hip::LaunchAttention(d_q + token * attention_width,
                                d_k + token * kv_width, d_v + token * kv_width,
                                d_gate + token * attention_width, d_cache_seq,
                                d_cache_seq + cache_elements, nullptr, nullptr,
                                d_out_seq + token * attention_width, 0,
                                static_cast<std::uint32_t>(token), max_context,
                                num_heads, num_kv_heads, head_dim);
  }
  const bool tile_launched = strix::hip::LaunchBatchedAttentionTile(
      d_q, d_k, d_v, d_gate, d_cache_tile, d_cache_tile + cache_elements,
      d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile, 0, 0, batch, max_context, num_heads, num_kv_heads, head_dim);
  if (!tile_launched) {
    std::cerr << "Tiled attention rejected the Qwen shape\n";
    std::abort();
  }
  const bool launched = strix::hip::LaunchBatchedAttentionCk(
      d_q, d_k, d_v, d_gate, d_cache_ck, d_cache_ck + cache_elements,
      d_cache_ck_f16,
      static_cast<std::uint16_t*>(d_cache_ck_f16) + cache_elements,
      d_scratch_ck_f16, d_out_ck, 0, 0, batch, max_context, num_heads,
      num_kv_heads, head_dim);
  if (!launched) {
    std::cerr << "Composable Kernel attention rejected the Qwen shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> sequential(q_size), tiled(q_size), ck(q_size);
  std::vector<float> sequential_cache(2 * cache_elements);
  std::vector<float> tile_cache(2 * cache_elements);
  std::vector<float> ck_cache(2 * cache_elements);
  HIP_CHECK(hipMemcpy(sequential.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tiled.data(), d_out_tile, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ck.data(), d_out_ck, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sequential_cache.data(), d_cache_seq,
                      sequential_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_cache.data(), d_cache_tile,
                      tile_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ck_cache.data(), d_cache_ck,
                      ck_cache.size() * sizeof(float), hipMemcpyDeviceToHost));
  float max_tile_diff = 0.0F;
  float max_ck_diff = 0.0F;
  for (std::size_t index = 0; index < q_size; ++index) {
    max_tile_diff =
        std::max(max_tile_diff, std::abs(sequential[index] - tiled[index]));
    max_ck_diff =
        std::max(max_ck_diff, std::abs(sequential[index] - ck[index]));
  }
  float max_tile_cache_diff = 0.0F;
  float max_cache_diff = 0.0F;
  for (std::size_t index = 0; index < sequential_cache.size(); ++index) {
    max_tile_cache_diff =
        std::max(max_tile_cache_diff,
                 std::abs(sequential_cache[index] - tile_cache[index]));
    max_cache_diff = std::max(
        max_cache_diff, std::abs(sequential_cache[index] - ck_cache[index]));
  }
  std::cout << "Attention Seq vs tile max diff: " << max_tile_diff << "\n";
  std::cout << "Attention Seq vs tile cache max diff: " << max_tile_cache_diff
            << "\n";
  std::cout << "Attention Seq vs CK max diff: " << max_ck_diff << "\n";
  std::cout << "Attention Seq vs CK cache max diff: " << max_cache_diff << "\n";
  if (max_tile_diff >= 5e-3F || max_tile_cache_diff != 0.0F) {
    std::cerr << "Tiled attention mismatch\n";
    std::abort();
  }
  if (max_ck_diff >= 2e-3F || max_cache_diff != 0.0F) {
    std::cerr << "Composable Kernel attention mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipMemset(d_cache_tile, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemset(d_out_tile, 0, q_size * sizeof(float)));
  constexpr std::size_t chunk_size = batch / 2;
  const bool first_chunk_launched = strix::hip::LaunchBatchedAttentionTile(
      d_q, d_k, d_v, d_gate, d_cache_tile, d_cache_tile + cache_elements,
      d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile, 0, 0, chunk_size, max_context, num_heads, num_kv_heads,
      head_dim);
  const bool second_chunk_launched = strix::hip::LaunchBatchedAttentionTile(
      d_q + chunk_size * attention_width, d_k + chunk_size * kv_width,
      d_v + chunk_size * kv_width, d_gate + chunk_size * attention_width,
      d_cache_tile, d_cache_tile + cache_elements, d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile + chunk_size * attention_width, 0,
      static_cast<std::uint32_t>(chunk_size), chunk_size, max_context,
      num_heads, num_kv_heads, head_dim);
  if (!first_chunk_launched || !second_chunk_launched) {
    std::cerr << "Tiled attention rejected a chunked Qwen shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(tiled.data(), d_out_tile, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_cache.data(), d_cache_tile,
                      tile_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  float max_chunked_diff = 0.0F;
  float max_chunked_cache_diff = 0.0F;
  for (std::size_t index = 0; index < q_size; ++index) {
    max_chunked_diff =
        std::max(max_chunked_diff, std::abs(sequential[index] - tiled[index]));
  }
  for (std::size_t index = 0; index < sequential_cache.size(); ++index) {
    max_chunked_cache_diff =
        std::max(max_chunked_cache_diff,
                 std::abs(sequential_cache[index] - tile_cache[index]));
  }
  std::cout << "Attention Seq vs chunked tile max diff: " << max_chunked_diff
            << "\n";
  std::cout << "Attention Seq vs chunked tile cache max diff: "
            << max_chunked_cache_diff << "\n";
  if (max_chunked_diff >= 5e-3F || max_chunked_cache_diff != 0.0F) {
    std::cerr << "Chunked tiled attention mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_cache_tile));
  HIP_CHECK(hipFree(d_cache_ck));
  HIP_CHECK(hipFree(d_cache_tile_f16));
  HIP_CHECK(hipFree(d_cache_ck_f16));
  HIP_CHECK(hipFree(d_scratch_ck_f16));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_tile));
  HIP_CHECK(hipFree(d_out_ck));
}

void TestLongContextDecodeAttention() {
  constexpr std::uint32_t max_position = 32767;
  constexpr std::uint32_t max_context = max_position + 129;
  constexpr std::uint32_t num_heads = 6;
  constexpr std::uint32_t num_kv_heads = 1;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t positions[] = {4095, 8191, 16383, max_position};
  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_elements =
      static_cast<std::size_t>(num_kv_heads) * max_context * head_dim;

  std::vector<float> h_q(attention_width);
  std::vector<float> h_k(kv_width);
  std::vector<float> h_v(kv_width);
  std::vector<float> h_gate(attention_width);
  std::vector<float> h_cache(cache_elements);
  for (std::size_t index = 0; index < attention_width; ++index) {
    h_q[index] =
        0.1F * std::sin(static_cast<float>((index % 257) + 1) * 0.013F);
    h_gate[index] =
        0.4F * std::cos(static_cast<float>((index % 193) + 1) * 0.017F);
  }
  for (std::size_t index = 0; index < kv_width; ++index) {
    h_k[index] =
        0.08F * std::cos(static_cast<float>((index % 251) + 1) * 0.019F);
    h_v[index] =
        0.2F * std::sin(static_cast<float>((index % 239) + 1) * 0.023F);
  }
  for (std::size_t index = 0; index < cache_elements; ++index) {
    h_cache[index] =
        0.03F * std::sin(static_cast<float>((index % 509) + 1) * 0.011F);
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_k_cache = nullptr, *d_v_cache = nullptr, *d_out = nullptr;
  float* d_split_k_scratch = nullptr;
  HIP_CHECK(hipMalloc(&d_q, attention_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, attention_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_cache, cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_cache, cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, attention_width * sizeof(float)));
  HIP_CHECK(hipMalloc(
      &d_split_k_scratch,
      strix::hip::detail::DecodeAttentionScratchElements(num_heads, head_dim) *
          sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), attention_width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), attention_width * sizeof(float),
                      hipMemcpyHostToDevice));
  std::vector<float> output(attention_width);
  std::vector<float> reference(attention_width);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  for (const std::uint32_t position : positions) {
    HIP_CHECK(hipMemcpy(d_k_cache, h_cache.data(),
                        cache_elements * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_v_cache, h_cache.data(),
                        cache_elements * sizeof(float), hipMemcpyHostToDevice));

    strix::hip::LaunchAttention(d_q, d_k, d_v, d_gate, d_k_cache, d_v_cache,
                                nullptr, nullptr, d_out, 0, position,
                                max_context, num_heads, num_kv_heads, head_dim,
                                nullptr, d_split_k_scratch);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(output.data(), d_out, attention_width * sizeof(float),
                        hipMemcpyDeviceToHost));

    std::vector<double> scores(static_cast<std::size_t>(position) + 1);
    for (std::uint32_t head = 0; head < num_heads; ++head) {
      const std::size_t query_offset =
          static_cast<std::size_t>(head) * head_dim;
      double max_score = -std::numeric_limits<double>::infinity();
      for (std::uint32_t token = 0; token <= position; ++token) {
        double dot = 0.0;
        const std::size_t cache_offset =
            static_cast<std::size_t>(token) * head_dim;
        for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
          const float key =
              token == position ? h_k[dim] : h_cache[cache_offset + dim];
          dot += static_cast<double>(h_q[query_offset + dim]) * key;
        }
        scores[token] = dot * scale;
        max_score = std::max(max_score, scores[token]);
      }

      double denominator = 0.0;
      for (double& score : scores) {
        score = std::exp(score - max_score);
        denominator += score;
      }
      for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
        double weighted_value = 0.0;
        for (std::uint32_t token = 0; token <= position; ++token) {
          const std::size_t cache_offset =
              static_cast<std::size_t>(token) * head_dim;
          const float value =
              token == position ? h_v[dim] : h_cache[cache_offset + dim];
          weighted_value += scores[token] * value;
        }
        const double sigmoid =
            1.0 / (1.0 + std::exp(-h_gate[query_offset + dim]));
        reference[query_offset + dim] =
            static_cast<float>((weighted_value / denominator) * sigmoid);
      }
    }

    bool has_nonzero = false;
    float max_reference_diff = 0.0F;
    for (std::size_t index = 0; index < output.size(); ++index) {
      const float value = output[index];
      max_reference_diff =
          std::max(max_reference_diff, std::abs(value - reference[index]));
      if (!std::isfinite(value)) {
        std::cerr
            << "Long-context decode attention produced non-finite output\n";
        std::abort();
      }
      has_nonzero = has_nonzero || std::abs(value) > 1e-8F;
    }
    std::cout << "Split-K decode attention context " << (position + 1)
              << " max reference diff: " << max_reference_diff << "\n";
    if (max_reference_diff >= 5e-4F) {
      std::cerr << "Long-context decode attention mismatch\n";
      std::abort();
    }
    if (!has_nonzero) {
      std::cerr << "Long-context decode attention produced only zeroes\n";
      std::abort();
    }
  }

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_k_cache));
  HIP_CHECK(hipFree(d_v_cache));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_split_k_scratch));
}

void TestBatchedFusedProjectionsEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t hidden_size = 256;
  constexpr std::size_t qkv_size = 10240;
  constexpr std::size_t inner_size = 6144;
  constexpr std::size_t time_step_rank = 48;

  std::vector<float> h_x(batch * hidden_size, 0.5F);
  std::vector<float> h_qkv_w(qkv_size * hidden_size, 0.01F);
  std::vector<float> h_gate_w(inner_size * hidden_size, 0.02F);
  std::vector<float> h_alpha_w(time_step_rank * hidden_size, 0.03F);
  std::vector<float> h_beta_w(time_step_rank * hidden_size, 0.04F);

  float *d_x = nullptr, *d_qkv_w = nullptr, *d_gate_w = nullptr,
        *d_alpha_w = nullptr, *d_beta_w = nullptr;
  float *d_qkv_seq = nullptr, *d_qkv_batch = nullptr;
  float *d_gate_seq = nullptr, *d_gate_batch = nullptr;
  float *d_alpha_seq = nullptr, *d_alpha_batch = nullptr;
  float *d_beta_seq = nullptr, *d_beta_batch = nullptr;

  HIP_CHECK(hipMalloc(&d_x, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv_w, qkv_size * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_w, inner_size * hidden_size * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_alpha_w, time_step_rank * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_w, time_step_rank * hidden_size * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_qkv_seq, batch * qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv_batch, batch * qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_seq, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_batch, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_seq, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_batch, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_seq, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_batch, batch * time_step_rank * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), batch * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qkv_w, h_qkv_w.data(),
                      qkv_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_w, h_gate_w.data(),
                      inner_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_alpha_w, h_alpha_w.data(),
                      time_step_rank * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_beta_w, h_beta_w.data(),
                      time_step_rank * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchFusedSSMInputProjections(
        d_qkv_w, strix::core::GgmlType::kF32, d_gate_w,
        strix::core::GgmlType::kF32, d_alpha_w, strix::core::GgmlType::kF32,
        d_beta_w, strix::core::GgmlType::kF32, d_x + t * hidden_size,
        d_qkv_seq + t * qkv_size, d_gate_seq + t * inner_size,
        d_alpha_seq + t * time_step_rank, d_beta_seq + t * time_step_rank,
        hidden_size, qkv_size, inner_size, time_step_rank);
  }

  // Batched
  strix::hip::LaunchBatchedFusedSSMInputProjections(
      d_qkv_w, false, d_gate_w, false, d_alpha_w, false, d_beta_w, false, d_x,
      d_qkv_batch, d_gate_batch, d_alpha_batch, d_beta_batch, batch,
      hidden_size, qkv_size, inner_size, time_step_rank);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(batch * qkv_size);
  std::vector<float> res_batch(batch * qkv_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_qkv_seq,
                      batch * qkv_size * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_qkv_batch,
                      batch * qkv_size * sizeof(float), hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Fused SSM Projections Seq vs Batch max diff: " << max_diff
            << "\n";
  assert(max_diff < 1e-4F);

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_qkv_w));
  HIP_CHECK(hipFree(d_gate_w));
  HIP_CHECK(hipFree(d_alpha_w));
  HIP_CHECK(hipFree(d_beta_w));
  HIP_CHECK(hipFree(d_qkv_seq));
  HIP_CHECK(hipFree(d_qkv_batch));
  HIP_CHECK(hipFree(d_gate_seq));
  HIP_CHECK(hipFree(d_gate_batch));
  HIP_CHECK(hipFree(d_alpha_seq));
  HIP_CHECK(hipFree(d_alpha_batch));
  HIP_CHECK(hipFree(d_beta_seq));
  HIP_CHECK(hipFree(d_beta_batch));
}

void TestBatchedFusedSwiGLUEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t hidden_size = 2560;
  constexpr std::size_t intermediate_size = 9216;

  std::vector<float> h_x(batch * hidden_size, 0.5F);
  std::vector<float> h_gate_w(intermediate_size * hidden_size, 0.01F);
  std::vector<float> h_up_w(intermediate_size * hidden_size, 0.02F);

  float *d_x = nullptr, *d_gate_w = nullptr, *d_up_w = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr;

  HIP_CHECK(hipMalloc(&d_x, batch * hidden_size * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_gate_w, intermediate_size * hidden_size * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_up_w, intermediate_size * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, batch * intermediate_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), batch * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_w, h_gate_w.data(),
                      intermediate_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up_w, h_up_w.data(),
                      intermediate_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchFusedSwiGLUGEMV(
        d_gate_w, strix::core::GgmlType::kF32, d_up_w,
        strix::core::GgmlType::kF32, d_x + t * hidden_size,
        d_out_seq + t * intermediate_size, intermediate_size, hidden_size);
  }

  // Batched
  strix::hip::LaunchBatchedFusedSwiGLUGEMM(d_gate_w, false, d_up_w, false, d_x,
                                           d_out_batch, nullptr, batch,
                                           intermediate_size, hidden_size);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(batch * intermediate_size),
      res_batch(batch * intermediate_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq,
                      batch * intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch,
                      batch * intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Fused SwiGLU Seq vs Batch max diff: " << max_diff << "\n";
  assert(max_diff < 1e-4F);

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_gate_w));
  HIP_CHECK(hipFree(d_up_w));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
}

void TestBatchedRoPEEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_heads = 16;
  constexpr std::uint32_t num_kv_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t rotary_dim = 64;
  constexpr float rope_theta = 1000000.0F;

  const std::size_t q_size = batch * num_heads * head_dim;
  const std::size_t kv_size = batch * num_kv_heads * head_dim;

  std::vector<float> h_q(q_size), h_k(kv_size);
  for (std::size_t i = 0; i < q_size; ++i)
    h_q[i] = std::sin(static_cast<float>(i + 1));
  for (std::size_t i = 0; i < kv_size; ++i)
    h_k[i] = std::cos(static_cast<float>(i + 1));

  float *d_q_seq = nullptr, *d_k_seq = nullptr;
  float *d_q_batch = nullptr, *d_k_batch = nullptr;

  HIP_CHECK(hipMalloc(&d_q_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_seq, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_batch, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_batch, kv_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q_seq, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_seq, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_batch, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_batch, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchRoPE(d_q_seq + t * (num_heads * head_dim),
                           d_k_seq + t * (num_kv_heads * head_dim), num_heads,
                           num_kv_heads, head_dim, rotary_dim,
                           static_cast<std::uint32_t>(t), rope_theta);
  }

  // Batched
  strix::hip::LaunchBatchedRoPE(d_q_batch, d_k_batch, batch, num_heads,
                                num_kv_heads, head_dim, rotary_dim, 0,
                                rope_theta);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_q_seq(q_size), res_q_batch(q_size);
  std::vector<float> res_k_seq(kv_size), res_k_batch(kv_size);
  HIP_CHECK(hipMemcpy(res_q_seq.data(), d_q_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_q_batch.data(), d_q_batch, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_seq.data(), d_k_seq, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_batch.data(), d_k_batch, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_q_diff = 0.0F, max_k_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    const float d = std::abs(res_q_seq[i] - res_q_batch[i]);
    if (d > max_q_diff)
      max_q_diff = d;
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    const float d = std::abs(res_k_seq[i] - res_k_batch[i]);
    if (d > max_k_diff)
      max_k_diff = d;
  }
  std::cout << "RoPE Seq vs Batch max Q diff: " << max_q_diff
            << " K diff: " << max_k_diff << "\n";
  assert(max_q_diff < 1e-4F);
  assert(max_k_diff < 1e-4F);

  HIP_CHECK(hipFree(d_q_seq));
  HIP_CHECK(hipFree(d_k_seq));
  HIP_CHECK(hipFree(d_q_batch));
  HIP_CHECK(hipFree(d_k_batch));
}

void TestBatchedPerHeadRMSNormEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr float eps = 1e-6F;

  const std::size_t total_size = batch * num_heads * head_dim;
  std::vector<float> h_x(total_size), h_w(head_dim);
  for (std::size_t i = 0; i < total_size; ++i)
    h_x[i] = std::sin(static_cast<float>(i + 1));
  for (std::size_t i = 0; i < head_dim; ++i)
    h_w[i] = 1.0F + 0.1F * std::cos(static_cast<float>(i + 1));

  float *d_x_seq = nullptr, *d_x_batch = nullptr, *d_w = nullptr;
  HIP_CHECK(hipMalloc(&d_x_seq, total_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x_batch, total_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, head_dim * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x_seq, h_x.data(), total_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x_batch, h_x.data(), total_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));

  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchPerHeadRMSNorm(d_x_seq + t * (num_heads * head_dim), d_w,
                                     d_x_seq + t * (num_heads * head_dim),
                                     num_heads, head_dim, eps);
  }

  strix::hip::LaunchBatchedPerHeadRMSNorm(d_x_batch, d_w, d_x_batch, batch,
                                          num_heads, head_dim, eps);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(total_size), res_batch(total_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_x_seq, total_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_x_batch, total_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < total_size; ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "PerHeadRMSNorm Seq vs Batch max diff: " << max_diff << "\n";
  assert(max_diff < 1e-4F);

  HIP_CHECK(hipFree(d_x_seq));
  HIP_CHECK(hipFree(d_x_batch));
  HIP_CHECK(hipFree(d_w));
}

void TestBaselineToTiledKvCacheTransition() {
  constexpr std::size_t chunk0_size = 512;
  constexpr std::size_t chunk1_size = 1024;
  constexpr std::size_t total_batch = chunk0_size + chunk1_size;
  constexpr std::uint32_t num_heads = 24;
  constexpr std::uint32_t num_kv_heads = 4;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t max_context = 2048;

  const std::size_t attention_width = num_heads * head_dim;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t q_size = total_batch * attention_width;
  const std::size_t kv_size = total_batch * kv_width;
  const std::size_t cache_elements = num_kv_heads * max_context * head_dim;

  std::vector<float> h_q(q_size);
  std::vector<float> h_k(kv_size);
  std::vector<float> h_v(kv_size);
  std::vector<float> h_gate(q_size);

  for (std::size_t index = 0; index < q_size; ++index) {
    h_q[index] =
        0.15F * std::sin(static_cast<float>((index % 257) + 1) * 0.017F);
    h_gate[index] =
        0.5F * std::cos(static_cast<float>((index % 193) + 1) * 0.013F);
  }
  for (std::size_t index = 0; index < kv_size; ++index) {
    h_k[index] =
        0.2F * std::cos(static_cast<float>((index % 251) + 1) * 0.019F);
    h_v[index] =
        0.25F * std::sin(static_cast<float>((index % 239) + 1) * 0.023F);
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache_seq = nullptr, *d_out_seq = nullptr;
  float *d_cache_trans = nullptr, *d_out_trans = nullptr;
  void* d_cache_trans_f16 = nullptr;

  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_trans, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_trans, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_trans_f16,
                      2 * cache_elements * sizeof(std::uint16_t)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMemset(d_cache_seq, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_out_seq, 0, q_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_trans, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_out_trans, 0, q_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_trans_f16, 0,
                      2 * cache_elements * sizeof(std::uint16_t)));

  // 1. Golden sequential reference token-by-token
  for (std::size_t token = 0; token < total_batch; ++token) {
    strix::hip::LaunchAttention(d_q + token * attention_width,
                                d_k + token * kv_width, d_v + token * kv_width,
                                d_gate + token * attention_width, d_cache_seq,
                                d_cache_seq + cache_elements, nullptr, nullptr,
                                d_out_seq + token * attention_width, 0,
                                static_cast<std::uint32_t>(token), max_context,
                                num_heads, num_kv_heads, head_dim);
  }

  // 2. Incremental transition:
  // Chunk 0 (0..512): LaunchBatchedAttention (Baseline)
  strix::hip::LaunchBatchedAttention(
      d_q, d_k, d_v, d_gate, d_cache_trans, d_cache_trans + cache_elements,
      d_cache_trans_f16,
      static_cast<std::uint16_t*>(d_cache_trans_f16) + cache_elements,
      d_out_trans, 0, 0, chunk0_size, max_context, num_heads, num_kv_heads,
      head_dim);

  // Chunk 1 (512..1536): LaunchBatchedAttentionTile (Tiled FP16 at start_pos =
  // 512)
  const bool chunk1_ok = strix::hip::LaunchBatchedAttentionTile(
      d_q + chunk0_size * attention_width, d_k + chunk0_size * kv_width,
      d_v + chunk0_size * kv_width, d_gate + chunk0_size * attention_width,
      d_cache_trans, d_cache_trans + cache_elements, d_cache_trans_f16,
      static_cast<std::uint16_t*>(d_cache_trans_f16) + cache_elements,
      d_out_trans + chunk0_size * attention_width, 0,
      static_cast<std::uint32_t>(chunk0_size), chunk1_size, max_context,
      num_heads, num_kv_heads, head_dim);
  assert(chunk1_ok);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> golden(q_size);
  std::vector<float> transition(q_size);
  HIP_CHECK(hipMemcpy(golden.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(transition.data(), d_out_trans, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t token = 0; token < total_batch; ++token) {
    for (std::size_t i = 0; i < attention_width; ++i) {
      const float g = golden[token * attention_width + i];
      const float tr = transition[token * attention_width + i];
      const float diff = std::abs(g - tr);
      if (diff > 1e-3F || !std::isfinite(tr)) {
        std::cout << "Token " << token << " index " << i << " head "
                  << (i / head_dim) << " dim " << (i % head_dim)
                  << " golden=" << g << " trans=" << tr << " diff=" << diff
                  << "\n";
        goto done_diff;
      }
    }
  }
done_diff:
  float max_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    max_diff = std::max(max_diff, std::abs(golden[i] - transition[i]));
  }
  std::cout << "Baseline-to-Tiled transition max diff: " << max_diff << "\n";
  assert(max_diff < 5e-3F);

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_cache_trans));
  HIP_CHECK(hipFree(d_out_trans));
  HIP_CHECK(hipFree(d_cache_trans_f16));
}

void TestHipGraphDecodeStep() {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  constexpr std::size_t hidden_size = 256;
  constexpr std::size_t vocab_size = 1024;
  std::vector<float> h_embd(vocab_size * hidden_size, 1.0F);
  for (std::size_t i = 0; i < vocab_size * hidden_size; ++i) {
    h_embd[i] = 0.01F * static_cast<float>(i % 37);
  }

  void* d_embd = nullptr;
  float* d_hidden = nullptr;
  float* d_normed = nullptr;
  float* d_weight = nullptr;
  std::uint32_t* d_params = nullptr;

  HIP_CHECK(hipMalloc(&d_embd, vocab_size * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_hidden, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_weight, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_params, 4 * sizeof(std::uint32_t)));

  std::vector<float> h_w(hidden_size, 1.5F);
  HIP_CHECK(hipMemcpy(d_embd, h_embd.data(),
                      vocab_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_weight, h_w.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::detail::HipGraphDecodeExecutor executor;
  assert(executor.IsEnabled());
  assert(!executor.IsCaptured());

  // Test across 5 consecutive tokens with graph replay
  for (std::uint32_t step = 0; step < 5; ++step) {
    const std::uint32_t token_id = step * 10 + 3;
    const std::uint32_t pos = step;
    const std::uint32_t in_params[2] = {token_id, pos};
    HIP_CHECK(hipMemcpyAsync(d_params, in_params, sizeof(in_params),
                             hipMemcpyHostToDevice, stream));

    auto StepOps = [&]() {
      strix::hip::LaunchEmbeddingLookup(d_embd, strix::core::GgmlType::kF32,
                                        d_params + 0, d_hidden, hidden_size,
                                        stream);
      strix::hip::LaunchRMSNorm(d_hidden, d_weight, d_normed, hidden_size,
                                1e-6F, stream);
    };

    if (!executor.IsCaptured()) {
      const bool ok = executor.TryCapture(stream, StepOps);
      assert(ok);
      assert(executor.IsCaptured());
      const bool l_ok = executor.Launch(stream);
      assert(l_ok);
    } else {
      const bool ok = executor.Launch(stream);
      assert(ok);
    }

    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> h_out(hidden_size);
    HIP_CHECK(hipMemcpy(h_out.data(), d_normed, hidden_size * sizeof(float),
                        hipMemcpyDeviceToHost));

    // Verify RMSNorm output
    float sum_sq = 0.0F;
    for (std::size_t i = 0; i < hidden_size; ++i) {
      const float val = h_embd[token_id * hidden_size + i];
      sum_sq += val * val;
    }
    const float rms =
        std::sqrt(sum_sq / static_cast<float>(hidden_size) + 1e-6F);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      const float expected = (h_embd[token_id * hidden_size + i] / rms) * 1.5F;
      assert(std::abs(h_out[i] - expected) < 1e-4F);
    }
  }

  executor.Reset();
  assert(!executor.IsCaptured());

  HIP_CHECK(hipFree(d_embd));
  HIP_CHECK(hipFree(d_hidden));
  HIP_CHECK(hipFree(d_normed));
  HIP_CHECK(hipFree(d_weight));
  HIP_CHECK(hipFree(d_params));
  HIP_CHECK(hipStreamDestroy(stream));
}

// opt-c010-qk-rope-kv: fused per-head Q/K RMSNorm + RoPE + KV-cache write must
// reproduce the unfused decode chain bit-for-bit (same norm reduction, same
// powf/cosf/sinf RoPE arithmetic, same __float2half_rn cache writes).
void TestFusedQKNormRoPEKvWriteEquivalence() {
  constexpr std::uint32_t num_heads = 4;
  constexpr std::uint32_t num_kv_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t rotary_dim = 64;
  constexpr float rope_theta = 1000000.0F;
  constexpr float eps = 1e-6F;
  constexpr std::uint32_t layer_idx = 1;
  constexpr std::uint32_t max_context = 16;
  constexpr std::uint32_t pos = 3;

  const std::size_t q_size = num_heads * head_dim;
  const std::size_t kv_size = num_kv_heads * head_dim;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t f32_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * num_kv_heads * max_context *
      head_dim;
  const std::size_t f16_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * max_context * kv_width;

  std::vector<float> h_q(q_size), h_k(kv_size), h_v(kv_size);
  std::vector<float> h_q_w(head_dim), h_k_w(head_dim);
  std::vector<float> h_gate(q_size);
  for (std::size_t i = 0; i < q_size; ++i) {
    h_q[i] = 0.18F * std::sin(static_cast<float>(i + 1) * 0.05F);
    h_gate[i] = 0.3F * std::cos(static_cast<float>(i + 1) * 0.03F);
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    h_k[i] = 0.11F * std::cos(static_cast<float>(i + 1) * 0.07F);
    h_v[i] = 0.24F * std::sin(static_cast<float>(i + 1) * 0.04F);
  }
  for (std::size_t i = 0; i < head_dim; ++i) {
    h_q_w[i] = 0.9F + 0.04F * static_cast<float>(i % 17);
    h_k_w[i] = 1.1F - 0.03F * static_cast<float>(i % 13);
  }

  float *d_q_ref = nullptr, *d_k_ref = nullptr, *d_v_ref = nullptr;
  float *d_q_fus = nullptr, *d_k_fus = nullptr, *d_v_fus = nullptr;
  float *d_wq = nullptr, *d_wk = nullptr, *d_gate = nullptr;
  float *d_kc_ref = nullptr, *d_vc_ref = nullptr;
  float *d_kc_fus = nullptr, *d_vc_fus = nullptr;
  void *d_kc16_ref = nullptr, *d_vc16_ref = nullptr;
  void *d_kc16_fus = nullptr, *d_vc16_fus = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;
  std::uint32_t* d_pos = nullptr;

  HIP_CHECK(hipMalloc(&d_q_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_fus, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wq, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wk, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_kc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_pos, sizeof(std::uint32_t)));

  HIP_CHECK(hipMemcpy(d_q_ref, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_ref, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_ref, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_fus, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_fus, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_fus, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wq, h_q_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wk, h_k_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_pos, &pos, sizeof(std::uint32_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_kc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_kc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));

  // Unfused reference chain
  strix::hip::LaunchPerHeadRMSNorm(d_q_ref, d_wq, d_q_ref, num_heads, head_dim,
                                   eps);
  strix::hip::LaunchPerHeadRMSNorm(d_k_ref, d_wk, d_k_ref, num_kv_heads,
                                   head_dim, eps);
  strix::hip::LaunchRoPE(d_q_ref, d_k_ref, num_heads, num_kv_heads, head_dim,
                         rotary_dim, d_pos, rope_theta);
  strix::hip::LaunchAttention(d_q_ref, d_k_ref, d_v_ref, d_gate, d_kc_ref,
                              d_vc_ref, d_kc16_ref, d_vc16_ref, d_out_ref,
                              layer_idx, d_pos, max_context, num_heads,
                              num_kv_heads, head_dim, nullptr,
                              /*skip_kv_write=*/false);

  // Fused chain
  strix::hip::LaunchFusedQKNormRoPEKvWrite(
      d_q_fus, d_k_fus, d_v_fus, d_wq, d_wk, d_q_fus, d_k_fus, d_kc_fus,
      d_vc_fus, d_kc16_fus, d_vc16_fus, layer_idx, d_pos, max_context,
      num_heads, num_kv_heads, head_dim, rotary_dim, rope_theta, eps);
  strix::hip::LaunchAttention(d_q_fus, d_k_fus, d_v_fus, d_gate, d_kc_fus,
                              d_vc_fus, d_kc16_fus, d_vc16_fus, d_out_fus,
                              layer_idx, d_pos, max_context, num_heads,
                              num_kv_heads, head_dim, nullptr,
                              /*skip_kv_write=*/true);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_q_ref(q_size), res_q_fus(q_size);
  std::vector<float> res_k_ref(kv_size), res_k_fus(kv_size);
  std::vector<float> res_kc_ref(f32_cache_elems), res_kc_fus(f32_cache_elems);
  std::vector<float> res_vc_ref(f32_cache_elems), res_vc_fus(f32_cache_elems);
  std::vector<std::uint16_t> res_kc16_ref(f16_cache_elems),
      res_kc16_fus(f16_cache_elems);
  std::vector<std::uint16_t> res_vc16_ref(f16_cache_elems),
      res_vc16_fus(f16_cache_elems);
  std::vector<float> res_out_ref(q_size), res_out_fus(q_size);
  HIP_CHECK(hipMemcpy(res_q_ref.data(), d_q_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_q_fus.data(), d_q_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_ref.data(), d_k_ref, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_fus.data(), d_k_fus, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_ref.data(), d_kc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_fus.data(), d_kc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_ref.data(), d_vc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_fus.data(), d_vc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_ref.data(), d_kc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_fus.data(), d_kc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_ref.data(), d_vc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_fus.data(), d_vc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_ref.data(), d_out_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_fus.data(), d_out_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < q_size; ++i) {
    if (res_q_ref[i] != res_q_fus[i]) {
      std::cerr << "Fused Q mismatch at " << i << ": " << res_q_ref[i] << " vs "
                << res_q_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    if (res_k_ref[i] != res_k_fus[i]) {
      std::cerr << "Fused K mismatch at " << i << ": " << res_k_ref[i] << " vs "
                << res_k_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f32_cache_elems; ++i) {
    if (res_kc_ref[i] != res_kc_fus[i] || res_vc_ref[i] != res_vc_fus[i]) {
      std::cerr << "Fused FP32 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f16_cache_elems; ++i) {
    if (res_kc16_ref[i] != res_kc16_fus[i] ||
        res_vc16_ref[i] != res_vc16_fus[i]) {
      std::cerr << "Fused FP16 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  float max_out_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    max_out_diff =
        std::max(max_out_diff, std::abs(res_out_ref[i] - res_out_fus[i]));
  }
  std::cout << "FusedQKNormRoPEKvWrite decode: exact Q/K/cache match, max out "
               "diff "
            << max_out_diff << "\n";
  if (max_out_diff >= 1e-4F) {
    std::cerr << "Fused decode attention output mismatch: " << max_out_diff
              << "\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_q_ref));
  HIP_CHECK(hipFree(d_k_ref));
  HIP_CHECK(hipFree(d_v_ref));
  HIP_CHECK(hipFree(d_q_fus));
  HIP_CHECK(hipFree(d_k_fus));
  HIP_CHECK(hipFree(d_v_fus));
  HIP_CHECK(hipFree(d_wq));
  HIP_CHECK(hipFree(d_wk));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_kc_ref));
  HIP_CHECK(hipFree(d_vc_ref));
  HIP_CHECK(hipFree(d_kc_fus));
  HIP_CHECK(hipFree(d_vc_fus));
  HIP_CHECK(hipFree(d_kc16_ref));
  HIP_CHECK(hipFree(d_vc16_ref));
  HIP_CHECK(hipFree(d_kc16_fus));
  HIP_CHECK(hipFree(d_vc16_fus));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
  HIP_CHECK(hipFree(d_pos));
}

// opt-c010-qk-rope-kv: batched fuse must reproduce the per-token unfused
// prefill chain bit-for-bit for Q/K and both cache representations.
void TestBatchedFusedQKNormRoPEKvWriteEquivalence() {
  constexpr std::size_t batch = 8;
  constexpr std::uint32_t num_heads = 4;
  constexpr std::uint32_t num_kv_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t rotary_dim = 64;
  constexpr float rope_theta = 1000000.0F;
  constexpr float eps = 1e-6F;
  constexpr std::uint32_t layer_idx = 2;
  constexpr std::uint32_t max_context = 64;
  constexpr std::uint32_t start_pos = 5;

  const std::size_t head_total = num_heads * head_dim;
  const std::size_t kv_total = num_kv_heads * head_dim;
  const std::size_t q_size = batch * head_total;
  const std::size_t kv_size = batch * kv_total;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t f32_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * num_kv_heads * max_context *
      head_dim;
  const std::size_t f16_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * max_context * kv_width;

  std::vector<float> h_q(q_size), h_k(kv_size), h_v(kv_size);
  std::vector<float> h_q_w(head_dim), h_k_w(head_dim);
  std::vector<float> h_gate(q_size);
  for (std::size_t i = 0; i < q_size; ++i) {
    h_q[i] = 0.16F * std::sin(static_cast<float>(i + 1) * 0.04F);
    h_gate[i] = 0.25F * std::cos(static_cast<float>(i + 1) * 0.06F);
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    h_k[i] = 0.13F * std::cos(static_cast<float>(i + 1) * 0.05F);
    h_v[i] = 0.21F * std::sin(static_cast<float>(i + 1) * 0.08F);
  }
  for (std::size_t i = 0; i < head_dim; ++i) {
    h_q_w[i] = 0.95F + 0.03F * static_cast<float>(i % 19);
    h_k_w[i] = 1.05F - 0.02F * static_cast<float>(i % 11);
  }

  float *d_q_ref = nullptr, *d_k_ref = nullptr, *d_v_ref = nullptr;
  float *d_q_fus = nullptr, *d_k_fus = nullptr, *d_v_fus = nullptr;
  float *d_wq = nullptr, *d_wk = nullptr, *d_gate = nullptr;
  float *d_kc_ref = nullptr, *d_vc_ref = nullptr;
  float *d_kc_fus = nullptr, *d_vc_fus = nullptr;
  void *d_kc16_ref = nullptr, *d_vc16_ref = nullptr;
  void *d_kc16_fus = nullptr, *d_vc16_fus = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;

  HIP_CHECK(hipMalloc(&d_q_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_fus, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wq, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wk, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_kc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, q_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q_ref, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_ref, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_ref, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_fus, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_fus, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_fus, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wq, h_q_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wk, h_k_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_kc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_kc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));

  // Per-token unfused reference chain
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchPerHeadRMSNorm(d_q_ref + t * head_total, d_wq,
                                     d_q_ref + t * head_total, num_heads,
                                     head_dim, eps);
    strix::hip::LaunchPerHeadRMSNorm(d_k_ref + t * kv_total, d_wk,
                                     d_k_ref + t * kv_total, num_kv_heads,
                                     head_dim, eps);
    strix::hip::LaunchRoPE(d_q_ref + t * head_total, d_k_ref + t * kv_total,
                           num_heads, num_kv_heads, head_dim, rotary_dim,
                           static_cast<std::uint32_t>(start_pos + t),
                           rope_theta);
    strix::hip::LaunchAttention(
        d_q_ref + t * head_total, d_k_ref + t * kv_total,
        d_v_ref + t * kv_total, d_gate + t * head_total, d_kc_ref, d_vc_ref,
        d_kc16_ref, d_vc16_ref, d_out_ref + t * head_total, layer_idx,
        static_cast<std::uint32_t>(start_pos + t), max_context, num_heads,
        num_kv_heads, head_dim, nullptr, nullptr, /*skip_kv_write=*/false);
  }

  // Batched fused chain
  strix::hip::LaunchBatchedFusedQKNormRoPEKvWrite(
      d_q_fus, d_k_fus, d_v_fus, d_wq, d_wk, d_q_fus, d_k_fus, d_kc_fus,
      d_vc_fus, d_kc16_fus, d_vc16_fus, layer_idx, start_pos, batch,
      max_context, num_heads, num_kv_heads, head_dim, rotary_dim, rope_theta,
      eps);
  strix::hip::LaunchBatchedAttention(
      d_q_fus, d_k_fus, d_v_fus, d_gate, d_kc_fus, d_vc_fus, d_kc16_fus,
      d_vc16_fus, d_out_fus, layer_idx, start_pos, batch, max_context,
      num_heads, num_kv_heads, head_dim, nullptr,
      /*skip_kv_write=*/true);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_q_ref(q_size), res_q_fus(q_size);
  std::vector<float> res_k_ref(kv_size), res_k_fus(kv_size);
  std::vector<float> res_kc_ref(f32_cache_elems), res_kc_fus(f32_cache_elems);
  std::vector<float> res_vc_ref(f32_cache_elems), res_vc_fus(f32_cache_elems);
  std::vector<std::uint16_t> res_kc16_ref(f16_cache_elems),
      res_kc16_fus(f16_cache_elems);
  std::vector<std::uint16_t> res_vc16_ref(f16_cache_elems),
      res_vc16_fus(f16_cache_elems);
  std::vector<float> res_out_ref(q_size), res_out_fus(q_size);
  HIP_CHECK(hipMemcpy(res_q_ref.data(), d_q_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_q_fus.data(), d_q_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_ref.data(), d_k_ref, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_fus.data(), d_k_fus, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_ref.data(), d_kc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_fus.data(), d_kc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_ref.data(), d_vc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_fus.data(), d_vc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_ref.data(), d_kc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_fus.data(), d_kc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_ref.data(), d_vc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_fus.data(), d_vc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_ref.data(), d_out_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_fus.data(), d_out_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < q_size; ++i) {
    if (res_q_ref[i] != res_q_fus[i]) {
      std::cerr << "Batched fused Q mismatch at " << i << ": " << res_q_ref[i]
                << " vs " << res_q_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    if (res_k_ref[i] != res_k_fus[i]) {
      std::cerr << "Batched fused K mismatch at " << i << ": " << res_k_ref[i]
                << " vs " << res_k_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f32_cache_elems; ++i) {
    if (res_kc_ref[i] != res_kc_fus[i] || res_vc_ref[i] != res_vc_fus[i]) {
      std::cerr << "Batched fused FP32 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f16_cache_elems; ++i) {
    if (res_kc16_ref[i] != res_kc16_fus[i] ||
        res_vc16_ref[i] != res_vc16_fus[i]) {
      std::cerr << "Batched fused FP16 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  float max_out_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    max_out_diff =
        std::max(max_out_diff, std::abs(res_out_ref[i] - res_out_fus[i]));
  }
  std::cout << "BatchedFusedQKNormRoPEKvWrite prefill: exact Q/K/cache match, "
               "max out diff "
            << max_out_diff << "\n";
  if (max_out_diff >= 1e-4F) {
    std::cerr << "Batched fused prefill attention output mismatch: "
              << max_out_diff << "\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_q_ref));
  HIP_CHECK(hipFree(d_k_ref));
  HIP_CHECK(hipFree(d_v_ref));
  HIP_CHECK(hipFree(d_q_fus));
  HIP_CHECK(hipFree(d_k_fus));
  HIP_CHECK(hipFree(d_v_fus));
  HIP_CHECK(hipFree(d_wq));
  HIP_CHECK(hipFree(d_wk));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_kc_ref));
  HIP_CHECK(hipFree(d_vc_ref));
  HIP_CHECK(hipFree(d_kc_fus));
  HIP_CHECK(hipFree(d_vc_fus));
  HIP_CHECK(hipFree(d_kc16_ref));
  HIP_CHECK(hipFree(d_vc16_ref));
  HIP_CHECK(hipFree(d_kc16_fus));
  HIP_CHECK(hipFree(d_vc16_fus));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
}

// opt-c010-residual-rmsnorm: fused residual add + RMSNorm must reproduce the
// unfused decode chain (ResidualAdd then RMSNorm) bit-for-bit for both the
// residual-updated sum and the normed output.
void TestFusedResidualAddRMSNormEquivalence() {
  constexpr std::size_t dim = 5120;
  std::vector<float> h_a(dim), h_b(dim), h_w(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    h_a[i] = 0.17F * std::sin(static_cast<float>(i + 1) * 0.013F);
    h_b[i] = 0.09F * std::cos(static_cast<float>(i + 1) * 0.007F);
    h_w[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_w = nullptr;
  float *d_sum_ref = nullptr, *d_sum_fus = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_a, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sum_ref, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sum_fus, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ref, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, dim * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_a, h_a.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_b, h_b.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_w, h_w.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  // Unfused reference chain.
  strix::hip::LaunchResidualAdd(d_a, d_b, d_sum_ref, dim);
  strix::hip::LaunchRMSNorm(d_sum_ref, d_w, d_out_ref, dim, 1e-6F);

  // Fused kernel.
  strix::hip::LaunchFusedResidualAddRMSNorm(d_a, d_b, d_sum_fus, d_w, d_out_fus,
                                            dim, 1e-6F);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_sum_ref(dim), res_sum_fus(dim);
  std::vector<float> res_out_ref(dim), res_out_fus(dim);
  HIP_CHECK(hipMemcpy(res_sum_ref.data(), d_sum_ref, dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_sum_fus.data(), d_sum_fus, dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_ref.data(), d_out_ref, dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_fus.data(), d_out_fus, dim * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < dim; ++i) {
    if (res_sum_ref[i] != res_sum_fus[i]) {
      std::cerr << "Fused residual sum mismatch at " << i << ": "
                << res_sum_ref[i] << " vs " << res_sum_fus[i] << "\n";
      std::abort();
    }
    if (res_out_ref[i] != res_out_fus[i]) {
      std::cerr << "Fused residual RMSNorm mismatch at " << i << ": "
                << res_out_ref[i] << " vs " << res_out_fus[i] << "\n";
      std::abort();
    }
  }
  std::cout << "FusedResidualAddRMSNorm decode: exact sum and normed match\n";

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_sum_ref));
  HIP_CHECK(hipFree(d_sum_fus));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
}

// opt-c010-residual-rmsnorm: batched fused residual add + RMSNorm must
// reproduce the unfused prefill chain bit-for-bit for the residual sum, the
// FP32 normed output, and the BF16 normed output.
void TestBatchedFusedResidualAddRMSNormEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t dim = 5120;
  std::vector<float> h_a(batch * dim), h_b(batch * dim), h_w(dim);
  for (std::size_t i = 0; i < batch * dim; ++i) {
    h_a[i] = 0.23F * std::cos(static_cast<float>(i + 1) * 0.011F);
    h_b[i] = 0.11F * std::sin(static_cast<float>(i + 1) * 0.019F);
  }
  for (std::size_t i = 0; i < dim; ++i) {
    h_w[i] = 1.1F - 0.04F * static_cast<float>(i % 19);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_w = nullptr;
  float *d_sum_ref = nullptr, *d_sum_fus = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;
  void *d_out_bf16_ref = nullptr, *d_out_bf16_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_a, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sum_ref, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sum_fus, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ref, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_bf16_ref, batch * dim * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_bf16_fus, batch * dim * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), batch * dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), batch * dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_w, h_w.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  // Unfused reference chain.
  strix::hip::LaunchBatchedResidualAdd(d_a, d_b, d_sum_ref, batch, dim);
  strix::hip::LaunchBatchedRMSNorm(d_sum_ref, d_w, d_out_ref, d_out_bf16_ref,
                                   batch, dim, 1e-6F);

  // Fused kernel.
  strix::hip::LaunchBatchedFusedResidualAddRMSNorm(
      d_a, d_b, d_sum_fus, d_w, d_out_fus, d_out_bf16_fus, batch, dim, 1e-6F);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_sum_ref(batch * dim), res_sum_fus(batch * dim);
  std::vector<float> res_out_ref(batch * dim), res_out_fus(batch * dim);
  std::vector<std::uint16_t> res_bf16_ref(batch * dim),
      res_bf16_fus(batch * dim);
  HIP_CHECK(hipMemcpy(res_sum_ref.data(), d_sum_ref,
                      batch * dim * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_sum_fus.data(), d_sum_fus,
                      batch * dim * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_ref.data(), d_out_ref,
                      batch * dim * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_fus.data(), d_out_fus,
                      batch * dim * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_bf16_ref.data(), d_out_bf16_ref,
                      batch * dim * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_bf16_fus.data(), d_out_bf16_fus,
                      batch * dim * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < batch * dim; ++i) {
    if (res_sum_ref[i] != res_sum_fus[i]) {
      std::cerr << "Batched fused residual sum mismatch at " << i << ": "
                << res_sum_ref[i] << " vs " << res_sum_fus[i] << "\n";
      std::abort();
    }
    if (res_out_ref[i] != res_out_fus[i]) {
      std::cerr << "Batched fused residual RMSNorm mismatch at " << i << ": "
                << res_out_ref[i] << " vs " << res_out_fus[i] << "\n";
      std::abort();
    }
    if (res_bf16_ref[i] != res_bf16_fus[i]) {
      std::cerr << "Batched fused residual BF16 mismatch at " << i << "\n";
      std::abort();
    }
  }
  std::cout << "BatchedFusedResidualAddRMSNorm prefill: exact sum, normed, and "
               "BF16 match\n";

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_sum_ref));
  HIP_CHECK(hipFree(d_sum_fus));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
  HIP_CHECK(hipFree(d_out_bf16_ref));
  HIP_CHECK(hipFree(d_out_bf16_fus));
}

// opt-c010-ffn-swiglu: batched fused FFN gate/up projection + SwiGLU must
// match the unfused production prefill chain (BF16 gate GEMM, BF16 up GEMM,
// SwiGLU activation) under the declared arithmetic contract. The fused kernel
// consumes the FP32 normed input while the unfused chain consumes the BF16
// normed input, so a finite tolerance applies.
void TestBatchedFusedSwiGLUProductionEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t hidden_size = 512;
  constexpr std::size_t intermediate_size = 1024;

  std::vector<float> h_x(batch * hidden_size);
  std::vector<std::uint16_t> h_gate_w(intermediate_size * hidden_size);
  std::vector<std::uint16_t> h_up_w(intermediate_size * hidden_size);
  for (std::size_t i = 0; i < batch * hidden_size; ++i) {
    h_x[i] = 0.5F * std::sin(static_cast<float>(i + 1) * 0.037F);
  }
  for (std::size_t i = 0; i < intermediate_size * hidden_size; ++i) {
    h_gate_w[i] =
        FloatToBf16Bits(0.012F * std::cos(static_cast<float>(i + 1) * 0.0021F));
    h_up_w[i] =
        FloatToBf16Bits(0.017F * std::sin(static_cast<float>(i + 1) * 0.0017F));
  }

  float *d_x = nullptr, *d_x_bf16 = nullptr;
  void *d_gate_w = nullptr, *d_up_w = nullptr;
  float *d_gate = nullptr, *d_up = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;
  void *d_out_bf16_ref = nullptr, *d_out_bf16_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_x, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x_bf16, batch * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_gate_w,
                      intermediate_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_up_w,
                      intermediate_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_gate, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_up, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ref, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_bf16_ref,
                      batch * intermediate_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_bf16_fus,
                      batch * intermediate_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), batch * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_w, h_gate_w.data(),
                      intermediate_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up_w, h_up_w.data(),
                      intermediate_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  strix::hip::LaunchFloatToBfloat16(d_x, d_x_bf16, batch * hidden_size);

  hipblasHandle_t handle = nullptr;
  HIPBLAS_CHECK(hipblasCreate(&handle));

  // Unfused production chain: gate GEMM, up GEMM, SwiGLU activation.
  strix::hip::LaunchHipblasGEMMBF16(handle, d_gate_w, d_x_bf16, d_gate, batch,
                                    intermediate_size, hidden_size);
  strix::hip::LaunchHipblasGEMMBF16(handle, d_up_w, d_x_bf16, d_up, batch,
                                    intermediate_size, hidden_size);
  strix::hip::LaunchBatchedSwiGLUActivation(
      d_gate, d_up, d_out_ref, d_out_bf16_ref, batch * intermediate_size);

  // Fused kernel (consumes the FP32 normed input).
  strix::hip::LaunchBatchedFusedSwiGLUGEMM(d_gate_w, true, d_up_w, true, d_x,
                                           d_out_fus, d_out_bf16_fus, batch,
                                           intermediate_size, hidden_size);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_ref(batch * intermediate_size);
  std::vector<float> res_fus(batch * intermediate_size);
  std::vector<float> res_bf16_ref(batch * intermediate_size);
  std::vector<float> res_bf16_fus(batch * intermediate_size);
  HIP_CHECK(hipMemcpy(res_ref.data(), d_out_ref,
                      batch * intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_fus.data(), d_out_fus,
                      batch * intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_bf16_ref.data(), d_out_bf16_ref,
                      batch * intermediate_size * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_bf16_fus.data(), d_out_bf16_fus,
                      batch * intermediate_size * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F, max_bf16_diff = 0.0F, mean_diff = 0.0F;
  for (std::size_t i = 0; i < batch * intermediate_size; ++i) {
    const float d = std::abs(res_ref[i] - res_fus[i]);
    max_diff = std::max(max_diff, d);
    mean_diff += d;
    const float db = std::abs(Bf16BitsToFloat(res_bf16_ref[i]) -
                              Bf16BitsToFloat(res_bf16_fus[i]));
    max_bf16_diff = std::max(max_bf16_diff, db);
  }
  mean_diff /= static_cast<float>(batch * intermediate_size);
  std::cout << "BatchedFusedSwiGLU vs production chain: max_diff=" << max_diff
            << " mean_diff=" << mean_diff << " max_bf16_diff=" << max_bf16_diff
            << "\n";
  if (max_diff >= 1e-2F || max_bf16_diff >= 1e-2F) {
    std::cerr << "Fused SwiGLU GEMM does not match the unfused production "
                 "chain (max_diff "
              << max_diff << ", max_bf16_diff " << max_bf16_diff << ")\n";
    std::abort();
  }

  HIPBLAS_CHECK(hipblasDestroy(handle));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_x_bf16));
  HIP_CHECK(hipFree(d_gate_w));
  HIP_CHECK(hipFree(d_up_w));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_up));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
  HIP_CHECK(hipFree(d_out_bf16_ref));
  HIP_CHECK(hipFree(d_out_bf16_fus));
}

// opt-c010-ssm-gate-residual: the fused batched recurrence with the per-head
// post-RMSNorm + SiLU gate folded into the epilogue must match the unfused
// chain (BatchedSSMConvRecurrence + BatchedSSMPostNormGateKernel) bit-for-bit.
void TestBatchedSSMRecurrenceNormGateEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_key_heads = 16;
  constexpr std::uint32_t num_heads = 48;
  constexpr std::uint32_t key_dim = 128;
  constexpr std::uint32_t val_dim = 128;
  constexpr std::size_t qkv_dim =
      (2 * num_key_heads * key_dim) + (num_heads * val_dim);
  constexpr std::size_t inner_size = num_heads * val_dim;

  std::vector<float> h_qkv(batch * qkv_dim);
  for (std::size_t i = 0; i < h_qkv.size(); ++i) {
    h_qkv[i] = std::sin(static_cast<float>(i) * 0.01F);
  }
  std::vector<float> h_weights(qkv_dim * 4, 0.25F);
  std::vector<float> h_ssm_a(num_heads, -0.05F);
  std::vector<float> h_ssm_dt(num_heads, 0.01F);
  std::vector<float> h_ssm_norm(val_dim);
  std::vector<float> h_gate(batch * inner_size);
  for (std::size_t i = 0; i < val_dim; ++i) {
    h_ssm_norm[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }
  for (std::size_t i = 0; i < batch * inner_size; ++i) {
    h_gate[i] = 0.5F * std::sin(static_cast<float>(i + 1) * 0.013F);
  }

  float *d_qkv = nullptr, *d_w = nullptr;
  float *d_state_ref = nullptr, *d_state_fus = nullptr;
  float *d_conv_out_ref = nullptr, *d_conv_out_fus = nullptr;
  float *d_delta_ref = nullptr, *d_delta_fus = nullptr;
  float *d_alpha = nullptr, *d_beta = nullptr;
  float *d_ssm_a = nullptr, *d_ssm_dt = nullptr, *d_ssm_norm = nullptr;
  float* d_gate = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;

  const std::size_t delta_size = num_heads * key_dim * val_dim;
  HIP_CHECK(hipMalloc(&d_qkv, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_ref, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_fus, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_ref, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_fus, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_ref, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_fus, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_a, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_dt, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_norm, val_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ref, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, batch * inner_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(), batch * qkv_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_weights.data(), qkv_dim * 4 * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_state_ref, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_state_fus, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_ref, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_fus, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_alpha, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemset(d_beta, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_ssm_a, h_ssm_a.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_dt, h_ssm_dt.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_norm, h_ssm_norm.data(), val_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), batch * inner_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Unfused reference chain: conv + recurrence + post-norm gate.
  strix::hip::LaunchBatchedSSMConvRecurrence(
      d_qkv, d_w, d_state_ref, d_conv_out_ref, d_delta_ref, d_alpha, d_beta,
      d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate, d_out_ref, 0, batch, qkv_dim,
      num_key_heads, num_heads, key_dim, val_dim);

  // Fused: conv + recurrence with the norm+gate in the epilogue.
  strix::hip::LaunchBatchedSSMConvRecurrenceNormGate(
      d_qkv, d_w, d_state_fus, d_conv_out_fus, d_delta_fus, d_alpha, d_beta,
      d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate, d_out_fus, 0, batch, qkv_dim,
      num_key_heads, num_heads, key_dim, val_dim);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_ref(batch * inner_size);
  std::vector<float> res_fus(batch * inner_size);
  HIP_CHECK(hipMemcpy(res_ref.data(), d_out_ref,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_fus.data(), d_out_fus,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_ref.size(); ++i) {
    const float d = std::abs(res_ref[i] - res_fus[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Batched SSM recurrence+norm+gate fused vs unfused max diff: "
            << max_diff << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Fused SSM recurrence+norm+gate mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_state_ref));
  HIP_CHECK(hipFree(d_state_fus));
  HIP_CHECK(hipFree(d_conv_out_ref));
  HIP_CHECK(hipFree(d_conv_out_fus));
  HIP_CHECK(hipFree(d_delta_ref));
  HIP_CHECK(hipFree(d_delta_fus));
  HIP_CHECK(hipFree(d_alpha));
  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_ssm_a));
  HIP_CHECK(hipFree(d_ssm_dt));
  HIP_CHECK(hipFree(d_ssm_norm));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
}

// opt-c010-ssm-gate-residual: the GEMV residual epilogue (y = A*x + residual)
// must match the unfused chain (GEMV then ResidualAdd) bit-for-bit for both
// the Wave32 single-row strategy (decode ssm_out shape) and the block
// strategy.
void TestGEMVResidualEquivalence() {
  // Wave32 single-row: BF16 weights, M=5120, K=6144 (decode ssm_out shape).
  constexpr std::size_t M = 5120;
  constexpr std::size_t K = 6144;
  std::vector<std::uint16_t> h_A(M * K);
  std::vector<float> h_x(K);
  std::vector<float> h_res(M);
  for (std::size_t i = 0; i < M * K; ++i) {
    h_A[i] = FloatToBf16Bits(0.01F * std::sin(static_cast<float>(i) * 0.0007F));
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = 0.3F * std::cos(static_cast<float>(i) * 0.011F);
  }
  for (std::size_t i = 0; i < M; ++i) {
    h_res[i] = 0.25F * std::sin(static_cast<float>(i) * 0.017F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_res = nullptr;
  float *d_y_ref = nullptr, *d_y_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_res, M * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_ref, M * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_fus, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_res, h_res.data(), M * sizeof(float), hipMemcpyHostToDevice));

  strix::hip::LaunchGEMV(d_A, strix::core::GgmlType::kBF16, d_x, d_y_ref, M, K);
  strix::hip::LaunchResidualAdd(d_res, d_y_ref, d_y_ref, M);
  strix::hip::LaunchGEMVResidual(d_A, strix::core::GgmlType::kBF16, d_x, d_y_fus, d_res, M, K);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ref(M);
  std::vector<float> fus(M);
  HIP_CHECK(
      hipMemcpy(ref.data(), d_y_ref, M * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(fus.data(), d_y_fus, M * sizeof(float), hipMemcpyDeviceToHost));
  float max_diff = 0.0F;
  for (std::size_t i = 0; i < M; ++i) {
    const float d = std::abs(ref[i] - fus[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Wave32 GEMV+residual fused vs unfused max diff: " << max_diff
            << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Wave32 GEMV+residual mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_res));
  HIP_CHECK(hipFree(d_y_ref));
  HIP_CHECK(hipFree(d_y_fus));

  // Block strategy: FP32 weights, K >= 8192.
  constexpr std::size_t M2 = 2048;
  constexpr std::size_t K2 = 12288;
  std::vector<float> h_A2(M2 * K2);
  std::vector<float> h_x2(K2);
  std::vector<float> h_res2(M2);
  for (std::size_t i = 0; i < M2 * K2; ++i) {
    h_A2[i] = 0.008F * std::sin(static_cast<float>(i) * 0.0003F);
  }
  for (std::size_t i = 0; i < K2; ++i) {
    h_x2[i] = 0.2F * std::cos(static_cast<float>(i) * 0.007F);
  }
  for (std::size_t i = 0; i < M2; ++i) {
    h_res2[i] = 0.15F * std::sin(static_cast<float>(i) * 0.019F);
  }

  float *d_A2 = nullptr, *d_x2 = nullptr, *d_res2 = nullptr;
  float *d_y_ref2 = nullptr, *d_y_fus2 = nullptr;
  HIP_CHECK(hipMalloc(&d_A2, M2 * K2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x2, K2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_res2, M2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_ref2, M2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_fus2, M2 * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A2, h_A2.data(), M2 * K2 * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x2, h_x2.data(), K2 * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_res2, h_res2.data(), M2 * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchGEMV(d_A2, strix::core::GgmlType::kF32, d_x2, d_y_ref2, M2, K2);
  strix::hip::LaunchResidualAdd(d_res2, d_y_ref2, d_y_ref2, M2);
  strix::hip::LaunchGEMVResidual(d_A2, strix::core::GgmlType::kF32, d_x2, d_y_fus2, d_res2, M2, K2);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ref2(M2);
  std::vector<float> fus2(M2);
  HIP_CHECK(hipMemcpy(ref2.data(), d_y_ref2, M2 * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(fus2.data(), d_y_fus2, M2 * sizeof(float),
                      hipMemcpyDeviceToHost));
  max_diff = 0.0F;
  for (std::size_t i = 0; i < M2; ++i) {
    const float d = std::abs(ref2[i] - fus2[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Block GEMV+residual fused vs unfused max diff: " << max_diff
            << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Block GEMV+residual mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A2));
  HIP_CHECK(hipFree(d_x2));
  HIP_CHECK(hipFree(d_res2));
  HIP_CHECK(hipFree(d_y_ref2));
  HIP_CHECK(hipFree(d_y_fus2));
}

void TestLayerWeightPrefetch() {
  // opt-c014-layer-prefetch: the async page-touch must never modify the buffer
  // and must be safe on a trailing partial page (bounds-clamped reads).
  constexpr std::size_t kBytes = 4096U * 3U + 123U;
  std::vector<unsigned char> h_buf(kBytes);
  for (std::size_t i = 0; i < kBytes; ++i) {
    h_buf[i] = static_cast<unsigned char>((i * 31U) & 0xFFU);
  }
  void* d_buf = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, kBytes));
  HIP_CHECK(hipMemcpy(d_buf, h_buf.data(), kBytes, hipMemcpyHostToDevice));

  strix::hip::LaunchLayerWeightPrefetch(d_buf, kBytes);
  strix::hip::LaunchLayerWeightPrefetch(nullptr, kBytes);
  strix::hip::LaunchLayerWeightPrefetch(d_buf, 0);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<unsigned char> h_out(kBytes);
  HIP_CHECK(hipMemcpy(h_out.data(), d_buf, kBytes, hipMemcpyDeviceToHost));
  assert(h_out == h_buf);

  HIP_CHECK(hipFree(d_buf));
}

// opt-c010-rmsnorm-projection: fused layer pre-RMSNorm + QKV projections must
// match the unfused chain (RMSNorm then FusedQKVProjections) bit-for-bit.
void TestFusedRMSNormQKVProjectionsEquivalence() {
  constexpr std::size_t hidden_size = 1024;
  constexpr std::size_t q_dim = 512;
  constexpr std::size_t kv_dim = 128;
  constexpr float eps = 1e-6F;

  std::vector<float> h_x(hidden_size);
  std::vector<float> h_w(hidden_size);
  for (std::size_t i = 0; i < hidden_size; ++i) {
    h_x[i] = 0.3F * std::sin(static_cast<float>(i) * 0.017F);
    h_w[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }
  std::vector<std::uint16_t> h_qw(q_dim * hidden_size);
  std::vector<std::uint16_t> h_kw(kv_dim * hidden_size);
  std::vector<std::uint16_t> h_vw(kv_dim * hidden_size);
  for (std::size_t i = 0; i < q_dim * hidden_size; ++i) {
    h_qw[i] =
        FloatToBf16Bits(0.01F * std::sin(static_cast<float>(i) * 0.0021F));
  }
  for (std::size_t i = 0; i < kv_dim * hidden_size; ++i) {
    h_kw[i] =
        FloatToBf16Bits(0.013F * std::cos(static_cast<float>(i) * 0.0017F));
    h_vw[i] =
        FloatToBf16Bits(0.011F * std::sin(static_cast<float>(i) * 0.0013F));
  }

  float *d_x = nullptr, *d_w = nullptr, *d_normed = nullptr;
  void *d_qw = nullptr, *d_kw = nullptr, *d_vw = nullptr;
  float *d_q_ref = nullptr, *d_k_ref = nullptr, *d_v_ref = nullptr;
  float *d_q_fus = nullptr, *d_k_fus = nullptr, *d_v_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_x, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qw, q_dim * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_kw, kv_dim * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vw, kv_dim * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_q_ref, q_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_ref, kv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_ref, kv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_fus, q_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_fus, kv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_fus, kv_dim * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qw, h_qw.data(),
                      q_dim * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_kw, h_kw.data(),
                      kv_dim * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_vw, h_vw.data(),
                      kv_dim * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchRMSNorm(d_x, d_w, d_normed, hidden_size, eps);
  strix::hip::LaunchFusedQKVProjections(
      d_qw, strix::core::GgmlType::kBF16, d_kw, strix::core::GgmlType::kBF16,
      d_vw, strix::core::GgmlType::kBF16, d_normed, d_q_ref, d_k_ref, d_v_ref,
      q_dim, kv_dim, hidden_size);
  strix::hip::LaunchFusedRMSNormQKVProjections(
      d_x, d_w, eps, d_qw, true, d_kw, true, d_vw, true, d_q_fus, d_k_fus,
      d_v_fus, q_dim, kv_dim, hidden_size);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> q_ref(q_dim), k_ref(kv_dim), v_ref(kv_dim);
  std::vector<float> q_fus(q_dim), k_fus(kv_dim), v_fus(kv_dim);
  HIP_CHECK(hipMemcpy(q_ref.data(), d_q_ref, q_dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(k_ref.data(), d_k_ref, kv_dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(v_ref.data(), d_v_ref, kv_dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(q_fus.data(), d_q_fus, q_dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(k_fus.data(), d_k_fus, kv_dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(v_fus.data(), d_v_fus, kv_dim * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < q_dim; ++i) {
    max_diff = std::max(max_diff, std::abs(q_ref[i] - q_fus[i]));
  }
  for (std::size_t i = 0; i < kv_dim; ++i) {
    max_diff = std::max(max_diff, std::abs(k_ref[i] - k_fus[i]));
    max_diff = std::max(max_diff, std::abs(v_ref[i] - v_fus[i]));
  }
  std::cout << "Fused RMSNorm+QKV vs unfused max diff: " << max_diff << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Fused RMSNorm+QKV mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_normed));
  HIP_CHECK(hipFree(d_qw));
  HIP_CHECK(hipFree(d_kw));
  HIP_CHECK(hipFree(d_vw));
  HIP_CHECK(hipFree(d_q_ref));
  HIP_CHECK(hipFree(d_k_ref));
  HIP_CHECK(hipFree(d_v_ref));
  HIP_CHECK(hipFree(d_q_fus));
  HIP_CHECK(hipFree(d_k_fus));
  HIP_CHECK(hipFree(d_v_fus));
}

// opt-c010-rmsnorm-projection: fused layer pre-RMSNorm + SSM input projections
// (QKV, gate, alpha, beta) must match the unfused chain bit-for-bit.
void TestFusedRMSNormSSMInputProjectionsEquivalence() {
  constexpr std::size_t hidden_size = 1024;
  constexpr std::size_t qkv_size = 2048;
  constexpr std::size_t inner_size = 512;
  constexpr std::size_t time_step_rank = 16;
  constexpr float eps = 1e-6F;

  std::vector<float> h_x(hidden_size);
  std::vector<float> h_w(hidden_size);
  for (std::size_t i = 0; i < hidden_size; ++i) {
    h_x[i] = 0.3F * std::sin(static_cast<float>(i) * 0.017F);
    h_w[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }
  std::vector<std::uint16_t> h_qkv(qkv_size * hidden_size);
  std::vector<std::uint16_t> h_gate(inner_size * hidden_size);
  std::vector<std::uint16_t> h_alpha(time_step_rank * hidden_size);
  std::vector<std::uint16_t> h_beta(time_step_rank * hidden_size);
  for (std::size_t i = 0; i < qkv_size * hidden_size; ++i) {
    h_qkv[i] =
        FloatToBf16Bits(0.009F * std::cos(static_cast<float>(i) * 0.0019F));
  }
  for (std::size_t i = 0; i < inner_size * hidden_size; ++i) {
    h_gate[i] =
        FloatToBf16Bits(0.012F * std::sin(static_cast<float>(i) * 0.0011F));
  }
  for (std::size_t i = 0; i < time_step_rank * hidden_size; ++i) {
    h_alpha[i] =
        FloatToBf16Bits(0.007F * std::cos(static_cast<float>(i) * 0.0023F));
    h_beta[i] =
        FloatToBf16Bits(0.006F * std::sin(static_cast<float>(i) * 0.0029F));
  }

  float *d_x = nullptr, *d_w = nullptr, *d_normed = nullptr;
  void *d_qkv = nullptr, *d_gate = nullptr, *d_alpha = nullptr;
  void* d_beta = nullptr;
  float *d_qkv_ref = nullptr, *d_gate_ref = nullptr, *d_alpha_ref = nullptr;
  float* d_beta_ref = nullptr;
  float *d_qkv_fus = nullptr, *d_gate_fus = nullptr, *d_alpha_fus = nullptr;
  float* d_beta_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_x, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv, qkv_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(
      hipMalloc(&d_gate, inner_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_alpha,
                      time_step_rank * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(
      hipMalloc(&d_beta, time_step_rank * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_qkv_ref, qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_ref, inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_ref, time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_ref, time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv_fus, qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_fus, inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_fus, time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_fus, time_step_rank * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(),
                      qkv_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(),
                      inner_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_alpha, h_alpha.data(),
                      time_step_rank * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_beta, h_beta.data(),
                      time_step_rank * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchRMSNorm(d_x, d_w, d_normed, hidden_size, eps);
  strix::hip::LaunchFusedSSMInputProjections(
      d_qkv, strix::core::GgmlType::kBF16, d_gate,
      strix::core::GgmlType::kBF16, d_alpha, strix::core::GgmlType::kBF16,
      d_beta, strix::core::GgmlType::kBF16, d_normed,
      d_qkv_ref, d_gate_ref, d_alpha_ref, d_beta_ref, hidden_size, qkv_size,
      inner_size, time_step_rank);
  strix::hip::LaunchFusedRMSNormSSMInputProjections(
      d_x, d_w, eps, d_qkv, true, d_gate, true, d_alpha, true, d_beta, true,
      d_qkv_fus, d_gate_fus, d_alpha_fus, d_beta_fus, hidden_size, qkv_size,
      inner_size, time_step_rank);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> qkv_ref(qkv_size), gate_ref(inner_size);
  std::vector<float> alpha_ref(time_step_rank), beta_ref(time_step_rank);
  std::vector<float> qkv_fus(qkv_size), gate_fus(inner_size);
  std::vector<float> alpha_fus(time_step_rank), beta_fus(time_step_rank);
  HIP_CHECK(hipMemcpy(qkv_ref.data(), d_qkv_ref, qkv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gate_ref.data(), d_gate_ref, inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(alpha_ref.data(), d_alpha_ref,
                      time_step_rank * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(beta_ref.data(), d_beta_ref,
                      time_step_rank * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(qkv_fus.data(), d_qkv_fus, qkv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gate_fus.data(), d_gate_fus, inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(alpha_fus.data(), d_alpha_fus,
                      time_step_rank * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(beta_fus.data(), d_beta_fus,
                      time_step_rank * sizeof(float), hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < qkv_size; ++i) {
    max_diff = std::max(max_diff, std::abs(qkv_ref[i] - qkv_fus[i]));
  }
  for (std::size_t i = 0; i < inner_size; ++i) {
    max_diff = std::max(max_diff, std::abs(gate_ref[i] - gate_fus[i]));
  }
  for (std::size_t i = 0; i < time_step_rank; ++i) {
    max_diff = std::max(max_diff, std::abs(alpha_ref[i] - alpha_fus[i]));
    max_diff = std::max(max_diff, std::abs(beta_ref[i] - beta_fus[i]));
  }
  std::cout << "Fused RMSNorm+SSM-input vs unfused max diff: " << max_diff
            << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Fused RMSNorm+SSM-input mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_normed));
  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_alpha));
  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_qkv_ref));
  HIP_CHECK(hipFree(d_gate_ref));
  HIP_CHECK(hipFree(d_alpha_ref));
  HIP_CHECK(hipFree(d_beta_ref));
  HIP_CHECK(hipFree(d_qkv_fus));
  HIP_CHECK(hipFree(d_gate_fus));
  HIP_CHECK(hipFree(d_alpha_fus));
  HIP_CHECK(hipFree(d_beta_fus));
}

// opt-c010-rmsnorm-projection: fused layer pre-RMSNorm + FFN SwiGLU gate/up
// must match the unfused chain (RMSNorm then FusedSwiGLUGEMV) bit-for-bit.
void TestFusedRMSNormSwiGLUEquivalence() {
  constexpr std::size_t hidden_size = 1024;
  constexpr std::size_t intermediate_size = 2048;
  constexpr float eps = 1e-6F;

  std::vector<float> h_x(hidden_size);
  std::vector<float> h_w(hidden_size);
  for (std::size_t i = 0; i < hidden_size; ++i) {
    h_x[i] = 0.3F * std::sin(static_cast<float>(i) * 0.017F);
    h_w[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }
  std::vector<std::uint16_t> h_gate_w(intermediate_size * hidden_size);
  std::vector<std::uint16_t> h_up_w(intermediate_size * hidden_size);
  for (std::size_t i = 0; i < intermediate_size * hidden_size; ++i) {
    h_gate_w[i] =
        FloatToBf16Bits(0.012F * std::cos(static_cast<float>(i) * 0.0021F));
    h_up_w[i] =
        FloatToBf16Bits(0.017F * std::sin(static_cast<float>(i) * 0.0017F));
  }

  float *d_x = nullptr, *d_w = nullptr, *d_normed = nullptr;
  void *d_gate_w = nullptr, *d_up_w = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_x, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_w,
                      intermediate_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_up_w,
                      intermediate_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, intermediate_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_w, h_gate_w.data(),
                      intermediate_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up_w, h_up_w.data(),
                      intermediate_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchRMSNorm(d_x, d_w, d_normed, hidden_size, eps);
  strix::hip::LaunchFusedSwiGLUGEMV(d_gate_w, strix::core::GgmlType::kBF16,
                                    d_up_w, strix::core::GgmlType::kBF16,
                                    d_normed, d_out_ref, intermediate_size,
                                    hidden_size);
  strix::hip::LaunchFusedRMSNormSwiGLUGEMV(d_x, d_w, eps, d_gate_w, d_up_w,
                                           d_out_fus, intermediate_size,
                                           hidden_size);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ref(intermediate_size);
  std::vector<float> fus(intermediate_size);
  HIP_CHECK(hipMemcpy(ref.data(), d_out_ref, intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(fus.data(), d_out_fus, intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  float max_diff = 0.0F;
  for (std::size_t i = 0; i < intermediate_size; ++i) {
    max_diff = std::max(max_diff, std::abs(ref[i] - fus[i]));
  }
  std::cout << "Fused RMSNorm+SwiGLU vs unfused max diff: " << max_diff << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Fused RMSNorm+SwiGLU mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_normed));
  HIP_CHECK(hipFree(d_gate_w));
  HIP_CHECK(hipFree(d_up_w));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
}

// opt-c162-q8k-model: isolate the Q8_K integer-dot GEMV kernel correctness
// BEFORE it is wired into the pipeline. The kernel (Q8KBlockGEMVKernel +
// LaunchQ8KBlockGEMV) runs the dot at Q8: each 256-wide fp32 x block is
// quantized in-register (max|x| -> scale = max/127, xq = clamp(round(x/scale),
// -127, 127)) and accumulated as an int32 integer MAC with qs; the single fp
// scale (d_w * scale) is applied at block end. These checks validate index
// math, the 292B block stride, scale derivation, and the int-MAC exactly.
void TestQ8KBlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 256;
  constexpr std::size_t K = 512;  // K % 256 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q8KBlockTest = strix::quant::block_q8_K;
  static_assert(sizeof(Q8KBlockTest) == 292, "Q8_K block must be 292 bytes");

  // Deterministic pseudo-random generator (same values on every run).
  std::uint32_t seed = 12345U;
  auto rnd = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto rnd_float = [&rnd](float lo, float hi) -> float {
    const float u = static_cast<float>(rnd() & 0xFFFFU) / 65535.0F;
    return lo + u * (hi - lo);
  };

  // Synthetic Q8_K weights: qs in [-127,127], d in ~[-2,2]; x is fp32 in [-1,1].
  std::vector<Q8KBlockTest> h_A(M * num_blocks);
  std::vector<float> h_x(K);
  for (auto& blk : h_A) {
    blk.d = rnd_float(-2.0F, 2.0F);
    for (std::size_t i = 0; i < QK; ++i) {
      blk.qs[i] =
          static_cast<std::int8_t>(static_cast<int>(rnd() % 255) - 127);
    }
    for (std::size_t i = 0; i < 16; ++i) {
      blk.bsums[i] = 0;
    }
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = rnd_float(-1.0F, 1.0F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * num_blocks * sizeof(Q8KBlockTest)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * num_blocks * sizeof(Q8KBlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), K * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchQ8KBlockGEMV(d_A, strix::core::GgmlType::kQ8_K, d_x, d_y,
                                 M, K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, M * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Check 3: finite, no NaN/Inf.
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q8K GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
  }

  // Check 1 (PRIMARY, tight): CPU reference replicating the integer-Q8
  // arithmetic (max|x| -> scale, xq clamp, int32 MAC, end scale).
  std::vector<float> y_q8_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    float sumf = 0.0F;
    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q8KBlockTest& wblk = h_A[m * num_blocks + b];
      const float d_w = wblk.d;
      const float* xb = h_x.data() + b * QK;
      float local_max = 0.0F;
      for (std::size_t i = 0; i < QK; ++i) {
        local_max = std::max(local_max, std::abs(xb[i]));
      }
      const float scale = (local_max > 0.0F) ? (local_max / 127.0F) : 0.0F;
      int acc = 0;
      for (std::size_t i = 0; i < QK; ++i) {
        const float xv = (scale > 0.0F) ? (xb[i] / scale) : 0.0F;
        int xq = static_cast<int>(std::round(xv));
        xq = (xq > 127) ? 127 : xq;
        xq = (xq < -127) ? -127 : xq;
        acc += static_cast<int>(wblk.qs[i]) * xq;
      }
      sumf += d_w * scale * static_cast<float>(acc);
    }
    y_q8_ref[m] = sumf;
  }

  // Check 2 (SANITY, loose): CPU DotProductQ8_K (exact fp dequant dot of the
  // weights against the raw x, no activation quant) vs GPU. The difference is
  // the activation-quantization error, expected ~1%, not a sign/index bug.
  std::vector<float> y_dequant_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_dequant_ref[m] = strix::quant::DotProductQ8_K(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float primary_max_rel = 0.0F, primary_max_abs = 0.0F;
  float sanity_max_rel = 0.0F, sanity_max_abs = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    const float abs_q8 = std::abs(y_gpu[m] - y_q8_ref[m]);
    const float rel_q8 = abs_q8 / std::max(1e-6F, std::abs(y_q8_ref[m]));
    primary_max_rel = std::max(primary_max_rel, rel_q8);
    primary_max_abs = std::max(primary_max_abs, abs_q8);

    const float abs_san = std::abs(y_gpu[m] - y_dequant_ref[m]);
    const float rel_san =
        abs_san / std::max(1e-3F, std::abs(y_dequant_ref[m]));
    sanity_max_rel = std::max(sanity_max_rel, rel_san);
    sanity_max_abs = std::max(sanity_max_abs, abs_san);
  }

  std::cout << "Q8K GEMV PRIMARY (int-Q8 vs GPU): max_rel=" << primary_max_rel
            << " max_abs=" << primary_max_abs << "\n";
  std::cout << "Q8K GEMV SANITY (dequant vs GPU): max_rel=" << sanity_max_rel
            << " max_abs=" << sanity_max_abs << "\n";
  if (primary_max_rel >= 1e-3F || primary_max_abs >= 1e-3F) {
    std::cerr << "Q8K GEMV integer-dot mismatch (primary)\n";
    std::abort();
  }
  if (sanity_max_rel >= 2e-2F) {
    std::cerr
        << "Q8K GEMV dequant delta out of the expected quantization band\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

// opt-r2-q8_0: isolate the Q8_0 integer-dot GEMV kernel correctness. The
// generalized kernel (Q8KBlockGEMVKernel + LaunchQ8KBlockGEMV) runs the dot at
// Q8 with Q8_0 blocks (34B, QK=32): each 32-wide fp32 x block is quantized
// in-register (max|x| -> scale = max/127, xq = clamp(round(x/scale), -127,
// 127)) and accumulated as an int32 integer MAC with qs; the single fp scale
// (d_w * scale) is applied at block end. Validates the Q8_0 block stride, the
// half d read, the scale derivation, and the int-MAC exactly.
void TestQ8_0BlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 32;
  constexpr std::size_t K = 512;  // K % 32 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q8_0BlockTest = strix::quant::block_q8_0;
  static_assert(sizeof(Q8_0BlockTest) == 34, "Q8_0 block must be 34 bytes");

  // Deterministic pseudo-random generator (same values on every run).
  std::uint32_t seed = 12345U;
  auto rnd = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto rnd_float = [&rnd](float lo, float hi) -> float {
    const float u = static_cast<float>(rnd() & 0xFFFFU) / 65535.0F;
    return lo + u * (hi - lo);
  };
  auto float_to_half_bits = [](float f) -> std::uint16_t {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp =
        static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = (x >> 13) & 0x3FFu;
    if (exp <= 0) {
      if (exp < -10) {
        return static_cast<std::uint16_t>(sign);
      }
      mant |= 0x400u;
      const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
      return static_cast<std::uint16_t>(sign | (mant >> shift));
    }
    if (exp >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    return static_cast<std::uint16_t>(sign |
                                      (static_cast<std::uint32_t>(exp) << 10) |
                                      mant);
  };
  auto half_to_float = [](std::uint16_t h) -> float {
    const std::uint32_t sign = (h >> 15) & 1u;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x3FFu;
    std::uint32_t f;
    if (exp == 0) {
      if (mant == 0) {
        f = sign << 31;
      } else {
        std::uint32_t e = 0;
        std::uint32_t m = mant;
        while ((m & 0x400u) == 0) {
          m <<= 1;
          ++e;
        }
        m &= 0x3FFu;
        f = (sign << 31) | ((127 - 15 - e) << 23) | (m << 13);
      }
    } else if (exp == 31) {
      f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
      f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
  };

  // Synthetic Q8_0 weights: qs in [-127,127], d (as fp16) in ~[-2,2]; x is fp32
  // in [-1,1].
  std::vector<Q8_0BlockTest> h_A(M * num_blocks);
  std::vector<float> h_x(K);
  for (auto& blk : h_A) {
    blk.d = float_to_half_bits(rnd_float(-2.0F, 2.0F));
    for (std::size_t i = 0; i < QK; ++i) {
      blk.qs[i] =
          static_cast<std::int8_t>(static_cast<int>(rnd() % 255) - 127);
    }
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = rnd_float(-1.0F, 1.0F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * num_blocks * sizeof(Q8_0BlockTest)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * num_blocks * sizeof(Q8_0BlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), K * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchQ8KBlockGEMV(d_A, strix::core::GgmlType::kQ8_0, d_x, d_y,
                                 M, K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, M * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Check 3: finite, no NaN/Inf.
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q8_0 GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
  }

  // Check 1 (PRIMARY, tight): CPU reference replicating the integer-Q8
  // arithmetic (max|x| -> scale, xq clamp, int32 MAC, end scale).
  std::vector<float> y_q8_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    float sumf = 0.0F;
    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q8_0BlockTest& wblk = h_A[m * num_blocks + b];
      const float d_w = half_to_float(wblk.d);
      const float* xb = h_x.data() + b * QK;
      float local_max = 0.0F;
      for (std::size_t i = 0; i < QK; ++i) {
        local_max = std::max(local_max, std::abs(xb[i]));
      }
      const float scale = (local_max > 0.0F) ? (local_max / 127.0F) : 0.0F;
      int acc = 0;
      for (std::size_t i = 0; i < QK; ++i) {
        const float xv = (scale > 0.0F) ? (xb[i] / scale) : 0.0F;
        int xq = static_cast<int>(std::round(xv));
        xq = (xq > 127) ? 127 : xq;
        xq = (xq < -127) ? -127 : xq;
        acc += static_cast<int>(wblk.qs[i]) * xq;
      }
      sumf += d_w * scale * static_cast<float>(acc);
    }
    y_q8_ref[m] = sumf;
  }

  // Check 2 (SANITY, loose): CPU DotProductQ8_0 (exact fp dequant dot, no
  // activation quant) vs GPU. The difference is the activation-quantization
  // error, expected ~1%.
  std::vector<float> y_dequant_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_dequant_ref[m] = strix::quant::DotProductQ8_0(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float primary_max_rel = 0.0F, primary_max_abs = 0.0F;
  float sanity_max_rel = 0.0F, sanity_max_abs = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    const float abs_q8 = std::abs(y_gpu[m] - y_q8_ref[m]);
    const float rel_q8 = abs_q8 / std::max(1e-6F, std::abs(y_q8_ref[m]));
    primary_max_rel = std::max(primary_max_rel, rel_q8);
    primary_max_abs = std::max(primary_max_abs, abs_q8);

    const float abs_san = std::abs(y_gpu[m] - y_dequant_ref[m]);
    const float rel_san =
        abs_san / std::max(1e-3F, std::abs(y_dequant_ref[m]));
    sanity_max_rel = std::max(sanity_max_rel, rel_san);
    sanity_max_abs = std::max(sanity_max_abs, abs_san);
  }

  std::cout << "Q8_0 GEMV PRIMARY (int-Q8 vs GPU): max_rel=" << primary_max_rel
            << " max_abs=" << primary_max_abs << "\n";
  std::cout << "Q8_0 GEMV SANITY (dequant vs GPU): max_rel=" << sanity_max_rel
            << " max_abs=" << sanity_max_abs << "\n";
  if (primary_max_rel >= 1e-3F || primary_max_abs >= 1e-3F) {
    std::cerr << "Q8_0 GEMV integer-dot mismatch (primary)\n";
    std::abort();
  }
  if (sanity_max_rel >= 2e-2F) {
    std::cerr
        << "Q8_0 GEMV dequant delta out of the expected quantization band\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

// opt-r4-q5k-q6k: isolate the Q5_K dequant-to-fp GEMV kernel correctness. The
// generalized kernel (Q8KBlockGEMVKernel + LaunchQ8KBlockGEMV) dequantizes each
// 256-wide Q5_K block to fp (Q5KValue) and accumulates a fp MAC against x, one
// warp per row. Validates the 176B block stride, the Q5KValue dequant (fp16
// d/dmin, packed qs/qh, GetQKScaleMin scale/min unpack), and the warp
// reduction. Oracle = CPU DotProductQ5_K (dequant + dot). Dequantized values
// are bit-identical between CPU and GPU, so the delta is pure fp reordering
// noise.
void TestQ5KBlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 256;
  constexpr std::size_t K = 256;  // K % 256 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q5KBlockTest = strix::quant::block_q5_K;
  static_assert(sizeof(Q5KBlockTest) == 176, "Q5_K block must be 176 bytes");

  auto float_to_half_bits = [](float f) -> std::uint16_t {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp =
        static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = (x >> 13) & 0x3FFu;
    if (exp <= 0) {
      if (exp < -10) {
        return static_cast<std::uint16_t>(sign);
      }
      mant |= 0x400u;
      const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
      return static_cast<std::uint16_t>(sign | (mant >> shift));
    }
    if (exp >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7C00u |
                                        (mant ? 0x200u : 0u));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exp) << 10) | mant);
  };

  // Deterministic Q5_K weights: d=0.25, dmin=0.125; scales set so
  // GetQKScaleMin(0..7) = (4,1) via the high-bit encoding; qh full (0xFF) on
  // even rows / zero (0x00) on odd rows; qs nibbles 0x11/0x22/0x33/0x44 across
  // the four 64-element groups.
  std::vector<Q5KBlockTest> h_A(M * num_blocks);
  std::vector<float> h_x(K);
  const std::uint16_t d16 = float_to_half_bits(0.25F);
  const std::uint16_t dmin16 = float_to_half_bits(0.125F);
  const std::uint8_t scales[12] = {0x04, 0x04, 0x04, 0x04, 0x01, 0x01,
                                   0x01, 0x01, 0x14, 0x14, 0x14, 0x14};
  const std::uint8_t nibbles[4] = {0x11, 0x22, 0x33, 0x44};
  for (std::size_t r = 0; r < M * num_blocks; ++r) {
    Q5KBlockTest& blk = h_A[r];
    blk.d = d16;
    blk.dmin = dmin16;
    std::memcpy(blk.scales, scales, sizeof(scales));
    std::memset(blk.qh, (r % 2 == 0) ? 0xFF : 0x00, sizeof(blk.qh));
    for (std::size_t g = 0; g < 4; ++g) {
      std::memset(blk.qs + (g * 32), nibbles[g], 32);
    }
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = 0.1F + (static_cast<float>((i * 37 + 11) % 1000) / 2000.0F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * num_blocks * sizeof(Q5KBlockTest)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * num_blocks * sizeof(Q5KBlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), K * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchQ8KBlockGEMV(d_A, strix::core::GgmlType::kQ5_K, d_x, d_y,
                                 M, K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, M * sizeof(float),
                      hipMemcpyDeviceToHost));

  // CPU oracle: exact fp dequant + dot (no activation quantization).
  std::vector<float> y_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_ref[m] = strix::quant::DotProductQ5_K(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float max_rel = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q5_K GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
    const float rel =
        std::abs(y_gpu[m] - y_ref[m]) / std::max(1e-6F, std::abs(y_ref[m]));
    max_rel = std::max(max_rel, rel);
  }

  std::cout << "Q5_K GEMV (dequant-dot vs GPU): max_rel=" << max_rel << "\n";
  if (max_rel >= 1e-3F) {
    std::cerr << "Q5_K GEMV dequant-dot mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

// opt-r4-q5k-q6k: isolate the Q6_K dequant-to-fp GEMV kernel correctness. The
// generalized kernel (Q8KBlockGEMVKernel + LaunchQ8KBlockGEMV) dequantizes each
// 256-wide Q6_K block to fp (Q6KValue) and accumulates a fp MAC against x, one
// warp per row. Validates the 210B block stride, the Q6KValue dequant (fp16 d,
// packed ql/qh, int8 scales), and the warp reduction. Oracle = CPU
// DotProductQ6_K (dequant + dot), matched modulo fp reordering noise.
void TestQ6KBlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 256;
  constexpr std::size_t K = 256;  // K % 256 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q6KBlockTest = strix::quant::block_q6_K;
  static_assert(sizeof(Q6KBlockTest) == 210, "Q6_K block must be 210 bytes");

  auto float_to_half_bits = [](float f) -> std::uint16_t {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp =
        static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = (x >> 13) & 0x3FFu;
    if (exp <= 0) {
      if (exp < -10) {
        return static_cast<std::uint16_t>(sign);
      }
      mant |= 0x400u;
      const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
      return static_cast<std::uint16_t>(sign | (mant >> shift));
    }
    if (exp >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7C00u |
                                        (mant ? 0x200u : 0u));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exp) << 10) | mant);
  };

  // Deterministic Q6_K weights: d=0.3; ql varies per byte, qh=0xFF (every
  // 2-bit field = 3 -> quant = 16 + ql nibble, always positive), scales are
  // positive int8 [7..4]. No cancellation -> robust fp reordering comparison.
  std::vector<Q6KBlockTest> h_A(M * num_blocks);
  std::vector<float> h_x(K);
  const std::uint16_t d16 = float_to_half_bits(0.3F);
  for (std::size_t r = 0; r < M * num_blocks; ++r) {
    Q6KBlockTest& blk = h_A[r];
    blk.d = d16;
    for (std::size_t i = 0; i < 128; ++i) {
      blk.ql[i] = static_cast<std::uint8_t>((i * 13 + 5) & 0xFF);
    }
    std::memset(blk.qh, 0xFF, sizeof(blk.qh));
    for (std::size_t i = 0; i < 16; ++i) {
      blk.scales[i] = static_cast<std::int8_t>(7 - (i % 4));
    }
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = 0.1F + (static_cast<float>((i * 37 + 11) % 1000) / 2000.0F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * num_blocks * sizeof(Q6KBlockTest)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * num_blocks * sizeof(Q6KBlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), K * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchQ8KBlockGEMV(d_A, strix::core::GgmlType::kQ6_K, d_x, d_y,
                                 M, K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, M * sizeof(float),
                      hipMemcpyDeviceToHost));

  // CPU oracle: exact fp dequant + dot (no activation quantization).
  std::vector<float> y_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_ref[m] = strix::quant::DotProductQ6_K(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float max_rel = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q6_K GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
    const float rel =
        std::abs(y_gpu[m] - y_ref[m]) / std::max(1e-6F, std::abs(y_ref[m]));
    max_rel = std::max(max_rel, rel);
  }

  std::cout << "Q6_K GEMV (dequant-dot vs GPU): max_rel=" << max_rel << "\n";
  if (max_rel >= 1e-3F) {
    std::cerr << "Q6_K GEMV dequant-dot mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

void TestDequantizeQ8KToBf16Equivalence() {
  constexpr std::size_t QK = 256;
  constexpr std::size_t num_blocks = 4;
  constexpr std::size_t n_elems = num_blocks * QK;

  using Q8KBlockTest = strix::quant::block_q8_K;
  static_assert(sizeof(Q8KBlockTest) == 292, "Q8_K block must be 292 bytes");

  // Deterministic pseudo-random weights (same values every run).
  std::uint32_t seed = 777U;
  auto rnd = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto rnd_float = [&rnd](float lo, float hi) -> float {
    const float u = static_cast<float>(rnd() & 0xFFFFU) / 65535.0F;
    return lo + u * (hi - lo);
  };

  std::vector<Q8KBlockTest> h_w(num_blocks);
  for (auto& blk : h_w) {
    blk.d = rnd_float(-1.0F, 1.0F);
    for (std::size_t i = 0; i < QK; ++i) {
      blk.qs[i] =
          static_cast<std::int8_t>(static_cast<int>(rnd() % 255) - 127);
    }
    for (std::size_t i = 0; i < 16; ++i) {
      blk.bsums[i] = 0;
    }
  }

  // CPU reference oracle (full-precision float dequant).
  std::vector<float> h_ref(n_elems);
  strix::quant::DequantizeQ8_K(h_w.data(), h_ref.data(), n_elems);

  void* d_w = nullptr;
  hip_bfloat16* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_w, num_blocks * sizeof(Q8KBlockTest)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_out),
                      n_elems * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), num_blocks * sizeof(Q8KBlockTest),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchDequantizeQ8KToBf16(d_w, d_out, n_elems, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<hip_bfloat16> h_gpu(n_elems);
  HIP_CHECK(hipMemcpy(h_gpu.data(), d_out, n_elems * sizeof(hip_bfloat16),
                      hipMemcpyDeviceToHost));

  float max_rel = 0.0F, max_abs = 0.0F;
  for (std::size_t i = 0; i < n_elems; ++i) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &h_gpu[i], sizeof(bits));
    const float gpu = Bf16BitsToFloat(bits);
    const float ref = h_ref[i];
    if (!std::isfinite(gpu)) {
      std::cerr << "Q8K dequant produced non-finite output at " << i << ": "
                << gpu << "\n";
      std::abort();
    }
    const float abs_d = std::abs(gpu - ref);
    const float rel_d = abs_d / std::max(1e-3F, std::abs(ref));
    max_rel = std::max(max_rel, rel_d);
    max_abs = std::max(max_abs, abs_d);
  }
  std::cout << "Q8K dequant (GPU BF16 vs CPU): max_rel=" << max_rel
            << " max_abs=" << max_abs << "\n";
  if (max_rel >= 1e-2F) {
    std::cerr << "Q8K dequant BF16 mismatch (relative)\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));
}

// opt-c162-q8k-prefill-dequant (extended): shared BF16-vs-CPU oracle check.
// The device kernel writes (hip_bfloat16)(per_value_dequant), so the GPU value
// is the CPU fp32 oracle rounded to BF16. Compare in the fp32 domain with a
// small relative tolerance (BF16 has ~2-3 significant digits).
static void CheckDequantToBf16(const char* name, const std::vector<float>& ref,
                               const std::vector<hip_bfloat16>& gpu) {
  float max_rel = 0.0F, max_abs = 0.0F;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &gpu[i], sizeof(bits));
    const float g = Bf16BitsToFloat(bits);
    const float r = ref[i];
    if (!std::isfinite(g)) {
      std::cerr << name << " dequant produced non-finite output at " << i
                << ": " << g << "\n";
      std::abort();
    }
    const float abs_d = std::abs(g - r);
    const float rel_d = abs_d / std::max(1e-3F, std::abs(r));
    max_rel = std::max(max_rel, rel_d);
    max_abs = std::max(max_abs, abs_d);
  }
  std::cout << name << " dequant (GPU BF16 vs CPU): max_rel=" << max_rel
            << " max_abs=" << max_abs << "\n";
  if (max_rel >= 1e-2F) {
    std::cerr << name << " dequant BF16 mismatch (relative)\n";
    std::abort();
  }
}

// opt-c162-q8k-prefill-dequant (extended): validate the type-dispatched
// LaunchDequantizeToBf16 against the CPU per-type oracles (DequantizeQ8_0 /
// Q5_K / Q6_K / Q8_K). For each type we build a small deterministic quantized
// region, dequantize on the device to a BF16 scratch, and compare to the CPU
// oracle rounded to BF16.
void TestDequantizeToBf16Equivalence() {
  auto float_to_half_bits = [](float f) -> std::uint16_t {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp =
        static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = (x >> 13) & 0x3FFu;
    if (exp <= 0) {
      if (exp < -10) {
        return static_cast<std::uint16_t>(sign);
      }
      mant |= 0x400u;
      const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
      return static_cast<std::uint16_t>(sign | (mant >> shift));
    }
    if (exp >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7C00u |
                                        (mant ? 0x200u : 0u));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exp) << 10) | mant);
  };

  // ---- Q8_0: 2 blocks (64 elems), d=0.5, deterministic qs across [-127,127].
  constexpr std::size_t Q80_QK = 32;
  constexpr std::size_t Q80_BLOCKS = 2;
  constexpr std::size_t Q80_ELEMS = Q80_BLOCKS * Q80_QK;
  using Q8_0BlockTest = strix::quant::block_q8_0;
  static_assert(sizeof(Q8_0BlockTest) == 34, "Q8_0 block must be 34 bytes");

  std::vector<Q8_0BlockTest> h_q80(Q80_BLOCKS);
  for (auto& blk : h_q80) {
    blk.d = float_to_half_bits(0.5F);
    for (std::size_t i = 0; i < Q80_QK; ++i) {
      blk.qs[i] = static_cast<std::int8_t>((i * 13 + 7) % 255 - 127);
    }
  }
  std::vector<float> h_ref_q80(Q80_ELEMS);
  strix::quant::DequantizeQ8_0(h_q80.data(), h_ref_q80.data(), Q80_ELEMS);

  void* d_w = nullptr;
  hip_bfloat16* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_w, Q80_BLOCKS * sizeof(Q8_0BlockTest)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_out),
                      Q80_ELEMS * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(d_w, h_q80.data(), Q80_BLOCKS * sizeof(Q8_0BlockTest),
                      hipMemcpyHostToDevice));
  strix::hip::LaunchDequantizeToBf16(strix::core::GgmlType::kQ8_0, d_w, d_out,
                                     Q80_ELEMS, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<hip_bfloat16> h_gpu_q80(Q80_ELEMS);
  HIP_CHECK(hipMemcpy(h_gpu_q80.data(), d_out, Q80_ELEMS * sizeof(hip_bfloat16),
                      hipMemcpyDeviceToHost));
  CheckDequantToBf16("Q8_0", h_ref_q80, h_gpu_q80);
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));

  // ---- Q5_K: 1 block (256 elems), d=0.25 dmin=0.125, high-bit scales,
  // qh=0xFF, qs nibbles across the four 64-element groups.
  constexpr std::size_t QK = 256;
  using Q5KBlockTest = strix::quant::block_q5_K;
  static_assert(sizeof(Q5KBlockTest) == 176, "Q5_K block must be 176 bytes");

  Q5KBlockTest q5k{};
  q5k.d = float_to_half_bits(0.25F);
  q5k.dmin = float_to_half_bits(0.125F);
  const std::uint8_t scales[12] = {0x04, 0x04, 0x04, 0x04, 0x01, 0x01,
                                   0x01, 0x01, 0x14, 0x14, 0x14, 0x14};
  std::memcpy(q5k.scales, scales, sizeof(scales));
  std::memset(q5k.qh, 0xFF, sizeof(q5k.qh));
  const std::uint8_t nibbles[4] = {0x11, 0x22, 0x33, 0x44};
  for (std::size_t g = 0; g < 4; ++g) {
    std::memset(q5k.qs + (g * 32), nibbles[g], 32);
  }
  std::vector<float> h_ref_q5k(QK);
  strix::quant::DequantizeQ5_K(&q5k, h_ref_q5k.data(), QK);

  HIP_CHECK(hipMalloc(&d_w, sizeof(Q5KBlockTest)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_out),
                      QK * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(d_w, &q5k, sizeof(Q5KBlockTest), hipMemcpyHostToDevice));
  strix::hip::LaunchDequantizeToBf16(strix::core::GgmlType::kQ5_K, d_w, d_out,
                                     QK, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<hip_bfloat16> h_gpu_q5k(QK);
  HIP_CHECK(hipMemcpy(h_gpu_q5k.data(), d_out, QK * sizeof(hip_bfloat16),
                      hipMemcpyDeviceToHost));
  CheckDequantToBf16("Q5_K", h_ref_q5k, h_gpu_q5k);
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));

  // ---- Q6_K: 1 block, d=0.3, ql varies, qh=0xFF, scales [7..4] (positive).
  using Q6KBlockTest = strix::quant::block_q6_K;
  static_assert(sizeof(Q6KBlockTest) == 210, "Q6_K block must be 210 bytes");

  Q6KBlockTest q6k{};
  q6k.d = float_to_half_bits(0.3F);
  for (std::size_t i = 0; i < 128; ++i) {
    q6k.ql[i] = static_cast<std::uint8_t>((i * 13 + 5) & 0xFF);
  }
  std::memset(q6k.qh, 0xFF, sizeof(q6k.qh));
  for (std::size_t i = 0; i < 16; ++i) {
    q6k.scales[i] = static_cast<std::int8_t>(7 - (i % 4));
  }
  std::vector<float> h_ref_q6k(QK);
  strix::quant::DequantizeQ6_K(&q6k, h_ref_q6k.data(), QK);

  HIP_CHECK(hipMalloc(&d_w, sizeof(Q6KBlockTest)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_out),
                      QK * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(d_w, &q6k, sizeof(Q6KBlockTest), hipMemcpyHostToDevice));
  strix::hip::LaunchDequantizeToBf16(strix::core::GgmlType::kQ6_K, d_w, d_out,
                                     QK, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<hip_bfloat16> h_gpu_q6k(QK);
  HIP_CHECK(hipMemcpy(h_gpu_q6k.data(), d_out, QK * sizeof(hip_bfloat16),
                      hipMemcpyDeviceToHost));
  CheckDequantToBf16("Q6_K", h_ref_q6k, h_gpu_q6k);
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));

  // ---- Q8_K via the dispatcher (delegates to the existing Q8_K kernel):
  // 2 blocks (512 elems), deterministic d/qs.
  constexpr std::size_t Q8K_QK = 256;
  constexpr std::size_t Q8K_BLOCKS = 2;
  constexpr std::size_t Q8K_ELEMS = Q8K_BLOCKS * Q8K_QK;
  using Q8KBlockTest = strix::quant::block_q8_K;
  static_assert(sizeof(Q8KBlockTest) == 292, "Q8_K block must be 292 bytes");

  std::vector<Q8KBlockTest> h_q8k(Q8K_BLOCKS);
  for (std::size_t blk_idx = 0; blk_idx < Q8K_BLOCKS; ++blk_idx) {
    Q8KBlockTest& blk = h_q8k[blk_idx];
    blk.d = 0.5F + static_cast<float>(blk_idx);
    for (std::size_t i = 0; i < Q8K_QK; ++i) {
      blk.qs[i] = static_cast<std::int8_t>((i * 7 + 13) % 255 - 127);
    }
    std::memset(blk.bsums, 0, sizeof(blk.bsums));
  }
  std::vector<float> h_ref_q8k(Q8K_ELEMS);
  strix::quant::DequantizeQ8_K(h_q8k.data(), h_ref_q8k.data(), Q8K_ELEMS);

  HIP_CHECK(hipMalloc(&d_w, Q8K_BLOCKS * sizeof(Q8KBlockTest)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_out),
                      Q8K_ELEMS * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(d_w, h_q8k.data(), Q8K_BLOCKS * sizeof(Q8KBlockTest),
                      hipMemcpyHostToDevice));
  strix::hip::LaunchDequantizeToBf16(strix::core::GgmlType::kQ8_K, d_w, d_out,
                                     Q8K_ELEMS, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<hip_bfloat16> h_gpu_q8k(Q8K_ELEMS);
  HIP_CHECK(hipMemcpy(h_gpu_q8k.data(), d_out, Q8K_ELEMS * sizeof(hip_bfloat16),
                      hipMemcpyDeviceToHost));
  CheckDequantToBf16("Q8_K", h_ref_q8k, h_gpu_q8k);
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));
}

int main() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "No HIP device found, skipping GPU kernel tests.\n";
    return 0;
  }

  TestHipGraphDecodeStep();
  TestBaselineToTiledKvCacheTransition();
  TestGpuRMSNorm();
  TestGpuResidualAdd();
  TestGpuGEMV();
  TestBatchedGEMM();
  TestHipblasGEMM();
  TestHipblasLtGEMM();
  TestBatchedSSMConvEquivalence();
  TestBatchedAttentionEquivalence();
  TestAttentionBackendEquivalence();
  TestLongContextDecodeAttention();
  TestBatchedFusedProjectionsEquivalence();
  TestBatchedFusedSwiGLUEquivalence();
  TestBatchedRoPEEquivalence();
  TestBatchedPerHeadRMSNormEquivalence();
  TestFusedQKNormRoPEKvWriteEquivalence();
  TestBatchedFusedQKNormRoPEKvWriteEquivalence();
  TestFusedResidualAddRMSNormEquivalence();
  TestBatchedFusedResidualAddRMSNormEquivalence();
  TestBatchedFusedSwiGLUProductionEquivalence();
  TestBatchedSSMRecurrenceNormGateEquivalence();
  TestGEMVResidualEquivalence();
  TestLayerWeightPrefetch();
  TestFusedRMSNormQKVProjectionsEquivalence();
  TestFusedRMSNormSSMInputProjectionsEquivalence();
  TestFusedRMSNormSwiGLUEquivalence();
  TestQ8KBlockGEMVEquivalence();
  TestQ8_0BlockGEMVEquivalence();
  TestQ5KBlockGEMVEquivalence();
  TestQ6KBlockGEMVEquivalence();
  TestDequantizeQ8KToBf16Equivalence();
  TestDequantizeToBf16Equivalence();
  std::cout << "All Qwen HIP GPU kernel tests passed on gfx1151.\n";
  return 0;
}
#else
int main() {
  std::cout << "HIP disabled, skipping GPU kernel tests.\n";
  return 0;
}
#endif
