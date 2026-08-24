#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/quant_gemm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"

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
      h_A[i] = strix::test::FloatToBf16Bits(val);
    }
    for (std::size_t k = 0; k < K; ++k) {
      h_x[k] = 0.1F * static_cast<float>(static_cast<int>(k % 17) - 8);
    }

    // CPU reference computation
    for (std::size_t m = 0; m < M; ++m) {
      float dot = 0.0F;
      for (std::size_t k = 0; k < K; ++k) {
        dot += strix::test::Bf16BitsToFloat(h_A[m * K + k]) * h_x[k];
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
              << " h_A[0]=" << strix::test::Bf16BitsToFloat(h_A[0])
              << " h_x[0]=" << h_x[0] << " ref[0]=" << h_y_ref[0]
              << " y[0]=" << h_y[0] << "\n";

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
      h_A[i] = strix::test::FloatToBf16Bits(1.5F);
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

  std::vector<std::uint16_t> h_a(m * k, strix::test::FloatToBf16Bits(1.5F));
  std::vector<std::uint16_t> h_x(batch * k);
  for (std::size_t b = 0; b < batch; ++b) {
    const auto value = strix::test::FloatToBf16Bits(static_cast<float>(b + 1));
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

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "No HIP device found, skipping Qwen basic ops test.\n";
    return 0;
  }

  TestGpuRMSNorm();
  TestGpuResidualAdd();
  TestGpuGEMV();
  TestBatchedGEMM();
  TestHipblasGEMM();
  TestHipblasLtGEMM();
  std::cout << "Qwen basic ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen basic ops test.\n";
  return 0;
#endif
}
