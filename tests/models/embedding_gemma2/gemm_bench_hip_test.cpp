// Times every GEMM shape the encoder issues through rocBLAS and through every
// workspace-free hipBLASLt heuristic result, and prints the best. It does not
// go through GemmLt. It asserts nothing, so it passes whenever the runs finish.
// Timings are meaningful only on gfx1151.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/embedding_gemma2/hip/blas.hpp"

namespace hip = gufo::models::embedding_gemma2::hip;

namespace {

constexpr int kMaxAlgorithms = 32;
constexpr int kRuns = 20;

struct Shape {
  const char* name;
  std::size_t m;
  std::size_t k;
};

double TimeMs(hipStream_t stream, const auto& launch) {
  hipEvent_t begin;
  hipEvent_t end;
  (void)hipEventCreate(&begin);
  (void)hipEventCreate(&end);
  for (int i = 0; i < 3; ++i) {
    launch();
  }
  (void)hipStreamSynchronize(stream);
  (void)hipEventRecord(begin, stream);
  for (int i = 0; i < kRuns; ++i) {
    launch();
  }
  (void)hipEventRecord(end, stream);
  (void)hipEventSynchronize(end);
  float milliseconds = 0.0F;
  (void)hipEventElapsedTime(&milliseconds, begin, end);
  (void)hipEventDestroy(begin);
  (void)hipEventDestroy(end);
  return milliseconds / kRuns;
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cerr << "SKIP embedding_gemma2_gemm_bench_hip_test: no GPU\n";
    return 77;
  }
  hipStream_t stream = nullptr;
  (void)hipStreamCreate(&stream);
  hipblasHandle_t blas = nullptr;
  (void)hipblasCreate(&blas);
  hipblasLtHandle_t lt = nullptr;
  (void)hipblasLtCreate(&lt);

  const std::vector<Shape> shapes = {
      {"qkv_sliding", 2048, 512}, {"qkv_full", 3072, 512},
      {"o_sliding", 512, 1024},   {"o_full", 512, 2048},
      {"gate_up", 4096, 512},     {"down", 512, 2048},
      {"ple_512", 512, 512},      {"final", 768, 512},
  };
  const std::vector<std::size_t> token_counts = {32, 128, 256, 512, 1024};

  void* weights = nullptr;
  void* inputs = nullptr;
  float* outputs = nullptr;
  (void)hipMalloc(&weights, 4096ULL * 2048 * 2);
  (void)hipMalloc(&inputs, 1024ULL * 2048 * 2);
  (void)hipMalloc(reinterpret_cast<void**>(&outputs), 1024ULL * 4096 * 4);
  (void)hipMemset(weights, 0x3c, 4096ULL * 2048 * 2);
  (void)hipMemset(inputs, 0x3c, 1024ULL * 2048 * 2);

  std::cout << "shape tokens | rocblas ms (TFLOPS) | lt rank: ms ...\n";
  for (const Shape& shape : shapes) {
    for (const std::size_t tokens : token_counts) {
      const double flops = 2.0 * double(tokens) * shape.m * shape.k;
      const double rocblas = TimeMs(stream, [&] {
        hip::LaunchGemmBf16(blas, weights, inputs, outputs, tokens, shape.m,
                            shape.k, stream);
      });
      std::cout << shape.name << " " << tokens << " | " << rocblas << " ("
                << flops / rocblas / 1e9 << ") |";

      hipblasLtMatmulDesc_t operation = nullptr;
      hipblasLtMatrixLayout_t a_layout = nullptr;
      hipblasLtMatrixLayout_t x_layout = nullptr;
      hipblasLtMatrixLayout_t y_layout = nullptr;
      (void)hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F,
                                      HIP_R_32F);
      (void)hipblasLtMatrixLayoutCreate(&a_layout, HIP_R_16BF, shape.k, shape.m,
                                        shape.k);
      (void)hipblasLtMatrixLayoutCreate(&x_layout, HIP_R_16BF, shape.k, tokens,
                                        shape.k);
      (void)hipblasLtMatrixLayoutCreate(&y_layout, HIP_R_32F, shape.m, tokens,
                                        shape.m);
      const hipblasOperation_t transpose = HIPBLAS_OP_T;
      (void)hipblasLtMatmulDescSetAttribute(operation,
                                            HIPBLASLT_MATMUL_DESC_TRANSA,
                                            &transpose, sizeof(transpose));
      hipblasLtMatmulPreference_t preference = nullptr;
      (void)hipblasLtMatmulPreferenceCreate(&preference);
      std::vector<hipblasLtMatmulHeuristicResult_t> results(kMaxAlgorithms);
      int found = 0;
      (void)hipblasLtMatmulAlgoGetHeuristic(
          lt, operation, a_layout, x_layout, y_layout, y_layout, preference,
          kMaxAlgorithms, results.data(), &found);
      const float alpha = 1.0F;
      const float beta = 0.0F;
      double best = 1e30;
      int best_rank = -1;
      for (int rank = 0; rank < found; ++rank) {
        if (results[rank].state != HIPBLAS_STATUS_SUCCESS ||
            results[rank].workspaceSize != 0) {
          continue;
        }
        const double ms = TimeMs(stream, [&] {
          (void)hipblasLtMatmul(lt, operation, &alpha, weights, a_layout,
                                inputs, x_layout, &beta, outputs, y_layout,
                                outputs, y_layout, &results[rank].algo, nullptr,
                                0, stream);
        });
        if (ms < best) {
          best = ms;
          best_rank = rank;
        }
        if (rank < 3) {
          std::cout << " r" << rank << ":" << ms;
        }
      }
      std::cout << " | best r" << best_rank << " " << best << " ("
                << flops / best / 1e9 << " TFLOPS) of " << found << '\n';
      (void)hipblasLtMatmulPreferenceDestroy(preference);
      (void)hipblasLtMatrixLayoutDestroy(y_layout);
      (void)hipblasLtMatrixLayoutDestroy(x_layout);
      (void)hipblasLtMatrixLayoutDestroy(a_layout);
      (void)hipblasLtMatmulDescDestroy(operation);
    }
  }
  (void)hipFree(outputs);
  (void)hipFree(inputs);
  (void)hipFree(weights);
  (void)hipblasLtDestroy(lt);
  (void)hipblasDestroy(blas);
  (void)hipStreamDestroy(stream);
  std::cout << "PASS embedding_gemma2_gemm_bench_hip_test\n";
  return 0;
}
