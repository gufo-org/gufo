// hipBLASLt F16 x F16 -> F32 sweep over the Qwen3.8-Flash-Next dense prefill
// shapes at 2048 tokens; build with tools/bench/build.sh inside nix develop,
// passing this file's path, then run /tmp/dense_blaslt_sweep [tokens].
// Weights rotate over four buffers so the 32 MiB MALL cannot serve them.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <vector>
void Check(hipError_t x) {
  if (x != hipSuccess) {
    fprintf(stderr, "%s\n", hipGetErrorString(x));
    exit(1);
  }
}
void Check(hipblasStatus_t x) {
  if (x != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "BLAS %d\n", int(x));
    exit(1);
  }
}
int main(int argc, char** argv) {
  setbuf(stdout, nullptr);
  hipStream_t stream;
  Check(hipStreamCreate(&stream));
  hipblasLtHandle_t handle;
  Check(hipblasLtCreate(&handle));
  const int n = argc > 1 ? atoi(argv[1]) : 2048;  // tokens
  struct Shape {
    int m, k;
    const char* name;
  };
  std::vector<Shape> shapes = {
      {16384, 2560, "ssm_in"},   {2560, 6144, "ssm_out/attn_out"},
      {13312, 2560, "attn_qkv"}, {10240, 320, "hc_up"},
      {320, 10240, "hc_down"},   {2560, 2560, "shexp_down-ish"}};
  constexpr int kRot = 4;  // rotate weight buffers past the 32 MiB MALL
  puts("name,m,n,k,algo,us,tflops");
  for (auto sh : shapes) {
    const int m = sh.m, k = sh.k;
    std::vector<__half> a(size_t(m) * k), b(size_t(n) * k);
    unsigned rng = 123;
    for (auto* values : {&a, &b})
      for (auto& v : *values) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        v = __float2half_rn(float(int(rng % 8193) - 4096) / 4096.f);
      }
    __half* da[kRot];
    __half* db;
    float* dc;
    for (int r = 0; r < kRot; ++r) {
      Check(hipMalloc(&da[r], a.size() * 2));
      Check(hipMemcpy(da[r], a.data(), a.size() * 2, hipMemcpyHostToDevice));
    }
    Check(hipMalloc(&db, b.size() * 2));
    Check(hipMalloc(&dc, size_t(m) * n * 4));
    Check(hipMemcpy(db, b.data(), b.size() * 2, hipMemcpyHostToDevice));
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t la, lb, lc;
    Check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    Check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                          &ta, sizeof(ta)));
    Check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                          &tb, sizeof(tb)));
    Check(hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, k, m, k));
    Check(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, k, n, k));
    Check(hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F, m, n, m));
    hipblasLtMatmulPreference_t pref;
    Check(hipblasLtMatmulPreferenceCreate(&pref));
    size_t ws = 64ULL << 20;
    void* dws;
    Check(hipMalloc(&dws, ws));
    Check(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    std::array<hipblasLtMatmulHeuristicResult_t, 32> choices{};
    int found = 0;
    Check(hipblasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, lc, pref,
                                          choices.size(), choices.data(),
                                          &found));
    float one = 1, zero = 0;
    hipEvent_t e0, e1;
    Check(hipEventCreate(&e0));
    Check(hipEventCreate(&e1));
    double best_us = 1e30;
    int best_algo = -1;
    for (int algo = 0; algo < found; ++algo) {
      auto run = [&](int r) {
        return hipblasLtMatmul(handle, desc, &one, da[r], la, db, lb, &zero, dc,
                               lc, dc, lc, &choices[algo].algo, dws, ws,
                               stream);
      };
      if (run(0) != HIPBLAS_STATUS_SUCCESS)
        continue;
      for (int w = 0; w < 3; ++w)
        Check(run(w % kRot));
      Check(hipStreamSynchronize(stream));
      constexpr int kIters = 12;
      Check(hipEventRecord(e0, stream));
      for (int i = 0; i < kIters; ++i)
        Check(run(i % kRot));
      Check(hipEventRecord(e1, stream));
      Check(hipEventSynchronize(e1));
      float ms = 0;
      Check(hipEventElapsedTime(&ms, e0, e1));
      const double us = ms * 1000.0 / kIters;
      const double tflops = 2.0 * m * double(n) * k / (us * 1e-6) / 1e12;
      printf("%s,%d,%d,%d,%d,%.1f,%.1f\n", sh.name, m, n, k, algo, us, tflops);
      if (us < best_us) {
        best_us = us;
        best_algo = algo;
      }
    }
    printf("BEST %s m=%d n=%d k=%d algo=%d us=%.1f tflops=%.1f\n", sh.name, m,
           n, k, best_algo, best_us,
           2.0 * m * double(n) * k / (best_us * 1e-6) / 1e12);
    for (int r = 0; r < kRot; ++r)
      Check(hipFree(da[r]));
    Check(hipFree(db));
    Check(hipFree(dc));
    Check(hipFree(dws));
  }
  return 0;
}
