// Qwen-Image projection/score GEMM policy audit on gfx1151.
// Build: nix develop -c tools/bench/build.sh
// tools/models/qwen_image_21/qwen_image_gemm_bench.hip Compare each library
// candidate to sampled FP64 dot products on BF16 inputs.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <tuple>
#include <vector>

#include "src/models/qwen_image_21/hip/dense.hip"
#define H(x)               \
  do {                     \
    if ((x) != hipSuccess) \
      std::abort();        \
  } while (0)
#define B(x)                           \
  do {                                 \
    if ((x) != HIPBLAS_STATUS_SUCCESS) \
      std::abort();                    \
  } while (0)
unsigned short pack(float x) {
  auto u = std::bit_cast<unsigned>(x);
  return (u + 32767 + ((u >> 16) & 1)) >> 16;
}
float unpack(unsigned short x) {
  return std::bit_cast<float>(unsigned(x) << 16);
}
int main() {
  hipblasLtHandle_t h;
  B(hipblasLtCreate(&h));
  hipEvent_t a, b;
  H(hipEventCreate(&a));
  H(hipEventCreate(&b));
  for (auto [n, m, k, fp32] : {std::tuple{256, 4096, 4096, false},
                               {256, 12288, 4096, false},
                               {1024, 4096, 4096, false},
                               {4096, 4096, 4096, false},
                               {4096, 12288, 4096, false},
                               {4096, 4096, 12288, false},
                               {1024, 144, 1296, true},
                               {1024, 1152, 10368, true},
                               {4096, 1152, 1536, true},
                               {128, 4096, 128, true}}) {
    size_t wc = size_t(m) * k, xc = size_t(n) * k, oc = size_t(n) * m;
    const int replicas = std::max(4, int((64U << 20) / (wc * 2)) + 1);
    std::vector<unsigned short> w(wc * replicas), x(xc);
    for (size_t i = 0; i < w.size(); i++)
      w[i] = pack(float(int((i * 17) % 101) - 50) / 100);
    for (size_t i = 0; i < x.size(); i++)
      x[i] = pack(float(int((i * 7) % 97) - 48) / 100);
    void *dw, *dx, *dy;
    H(hipMalloc(&dw, w.size() * 2));
    H(hipMalloc(&dx, x.size() * 2));
    H(hipMalloc(&dy, oc * 4));
    H(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
    H(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    hipblasLtMatmulDesc_t op;
    hipblasLtMatrixLayout_t W, X, Y;
    hipblasLtMatmulPreference_t pref;
    B(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t tr = HIPBLAS_OP_T;
    B(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &tr,
                                      sizeof(tr)));
    B(hipblasLtMatrixLayoutCreate(&W, HIP_R_16BF, k, m, k));
    B(hipblasLtMatrixLayoutCreate(&X, HIP_R_16BF, k, n, k));
    B(hipblasLtMatrixLayoutCreate(&Y, fp32 ? HIP_R_32F : HIP_R_16BF, m, n, m));
    B(hipblasLtMatmulPreferenceCreate(&pref));
    size_t ws = 0;
    B(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    hipblasLtMatmulHeuristicResult_t r[16];
    int found = 0;
    B(hipblasLtMatmulAlgoGetHeuristic(h, op, W, X, Y, Y, pref, 16, r, &found));
    const float one = 1, zero = 0;
    printf("shape n=%d m=%d k=%d fp32=%d found=%d\n", n, m, k, fp32, found);
    for (int j = 0; j < found; j++) {
      if (r[j].state != HIPBLAS_STATUS_SUCCESS || r[j].workspaceSize)
        continue;
      auto name = hipblaslt_ext::getSolutionNameFromAlgo(h, r[j].algo);
      auto call = [&](int t) {
        B(hipblasLtMatmul(
            h, op, &one, static_cast<unsigned short*>(dw) + (t % replicas) * wc,
            W, dx, X, &zero, dy, Y, dy, Y, &r[j].algo, nullptr, 0, nullptr));
      };
      call(0);
      H(hipDeviceSynchronize());
      std::vector<unsigned short> hb;
      std::vector<float> hf;
      if (fp32) {
        hf.resize(oc);
        H(hipMemcpy(hf.data(), dy, oc * 4, hipMemcpyDeviceToHost));
      } else {
        hb.resize(oc);
        H(hipMemcpy(hb.data(), dy, oc * 2, hipMemcpyDeviceToHost));
      }
      double delta = 0, mag = 0;
      for (int s = 0; s < 64; s++) {
        int row = (s * 73) % n, col = (s * 199) % m;
        double ref = 0;
        for (int l = 0; l < k; l++)
          ref += double(unpack(w[size_t(col) * k + l])) *
                 unpack(x[size_t(row) * k + l]);
        double actual = fp32 ? hf[size_t(row) * m + col]
                             : unpack(hb[size_t(row) * m + col]);
        delta += (actual - ref) * (actual - ref);
        mag += ref * ref;
      }
      if (!std::isfinite(delta) ||
          std::sqrt(delta / mag) > (fp32 ? 1e-4 : 0.005)) {
        std::fprintf(stderr, "GEMM failed the independent FP64 check\n");
        return 1;
      }
      H(hipEventRecord(a));
      for (int t = 0; t < 5; t++)
        call(t);
      H(hipEventRecord(b));
      H(hipEventSynchronize(b));
      float ms;
      H(hipEventElapsedTime(&ms, a, b));
      printf("rank=%d ms=%.4f rel=%.8f algo=%s\n", j, ms / 5, sqrt(delta / mag),
             name.c_str());
      fflush(stdout);
    }
    if (!fp32 && gufo::models::qwen_image_21::hip::DenseShape(m, k)) {
      unsigned short *packed_w, *packed_x;
      H(hipMalloc(&packed_w, w.size() * 2));
      H(hipMalloc(&packed_x, xc * 2));
      for (int replica = 0; replica < replicas; ++replica)
        gufo::models::qwen_image_21::hip::PackDense16(
            static_cast<unsigned short*>(dw) + replica * wc,
            packed_w + replica * wc, m, k, nullptr);
      auto call = [&](int t) {
        // Include activation packing; weights are packed once when loaded.
        gufo::models::qwen_image_21::hip::PackDense16(
            static_cast<unsigned short*>(dx), packed_x, n, k, nullptr);
        return gufo::models::qwen_image_21::hip::DenseBf16(
            packed_w + (t % replicas) * wc, packed_x,
            static_cast<unsigned short*>(dy), n, m, k, nullptr);
      };
      if (call(0)) {
        H(hipGetLastError());
        std::vector<unsigned short> output(oc);
        H(hipMemcpy(output.data(), dy, oc * 2, hipMemcpyDeviceToHost));
        double error = 0, magnitude = 0;
        for (int s = 0; s < 64; ++s) {
          const int row = s * 73 % n, col = s * 199 % m;
          double expected = 0;
          for (int l = 0; l < k; ++l)
            expected += double(unpack(w[size_t(col) * k + l])) *
                        unpack(x[size_t(row) * k + l]);
          const double delta = unpack(output[size_t(row) * m + col]) - expected;
          error += delta * delta;
          magnitude += expected * expected;
        }
        const double relative = std::sqrt(error / magnitude);
        if (!std::isfinite(relative) || relative > 0.005)
          return 1;
        H(hipEventRecord(a));
        for (int t = 0; t < 5; ++t)
          call(t);
        H(hipEventRecord(b));
        H(hipEventSynchronize(b));
        float ms;
        H(hipEventElapsedTime(&ms, a, b));
        printf("native ms=%.4f rel=%.8f\n", ms / 5, relative);
        fflush(stdout);
      }
      hipFree(packed_x);
      hipFree(packed_w);
    }
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(Y);
    hipblasLtMatrixLayoutDestroy(X);
    hipblasLtMatrixLayoutDestroy(W);
    hipblasLtMatmulDescDestroy(op);
    hipFree(dy);
    hipFree(dx);
    hipFree(dw);
  }
  hipEventDestroy(a);
  hipEventDestroy(b);
  hipblasLtDestroy(h);
}
