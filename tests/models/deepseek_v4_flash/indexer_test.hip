#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_indexer_topk.hip.hpp"

namespace {
void check(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(status));
    std::exit(1);
  }
}

// Previous dispatch for the measured range. These production fallback kernels
// also serve wider batches and contexts; the CPU ordering is the quality
// oracle.
void reference_topk(uint32_t* out, const float* scores, uint32_t* scratch,
                    uint32_t n, uint32_t tokens, int radix_bytes) {
  if (n <= 2048)
    indexer_topk_pow2_kernel<2048>
        <<<tokens, 1024>>>(out, scores, n, tokens, 512);
  else if (n < 4096)
    indexer_topk_pow2_kernel<4096>
        <<<tokens, 1024>>>(out, scores, n, tokens, 512);
  else if (n <= 8192 && radix_bytes)
    indexer_topk_8192_cub_kernel<<<tokens, 512, radix_bytes>>>(out, scores, n,
                                                               tokens, 512);
  else if (n == 4096)
    indexer_topk_pow2_kernel<4096>
        <<<tokens, 1024>>>(out, scores, n, tokens, 512);
  else if (n <= 8192)
    indexer_topk_pow2_u16_kernel<8192>
        <<<tokens, 1024>>>(out, scores, n, tokens, 512);
  else {
    const uint32_t chunks = (n + 4095) / 4096, stride = chunks * 512;
    indexer_topk_chunk_pow2_kernel<4096><<<dim3(tokens, chunks), 1024>>>(
        scratch, scores, n, tokens, 512, stride);
    indexer_topk_merge_pow2_kernel<4096><<<tokens, 1024>>>(
        out, scratch, scores, n, tokens, 512, stride, stride);
  }
}
}  // namespace

void check_indexer_selection(bool benchmark) {
  int device = 0, max_shared = 0;
  check(hipGetDevice(&device));
  check(hipDeviceGetAttribute(
      &max_shared, hipDeviceAttributeSharedMemPerBlockOptin, device));
  int radix_bytes =
      sizeof(hipcub::BlockRadixSort<uint64_t, 512, 16>::TempStorage);
  if (max_shared < radix_bytes ||
      hipFuncSetAttribute(
          reinterpret_cast<const void*>(indexer_topk_8192_cub_kernel),
          hipFuncAttributeMaxDynamicSharedMemorySize,
          radix_bytes) != hipSuccess)
    radix_bytes = 0;
  std::mt19937 random(731);
  std::uniform_real_distribution<float> values(-4.F, 4.F);
  for (uint32_t tokens : {1u, 6u}) {
    for (uint32_t n : {1025u, 2048u, 2049u, 4096u, 8192u, 16384u}) {
      std::vector<float> input(n * tokens);
      std::vector<uint32_t> expected(512 * tokens), actual(expected.size()),
          reference(expected.size());
      float* scores;
      uint32_t *out, *ref, *scratch;
      check(hipMalloc(&scores, input.size() * sizeof(float)));
      check(hipMalloc(&out, actual.size() * sizeof(uint32_t)));
      check(hipMalloc(&ref, actual.size() * sizeof(uint32_t)));
      check(hipMalloc(&scratch, tokens * 2048 * sizeof(uint32_t)));
      for (unsigned mode = 0; mode < 6; ++mode) {
        for (uint32_t t = 0; t < tokens; ++t) {
          for (uint32_t i = 0; i < n; ++i) {
            float value = values(random);
            if (mode == 1)
              value = std::round(value);
            if (mode == 2)
              value = (i % 2) ? -0.F : +0.F;
            if (mode == 3)
              value = (i % 11) ? -INFINITY : INFINITY;
            if (mode == 4)
              value = 128.F + 1e-3F * value;
            if (mode == 5 && i >= 256 + 5 * t)
              value = -INFINITY;
            input[t * n + i] = value;
          }
          std::vector<uint32_t> indices(n);
          std::iota(indices.begin(), indices.end(), 0u);
          std::partial_sort(indices.begin(), indices.begin() + 512,
                            indices.end(), [&](uint32_t a, uint32_t b) {
                              const float av = input[t * n + a],
                                          bv = input[t * n + b];
                              return av > bv || (av == bv && a < b);
                            });
          std::copy_n(indices.begin(), 512, expected.begin() + t * 512);
        }
        check(hipMemcpy(scores, input.data(), input.size() * sizeof(float),
                        hipMemcpyHostToDevice));
        reference_topk(ref, scores, scratch, n, tokens, radix_bytes);
        check(hipGetLastError());
        check(hipMemcpy(reference.data(), ref,
                        reference.size() * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
        for (int repeat = 0; repeat < 2; ++repeat) {
          check(hipMemset(out, 0xff, actual.size() * sizeof(uint32_t)));
          if (!indexer_partial_topk_launch(out, scores, n, tokens, 512)) {
            std::fprintf(stderr, "Partial selection did not dispatch\n");
            std::exit(8);
          }
          check(hipGetLastError());
          check(hipMemcpy(actual.data(), out, actual.size() * sizeof(uint32_t),
                          hipMemcpyDeviceToHost));
          if (actual != expected || reference != expected) {
            std::fprintf(
                stderr,
                "Top-k ordering mismatch: n=%u tokens=%u mode=%u repeat=%d\n",
                n, tokens, mode, repeat);
            std::exit(8);
          }
        }
        if (benchmark && (mode == 0 || mode == 2 || mode == 4)) {
          hipEvent_t start, end;
          check(hipEventCreate(&start));
          check(hipEventCreate(&end));
          for (int pass = 0; pass < 2; ++pass) {
            float ms[2]{};
            for (int order = 0; order < 2; ++order) {
              const bool candidate = (order + pass) % 2;
              auto launch = [&] {
                if (candidate)
                  (void)indexer_partial_topk_launch(out, scores, n, tokens,
                                                    512);
                else
                  reference_topk(ref, scores, scratch, n, tokens, radix_bytes);
              };
              for (int i = 0; i < 5; ++i)
                launch();
              check(hipEventRecord(start));
              for (int i = 0; i < 50; ++i)
                launch();
              check(hipEventRecord(end));
              check(hipEventSynchronize(end));
              check(hipEventElapsedTime(&ms[candidate], start, end));
            }
            std::printf(
                "selection n=%u tokens=%u mode=%u pass=%d old_us=%.3f "
                "new_us=%.3f\n",
                n, tokens, mode, pass, ms[0] * 20.F, ms[1] * 20.F);
          }
          check(hipEventDestroy(start));
          check(hipEventDestroy(end));
        }
      }
      for (void* ptr : {static_cast<void*>(scores), static_cast<void*>(out),
                        static_cast<void*>(ref), static_cast<void*>(scratch)})
        check(hipFree(ptr));
    }
  }
  std::puts(
      "Indexer selection: 72 cases exact twice against CPU ordering and "
      "fallback kernels, including signed-zero ties and causal masks");
}
