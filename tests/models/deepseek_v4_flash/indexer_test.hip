#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_dspark_sampling.hip.hpp"
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

constexpr uint32_t reference_topk_max_keys = 32768;
}  // namespace

void check_dspark_candidates() {
  constexpr unsigned vocab = 523, rank = 256, hidden_width = 128;
  constexpr unsigned blocks = (vocab + 255) / 256, stride = vocab + 17;
  std::vector<unsigned char> weights(vocab * rank / 32 * 34);
  for (unsigned b = 0; b < vocab * rank / 32; ++b) {
    const __half scale = __float2half(0.001F * (1 + b % 7));
    std::memcpy(weights.data() + b * 34, &scale, 2);
    for (unsigned j = 0; j < 32; ++j)
      weights[b * 34 + 2 + j] = (unsigned char)(int8_t)((b + 3 * j) % 31 - 15);
  }
  const auto unpack = [&](unsigned i) {
    __half scale;
    std::memcpy(&scale, weights.data() + i / 32 * 34, 2);
    return __half2float(scale) * (int8_t)weights[i / 32 * 34 + 2 + i % 32];
  };
  unsigned char* gpu_weights;
  check(hipMalloc(&gpu_weights, weights.size()));
  check(hipMemcpy(gpu_weights, weights.data(), weights.size(),
                  hipMemcpyHostToDevice));
  for (unsigned requests : {1u, 2u, 3u, 4u, 6u, 8u}) {
    std::vector<float> logits(requests * stride),
        hidden(requests * hidden_width);
    std::vector<int32_t> previous(requests);
    for (unsigned c = 0; c < requests; ++c) {
      previous[c] = 7 + c * 13;
      for (unsigned v = 0; v < vocab; ++v)
        logits[c * stride + v] = 0.01F * ((v * 11 + c * 19) % 59);
      for (unsigned j = 0; j < hidden_width; ++j)
        hidden[c * hidden_width + j] = 0.02F * ((j + 17 * c) % 19);
    }
    float *gpu_logits, *gpu_hidden;
    int32_t* gpu_previous;
    uint64_t* partial;
    ds4_dspark_candidates* out;
    check(hipMalloc(&gpu_logits, logits.size() * 4));
    check(hipMalloc(&gpu_hidden, hidden.size() * 4));
    check(hipMalloc(&gpu_previous, previous.size() * 4));
    check(hipMalloc(&partial, requests * blocks * 8 * 8));
    check(hipMalloc(&out, requests * sizeof(*out)));
    check(hipMemcpy(gpu_logits, logits.data(), logits.size() * 4,
                    hipMemcpyHostToDevice));
    check(hipMemcpy(gpu_hidden, hidden.data(), hidden.size() * 4,
                    hipMemcpyHostToDevice));
    check(hipMemcpy(gpu_previous, previous.data(), previous.size() * 4,
                    hipMemcpyHostToDevice));
    dspark_markov_candidates_kernel<<<dim3(blocks, requests), 256>>>(
        partial, gpu_logits, stride, gpu_previous, gpu_weights, gpu_weights,
        vocab, rank);
    dspark_merge_candidates_kernel<<<requests, 256>>>(
        out, partial, blocks, gpu_hidden, hidden_width, hidden_width,
        gpu_previous, gpu_weights, gpu_weights, rank);
    std::vector<ds4_dspark_candidates> actual(requests);
    check(hipMemcpy(actual.data(), out, requests * sizeof(*out),
                    hipMemcpyDeviceToHost));
    for (unsigned c = 0; c < requests; ++c) {
      std::vector<float> corrected(vocab);
      std::vector<unsigned> indices(vocab);
      std::iota(indices.begin(), indices.end(), 0);
      for (unsigned v = 0; v < vocab; ++v) {
        double sum = logits[c * stride + v];
        for (unsigned j = 0; j < rank; ++j)
          sum += (double)unpack(v * rank + j) * unpack(previous[c] * rank + j);
        corrected[v] = sum;
      }
      std::partial_sort(indices.begin(), indices.begin() + 8, indices.end(),
                        [&](unsigned a, unsigned b) {
                          return corrected[a] > corrected[b] ||
                                 (corrected[a] == corrected[b] && a < b);
                        });
      double confidence = 0;
      for (unsigned j = 0; j < hidden_width + rank; ++j)
        confidence +=
            (double)unpack(j) *
            (j < hidden_width ? hidden[c * hidden_width + j]
                              : unpack(previous[c] * rank + j - hidden_width));
      confidence = 1.0 / (1 + std::exp(-confidence));
      for (unsigned k = 0; k < 8; ++k) {
        if (actual[c].ids[k] != (int32_t)indices[k] ||
            std::abs(actual[c].logits[k] - corrected[indices[k]]) > 2e-5F) {
          std::fprintf(stderr,
                       "DSpark Markov top-k oracle failed C=%u row=%u k=%u\n",
                       requests, c, k);
          std::exit(1);
        }
      }
      if (std::abs(actual[c].confidence - confidence) > 2e-6) {
        std::fprintf(stderr, "DSpark pre-norm confidence oracle failed\n");
        std::exit(1);
      }
    }
    check(hipFree(out));
    check(hipFree(partial));
    check(hipFree(gpu_previous));
    check(hipFree(gpu_hidden));
    check(hipFree(gpu_logits));
  }
  check(hipFree(gpu_weights));
  std::puts(
      "DSpark Markov candidates and pre-norm confidence: C1/2/3/4/6/8 pass");
}

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
  unsigned cases_passed = 0;
  for (uint32_t tokens : {1u, 6u, 33u, 2048u}) {
    for (uint32_t n : {1025u, 2048u, 2049u, 4096u, 8192u, 8704u, 9217u, 16384u,
                       16385u, 32768u, 32769u, 47104u, 65536u}) {
      if ((tokens == 33 && n < 4096) ||
          (tokens == 2048 && n != 8704 && n != 16385))
        continue;
      std::vector<float> input(n * tokens);
      std::vector<uint32_t> expected(512 * tokens), actual(expected.size()),
          reference(expected.size());
      float* scores;
      uint32_t *out, *ref, *scratch;
      check(hipMalloc(&scores, input.size() * sizeof(float)));
      check(hipMalloc(&out, actual.size() * sizeof(uint32_t)));
      check(hipMalloc(&ref, actual.size() * sizeof(uint32_t)));
      check(hipMalloc(&scratch, tokens * 4096 * sizeof(uint32_t)));
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
        if (n <= reference_topk_max_keys) {
          reference_topk(ref, scores, scratch, n, tokens, radix_bytes);
          check(hipGetLastError());
          check(hipMemcpy(reference.data(), ref,
                          reference.size() * sizeof(uint32_t),
                          hipMemcpyDeviceToHost));
        } else {
          reference = expected;
        }
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
          if (indexer_seeded_topk_fits(n, tokens, 512)) {
            check(hipMemset(out, 0xff, actual.size() * sizeof(uint32_t)));
            check(indexer_seeded_topk_launch(out, scores, n, tokens, scratch));
            check(hipMemcpy(actual.data(), out,
                            actual.size() * sizeof(uint32_t),
                            hipMemcpyDeviceToHost));
            if (actual != expected) {
              std::fprintf(stderr,
                           "Seeded top-k ordering mismatch: n=%u tokens=%u "
                           "mode=%u repeat=%d\n",
                           n, tokens, mode, repeat);
              std::exit(8);
            }
          }
        }
        ++cases_passed;
        if (benchmark && n <= reference_topk_max_keys &&
            (mode == 0 || mode == 2 || mode == 4)) {
          const int repetitions = tokens > 6 ? 3 : 50;
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
              for (int i = 0; i < repetitions; ++i)
                launch();
              check(hipEventRecord(end));
              check(hipEventSynchronize(end));
              check(hipEventElapsedTime(&ms[candidate], start, end));
            }
            std::printf(
                "selection n=%u tokens=%u mode=%u pass=%d old_us=%.3f "
                "new_us=%.3f\n",
                n, tokens, mode, pass, ms[0] * 1000.F / repetitions,
                ms[1] * 1000.F / repetitions);
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
  std::printf(
      "Indexer selection: %u cases exact twice against CPU ordering and "
      "fallback kernels, including signed-zero ties and causal masks\n",
      cases_passed);
}
