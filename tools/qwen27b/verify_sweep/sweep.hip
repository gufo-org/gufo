// Launch-geometry sweep for the Qwen GPU verification projections.
//
// Loads one weight matrix from a GGUF, runs the production path
// (TryLaunchVerifyGemm27, then LaunchBatchedQuantGEMMFp32), the generic
// LaunchBatchedQuantGEMMFp32 dispatch alone and a set of
// SmallBatchKQuantExactFp32GEMMKernel launch shapes on the same input, checks
// every candidate bit for bit against production and times them with HIP
// events: alternating rounds, median, weights rotated through >= 512 MiB of
// copies so no run reads from the 32 MB Infinity Cache.
//
//   qwen27b_verify_sweep <gguf> <offset> <type> <m> <k> <batch> [rounds]
//
// <type> is the GGUF type id: 12 Q4_K, 13 Q5_K, 14 Q6_K, 23 IQ4_XS. Output:
// one "base" line, a "listed" line (1 when verify_gemm_27b.hip lists the
// shape) and one line per candidate ("generic" or Launch<> parameters, time,
// GB/s, gain); see README.md.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "src/families/qwen/kernels/small_batch_gemm.hpp"
#include "src/families/qwen/kernels/verify_gemm_27b.hpp"
#include "src/families/qwen/kernels/ops/gemm.hpp"

using gufo::core::GgmlType;

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      std::fprintf(stderr, "HIP %s at %d\n", hipGetErrorString(e_), __LINE__); \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

namespace {

using Run = std::function<void(const void*, const float*, float*, std::size_t,
                               std::size_t, hipStream_t)>;

struct Candidate {
  std::string
      name;  // Launch<T, Batch, Waves, Rows, Tiles, Tokens, MinWaves, Lanes>
  Run run;
};

template<GgmlType T, std::size_t Batch, std::uint32_t Waves, std::size_t Rows,
         std::size_t Tiles, std::size_t TokensPerStep, std::uint32_t MinWaves,
         std::uint32_t Lanes>
Candidate Make() {
  char name[96];
  std::snprintf(name, sizeof(name), "Launch<T, %zu, %u, %zu, %zu, %zu, %u, %u>",
                Batch, Waves, Rows, Tiles, TokensPerStep, MinWaves, Lanes);
  return {
      name, [](const void* w, const float* x, float* y, std::size_t m,
               std::size_t k, hipStream_t s) {
        constexpr std::size_t kRowsPerBlock = Waves * (32 / Lanes) * Rows;
        gufo::hip::SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, Rows, T,
                                                       Tiles, MinWaves, 1, true,
                                                       TokensPerStep, 32, Lanes>
            <<<static_cast<unsigned>((m + kRowsPerBlock - 1) / kRowsPerBlock),
               Waves * 32, 0, s>>>(w, x, y, m, k);
      }};
}

// The shapes picked in verify_gemm_27b.hip so far, plus their neighbours.
template<GgmlType T, std::size_t B>
std::vector<Candidate> Candidates() {
  return {Make<T, B, 4, 4, 1, B, 8, 16>(),  Make<T, B, 4, 4, 1, B, 4, 16>(),
          Make<T, B, 8, 4, 1, B, 8, 16>(),  Make<T, B, 8, 4, 1, B, 4, 16>(),
          Make<T, B, 16, 4, 1, B, 4, 16>(), Make<T, B, 16, 4, 1, B, 8, 16>(),
          Make<T, B, 4, 2, 1, B, 8, 16>(),  Make<T, B, 8, 2, 1, B, 8, 16>(),
          Make<T, B, 4, 3, 1, B, 8, 16>(),  Make<T, B, 8, 3, 1, B, 4, 16>(),
          Make<T, B, 8, 4, 1, 1, 4, 32>(),  Make<T, B, 16, 4, 1, 1, 4, 32>(),
          Make<T, B, 8, 4, 2, B, 4, 32>(),  Make<T, B, 16, 4, 2, B, 4, 32>()};
}

template<GgmlType T>
std::vector<Candidate> ForBatch(std::size_t batch) {
  switch (batch) {
    case 2:
      return Candidates<T, 2>();
    case 3:
      return Candidates<T, 3>();
    case 4:
      return Candidates<T, 4>();
    case 5:
      return Candidates<T, 5>();
    case 6:
      return Candidates<T, 6>();
    case 7:
      return Candidates<T, 7>();
    case 8:
      return Candidates<T, 8>();
    default:
      return {};
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr,
                 "usage: qwen27b_verify_sweep <gguf> <offset> <type> <m> <k> "
                 "<batch> [rounds]\n");
    return 1;
  }
  const std::string path = argv[1];
  const std::uint64_t offset = std::strtoull(argv[2], nullptr, 10);
  const int type_id = std::atoi(argv[3]);
  const std::size_t m = std::strtoull(argv[4], nullptr, 10);
  const std::size_t k = std::strtoull(argv[5], nullptr, 10);
  const std::size_t batch = std::strtoull(argv[6], nullptr, 10);
  const int rounds = argc > 7 ? std::atoi(argv[7]) : 9;
  GgmlType type;
  std::size_t block_bytes;  // per 256 weights
  switch (type_id) {
    case 12:
      type = GgmlType::kQ4_K;
      block_bytes = 144;
      break;
    case 13:
      type = GgmlType::kQ5_K;
      block_bytes = 176;
      break;
    case 14:
      type = GgmlType::kQ6_K;
      block_bytes = 210;
      break;
    case 23:
      type = GgmlType::kIQ4_XS;
      block_bytes = 136;
      break;
    default:
      std::fprintf(stderr, "unsupported type %d\n", type_id);
      return 1;
  }
  if (batch < 2 || batch > 8 || k % 256 != 0) {
    std::fprintf(stderr, "batch must be 2-8 and k a multiple of 256\n");
    return 1;
  }
  const std::size_t bytes = m * (k / 256) * block_bytes;

  std::vector<char> host_w(bytes);
  std::ifstream file(path, std::ios::binary);
  file.seekg(static_cast<std::streamoff>(offset));
  file.read(host_w.data(), static_cast<std::streamsize>(bytes));
  if (!file) {
    std::fprintf(stderr, "could not read %zu bytes at %llu\n", bytes,
                 static_cast<unsigned long long>(offset));
    return 1;
  }
  std::vector<float> host_x(batch * k);
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  for (auto& v : host_x)
    v = dist(rng);

  const std::size_t copies =
      std::max<std::size_t>(2, (512ull << 20) / bytes + 1);
  std::vector<void*> weights(copies);
  for (auto& p : weights) {
    CHECK(hipMalloc(&p, bytes));
    CHECK(hipMemcpy(p, host_w.data(), bytes, hipMemcpyHostToDevice));
  }
  std::size_t next = 0;
  const auto rotate = [&] {
    next = (next + 1) % copies;
    return weights[next];
  };
  float *x = nullptr, *y = nullptr;
  CHECK(hipMalloc(&x, host_x.size() * sizeof(float)));
  CHECK(hipMalloc(&y, batch * m * sizeof(float)));
  CHECK(hipMemcpy(x, host_x.data(), host_x.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  hipStream_t s;
  CHECK(hipStreamCreate(&s));

  // The production path, as LaunchProjection takes it.
  const auto production = [&](const void* w, float* out) {
    if (!gufo::hip::TryLaunchVerifyGemm27(type, w, x, out, batch, m, k, s))
      gufo::hip::LaunchBatchedQuantGEMMFp32(type, w, x, out, batch, m, k, s);
  };
  auto candidates = type == GgmlType::kQ4_K   ? ForBatch<GgmlType::kQ4_K>(batch)
                    : type == GgmlType::kQ5_K ? ForBatch<GgmlType::kQ5_K>(batch)
                    : type == GgmlType::kQ6_K
                        ? ForBatch<GgmlType::kQ6_K>(batch)
                        : ForBatch<GgmlType::kIQ4_XS>(batch);
  // The dispatch without any listed shape; a listed shape slower than it is a
  // regression.
  candidates.insert(
      candidates.begin(),
      {"generic",
       [type, batch](const void* w, const float* in, float* out,
                     std::size_t rows, std::size_t cols, hipStream_t st) {
         gufo::hip::LaunchBatchedQuantGEMMFp32(type, w, in, out, batch, rows,
                                               cols, st);
       }});

  std::vector<float> reference(batch * m), got(batch * m);
  const bool listed =
      gufo::hip::TryLaunchVerifyGemm27(type, weights[0], x, y, batch, m, k, s);
  if (!listed)
    gufo::hip::LaunchBatchedQuantGEMMFp32(type, weights[0], x, y, batch, m, k,
                                          s);
  CHECK(hipStreamSynchronize(s));
  CHECK(hipMemcpy(reference.data(), y, reference.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  std::vector<bool> exact(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    CHECK(hipMemset(y, 0, got.size() * sizeof(float)));
    candidates[i].run(weights[0], x, y, m, k, s);
    CHECK(hipStreamSynchronize(s));
    CHECK(hipMemcpy(got.data(), y, got.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    exact[i] = std::memcmp(got.data(), reference.data(),
                           got.size() * sizeof(float)) == 0;
  }

  hipEvent_t start, stop;
  CHECK(hipEventCreate(&start));
  CHECK(hipEventCreate(&stop));
  constexpr int kWarmup = 3, kRepetitions = 20;
  const auto time_us = [&](const std::function<void()>& launch) {
    for (int i = 0; i < kWarmup; ++i)
      launch();
    CHECK(hipEventRecord(start, s));
    for (int i = 0; i < kRepetitions; ++i)
      launch();
    CHECK(hipEventRecord(stop, s));
    CHECK(hipEventSynchronize(stop));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms * 1000.0F / kRepetitions;
  };
  std::vector<std::vector<float>> samples(candidates.size() + 1);
  for (int r = 0; r < rounds; ++r) {
    samples[0].push_back(time_us([&] { production(rotate(), y); }));
    for (std::size_t i = 0; i < candidates.size(); ++i)
      if (exact[i])
        samples[i + 1].push_back(
            time_us([&] { candidates[i].run(rotate(), x, y, m, k, s); }));
  }
  const auto median = [](std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0F : v[v.size() / 2];
  };
  const float base = median(samples[0]);
  std::printf(
      "base type=%d m=%zu k=%zu batch=%zu MB=%.1f copies=%zu %.1f us %.0f "
      "GB/s\n",
      type_id, m, k, batch, bytes / 1e6, copies, base, bytes / (base * 1e3));
  std::printf("listed %d\n", listed ? 1 : 0);
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (!exact[i]) {
      std::printf("cand %s NOT-EXACT\n", candidates[i].name.c_str());
      continue;
    }
    const float t = median(samples[i + 1]);
    std::printf("cand %s %.1f us %.0f GB/s %+.1f%%\n",
                candidates[i].name.c_str(), t, bytes / (t * 1e3),
                (base - t) / base * 100.0F);
  }
  return 0;
}
