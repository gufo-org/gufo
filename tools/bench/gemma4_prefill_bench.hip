// Standalone timing of Gemma 4's dense prefill projection kernel
// (LaunchDenseHalfGemm) on production shapes with production-format weights,
// against its prefill weight copies (LaunchPackedPrefillGemm, compact and
// decoded) on the same weights.
//
//   nix develop -c tools/bench/build.sh gemma4_prefill_bench
//   /tmp/gemma4_prefill_bench [iterations] [rows] [ud|qat|q6] [group] [rounds]
//
// ud, qat and q6 are the 31B UD-Q4_K_XL, QAT and llama.cpp Q6_K shape mixes; group overrides the
// copies' token tiles per row-block sweep (0: the launcher's default); each
// layout keeps its best time over `rounds` alternating rounds. GUFO_G4_SHAPE
// keeps only the shapes whose name contains it (e.g. for counter profiles).
//
// Cold-weight reads: every shape rotates over enough copies of its matrix to
// exceed the 32 MiB MALL, so a repeat does not read a cached tensor (the small
// attention projections sit entirely in the MALL otherwise). Outputs follow
// production: FP32 rows for a plain projection, binary16 GeGLU rows for the
// fused [gate | up] shape.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// The production translation unit itself: the bench builds the exact kernel and
// flags the model ships, so a change is measured here before it lands.
#include "src/models/gemma4/kernels/rocm/moe_prefill.hip.cpp"

using gufo::models::gemma4::rocm::ExpertFormat;
using gufo::models::gemma4::rocm::LaunchDenseHalfGemm;
using gufo::models::gemma4::rocm::LaunchPackedPrefillGemm;
using gufo::models::gemma4::rocm::PackPrefillWeights;
using gufo::models::gemma4::rocm::PrefillLayout;
using gufo::models::gemma4::rocm::PrefillLayoutSupports;
using gufo::models::gemma4::rocm::PrefillWeightBytes;

namespace {

constexpr std::uint32_t kThreads = 256;

__global__ void Fill(std::uint8_t* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    std::uint32_t v = static_cast<std::uint32_t>(i) * 2654435761U + seed;
    v ^= v >> 13;
    p[i] = static_cast<std::uint8_t>(v);
  }
}

void FillBuffer(void* p, std::size_t n, std::uint32_t seed) {
  const unsigned blocks = static_cast<unsigned>((n + kThreads - 1) / kThreads);
  Fill<<<blocks, kThreads>>>(static_cast<std::uint8_t*>(p), n, seed);
}

void Alloc(void** p, std::size_t bytes, const char* what) {
  if (hipMalloc(p, bytes) != hipSuccess) {
    std::printf("alloc failed (%s, %zu bytes)\n", what, bytes);
    std::exit(1);
  }
}

std::uint32_t BlockBytes(ExpertFormat f) {
  switch (f) {
    case ExpertFormat::kQ4_K:
      return 144;
    case ExpertFormat::kQ5_K:
      return 176;
    case ExpertFormat::kQ6_K:
      return 210;
    default:
      return 0;
  }
}

/// Bytes of one row of k values.
std::size_t RowBytes(ExpertFormat f, std::uint32_t k) {
  switch (f) {
    case ExpertFormat::kQ4_0:
      return std::size_t{k} / 32 * 18;
    case ExpertFormat::kQ5_0:
      return std::size_t{k} / 32 * 22;
    default:
      return std::size_t{k} / 256 * BlockBytes(f);
  }
}

struct Shape {
  const char* name;
  ExpertFormat format;
  std::uint32_t m;
  std::uint32_t k;
  bool geglu;
  int count;  // launches per 2048-token chunk, for the predicted total
};

constexpr PrefillLayout kLayouts[] = {
    PrefillLayout::kNone, PrefillLayout::kCompact, PrefillLayout::kDecoded};
constexpr const char* kLayoutNames[] = {"gguf", "compact", "decoded"};
constexpr int kLayoutCount = 3;

/// One shape's weights (GGUF and every supported prefill copy, enough
/// copies of each to exceed the MALL), activations and outputs.
struct Buffers {
  std::vector<void*> w[kLayoutCount];
  void* x = nullptr;
  void* out_half = nullptr;
  void* out_f32 = nullptr;
};

Buffers Setup(const Shape& s, std::uint32_t rows) {
  Buffers b;
  const std::size_t weight_bytes = std::size_t{s.m} * RowBytes(s.format, s.k);
  for (int l = 0; l < kLayoutCount; ++l) {
    if (l != 0 && !PrefillLayoutSupports(s.format, kLayouts[l], s.k)) {
      continue;
    }
    const std::size_t bytes =
        l == 0 ? weight_bytes
               : PrefillWeightBytes(s.format, kLayouts[l], s.m, s.k);
    const int copies = std::max(
        1, static_cast<int>(std::ceil(96.0e6 / static_cast<double>(bytes))));
    for (int c = 0; c < copies; ++c) {
      void* p = nullptr;
      Alloc(&p, bytes, s.name);
      b.w[l].push_back(p);
    }
  }
  // Every packed copy derives from the first GGUF copy.
  FillBuffer(b.w[0][0], weight_bytes, 7);
  for (std::size_t c = 1; c < b.w[0].size(); ++c) {
    hipMemcpy(b.w[0][c], b.w[0][0], weight_bytes, hipMemcpyDeviceToDevice);
  }
  for (int l = 1; l < kLayoutCount; ++l) {
    for (void* p : b.w[l]) {
      if (!PackPrefillWeights(s.format, kLayouts[l], b.w[0][0], p, s.m, s.k,
                              nullptr)) {
        std::printf("pack failed (%s)\n", s.name);
        std::exit(1);
      }
    }
  }
  const std::size_t x_bytes = std::size_t{rows} * s.k * 2;
  Alloc(&b.x, x_bytes, "x");
  FillBuffer(b.x, x_bytes, 11);
  Alloc(&b.out_half, std::size_t{rows} * s.m * 2, "out_half");
  Alloc(&b.out_f32, std::size_t{rows} * s.m * 4, "out_f32");
  hipDeviceSynchronize();
  return b;
}

void Release(Buffers& b) {
  for (int l = 0; l < kLayoutCount; ++l) {
    for (void* p : b.w[l]) {
      hipFree(p);
    }
  }
  hipFree(b.x);
  hipFree(b.out_half);
  hipFree(b.out_f32);
}

/// Best of `iterations` event-timed launches of one layout.
double Time(const Shape& s, Buffers& b, int layout, std::uint32_t rows,
            int iterations, std::uint32_t group) {
  const auto& w = b.w[layout];
  auto run = [&](int i) {
    void* wi = w[static_cast<std::size_t>(i) % w.size()];
    float* out = s.geglu ? nullptr : static_cast<float*>(b.out_f32);
    void* out_half = s.geglu ? b.out_half : nullptr;
    const bool ok =
        layout == 0
            ? LaunchDenseHalfGemm(s.format, wi, b.x, out, out_half, rows, s.m,
                                  s.k, nullptr, s.geglu)
            : LaunchPackedPrefillGemm(s.format, kLayouts[layout], wi, b.x, out,
                                      out_half, rows, s.m, s.k, nullptr,
                                      s.geglu, group);
    if (!ok) {
      std::printf("launch rejected (%s)\n", s.name);
      std::exit(1);
    }
  };
  for (int i = 0; i < 3; ++i) {
    run(i);
  }
  hipDeviceSynchronize();
  hipEvent_t a = nullptr;
  hipEvent_t e = nullptr;
  hipEventCreate(&a);
  hipEventCreate(&e);
  double best = 1e30;
  for (int i = 0; i < iterations; ++i) {
    hipEventRecord(a, nullptr);
    run(i);
    hipEventRecord(e, nullptr);
    hipEventSynchronize(e);
    float elapsed = 0.0F;
    hipEventElapsedTime(&elapsed, a, e);
    best = std::min(best, static_cast<double>(elapsed));
  }
  hipEventDestroy(a);
  hipEventDestroy(e);
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 5;
  const std::uint32_t rows =
      argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 2048;
  const std::string profile = argc > 3 ? argv[3] : "ud";
  const auto group =
      argc > 4 ? static_cast<std::uint32_t>(std::atoi(argv[4])) : 0U;
  // Alternating rounds; each layout keeps its best.
  const int rounds = argc > 5 ? std::atoi(argv[5]) : 2;
  hipDeviceProp_t prop{};
  hipGetDeviceProperties(&prop, 0);
  std::printf(
      "device: %s  CUs=%d  clock=%.2f GHz  iterations=%d  rows=%u  "
      "profile=%s  group=%u  rounds=%d\n",
      prop.name, prop.multiProcessorCount, prop.clockRate / 1e6, iterations,
      rows, profile.c_str(), group, rounds);
  // One 2048-token chunk of 31B UD-Q4_K_XL: the per-layer mix of formats and
  // shapes (10 global layers with an 18432-row fused q/k and a 16384-wide o,
  // 50 sliding with 12288 and 8192; V is separate where its format differs).
  const std::vector<Shape> ud = {
      {"gate|up Q4_K [43008x5376]", ExpertFormat::kQ4_K, 43008, 5376, true, 54},
      {"gate|up Q5_K [43008x5376]", ExpertFormat::kQ5_K, 43008, 5376, true, 6},
      {"down Q4_K [5376x21504]", ExpertFormat::kQ4_K, 5376, 21504, false, 28},
      {"down Q5_K [5376x21504]", ExpertFormat::kQ5_K, 5376, 21504, false, 6},
      {"down Q6_K [5376x21504]", ExpertFormat::kQ6_K, 5376, 21504, false, 26},
      {"q|k sliding [12288x5376]", ExpertFormat::kQ4_K, 12288, 5376, false, 46},
      {"q|k sliding Q5_K [12288x5376]", ExpertFormat::kQ5_K, 12288, 5376, false,
       4},
      {"q|k global [18432x5376]", ExpertFormat::kQ4_K, 18432, 5376, false, 8},
      {"q|k global Q5_K [18432x5376]", ExpertFormat::kQ5_K, 18432, 5376, false,
       2},
      {"o sliding [5376x8192]", ExpertFormat::kQ4_K, 5376, 8192, false, 50},
      {"o global [5376x16384]", ExpertFormat::kQ4_K, 5376, 16384, false, 10},
      {"v sliding Q6_K [4096x5376]", ExpertFormat::kQ6_K, 4096, 5376, false,
       44},
      {"v sliding Q5_K [4096x5376]", ExpertFormat::kQ5_K, 4096, 5376, false, 6},
  };
  // 31B QAT UD-Q4_K_XL: Q4_0 throughout, V fused into [q | k | v] (sliding)
  // or [q | k] (global, V shared with K).
  const std::vector<Shape> qat = {
      {"gate|up Q4_0 [43008x5376]", ExpertFormat::kQ4_0, 43008, 5376, true, 60},
      {"down Q4_0 [5376x21504]", ExpertFormat::kQ4_0, 5376, 21504, false, 60},
      {"q|k|v sliding [16384x5376]", ExpertFormat::kQ4_0, 16384, 5376, false,
       50},
      {"q|k global [18432x5376]", ExpertFormat::kQ4_0, 18432, 5376, false, 10},
      {"o sliding [5376x8192]", ExpertFormat::kQ4_0, 5376, 8192, false, 50},
      {"o global [5376x16384]", ExpertFormat::kQ4_0, 5376, 16384, false, 10},
  };
  // A single-format Q6_K 31B (llama.cpp Q6_K): [q | k | v] fused in sliding
  // layers, [q | k] in global ones.
  const std::vector<Shape> q6 = {
      {"gate|up Q6_K [43008x5376]", ExpertFormat::kQ6_K, 43008, 5376, true, 60},
      {"down Q6_K [5376x21504]", ExpertFormat::kQ6_K, 5376, 21504, false, 60},
      {"q|k|v sliding Q6_K [16384x5376]", ExpertFormat::kQ6_K, 16384, 5376,
       false, 50},
      {"q|k global Q6_K [18432x5376]", ExpertFormat::kQ6_K, 18432, 5376, false,
       10},
      {"o sliding Q6_K [5376x8192]", ExpertFormat::kQ6_K, 5376, 8192, false, 50},
      {"o global Q6_K [5376x16384]", ExpertFormat::kQ6_K, 5376, 16384, false,
       10},
  };
  const auto& shapes = profile == "qat" ? qat : profile == "q6" ? q6 : ud;
  std::printf("\n%-30s %8s %8s %8s %7s %7s\n", "shape (ms, best)", "gguf",
              "compact", "decoded", "compact", "decoded");
  // gguf, compact where supported, decoded where supported (else compact)
  double total[kLayoutCount] = {};
  double best_total = 0.0;
  double flops = 0.0;
  // GUFO_G4_SHAPE: only shapes whose name contains it.
  const char* only = std::getenv("GUFO_G4_SHAPE");
  for (const Shape& s : shapes) {
    if (only != nullptr && std::string(s.name).find(only) == std::string::npos) {
      continue;
    }
    Buffers b = Setup(s, rows);
    double ms[kLayoutCount] = {1e30, 1e30, 1e30};
    for (int r = 0; r < rounds; ++r) {
      for (int i = 0; i < kLayoutCount; ++i) {
        const int l = r % 2 == 0 ? i : kLayoutCount - 1 - i;
        if (!b.w[l].empty()) {
          ms[l] = std::min(ms[l], Time(s, b, l, rows, iterations, group));
        }
      }
    }
    Release(b);
    const auto gain = [&](int l) {
      return ms[l] < 1e29 ? (ms[0] / ms[l] - 1.0) * 100.0 : 0.0;
    };
    const auto shown = [&](int l) { return ms[l] < 1e29 ? ms[l] : 0.0; };
    std::printf("%-30s %8.3f %8.3f %8.3f %+6.1f%% %+6.1f%%\n", s.name, ms[0],
                shown(1), shown(2), gain(1), gain(2));
    const double compact = ms[1] < 1e29 ? ms[1] : ms[0];
    const double decoded = ms[2] < 1e29 ? ms[2] : compact;
    total[0] += ms[0] * s.count;
    total[1] += compact * s.count;
    total[2] += decoded * s.count;
    best_total += std::min({ms[0], ms[1], ms[2]}) * s.count;
    flops += 2.0 * rows * static_cast<double>(s.m) * s.k * s.count;
  }
  std::printf("\npredicted chunk projections (ms, TFLOP/s):\n");
  for (int l = 0; l < kLayoutCount; ++l) {
    std::printf("  %-8s %8.0f %6.2f  %+6.2f%%\n", kLayoutNames[l], total[l],
                flops / (total[l] * 1e-3) / 1e12,
                (total[0] / total[l] - 1.0) * 100.0);
  }
  std::printf("  %-8s %8.0f %6.2f  %+6.2f%%\n", "best", best_total,
              flops / (best_total * 1e-3) / 1e12,
              (total[0] / best_total - 1.0) * 100.0);
  return 0;
}
