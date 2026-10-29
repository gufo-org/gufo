// SPDX-License-Identifier: MIT
// GPU check of Nemotron 3.5 Bolt's short-chunk vector products (gemv.hpp)
// and prefill matrix products (gemm.hpp) against an independent CPU oracle:
// the core ggml dequantizers and double accumulation over the activations as
// the kernel rounds them (BF16 or FP16), so the only difference left is F32
// accumulation order. Shapes cover partial
// activation chunks, every token count up to the maximum, routed slots with
// broadcast and per-slot activations, and the fused ReLU².

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/gemm.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/gemv.hpp"

namespace {

namespace k = gufo::models::nemotron35_bolt::rocm;

/// Accumulation-order noise allowed per output, relative to sum |w * x|.
constexpr double kMaxRelError = 2e-6;

#define HIP_OK(expr)                                                           \
  do {                                                                         \
    const hipError_t err_ = (expr);                                            \
    if (err_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s failed: %s\n", #expr, hipGetErrorString(err_)); \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Rng {
  std::uint64_t state;
  std::uint64_t Next() {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
  }
  std::uint32_t Below(std::uint32_t n) {
    return static_cast<std::uint32_t>(Next() % n);
  }
  float Uniform(float lo, float hi) {
    return lo + (hi - lo) * static_cast<float>(
                                static_cast<double>(Next() >> 11U) * 0x1.0p-53);
  }
};

int failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

/// Random valid IQ2_XXS rows: every grid index, sign index and group scale
/// is valid, so the payload is uniform random bits under an fp16 scale.
std::vector<std::uint8_t> RandomIq2xxs(std::size_t rows, int cols, Rng& rng) {
  std::vector<std::uint8_t> w(rows * static_cast<std::size_t>(cols) / 256 * 66);
  for (std::size_t b = 0; b < w.size(); b += 66) {
    const unsigned e = 4 + rng.Below(5);
    const auto d = static_cast<std::uint16_t>((e << 10U) | rng.Below(1024));
    w[b] = static_cast<std::uint8_t>(d & 0xFFU);
    w[b + 1] = static_cast<std::uint8_t>(d >> 8U);
    for (int i = 2; i < 66; ++i)
      w[b + i] = static_cast<std::uint8_t>(rng.Next());
  }
  return w;
}

float Bf16Round(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  u = (u + 0x7FFFU + ((u >> 16) & 1U)) & 0xFFFF0000U;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

template<class T>
T* Device(const std::vector<T>& h) {
  T* d = nullptr;
  HIP_OK(hipMalloc(&d, std::max<std::size_t>(h.size(), 1) * sizeof(T)));
  HIP_OK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

/// Compares every output with the oracle; `slot_expert` maps output rows of
/// `rows` values to (expert, x row).
void Compare(const char* label, const std::vector<float>& got,
             const std::vector<std::uint8_t>& w, int rows, int cols,
             const std::vector<float>& x, const std::vector<int>& expert,
             const std::vector<int>& x_row, bool relu2) {
  const std::size_t row_bytes = static_cast<std::size_t>(cols) / 256 * 66;
  std::vector<float> wrow(cols);
  double worst = 0.0;
  for (std::size_t s = 0; s < expert.size(); ++s) {
    for (int r = 0; r < rows; ++r) {
      gufo::quant::DequantizeIQ2_XXS(
          w.data() +
              (static_cast<std::size_t>(expert[s]) * rows + r) * row_bytes,
          wrow.data(), static_cast<std::size_t>(cols));
      double sum = 0.0;
      double mag = 0.0;
      for (int c = 0; c < cols; ++c) {
        const double p =
            double(wrow[c]) *
            Bf16Round(x[static_cast<std::size_t>(x_row[s]) * cols + c]);
        sum += p;
        mag += std::fabs(p);
      }
      double want = sum;
      double g = got[s * rows + r];
      if (relu2) {
        // Compare before the square, where the error is linear.
        want = std::max(want, 0.0);
        g = std::sqrt(std::max(g, 0.0));
      }
      worst = std::max(worst, std::fabs(g - want) / std::max(mag, 1e-30));
    }
  }
  std::printf("%-40s max rel error %.2e\n", label, worst);
  Check(worst < kMaxRelError, label);
}

void DenseCase(int rows, int cols, int n, bool relu2, Rng& rng,
               const std::uint32_t* grid) {
  const auto w = RandomIq2xxs(rows, cols, rng);
  std::vector<float> x(static_cast<std::size_t>(n) * cols);
  for (float& v : x)
    v = rng.Uniform(-2.0F, 2.0F);
  auto* dw = Device(w);
  auto* dx = Device(x);
  float* dout = nullptr;
  HIP_OK(hipMalloc(&dout, static_cast<std::size_t>(n) * rows * sizeof(float)));
  k::Iq2xxsGemvArgs a{};
  a.weights = dw;
  a.grid = grid;
  a.x = dx;
  a.out = dout;
  a.rows = rows;
  a.cols = cols;
  a.n_tokens = n;
  a.relu2 = relu2;
  HIP_OK(k::Iq2xxsGemv(a, nullptr));
  std::vector<float> got(static_cast<std::size_t>(n) * rows);
  HIP_OK(hipMemcpy(got.data(), dout, got.size() * sizeof(float),
                   hipMemcpyDeviceToHost));
  std::vector<int> expert(n, 0);
  std::vector<int> x_row(n);
  for (int t = 0; t < n; ++t)
    x_row[t] = t;
  char label[96];
  std::snprintf(label, sizeof(label), "dense %dx%d, %d tokens%s", rows, cols, n,
                relu2 ? ", relu2" : "");
  Compare(label, got, w, rows, cols, x, expert, x_row, relu2);
  HIP_OK(hipFree(dw));
  HIP_OK(hipFree(dx));
  HIP_OK(hipFree(dout));
}

void RoutedCase(int experts, int rows, int cols, int n, int used, bool per_slot,
                bool relu2, Rng& rng, const std::uint32_t* grid) {
  const auto w =
      RandomIq2xxs(static_cast<std::size_t>(experts) * rows, cols, rng);
  const int x_rows = per_slot ? n * used : n;
  std::vector<float> x(static_cast<std::size_t>(x_rows) * cols);
  for (float& v : x)
    v = rng.Uniform(-2.0F, 2.0F);
  // Distinct experts per token, as top-k selects them.
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n) * used);
  for (int t = 0; t < n; ++t) {
    for (int s = 0; s < used; ++s) {
      std::int32_t e;
      do {
        e = static_cast<std::int32_t>(rng.Below(experts));
      } while (std::find(ids.begin() + t * used, ids.begin() + t * used + s,
                         e) != ids.begin() + t * used + s);
      ids[t * used + s] = e;
    }
  }
  auto* dw = Device(w);
  auto* dx = Device(x);
  auto* dids = Device(ids);
  float* dout = nullptr;
  HIP_OK(hipMalloc(&dout,
                   static_cast<std::size_t>(n) * used * rows * sizeof(float)));
  k::Iq2xxsGemvArgs a{};
  a.weights = dw;
  a.grid = grid;
  a.x = dx;
  a.ids = dids;
  a.used = used;
  a.x_per_slot = per_slot;
  a.out = dout;
  a.rows = rows;
  a.cols = cols;
  a.n_tokens = n;
  a.relu2 = relu2;
  HIP_OK(k::Iq2xxsGemv(a, nullptr));
  std::vector<float> got(static_cast<std::size_t>(n) * used * rows);
  HIP_OK(hipMemcpy(got.data(), dout, got.size() * sizeof(float),
                   hipMemcpyDeviceToHost));
  std::vector<int> expert(ids.begin(), ids.end());
  std::vector<int> x_row(ids.size());
  for (std::size_t s = 0; s < ids.size(); ++s)
    x_row[s] = per_slot ? static_cast<int>(s) : static_cast<int>(s) / used;
  char label[96];
  std::snprintf(label, sizeof(label), "routed %d experts %dx%d, %dx%d%s%s",
                experts, rows, cols, n, used, per_slot ? ", per slot" : "",
                relu2 ? ", relu2" : "");
  Compare(label, got, w, rows, cols, x, expert, x_row, relu2);
  HIP_OK(hipFree(dw));
  HIP_OK(hipFree(dx));
  HIP_OK(hipFree(dids));
  HIP_OK(hipFree(dout));
}

/// The NVFP4 prefill product on FP16 activations: decoded weights are exact
/// in FP16, so only F32 accumulation order remains.
void Nvfp4GemmCase(int rows, int cols, int n, Rng& rng) {
  std::vector<std::uint8_t> w(static_cast<std::size_t>(rows) * cols / 64 * 36);
  for (std::size_t b = 0; b < w.size(); b += 36) {
    for (int s = 0; s < 4; ++s)  // UE4M3 exponents 3..10, any mantissa
      w[b + s] =
          static_cast<std::uint8_t>(((3 + rng.Below(8)) << 3) | rng.Below(8));
    for (int i = 4; i < 36; ++i)
      w[b + i] = static_cast<std::uint8_t>(rng.Next());
  }
  std::vector<float> x(static_cast<std::size_t>(n) * cols);
  std::vector<_Float16> x16(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    x16[i] = static_cast<_Float16>(rng.Uniform(-3.0F, 3.0F));
    x[i] = static_cast<float>(x16[i]);
  }
  auto* dw = Device(w);
  auto* dx = Device(x16);
  float* dout = nullptr;
  HIP_OK(hipMalloc(&dout, static_cast<std::size_t>(n) * rows * sizeof(float)));
  HIP_OK(k::Nvfp4Gemm(dw, reinterpret_cast<const __half*>(dx), dout, rows, cols,
                      n, nullptr));
  std::vector<float> got(static_cast<std::size_t>(n) * rows);
  HIP_OK(hipMemcpy(got.data(), dout, got.size() * sizeof(float),
                   hipMemcpyDeviceToHost));
  const std::size_t row_bytes = static_cast<std::size_t>(cols) / 64 * 36;
  std::vector<float> wrow(cols);
  double worst = 0.0;
  for (int r = 0; r < rows; ++r) {
    gufo::quant::DequantizeNVFP4(w.data() + r * row_bytes, wrow.data(),
                                 static_cast<std::size_t>(cols));
    for (int t = 0; t < n; ++t) {
      double sum = 0.0;
      double mag = 0.0;
      for (int c = 0; c < cols; ++c) {
        const double p =
            double(wrow[c]) * x[static_cast<std::size_t>(t) * cols + c];
        sum += p;
        mag += std::fabs(p);
      }
      worst = std::max(
          worst, std::fabs(got[static_cast<std::size_t>(t) * rows + r] - sum) /
                     std::max(mag, 1e-30));
    }
  }
  char label[96];
  std::snprintf(label, sizeof(label), "nvfp4 gemm %dx%d, %d tokens", rows, cols,
                n);
  std::printf("%-40s max rel error %.2e\n", label, worst);
  Check(worst < kMaxRelError, label);
  HIP_OK(hipFree(dw));
  HIP_OK(hipFree(dx));
  HIP_OK(hipFree(dout));
}

/// The routed IQ2_XXS prefill product: exact weights, activations rounded
/// to BF16, F32 accumulation.
void Iq2xxsGemmCase(int experts, int rows, int cols, int n, int used,
                    bool per_slot, bool relu2, Rng& rng,
                    const std::uint32_t* grid) {
  const auto w =
      RandomIq2xxs(static_cast<std::size_t>(experts) * rows, cols, rng);
  const int slots = n * used;
  const int x_rows = per_slot ? slots : n;
  std::vector<float> x(static_cast<std::size_t>(x_rows) * cols);
  for (float& v : x)
    v = rng.Uniform(-2.0F, 2.0F);
  // Repeated rows land in other tile columns: their outputs must be equal.
  const int copy_from = 3 % x_rows;
  const int copy_to[] = {x_rows / 2, x_rows - 1};
  for (const int to : copy_to)
    std::copy_n(x.begin() + static_cast<std::ptrdiff_t>(copy_from) * cols, cols,
                x.begin() + static_cast<std::ptrdiff_t>(to) * cols);
  // Distinct experts per token, skewed so that some take several tiles.
  std::vector<std::int32_t> ids(static_cast<std::size_t>(slots), 0);
  for (int t = 0; t < n; ++t) {
    for (int s = 0; s < used; ++s) {
      std::int32_t e;
      do {
        e = static_cast<std::int32_t>(rng.Below(4) == 0 ? 0
                                                        : rng.Below(experts));
      } while (std::find(ids.begin() + t * used, ids.begin() + t * used + s,
                         e) != ids.begin() + t * used + s);
      ids[t * used + s] = e;
    }
  }
  auto* dw = Device(w);
  auto* dx = Device(x);
  auto* dids = Device(ids);
  float* dout = nullptr;
  HIP_OK(
      hipMalloc(&dout, static_cast<std::size_t>(slots) * rows * sizeof(float)));
  k::Iq2xxsRoute route{};
  route.max_tiles = k::Iq2xxsRouteMaxTiles(slots, experts);
  HIP_OK(hipMalloc(&route.cursor, experts * sizeof(std::int32_t)));
  HIP_OK(hipMalloc(&route.order, slots * sizeof(std::int32_t)));
  HIP_OK(hipMalloc(&route.tiles, route.max_tiles * sizeof(int4)));
  HIP_OK(hipMalloc(&route.num_tiles, sizeof(std::int32_t)));
  HIP_OK(k::Iq2xxsRouteSlots(dids, slots, experts, route, nullptr));
  k::Iq2xxsGemmArgs a{};
  a.weights = dw;
  a.grid = grid;
  a.x = dx;
  a.out = dout;
  a.rows = rows;
  a.cols = cols;
  a.relu2 = relu2;
  a.route = &route;
  a.used = used;
  a.x_per_slot = per_slot;
  HIP_OK(k::Iq2xxsRoutedGemm(a, nullptr));
  std::vector<float> got(static_cast<std::size_t>(slots) * rows);
  HIP_OK(hipMemcpy(got.data(), dout, got.size() * sizeof(float),
                   hipMemcpyDeviceToHost));
  std::vector<int> expert(ids.begin(), ids.end());
  std::vector<int> x_row(slots);
  for (int s = 0; s < slots; ++s)
    x_row[s] = per_slot ? s : s / used;
  char label[96];
  std::snprintf(label, sizeof(label), "iq2 routed gemm %dx%d, %dx%d%s%s", rows,
                cols, n, used, per_slot ? ", per slot" : "",
                relu2 ? ", relu2" : "");
  Compare(label, got, w, rows, cols, x, expert, x_row, relu2);
  // Same input row and expert: bit-identical outputs.
  bool same = true;
  for (int s = 0; s < slots; ++s) {
    for (int o = 0; o < slots; ++o) {
      if (x_row[s] != copy_from || expert[o] != expert[s] ||
          (x_row[o] != copy_to[0] && x_row[o] != copy_to[1]))
        continue;
      same = same && std::equal(got.begin() + std::ptrdiff_t(s) * rows,
                                got.begin() + std::ptrdiff_t(s + 1) * rows,
                                got.begin() + std::ptrdiff_t(o) * rows);
    }
  }
  Check(same, "equal rows give equal outputs in any tile column");
  HIP_OK(hipFree(route.cursor));
  HIP_OK(hipFree(route.order));
  HIP_OK(hipFree(route.tiles));
  HIP_OK(hipFree(route.num_tiles));
  HIP_OK(hipFree(dw));
  HIP_OK(hipFree(dx));
  HIP_OK(hipFree(dids));
  HIP_OK(hipFree(dout));
}

}  // namespace

int main() {
  Rng rng{20261029};
  std::vector<std::uint32_t> words(k::kIq2xxsGridWords);
  k::Iq2xxsGridBf16(gufo::quant::Iq2xxsGrid(), words.data());
  auto* grid = Device(words);

  // Rows of 4k super-blocks take whole 32-value groups per lane, others
  // eight values. 768 columns: one partial chunk; 2304: a full chunk and one
  // block; 3072: a full chunk and a half one.
  for (int n : {1, 2, 5, 8}) {
    DenseCase(64, 768, n, false, rng, grid);
    DenseCase(32, 3072, n, false, rng, grid);
  }
  DenseCase(96, 2304, 1, true, rng, grid);
  DenseCase(32, 2304, 8, true, rng, grid);
  DenseCase(64, 1024, 1, true, rng, grid);
  RoutedCase(12, 96, 512, 1, 9, false, true, rng, grid);
  RoutedCase(12, 64, 1024, 1, 9, false, true, rng, grid);
  RoutedCase(12, 64, 768, 1, 9, true, false, rng, grid);
  RoutedCase(7, 32, 2304, 3, 4, true, false, rng, grid);
  RoutedCase(7, 32, 2048, 2, 3, true, false, rng, grid);
  RoutedCase(7, 32, 256, 8, 2, false, true, rng, grid);

  // Partial token tiles, several row tiles, K spanning many stages.
  Nvfp4GemmCase(256, 512, 9, rng);
  Nvfp4GemmCase(128, 1024, 200, rng);
  Nvfp4GemmCase(384, 192, 1, rng);
  Nvfp4GemmCase(8192, 128, 300, rng);  // 256-token tiles

  // Experts over several tiles, partial tiles, K over several super-blocks,
  // per-token and per-slot activations.
  Iq2xxsGemmCase(7, 256, 512, 60, 3, false, true, rng, grid);
  Iq2xxsGemmCase(3, 256, 256, 5, 2, false, false, rng, grid);
  Iq2xxsGemmCase(7, 512, 768, 60, 3, true, false, rng, grid);
  Iq2xxsGemmCase(31, 256, 1024, 130, 9, false, true, rng, grid);

  HIP_OK(hipFree(grid));
  if (failures != 0) {
    std::fprintf(stderr, "%d vector product check(s) failed\n", failures);
    return 1;
  }
  std::printf("all vector product checks passed\n");
  return 0;
}
