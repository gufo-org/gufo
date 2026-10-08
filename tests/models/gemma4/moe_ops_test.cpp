// Routed expert kernels of the 26B-A4B at its shapes: routing against an
// FP64 host reference, every routed weight format against FP64 dots of the
// CPU-dequantized rows, and batch invariance -- a row routed and projected
// alone equals the same row inside a verification batch bit for bit, so
// greedy speculation reproduces decode. Prints cold-weight timings of the
// decode (one row, eight experts) and five-row verification projections.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/gemma4/kernels/rocm/kernels.hpp"
#include "src/models/gemma4/kernels/rocm/moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4k = gufo::models::gemma4::rocm;
using gemma4_test::Require;
using gufo::core::GgmlType;

namespace {

constexpr std::uint32_t kHidden = 2816;
constexpr std::uint32_t kExperts = 128;
constexpr std::uint32_t kUsed = 8;
constexpr std::uint32_t kWidth = 704;
constexpr float kEps = 1e-6F;

struct Format {
  g4k::ExpertFormat format;
  GgmlType type;
  std::size_t block;
  std::size_t bytes;
  std::vector<std::size_t> half_offsets;
  const char* name;
};

std::uint16_t Half(float x) {
  const _Float16 h = static_cast<_Float16>(x);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

std::vector<std::uint8_t> RandomMatrix(const Format& f, std::size_t rows,
                                       std::size_t cols, std::mt19937& rng) {
  const std::size_t blocks = rows * cols / f.block;
  std::vector<std::uint8_t> data(blocks * f.bytes + 4096, 0);
  if (f.type == GgmlType::kF16) {
    // Trained-weight magnitudes rather than random bit patterns.
    std::normal_distribution<float> normal(0.0F, 0.02F);
    for (std::size_t i = 0; i < rows * cols; ++i) {
      const std::uint16_t h = Half(normal(rng));
      std::memcpy(&data[2 * i], &h, 2);
    }
    return data;
  }
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> scale(0.0005F, 0.004F);
  for (std::size_t i = 0; i < blocks * f.bytes; ++i) {
    data[i] = static_cast<std::uint8_t>(byte(rng));
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    for (std::size_t off : f.half_offsets) {
      const std::uint16_t h = Half(scale(rng));
      std::memcpy(&data[b * f.bytes + off], &h, 2);
    }
  }
  return data;
}

template<class T>
T* Device(const T* host, std::size_t n) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, std::max<std::size_t>(n, 1) * sizeof(T)));
  HIP_CHECK(hipMemcpy(ptr, host, n * sizeof(T), hipMemcpyHostToDevice));
  return ptr;
}

template<class T>
std::vector<T> Host(const T* device, std::size_t n) {
  std::vector<T> out(n);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(
      hipMemcpy(out.data(), device, n * sizeof(T), hipMemcpyDeviceToHost));
  return out;
}

std::vector<float> Normal(std::size_t n, float sigma, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0F, sigma);
  std::vector<float> v(n);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

struct Routing {
  std::vector<std::int32_t> ids;
  std::vector<float> weights;
  std::vector<std::int32_t> groups;
  std::vector<float> logits;
};

/// Routes `rows` rows on the device (with the group table).
Routing Route(const float* dx, const float* router, const float* expert_scale,
              std::uint32_t rows, std::uint32_t* sync = nullptr) {
  float* logits = nullptr;
  std::int32_t* ids = nullptr;
  float* weights = nullptr;
  std::int32_t* groups = nullptr;
  const std::size_t group_ints = g4k::ExpertGroupInts(kExperts);
  HIP_CHECK(hipMalloc(&logits, rows * kExperts * sizeof(float)));
  HIP_CHECK(hipMalloc(&ids, rows * kUsed * sizeof(std::int32_t)));
  HIP_CHECK(hipMalloc(&weights, rows * kUsed * sizeof(float)));
  HIP_CHECK(hipMalloc(&groups, group_ints * sizeof(std::int32_t)));
  g4k::MoeRouteArgs a{};
  a.x = dx;
  a.router = router;
  a.expert_scale = expert_scale;
  a.logits = logits;
  a.ids = ids;
  a.weights = weights;
  a.groups = groups;
  a.rows = rows;
  a.hidden = kHidden;
  a.experts = kExperts;
  a.used = kUsed;
  a.eps = kEps;
  a.sync = sync;
  g4k::MoeRoute(a, nullptr);
  Routing r;
  r.ids = Host(ids, rows * kUsed);
  r.weights = Host(weights, rows * kUsed);
  r.groups = Host(groups, group_ints);
  r.logits = Host(logits, rows * kExperts);
  HIP_CHECK(hipFree(logits));
  HIP_CHECK(hipFree(ids));
  HIP_CHECK(hipFree(weights));
  HIP_CHECK(hipFree(groups));
  return r;
}

void CheckRouting(std::mt19937& rng) {
  constexpr std::uint32_t kRows = 16;
  const auto router = Normal(std::size_t{kExperts} * kHidden, 0.05F, rng);
  auto scale = Normal(kHidden, 1.0F, rng);
  for (float& s : scale) {
    s = 30.0F + s;
  }
  auto expert_scale = Normal(kExperts, 0.01F, rng);
  for (float& s : expert_scale) {
    s += 1.0F;
  }
  const auto x = Normal(std::size_t{kRows} * kHidden, 3.0F, rng);
  float* d_router = Device(router.data(), router.size());
  float* d_scale = Device(scale.data(), scale.size());
  g4k::ScaleRouter(d_router, d_scale, kExperts, kHidden, nullptr);
  float* d_es = Device(expert_scale.data(), expert_scale.size());
  float* d_x = Device(x.data(), x.size());
  const Routing all = Route(d_x, d_router, d_es, kRows);

  // FP64 reference routing.
  for (std::uint32_t r = 0; r < kRows; ++r) {
    const float* xr = &x[std::size_t{r} * kHidden];
    double ss = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      ss += double{xr[i]} * xr[i];
    }
    const double rs = 1.0 / std::sqrt(ss / kHidden + kEps) / std::sqrt(kHidden);
    std::vector<double> logits(kExperts);
    for (std::uint32_t e = 0; e < kExperts; ++e) {
      double dot = 0.0;
      for (std::uint32_t i = 0; i < kHidden; ++i) {
        dot += double{router[std::size_t{e} * kHidden + i]} * scale[i] * xr[i];
      }
      logits[e] = dot * rs;
      const double got = all.logits[std::size_t{r} * kExperts + e];
      Require(std::fabs(got - logits[e]) < 1e-4 * (1.0 + std::fabs(logits[e])),
              "router logit error");
    }
    std::vector<std::uint32_t> order(kExperts);
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(),
                     [&](auto a, auto b) { return logits[a] > logits[b]; });
    double total = 0.0;
    for (std::uint32_t j = 0; j < kUsed; ++j) {
      total += std::exp(logits[order[j]] - logits[order[0]]);
    }
    for (std::uint32_t j = 0; j < kUsed; ++j) {
      const std::size_t slot = std::size_t{r} * kUsed + j;
      Require(all.ids[slot] == static_cast<std::int32_t>(order[j]),
              "top-k expert differs from the reference");
      const double want = std::exp(logits[order[j]] - logits[order[0]]) /
                          total * expert_scale[order[j]];
      Require(std::fabs(all.weights[slot] - want) < 1e-5,
              "mixture weight error");
    }
  }
  // Group table: every selected expert once, ascending, slots ascending.
  std::uint32_t seen = 0;
  std::int32_t previous = -1;
  for (std::int32_t g = 0; g < all.groups[0]; ++g) {
    const std::int32_t* group = &all.groups[1 + g * g4k::kGroupInts];
    Require(group[0] > previous && group[1] > 0, "group order");
    previous = group[0];
    for (std::int32_t i = 0; i < group[1]; ++i) {
      Require(all.ids[group[2 + i]] == group[0] &&
                  (i == 0 || group[2 + i] > group[1 + i]),
              "group slots");
      ++seen;
    }
  }
  Require(seen == kRows * kUsed, "groups do not cover every slot");

  // One-launch routing (the decode path) gives the same bits, twice in a
  // row (its counter returns to zero).
  {
    std::uint32_t* sync = nullptr;
    HIP_CHECK(hipMalloc(&sync, sizeof(std::uint32_t)));
    HIP_CHECK(hipMemset(sync, 0, sizeof(std::uint32_t)));
    for (int round = 0; round < 2; ++round) {
      const Routing fused = Route(d_x, d_router, d_es, kRows, sync);
      Require(fused.ids == all.ids, "one-launch routing: ids differ");
      // Each group's expert, slot count and slots.
      bool same = fused.groups[0] == all.groups[0];
      for (std::int32_t g = 0; same && g < all.groups[0]; ++g) {
        const std::int32_t* x = &all.groups[1 + g * g4k::kGroupInts];
        const std::int32_t* y = &fused.groups[1 + g * g4k::kGroupInts];
        same = std::equal(x, x + 2 + x[1], y);
      }
      Require(same, "one-launch routing: groups differ");
      Require(std::memcmp(fused.weights.data(), all.weights.data(),
                          all.weights.size() * 4) == 0,
              "one-launch routing: weights differ");
      Require(std::memcmp(fused.logits.data(), all.logits.data(),
                          all.logits.size() * 4) == 0,
              "one-launch routing: logits differ");
    }
    HIP_CHECK(hipFree(sync));
  }

  // Batch invariance: each row routed alone gives identical bits.
  for (std::uint32_t r = 0; r < kRows; r += 5) {
    const Routing one =
        Route(d_x + std::size_t{r} * kHidden, d_router, d_es, 1);
    Require(std::memcmp(one.logits.data(), &all.logits[r * kExperts],
                        kExperts * 4) == 0 &&
                std::memcmp(one.weights.data(), &all.weights[r * kUsed],
                            kUsed * 4) == 0 &&
                std::equal(one.ids.begin(), one.ids.end(),
                           all.ids.begin() + r * kUsed),
            "routing depends on the batch");
  }
  // The prefill form: raw dot products from a GEMM, scaled and selected.
  {
    std::vector<float> raw(std::size_t{kRows} * kExperts);
    for (std::uint32_t r = 0; r < kRows; ++r) {
      for (std::uint32_t e = 0; e < kExperts; ++e) {
        double dot = 0.0;
        for (std::uint32_t i = 0; i < kHidden; ++i) {
          dot += double{router[std::size_t{e} * kHidden + i]} * scale[i] *
                 x[std::size_t{r} * kHidden + i];
        }
        raw[std::size_t{r} * kExperts + e] = static_cast<float>(dot);
      }
    }
    float* d_raw = Device(raw.data(), raw.size());
    std::int32_t* ids = nullptr;
    float* weights = nullptr;
    HIP_CHECK(hipMalloc(&ids, kRows * kUsed * sizeof(std::int32_t)));
    HIP_CHECK(hipMalloc(&weights, kRows * kUsed * sizeof(float)));
    g4k::MoeRouteArgs a{};
    a.x = d_x;
    a.expert_scale = d_es;
    a.logits = d_raw;
    a.raw_logits = true;
    a.ids = ids;
    a.weights = weights;
    a.rows = kRows;
    a.hidden = kHidden;
    a.experts = kExperts;
    a.used = kUsed;
    a.eps = kEps;
    g4k::MoeRoute(a, nullptr);
    Require(Host(ids, kRows * kUsed) == all.ids,
            "raw-logit routing selects other experts");
    const auto w = Host(weights, kRows * kUsed);
    for (std::size_t i = 0; i < w.size(); ++i) {
      Require(std::fabs(w[i] - all.weights[i]) < 1e-5,
              "raw-logit mixture weights");
    }
    HIP_CHECK(hipFree(d_raw));
    HIP_CHECK(hipFree(ids));
    HIP_CHECK(hipFree(weights));
  }
  std::cout << "routing: " << all.groups[0] << " expert groups for " << kRows
            << " rows\n";
  HIP_CHECK(hipFree(d_router));
  HIP_CHECK(hipFree(d_scale));
  HIP_CHECK(hipFree(d_es));
  HIP_CHECK(hipFree(d_x));
}

/// Group table for `rows` rows with the given expert ids ([rows][kUsed]).
std::vector<std::int32_t> Groups(const std::vector<std::int32_t>& ids) {
  std::vector<std::int32_t> table(g4k::ExpertGroupInts(kExperts), 0);
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    std::vector<std::int32_t> slots;
    for (std::size_t s = 0; s < ids.size(); ++s) {
      if (ids[s] == static_cast<std::int32_t>(e)) {
        slots.push_back(static_cast<std::int32_t>(s));
      }
    }
    if (slots.empty()) {
      continue;
    }
    std::int32_t* g = &table[1 + table[0] * g4k::kGroupInts];
    g[0] = static_cast<std::int32_t>(e);
    g[1] = static_cast<std::int32_t>(slots.size());
    std::copy(slots.begin(), slots.end(), g + 2);
    ++table[0];
  }
  return table;
}

/// One projection shape: m outputs over k per expert; `x_div` maps a slot
/// to its activation row (kUsed for the gate/up input shared by a token's
/// slots, 1 for the down projection's per-slot input).
void CheckProjection(const Format& f, std::uint32_t m, std::uint32_t k,
                     std::uint32_t x_div, std::mt19937& rng) {
  const std::string name =
      std::string(f.name) + " " + std::to_string(m) + "x" + std::to_string(k);
  constexpr std::uint32_t kRows = 16;
  // Expert ids with repeats across rows, as in verification batches.
  std::vector<std::int32_t> ids(kRows * kUsed);
  std::uniform_int_distribution<int> pick(0, 23);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    std::vector<int> row;
    while (row.size() < kUsed) {
      const int e = r % 3 == 0 ? pick(rng) : pick(rng) * 5;
      if (std::find(row.begin(), row.end(), e) == row.end()) {
        row.push_back(e);
      }
    }
    std::copy(row.begin(), row.end(), ids.begin() + r * kUsed);
  }
  const std::uint32_t max_expert =
      static_cast<std::uint32_t>(*std::max_element(ids.begin(), ids.end())) + 1;
  const auto w = RandomMatrix(f, std::size_t{max_expert} * m, k, rng);
  const std::uint32_t x_rows = kRows * kUsed / x_div;
  const auto x = Normal(std::size_t{x_rows} * k, 1.0F, rng);
  auto* dw = Device(w.data(), w.size());
  float* dx = Device(x.data(), x.size());
  float* dy = nullptr;
  HIP_CHECK(hipMalloc(&dy, std::size_t{kRows} * kUsed * m * sizeof(float)));
  std::int32_t* dg = nullptr;
  HIP_CHECK(hipMalloc(&dg, g4k::ExpertGroupInts(kExperts) * 4));

  const auto run = [&](std::uint32_t first_row, std::uint32_t rows,
                       bool single = false) {
    std::vector<std::int32_t> sub(ids.begin() + first_row * kUsed,
                                  ids.begin() + (first_row + rows) * kUsed);
    const auto table = Groups(sub);
    HIP_CHECK(
        hipMemcpy(dg, table.data(), table.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dy, 0xFF, std::size_t{rows} * kUsed * m * 4));
    const float* xin = dx + std::size_t{first_row} * kUsed / x_div * k;
    Require(g4k::LaunchRoutedGemv(f.format, dw, dg, rows * kUsed, xin, x_div,
                                  dy, m, k, nullptr, false, single),
            name + ": shape rejected");
    return Host(dy, std::size_t{rows} * kUsed * m);
  };
  const auto all = run(0, kRows);

  // Against FP64 dots of the dequantized rows (sampled).
  const std::size_t row_bytes = k / f.block * f.bytes;
  std::vector<float> row(k);
  double worst = 0.0;
  for (std::uint32_t s = 0; s < kRows * kUsed; s += 7) {
    const std::size_t e = static_cast<std::size_t>(ids[s]);
    const float* xs = &x[std::size_t{s / x_div} * k];
    for (std::uint32_t o = 0; o < m; o += std::max<std::uint32_t>(1, m / 61)) {
      gufo::quant::Dequantize(f.type, w.data() + (e * m + o) * row_bytes,
                              row.data(), k);
      double want = 0.0, magnitude = 0.0;
      for (std::uint32_t c = 0; c < k; ++c) {
        const double p = double{row[c]} * xs[c];
        want += p;
        magnitude += std::fabs(p);
      }
      worst = std::max(
          worst, std::fabs(all[std::size_t{s} * m + o] - want) / magnitude);
    }
  }
  Require(worst < 2e-6, name + ": error " + std::to_string(worst));

  // A row alone, and inside narrower batches, matches the 16-row batch.
  for (std::uint32_t rows : {1U, 2U, 5U, 8U}) {
    for (std::uint32_t first = 0; first + rows <= kRows; first += rows + 3) {
      const auto part = run(first, rows);
      Require(std::memcmp(part.data(), &all[std::size_t{first} * kUsed * m],
                          part.size() * 4) == 0,
              name + ": " + std::to_string(rows) +
                  "-row batch differs from the 16-row batch");
    }
  }
  // One-row decode runs a single slot per pass.
  for (std::uint32_t first = 0; first < kRows; first += 5) {
    const auto part = run(first, 1, true);
    Require(std::memcmp(part.data(), &all[std::size_t{first} * kUsed * m],
                        part.size() * 4) == 0,
            name + ": the one-slot pass differs from the 16-row batch");
  }

  // The fused gate/up form: GeGLU of the same rows, batch invariant too.
  if (x_div == kUsed) {
    const std::uint32_t width = m / 2;
    const auto run_geglu = [&](std::uint32_t first_row, std::uint32_t rows,
                               bool single = false) {
      std::vector<std::int32_t> sub(ids.begin() + first_row * kUsed,
                                    ids.begin() + (first_row + rows) * kUsed);
      const auto table = Groups(sub);
      HIP_CHECK(
          hipMemcpy(dg, table.data(), table.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(dy, 0xFF, std::size_t{rows} * kUsed * width * 4));
      const float* xin = dx + std::size_t{first_row} * kUsed / x_div * k;
      Require(g4k::LaunchRoutedGemv(f.format, dw, dg, rows * kUsed, xin, x_div,
                                    dy, m, k, nullptr, true, single),
              name + ": GeGLU shape rejected");
      return Host(dy, std::size_t{rows} * kUsed * width);
    };
    const auto fused = run_geglu(0, kRows);
    double geglu_worst = 0.0;
    for (std::size_t s = 0; s < std::size_t{kRows} * kUsed; ++s) {
      for (std::uint32_t c = 0; c < width; ++c) {
        const double g = all[s * m + c];
        const double u = all[s * m + width + c];
        const double want = 0.5 * g *
                            (1.0 + std::tanh(0.7978845608028654 * g *
                                             (1.0 + 0.044715 * g * g))) *
                            u;
        // FP32 1 + tanh cancels for very negative gates: scale the error by
        // |gate * up|.
        geglu_worst =
            std::max(geglu_worst, std::fabs(fused[s * width + c] - want) /
                                      (std::fabs(g * u) + 1e-3));
      }
    }
    Require(geglu_worst < 1e-6,
            name + ": GeGLU error " + std::to_string(geglu_worst));
    for (std::uint32_t rows : {1U, 5U}) {
      const auto part = run_geglu(3, rows);
      Require(std::memcmp(part.data(), &fused[std::size_t{3} * kUsed * width],
                          part.size() * 4) == 0,
              name + ": fused GeGLU depends on the batch");
    }
    const auto single = run_geglu(3, 1, true);
    Require(std::memcmp(single.data(), &fused[std::size_t{3} * kUsed * width],
                        single.size() * 4) == 0,
            name + ": the one-slot fused GeGLU pass differs");
  }

  // Cold weights as in decode: a full 128-expert tensor, fresh random
  // experts every launch, so the 32 MiB MALL holds none of them.
  const std::size_t expert_bytes = std::size_t{m} * row_bytes;
  const auto full = RandomMatrix(f, std::size_t{kExperts} * m, k, rng);
  auto* dfull = Device(full.data(), full.size());
  constexpr int kIters = 48;
  const auto time = [&](std::uint32_t rows) {
    std::vector<std::vector<std::int32_t>> tables;
    std::vector<std::int32_t*> device_tables;
    for (int i = 0; i < kIters + 3; ++i) {
      std::vector<std::int32_t> sub(rows * kUsed);
      for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<int> all(kExperts);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        std::copy_n(all.begin(), kUsed, sub.begin() + r * kUsed);
      }
      const auto table = Groups(sub);
      device_tables.push_back(Device(table.data(), table.size()));
    }
    const auto launch = [&](int i) {
      (void)g4k::LaunchRoutedGemv(f.format, dfull, device_tables[i],
                                  std::min(kExperts, rows * kUsed), dx, x_div,
                                  dy, m, k, nullptr);
    };
    for (int i = 0; i < 3; ++i) {
      launch(kIters + i);
    }
    HIP_CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
      launch(i);
    }
    HIP_CHECK(hipDeviceSynchronize());
    const double us =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() /
        kIters * 1e6;
    for (auto* t : device_tables) {
      HIP_CHECK(hipFree(t));
    }
    return us;
  };
  const double one = time(1);
  const double five = time(5);
  std::cout << name << ": 8 experts " << one << " us ("
            << 8 * expert_bytes / one / 1e3 << " GB/s); 5 rows " << five
            << " us\n";
  HIP_CHECK(hipFree(dfull));
  HIP_CHECK(hipFree(dw));
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dy));
  HIP_CHECK(hipFree(dg));
}

void CheckFinish(std::mt19937& rng) {
  constexpr std::uint32_t kRows = 5;
  const auto dense = Normal(kRows * kHidden, 2.0F, rng);
  const auto experts = Normal(kRows * kUsed * kHidden, 1.0F, rng);
  const auto weights = Normal(kRows * kUsed, 0.3F, rng);
  const auto x = Normal(kRows * kHidden, 5.0F, rng);
  const auto n1 = Normal(kHidden, 1.0F, rng);
  const auto n2 = Normal(kHidden, 1.0F, rng);
  const auto post = Normal(kHidden, 1.0F, rng);
  const auto next = Normal(kHidden, 1.0F, rng);
  constexpr float kScale = 0.25F;
  float* dd = Device(dense.data(), dense.size());
  float* de = Device(experts.data(), experts.size());
  float* dw = Device(weights.data(), weights.size());
  float* dx = Device(x.data(), x.size());
  float* d1 = Device(n1.data(), n1.size());
  float* d2 = Device(n2.data(), n2.size());
  float* dp = Device(post.data(), post.size());
  float* dn = Device(next.data(), next.size());
  float* dh = nullptr;
  HIP_CHECK(hipMalloc(&dh, x.size() * sizeof(float)));
  g4k::MoeFinishArgs a{};
  a.dense = dd;
  a.experts = de;
  a.weights = dw;
  a.norm1 = d1;
  a.norm2 = d2;
  a.post_norm = dp;
  a.scale = kScale;
  a.x = dx;
  a.next_norm = dn;
  a.h = dh;
  a.rows = kRows;
  a.hidden = kHidden;
  a.used = kUsed;
  a.eps = kEps;
  g4k::MoeFinish(a, nullptr);
  const auto got_x = Host(dx, x.size());
  const auto got_h = Host(dh, x.size());
  const auto rms = [](const std::vector<double>& v) {
    double s = 0.0;
    for (double e : v) {
      s += e * e;
    }
    return 1.0 / std::sqrt(s / static_cast<double>(v.size()) + kEps);
  };
  double worst = 0.0;
  for (std::uint32_t r = 0; r < kRows; ++r) {
    std::vector<double> d(kHidden), mix(kHidden, 0.0), f(kHidden), xr(kHidden);
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      d[i] = dense[r * kHidden + i];
      for (std::uint32_t j = 0; j < kUsed; ++j) {
        mix[i] += double{weights[r * kUsed + j]} *
                  experts[(std::size_t{r} * kUsed + j) * kHidden + i];
      }
    }
    const double rd = rms(d), rm = rms(mix);
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      f[i] = d[i] * rd * n1[i] + mix[i] * rm * n2[i];
    }
    const double rf = rms(f);
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      xr[i] = (x[r * kHidden + i] + f[i] * rf * post[i]) * kScale;
    }
    const double rx = rms(xr);
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      const double want_h = xr[i] * rx * next[i];
      worst = std::max({worst,
                        std::fabs(got_x[r * kHidden + i] - xr[i]) /
                            (std::fabs(xr[i]) + 1e-2),
                        std::fabs(got_h[r * kHidden + i] - want_h) /
                            (std::fabs(want_h) + 1e-2)});
    }
  }
  Require(worst < 1e-4, "feed-forward residual error " + std::to_string(worst));
  for (float* p : {dd, de, dw, dx, d1, d2, dp, dn, dh}) {
    HIP_CHECK(hipFree(p));
  }
}

/// The K-quant routed prefill GEMM over binary16 rows, in the routing layout
/// the executor builds (Flash-Next compaction, 96-row tiles), against FP64
/// dots of the dequantized rows and the binary16 inputs; Q4_K, Q5_K, Q5_1 and
/// Q8_0 also bit for bit against Flash-Next's routed GEMM. At the gate/up
/// shape the fused GeGLU matches the separate pass (for Q8_0 over Flash-Next's
/// binary16 rows, the path it replaces).
void CheckRoutedPrefill(const Format& f, std::uint32_t m, std::uint32_t k,
                        bool gate_up, std::mt19937& rng) {
  namespace fn = gufo::models::qwen38_flash_next::rocm;
  // About 150 rows per expert: a full 96-row tile and a partly live one.
  constexpr std::uint32_t kTokens = 300;
  constexpr std::uint32_t kExpertsHere = 16;
  constexpr std::uint32_t kTile = 96;
  std::vector<std::int32_t> ids(kTokens * kUsed);
  std::vector<std::uint32_t> counts(kExpertsHere, 0);
  for (std::uint32_t t = 0; t < kTokens; ++t) {
    std::vector<int> all(kExpertsHere);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    for (std::uint32_t j = 0; j < kUsed; ++j) {
      ids[t * kUsed + j] = all[j];
      ++counts[all[j]];
    }
  }
  const auto tile_map = [&](std::uint32_t rows) {
    std::vector<std::int32_t> map;
    for (std::uint32_t e = 0; e < kExpertsHere; ++e) {
      const std::uint32_t padded = (counts[e] + 15) / 16 * 16;
      for (std::uint32_t j = 0; j * rows < padded; ++j) {
        map.push_back(static_cast<std::int32_t>(e | (j << 16)));
      }
    }
    return map;
  };
  const std::vector<std::int32_t> tiles = tile_map(kTile);
  const auto w = RandomMatrix(f, std::size_t{kExpertsHere} * m, k, rng);
  const auto x = Normal(std::size_t{kTokens} * k, 1.0F, rng);
  std::vector<__half> xh(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    xh[i] = __float2half(x[i]);
  }
  const std::size_t slots = kTokens * kUsed;
  const std::size_t compact = fn::RoutedCompactRows(slots, kExpertsHere);
  auto* dw = Device(w.data(), w.size());
  auto* dx = Device(xh.data(), xh.size());
  auto* dids = Device(ids.data(), ids.size());
  auto* dcounts = Device(counts.data(), counts.size());
  auto* dtiles = Device(tiles.data(), tiles.size());
  std::int32_t *bounds = nullptr, *cursors = nullptr, *rows_token = nullptr,
               *rows_slot = nullptr;
  float* dy = nullptr;
  HIP_CHECK(hipMalloc(&bounds, (kExpertsHere + 1) * 4));
  HIP_CHECK(hipMalloc(&cursors, kExpertsHere * 4));
  HIP_CHECK(hipMalloc(&rows_token, compact * 4));
  HIP_CHECK(hipMalloc(&rows_slot, compact * 4));
  HIP_CHECK(hipMalloc(&dy, slots * m * sizeof(float)));
  fn::RoutedCompact(dids, dcounts, bounds, cursors, rows_token, rows_slot,
                    kTokens, kUsed, kExpertsHere, nullptr);
  Require(g4k::LaunchRoutedHalfGemm(f.format, dw, dx, dtiles,
                                    static_cast<std::uint32_t>(tiles.size()),
                                    kTile, bounds, rows_token, rows_slot, dy,
                                    nullptr, m, k, nullptr),
          "routed prefill GEMM rejected " + std::string(f.name));
  const auto got = Host(dy, slots * m);
  // Gate/up with GeGLU in the epilogue: the bytes the separate binary16 pass
  // makes of the binary16 [gate | up] rows.
  if (gate_up) {
    const std::uint32_t width = m / 2;
    __half* gu = nullptr;
    __half* want = nullptr;
    __half* fused = nullptr;
    HIP_CHECK(hipMalloc(&gu, slots * m * 2));
    HIP_CHECK(hipMalloc(&want, slots * width * 2));
    HIP_CHECK(hipMalloc(&fused, slots * width * 2));
    const auto n_tiles = static_cast<std::uint32_t>(tiles.size());
    if (f.format == g4k::ExpertFormat::kF16) {
      // The FP32 rows above, rounded as the binary16 output would be.
      std::vector<__half> rows(got.size());
      for (std::size_t i = 0; i < got.size(); ++i) {
        rows[i] = __float2half(std::clamp(got[i], -65504.0F, 65504.0F));
      }
      HIP_CHECK(
          hipMemcpy(gu, rows.data(), rows.size() * 2, hipMemcpyHostToDevice));
    } else if (f.format == g4k::ExpertFormat::kQ8_0) {
      const std::vector<std::int32_t> tiles64 = tile_map(64);
      auto* dtiles64 = Device(tiles64.data(), tiles64.size());
      Require(fn::RoutedF16Gemm(dw, fn::WeightType::kQ8_0,
                                reinterpret_cast<const __half*>(dx), dtiles64,
                                static_cast<std::uint32_t>(tiles64.size()), 64,
                                bounds, rows_token, rows_slot, nullptr, nullptr,
                                gu, m, k, nullptr),
              "Flash-Next binary16 routed GEMM rejected");
      HIP_CHECK(hipFree(dtiles64));
    } else {
      Require(g4k::LaunchRoutedHalfGemm(f.format, dw, dx, dtiles, n_tiles,
                                        kTile, bounds, rows_token, rows_slot,
                                        nullptr, gu, m, k, nullptr),
              "binary16 routed prefill rejected");
    }
    g4k::GeGluPackedHalf(gu, want, static_cast<std::uint32_t>(slots), width,
                         nullptr);
    Require(g4k::LaunchRoutedHalfGemm(f.format, dw, dx, dtiles, n_tiles, kTile,
                                      bounds, rows_token, rows_slot, nullptr,
                                      fused, m, k, nullptr, true),
            "fused GeGLU routed prefill rejected");
    const auto a = Host(want, slots * width);
    const auto b = Host(fused, slots * width);
    Require(
        std::memcmp(a.data(), b.data(), a.size() * 2) == 0,
        std::string(f.name) + ": fused GeGLU differs from the separate pass");
    for (void* p : {static_cast<void*>(gu), static_cast<void*>(want),
                    static_cast<void*>(fused)}) {
      HIP_CHECK(hipFree(p));
    }
  }
  if (f.format != g4k::ExpertFormat::kQ6_K &&
      f.format != g4k::ExpertFormat::kF16) {
    // Flash-Next's widest tile per format: 48 rows (K-quants), 64 (Q5_1,
    // Q8_0).
    const bool down = f.format == g4k::ExpertFormat::kQ5_1 ||
                      f.format == g4k::ExpertFormat::kQ8_0;
    const std::uint32_t rows = down ? 64 : 48;
    const std::vector<std::int32_t> tiles48 = tile_map(rows);
    auto* dtiles48 = Device(tiles48.data(), tiles48.size());
    HIP_CHECK(hipMemset(dy, 0, slots * m * sizeof(float)));
    const fn::WeightType type =
        f.format == g4k::ExpertFormat::kQ4_K   ? fn::WeightType::kQ4_K
        : f.format == g4k::ExpertFormat::kQ5_K ? fn::WeightType::kQ5_K
        : f.format == g4k::ExpertFormat::kQ5_1 ? fn::WeightType::kQ5_1
                                               : fn::WeightType::kQ8_0;
    Require(fn::RoutedF16Gemm(
                dw, type, reinterpret_cast<const __half*>(dx), dtiles48,
                static_cast<std::uint32_t>(tiles48.size()), rows, bounds,
                rows_token, rows_slot, nullptr, dy, nullptr, m, k, nullptr),
            "Flash-Next routed GEMM rejected " + std::string(f.name));
    const auto flash = Host(dy, slots * m);
    Require(
        std::memcmp(flash.data(), got.data(), got.size() * sizeof(float)) == 0,
        std::string(f.name) + " routed prefill differs from Flash-Next");
    HIP_CHECK(hipFree(dtiles48));
  }
  const std::size_t row_bytes = k / f.block * f.bytes;
  std::vector<float> row(k);
  double worst = 0.0;
  for (std::size_t s = 0; s < slots; s += 13) {
    const std::size_t e = static_cast<std::size_t>(ids[s]);
    for (std::uint32_t o = 0; o < m; o += 37) {
      gufo::quant::Dequantize(f.type, w.data() + (e * m + o) * row_bytes,
                              row.data(), k);
      double want = 0.0, magnitude = 0.0;
      for (std::uint32_t c = 0; c < k; ++c) {
        const double p = double{row[c]} * __half2float(xh[(s / kUsed) * k + c]);
        want += p;
        magnitude += std::fabs(p);
      }
      worst = std::max(worst, std::fabs(got[s * m + o] - want) / magnitude);
    }
  }
  std::cout << "routed prefill " << f.name << ": error " << worst << "\n";
  Require(worst < 2e-3, "routed prefill error " + std::to_string(worst));
  for (void* p : {static_cast<void*>(dw), static_cast<void*>(dx),
                  static_cast<void*>(dids), static_cast<void*>(dcounts),
                  static_cast<void*>(dtiles), static_cast<void*>(bounds),
                  static_cast<void*>(cursors), static_cast<void*>(rows_token),
                  static_cast<void*>(rows_slot), static_cast<void*>(dy)}) {
    HIP_CHECK(hipFree(p));
  }
}

/// The dense binary16 prefill GEMM (dense models' K-quant projections)
/// against the routed GEMM over an identity routing, bit for bit, with FP32
/// and fused-GeGLU outputs, at row counts that end inside a token tile and
/// an m whose last row block is partial.
void CheckDensePrefill(const Format& f, std::uint32_t m, std::uint32_t k,
                       std::mt19937& rng) {
  constexpr std::uint32_t kTile = 96;
  const auto w = RandomMatrix(f, m, k, rng);
  auto* dw = Device(w.data(), w.size());
  for (const std::uint32_t rows : {1U, 97U, 300U}) {
    const auto x = Normal(std::size_t{rows} * k, 1.0F, rng);
    std::vector<__half> xh(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      xh[i] = __float2half(x[i]);
    }
    // One expert whose 16-padded bucket holds every row in order.
    const std::uint32_t padded = (rows + 15) / 16 * 16;
    std::vector<std::int32_t> ids(padded, -1);
    std::iota(ids.begin(), ids.begin() + rows, 0);
    std::vector<std::int32_t> tiles;
    for (std::uint32_t j = 0; j * kTile < padded; ++j) {
      tiles.push_back(static_cast<std::int32_t>(j << 16));
    }
    const std::vector<std::int32_t> bounds = {
        0, static_cast<std::int32_t>(padded)};
    auto* dx = Device(xh.data(), xh.size());
    auto* dids = Device(ids.data(), ids.size());
    auto* dtiles = Device(tiles.data(), tiles.size());
    auto* dbounds = Device(bounds.data(), bounds.size());
    const auto n_tiles = static_cast<std::uint32_t>(tiles.size());
    float *routed = nullptr, *dense = nullptr;
    __half *routed_half = nullptr, *dense_half = nullptr;
    HIP_CHECK(hipMalloc(&routed, std::size_t{rows} * m * 4));
    HIP_CHECK(hipMalloc(&dense, std::size_t{rows} * m * 4));
    HIP_CHECK(hipMalloc(&routed_half, std::size_t{rows} * m));
    HIP_CHECK(hipMalloc(&dense_half, std::size_t{rows} * m));
    Require(g4k::LaunchRoutedHalfGemm(f.format, dw, dx, dtiles, n_tiles, kTile,
                                      dbounds, dids, dids, routed, nullptr, m,
                                      k, nullptr) &&
                g4k::LaunchDenseHalfGemm(f.format, dw, dx, dense, nullptr, rows,
                                         m, k, nullptr) &&
                g4k::LaunchRoutedHalfGemm(f.format, dw, dx, dtiles, n_tiles,
                                          kTile, dbounds, dids, dids, nullptr,
                                          routed_half, m, k, nullptr, true) &&
                g4k::LaunchDenseHalfGemm(f.format, dw, dx, nullptr, dense_half,
                                         rows, m, k, nullptr, true),
            "dense prefill GEMM rejected " + std::string(f.name));
    const auto a = Host(routed, std::size_t{rows} * m);
    const auto b = Host(dense, std::size_t{rows} * m);
    Require(std::memcmp(a.data(), b.data(), a.size() * 4) == 0,
            std::string(f.name) + ": dense prefill differs from routed at " +
                std::to_string(rows) + " rows");
    const auto ah = Host(routed_half, std::size_t{rows} * m / 2);
    const auto bh = Host(dense_half, std::size_t{rows} * m / 2);
    Require(std::memcmp(ah.data(), bh.data(), ah.size() * 2) == 0,
            std::string(f.name) + ": dense GeGLU differs from routed at " +
                std::to_string(rows) + " rows");
    for (void* p :
         {static_cast<void*>(dx), static_cast<void*>(dids),
          static_cast<void*>(dtiles), static_cast<void*>(dbounds),
          static_cast<void*>(routed), static_cast<void*>(dense),
          static_cast<void*>(routed_half), static_cast<void*>(dense_half)}) {
      HIP_CHECK(hipFree(p));
    }
  }
  std::cout << "dense prefill " << f.name << ": equals routed\n";
  HIP_CHECK(hipFree(dw));
}

/// The dense Q4_0 prefill GEMM (the QAT targets) against FP64 dots of the
/// dequantized rows and the binary16 inputs; its fused GeGLU bit for bit
/// against the separate pass over its own binary16-rounded FP32 rows.
void CheckDenseQ40(const Format& f, std::uint32_t m, std::uint32_t k,
                   std::mt19937& rng) {
  const auto w = RandomMatrix(f, m, k, rng);
  auto* dw = Device(w.data(), w.size());
  const std::size_t row_bytes = k / f.block * f.bytes;
  for (const std::uint32_t rows : {1U, 97U, 300U}) {
    const auto x = Normal(std::size_t{rows} * k, 1.0F, rng);
    std::vector<__half> xh(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      xh[i] = __float2half(x[i]);
    }
    auto* dx = Device(xh.data(), xh.size());
    const std::uint32_t width = m / 2;
    float* dy = nullptr;
    __half *gu = nullptr, *want = nullptr, *fused = nullptr;
    HIP_CHECK(hipMalloc(&dy, std::size_t{rows} * m * 4));
    HIP_CHECK(hipMalloc(&gu, std::size_t{rows} * m * 2));
    HIP_CHECK(hipMalloc(&want, std::size_t{rows} * width * 2));
    HIP_CHECK(hipMalloc(&fused, std::size_t{rows} * width * 2));
    Require(g4k::LaunchDenseHalfGemm(f.format, dw, dx, dy, nullptr, rows, m, k,
                                     nullptr) &&
                g4k::LaunchDenseHalfGemm(f.format, dw, dx, nullptr, fused, rows,
                                         m, k, nullptr, true),
            "dense Q4_0 prefill GEMM rejected");
    const auto got = Host(dy, std::size_t{rows} * m);
    std::vector<float> row(k);
    double worst = 0.0;
    for (std::uint32_t r = 0; r < rows; r += 7) {
      for (std::uint32_t o = 0; o < m; o += 13) {
        gufo::quant::Dequantize(f.type, w.data() + o * row_bytes, row.data(),
                                k);
        double expect = 0.0, magnitude = 0.0;
        for (std::uint32_t c = 0; c < k; ++c) {
          const double p = double{row[c]} * __half2float(xh[r * k + c]);
          expect += p;
          magnitude += std::fabs(p);
        }
        worst = std::max(
            worst, std::fabs(got[std::size_t{r} * m + o] - expect) / magnitude);
      }
    }
    Require(worst < 2e-3, "dense Q4_0 prefill error " + std::to_string(worst));
    std::vector<__half> rounded(got.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      rounded[i] = __float2half(std::clamp(got[i], -65504.0F, 65504.0F));
    }
    HIP_CHECK(hipMemcpy(gu, rounded.data(), rounded.size() * 2,
                        hipMemcpyHostToDevice));
    g4k::GeGluPackedHalf(gu, want, rows, width, nullptr);
    const auto a = Host(want, std::size_t{rows} * width);
    const auto b = Host(fused, std::size_t{rows} * width);
    Require(std::memcmp(a.data(), b.data(), a.size() * 2) == 0,
            "dense Q4_0 GeGLU differs from the separate pass at " +
                std::to_string(rows) + " rows");
    std::cout << "dense prefill Q4_0, " << rows << " rows: error " << worst
              << "\n";
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dy),
                    static_cast<void*>(gu), static_cast<void*>(want),
                    static_cast<void*>(fused)}) {
      HIP_CHECK(hipFree(p));
    }
  }
  HIP_CHECK(hipFree(dw));
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "SKIP: no HIP device\n";
    return 77;
  }
  return gemma4_test::Run([] {
    std::mt19937 rng(26);
    CheckRouting(rng);
    CheckFinish(rng);
    const Format q4k{
        g4k::ExpertFormat::kQ4_K, GgmlType::kQ4_K, 256, 144, {0, 2}, "Q4_K"};
    const Format q5k{
        g4k::ExpertFormat::kQ5_K, GgmlType::kQ5_K, 256, 176, {0, 2}, "Q5_K"};
    const Format q6k{
        g4k::ExpertFormat::kQ6_K, GgmlType::kQ6_K, 256, 210, {208}, "Q6_K"};
    const Format q80{
        g4k::ExpertFormat::kQ8_0, GgmlType::kQ8_0, 32, 34, {0}, "Q8_0"};
    const Format q51{
        g4k::ExpertFormat::kQ5_1, GgmlType::kQ5_1, 32, 24, {0, 2}, "Q5_1"};
    // BF16 experts (the UD-Q8_K_XL last layer) run as binary16.
    const Format f16{
        g4k::ExpertFormat::kF16, GgmlType::kF16, 32, 64, {}, "F16"};
    // Fused gate/up [2 * 704, 2816] and down [2816, 704] of every format the
    // Unsloth quants use.
    for (const Format& f : {q4k, q5k, q6k, q80, f16}) {
      CheckProjection(f, 2 * kWidth, kHidden, kUsed, rng);
    }
    for (const Format& f : {q51, q80, f16}) {
      CheckProjection(f, kHidden, kWidth, 1, rng);
    }
    for (const Format& f : {q4k, q5k, q6k, q80, f16}) {
      CheckRoutedPrefill(f, 2 * kWidth, kHidden, true, rng);
    }
    for (const Format& f : {q51, q80, f16}) {
      CheckRoutedPrefill(f, kHidden, kWidth, false, rng);
    }
    // A fused gate/up of 2 x 1000 rows: partial row blocks both ways.
    for (const Format& f : {q4k, q5k, q6k}) {
      CheckDensePrefill(f, 2000, kHidden, rng);
    }
    const Format q40{
        g4k::ExpertFormat::kQ4_0, GgmlType::kQ4_0, 32, 18, {0}, "Q4_0"};
    CheckDenseQ40(q40, 2000, kHidden, rng);
  });
}
