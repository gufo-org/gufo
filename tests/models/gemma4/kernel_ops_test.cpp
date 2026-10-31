// Gemma 4 elementwise and per-row kernels against FP64 formulas: Q/K/V
// post-processing (weighted and unweighted norms, NEOX rope with frequency
// divisors, binary16 cache slots in linear and ring caches), the fused
// residual norms, GeGLU and the logit softcap; the fused prefill
// quantization writes the shared Q8_1 activation; sampling candidates hold
// each row's exact top logits and give the full row's distributions.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/sampling.hpp"
#include "src/models/gemma4/kernels/rocm/candidates.hpp"
#include "src/models/gemma4/kernels/rocm/kernels.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "tests/models/gemma4/check.hpp"

namespace k = gufo::models::gemma4::rocm;
using gemma4_test::Require;

namespace {

constexpr std::uint16_t kSentinel = 0x7C01;  // a NaN payload never written

float FromHalf(std::uint16_t bits) {
  _Float16 h;
  std::memcpy(&h, &bits, 2);
  return static_cast<float>(h);
}

template<class T>
T* Device(const std::vector<T>& host) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, host.size() * sizeof(T) + 16));
  HIP_CHECK(hipMemcpy(ptr, host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  return ptr;
}

template<class T>
std::vector<T> Host(const T* ptr, std::size_t n) {
  std::vector<T> out(n);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(out.data(), ptr, n * sizeof(T), hipMemcpyDeviceToHost));
  return out;
}

std::vector<double> RmsNorm(const float* x, const float* w, std::size_t n) {
  double ss = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    ss += double{x[i]} * x[i];
  const double r = 1.0 / std::sqrt(ss / n + 1e-6);
  std::vector<double> y(n);
  for (std::size_t i = 0; i < n; ++i)
    y[i] = x[i] * r * (w ? w[i] : 1.0);
  return y;
}

/// The angle follows ggml's float formula (it is part of the rope contract
/// and dominates the error at large positions); the rotation is FP64.
void RopeHead(std::vector<double>& h, std::uint32_t pos, float theta,
              const float* factors) {
  const std::size_t half = h.size() / 2;
  const float scale = std::pow(theta, -2.0F / static_cast<float>(h.size()));
  for (std::size_t i = 0; i < half; ++i) {
    float angle_f =
        static_cast<float>(pos) * std::pow(scale, static_cast<float>(i));
    if (factors != nullptr)
      angle_f /= factors[i];
    const double angle = angle_f;
    const double a = h[i];
    const double b = h[i + half];
    h[i] = a * std::cos(angle) - b * std::sin(angle);
    h[i + half] = a * std::sin(angle) + b * std::cos(angle);
  }
}

void CheckQkvPost(std::uint32_t dim, bool global, std::mt19937& rng) {
  const std::uint32_t rows = 3, heads = 4, kv_heads = 2;
  const std::uint32_t ring = global ? 0 : 5;
  const std::uint32_t first = global ? 1000 : 7;  // the ring case wraps
  const float theta = global ? 1e6F : 1e4F;
  std::normal_distribution<float> normal(0.0F, 1.0F);
  const auto fill = [&](std::size_t n, float scale) {
    std::vector<float> v(n);
    for (float& x : v)
      x = normal(rng) * scale;
    return v;
  };
  auto q = fill(std::size_t{rows} * heads * dim, 3.0F);
  const auto kraw = fill(std::size_t{rows} * kv_heads * dim, 2.0F);
  const auto vraw = fill(kraw.size(), 2.0F);
  const auto qn = fill(dim, 1.0F);
  const auto kn = fill(dim, 0.2F);
  std::vector<float> factors(dim / 2, 1e30F);
  for (std::uint32_t i = 0; i < dim / 8; ++i)
    factors[i] = 1.0F;
  const std::uint32_t slots = global ? first + rows : ring;
  std::vector<std::uint16_t> cache(std::size_t{slots} * kv_heads * dim,
                                   kSentinel);

  float* dq = Device(q);
  float* dk = Device(kraw);
  float* dv = global ? dk : Device(vraw);
  float* dqn = Device(qn);
  float* dkn = Device(kn);
  float* dff = Device(factors);
  auto* kc = Device(cache);
  auto* vc = Device(cache);
  k::QkvPostArgs a{};
  a.q = dq;
  a.k = dk;
  a.v = dv;
  a.q_norm = dqn;
  a.k_norm = dkn;
  a.theta_scale = std::pow(theta, -2.0F / static_cast<float>(dim));
  a.freq_factors = global ? dff : nullptr;
  a.k_cache = kc;
  a.v_cache = vc;
  a.rows = rows;
  a.heads = heads;
  a.kv_heads = kv_heads;
  a.head_dim = dim;
  a.first_position = first;
  a.ring = ring;
  a.eps = 1e-6F;
  k::QkvPost(a, nullptr);
  const auto q_out = Host(dq, q.size());
  const auto k_out = Host(kc, cache.size());
  const auto v_out = Host(vc, cache.size());

  const std::string name = global ? "global" : "sliding";
  const float* ff = global ? factors.data() : nullptr;
  double worst_q = 0.0, worst_kv = 0.0;
  std::vector<bool> written(slots, false);
  for (std::uint32_t r = 0; r < rows; ++r) {
    const std::uint32_t pos = first + r;
    for (std::uint32_t h = 0; h < heads; ++h) {
      auto want = RmsNorm(&q[(r * heads + h) * dim], qn.data(), dim);
      RopeHead(want, pos, theta, ff);
      for (std::uint32_t d = 0; d < dim; ++d) {
        worst_q = std::max(
            worst_q, std::fabs(q_out[(r * heads + h) * dim + d] - want[d]));
      }
    }
    const std::uint32_t slot = ring != 0 ? pos % ring : pos;
    written[slot] = true;
    for (std::uint32_t h = 0; h < kv_heads; ++h) {
      const float* kin = &kraw[(r * kv_heads + h) * dim];
      const float* vin = global ? kin : &vraw[(r * kv_heads + h) * dim];
      auto kw = RmsNorm(kin, kn.data(), dim);
      RopeHead(kw, pos, theta, ff);
      const auto vw = RmsNorm(vin, nullptr, dim);
      for (std::uint32_t d = 0; d < dim; ++d) {
        const std::size_t at = (std::size_t{slot} * kv_heads + h) * dim + d;
        worst_kv = std::max(worst_kv, std::fabs(FromHalf(k_out[at]) - kw[d]) /
                                          std::max(1.0, std::fabs(kw[d])));
        worst_kv = std::max(worst_kv, std::fabs(FromHalf(v_out[at]) - vw[d]) /
                                          std::max(1.0, std::fabs(vw[d])));
      }
    }
  }
  Require(worst_q < 2e-4, name + " Q error " + std::to_string(worst_q));
  Require(worst_kv < 2e-3, name + " K/V error " + std::to_string(worst_kv));
  for (std::uint32_t s = 0; s < slots; ++s) {
    if (!written[s]) {
      Require(k_out[std::size_t{s} * kv_heads * dim] == kSentinel,
              name + ": unrelated cache slot written");
    }
  }
  for (void* p :
       {static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dqn),
        static_cast<void*>(dkn), static_cast<void*>(dff),
        static_cast<void*>(kc), static_cast<void*>(vc)}) {
    HIP_CHECK(hipFree(p));
  }
  if (!global)
    HIP_CHECK(hipFree(dv));
}

void CheckNorms(std::mt19937& rng) {
  const std::uint32_t rows = 3, dim = 5376;
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(rows * dim), o(rows * dim), w1(dim), w2(dim);
  for (float& v : x)
    v = normal(rng) * 40.0F;
  for (float& v : o)
    v = normal(rng) * 3.0F;
  for (float& v : w1)
    v = normal(rng);
  for (float& v : w2)
    v = normal(rng);
  float* dx = Device(x);
  float* dox = Device(o);
  float* dw1 = Device(w1);
  float* dw2 = Device(w2);
  std::vector<float> zero(rows * dim, 0.0F);
  float* dh = Device(zero);

  k::PostAttentionNorm(dox, dw1, dx, dw2, dh, rows, dim, 1e-6F, nullptr);
  auto xg = Host(dx, x.size());
  auto hg = Host(dh, x.size());
  double worst = 0.0;
  for (std::uint32_t r = 0; r < rows; ++r) {
    const auto on = RmsNorm(&o[r * dim], w1.data(), dim);
    std::vector<float> xs(dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
      const double want = x[r * dim + i] + on[i];
      xs[i] = static_cast<float>(want);
      worst = std::max(worst, std::fabs(xg[r * dim + i] - want) / 40.0);
    }
    const auto hn = RmsNorm(xs.data(), w2.data(), dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
      worst = std::max(worst, std::fabs(hg[r * dim + i] - hn[i]));
    }
  }
  Require(worst < 1e-4, "PostAttentionNorm error " + std::to_string(worst));

  // x <- (x + rms(f) w1) * 0.25 ; h <- rms(x) w2
  HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  k::PostFeedForwardNorm(dox, dw1, 0.25F, dx, dw2, dh, rows, dim, 1e-6F,
                         nullptr);
  xg = Host(dx, x.size());
  hg = Host(dh, x.size());
  worst = 0.0;
  for (std::uint32_t r = 0; r < rows; ++r) {
    const auto fn = RmsNorm(&o[r * dim], w1.data(), dim);
    std::vector<float> xs(dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
      const double want = (x[r * dim + i] + fn[i]) * 0.25;
      xs[i] = static_cast<float>(want);
      worst = std::max(worst, std::fabs(xg[r * dim + i] - want) / 10.0);
    }
    const auto hn = RmsNorm(xs.data(), w2.data(), dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
      worst = std::max(worst, std::fabs(hg[r * dim + i] - hn[i]));
    }
  }
  Require(worst < 1e-4, "PostFeedForwardNorm error " + std::to_string(worst));

  // Embedding scale then first norm.
  HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  k::ScaleRmsNorm(dx, 2.5F, dw1, dh, rows, dim, 1e-6F, nullptr);
  xg = Host(dx, x.size());
  hg = Host(dh, x.size());
  worst = 0.0;
  for (std::uint32_t r = 0; r < rows; ++r) {
    std::vector<float> xs(dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
      xs[i] = x[r * dim + i] * 2.5F;
      worst = std::max(worst, double(std::fabs(xg[r * dim + i] - xs[i])));
    }
    const auto hn = RmsNorm(xs.data(), w1.data(), dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
      worst = std::max(worst, std::fabs(hg[r * dim + i] - hn[i]));
    }
  }
  Require(worst < 1e-4, "ScaleRmsNorm error " + std::to_string(worst));
  for (void* p : {static_cast<void*>(dx), static_cast<void*>(dox),
                  static_cast<void*>(dw1), static_cast<void*>(dw2),
                  static_cast<void*>(dh)}) {
    HIP_CHECK(hipFree(p));
  }
}

void CheckElementwise() {
  std::vector<float> g, u;
  for (int i = -2000; i <= 2000; ++i) {
    g.push_back(static_cast<float>(i) * 0.01F);
    u.push_back(1.0F - static_cast<float>(i) * 0.0005F);
  }
  float* dg = Device(g);
  float* du = Device(u);
  float* dout = Device(g);
  k::GeGlu(dg, du, dout, g.size(), nullptr);
  auto out = Host(dout, g.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < g.size(); ++i) {
    const double x = g[i];
    const double want =
        0.5 * x *
        (1 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))) * u[i];
    worst = std::max(worst, std::fabs(out[i] - want));
  }
  Require(worst < 2e-6, "GeGLU error " + std::to_string(worst));

  std::vector<float> logits;
  for (int i = -500; i <= 500; ++i)
    logits.push_back(static_cast<float>(i) * 0.3F);
  float* dl = Device(logits);
  k::Softcap(dl, logits.size(), 30.0F, nullptr);
  out = Host(dl, logits.size());
  worst = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    worst =
        std::max(worst, std::fabs(out[i] - 30.0 * std::tanh(logits[i] / 30.0)));
  }
  Require(worst < 1e-5, "softcap error " + std::to_string(worst));
  for (void* p : {static_cast<void*>(dg), static_cast<void*>(du),
                  static_cast<void*>(dout), static_cast<void*>(dl)}) {
    HIP_CHECK(hipFree(p));
  }
}

}  // namespace

/// Prefill producers that write the Q8_1 activation directly match the FP32
/// producer followed by the shared quantizer, including the zeroed padding
/// rows of a partial 16-row tile.
void CheckFusedQuantize(std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0F, 1.0F);
  for (const std::uint32_t rows : {37U, 64U}) {
    constexpr std::uint32_t kDim = 5376;
    constexpr std::uint32_t kFfn = 2048;
    const std::size_t bytes =
        std::max(gufo::hip::QuantizedActivationBytes(rows, kDim),
                 gufo::hip::QuantizedActivationBytes(rows, kFfn));
    const auto random = [&](std::size_t n, float scale) {
      std::vector<float> v(n);
      for (float& x : v)
        x = normal(rng) * scale;
      return v;
    };
    void* want = nullptr;
    void* got = nullptr;
    HIP_CHECK(hipMalloc(&want, bytes));
    HIP_CHECK(hipMalloc(&got, bytes));
    // Dequantized values agree within one quantization step (the fused
    // producers build with fast-math, so a block scale may differ by an ulp);
    // padding rows of the last 16-row tile are exactly zero in both.
    const auto compare = [&](std::uint32_t cols, const std::string& what) {
      HIP_CHECK(hipDeviceSynchronize());
      const std::size_t n = gufo::hip::QuantizedActivationBytes(rows, cols);
      std::vector<std::uint8_t> a(n), b(n);
      HIP_CHECK(hipMemcpy(a.data(), want, n, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(b.data(), got, n, hipMemcpyDeviceToHost));
      constexpr std::size_t kTile = 576;
      const std::size_t blocks = cols / 32;
      const std::size_t tiles = (rows + 15) / 16;
      double worst = 0.0;
      std::size_t differing = 0;
      for (std::size_t tt = 0; tt < tiles; ++tt) {
        for (std::size_t blk = 0; blk < blocks; ++blk) {
          const std::uint8_t* ta = a.data() + (tt * blocks + blk) * kTile;
          const std::uint8_t* tb = b.data() + (tt * blocks + blk) * kTile;
          for (std::size_t tl = 0; tl < 16; ++tl) {
            float da, db;
            std::memcpy(&da, ta + 512 + tl * 4, 4);
            std::memcpy(&db, tb + 512 + tl * 4, 4);
            for (std::size_t lane = 0; lane < 32; ++lane) {
              const std::size_t at = (lane >> 4) * 256 + tl * 16 + (lane & 15);
              const double va = static_cast<std::int8_t>(ta[at]) * double{da};
              const double vb = static_cast<std::int8_t>(tb[at]) * double{db};
              if (tt * 16 + tl >= rows) {
                Require(va == 0.0 && vb == 0.0,
                        what + ": padding row not zero");
              }
              worst = std::max(worst, std::fabs(va - vb) / (da + 1e-30));
              differing += ta[at] != tb[at] || da != db;
            }
          }
        }
      }
      std::cout << what << " at " << rows << " rows: " << differing
                << " values differ from the unfused quantization\n";
      Require(worst <= 1.01, what + ": fused quantization off by " +
                                 std::to_string(worst) + " steps at " +
                                 std::to_string(rows) + " rows");
    };
    // GeGLU.
    float* gate = Device(random(std::size_t{rows} * kFfn, 1.5F));
    float* up = Device(random(std::size_t{rows} * kFfn, 1.0F));
    float* act = Device(std::vector<float>(std::size_t{rows} * kFfn));
    HIP_CHECK(hipMemset(want, 0x5A, bytes));
    HIP_CHECK(hipMemset(got, 0x5A, bytes));
    k::GeGlu(gate, up, act, std::size_t{rows} * kFfn, nullptr);
    gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(act, want, rows, kFfn,
                                                    nullptr);
    k::GeGluQuantize(gate, up, got, rows, kFfn, nullptr);
    compare(kFfn, "GeGluQuantize");
    // Residual norms.
    const auto o = random(std::size_t{rows} * kDim, 3.0F);
    const auto x0 = random(std::size_t{rows} * kDim, 10.0F);
    const auto w1 = random(kDim, 0.3F);
    const auto w2 = random(kDim, 0.3F);
    float* d_o = Device(o);
    float* d_w1 = Device(w1);
    float* d_w2 = Device(w2);
    float* x = Device(x0);
    float* h = Device(std::vector<float>(std::size_t{rows} * kDim));
    for (int kind = 0; kind < 2; ++kind) {
      HIP_CHECK(hipMemcpy(x, x0.data(), x0.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(want, 0x5A, bytes));
      HIP_CHECK(hipMemset(got, 0x5A, bytes));
      if (kind == 0) {
        k::PostAttentionNorm(d_o, d_w1, x, d_w2, h, rows, kDim, 1e-6F, nullptr);
      } else {
        k::PostFeedForwardNorm(d_o, d_w1, 0.7F, x, d_w2, h, rows, kDim, 1e-6F,
                               nullptr);
      }
      gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(h, want, rows, kDim,
                                                      nullptr);
      HIP_CHECK(hipMemcpy(x, x0.data(), x0.size() * 4, hipMemcpyHostToDevice));
      if (kind == 0) {
        k::PostAttentionNorm(d_o, d_w1, x, d_w2, h, rows, kDim, 1e-6F, nullptr,
                             got);
      } else {
        k::PostFeedForwardNorm(d_o, d_w1, 0.7F, x, d_w2, h, rows, kDim, 1e-6F,
                               nullptr, got);
      }
      compare(kDim, kind == 0 ? "PostAttentionNorm" : "PostFeedForwardNorm");
    }
    for (void* p : {static_cast<void*>(gate), static_cast<void*>(up),
                    static_cast<void*>(act), static_cast<void*>(d_o),
                    static_cast<void*>(d_w1), static_cast<void*>(d_w2),
                    static_cast<void*>(x), static_cast<void*>(h), want, got}) {
      HIP_CHECK(hipFree(p));
    }
  }
}

/// LogitCandidates against a CPU ranking in sampling order, then the target
/// distributions FinishCycle builds from them against the full rows'.
void CheckLogitCandidates(std::mt19937& rng) {
  constexpr std::uint32_t kVocab = 262144;
  constexpr std::uint32_t kRows = 16;
  std::normal_distribution<float> normal(0.0F, 3.0F);
  std::vector<float> logits(std::size_t{kRows} * kVocab);
  for (float& v : logits) {
    v = normal(rng);
  }
  const auto row = [&](std::uint32_t r) {
    return std::span<float>(logits).subspan(std::size_t{r} * kVocab, kVocab);
  };
  // Ties across the top-64 boundary, signed zeros and non-finite logits.
  {
    auto x = row(1);
    std::vector<std::uint32_t> order(kVocab);
    for (std::uint32_t i = 0; i < kVocab; ++i) {
      order[i] = i;
    }
    std::ranges::partial_sort(order, order.begin() + 80,
                              [&](auto a, auto b) { return x[a] > x[b]; });
    for (std::uint32_t i = 50; i < 80; ++i) {
      x[order[i]] = x[order[63]];
    }
    x[3] = std::numeric_limits<float>::infinity();
    x[4] = std::numeric_limits<float>::quiet_NaN();
    x[5] = -std::numeric_limits<float>::infinity();
  }
  {
    // 300 signed zeros tie at the boundary below 40 positive logits.
    auto x = row(2);
    for (float& v : x) {
      v = -20.0F - std::abs(v);
    }
    for (std::uint32_t i = 0; i < 300; ++i) {
      x[i * 811 + 1] = i % 2 == 0 ? 0.0F : -0.0F;
    }
    for (std::uint32_t i = 0; i < 40; ++i) {
      x[i * 4099] = 1.0F + static_cast<float>(i);
    }
  }
  // Fewer finite logits than candidates, and a threshold bin too full.
  std::ranges::fill(row(3), std::numeric_limits<float>::quiet_NaN());
  for (std::uint32_t i = 0; i < 10; ++i) {
    row(3)[i * 1000 + 7] = static_cast<float>(i);
  }
  std::ranges::fill(row(4), 2.5F);
  const std::vector<std::uint32_t> extra = {0, 7, 1000, 4099, 131072, 262143};
  const auto n_extra = static_cast<std::uint32_t>(extra.size());
  const std::uint32_t top = k::kCandidateTop + n_extra;
  const std::uint32_t stride = top + n_extra;

  float* d_logits = Device(logits);
  std::uint32_t* d_extra = Device(extra);
  void* scratch = nullptr;
  HIP_CHECK(hipMalloc(&scratch, k::CandidateWorkBytes(kRows)));
  std::uint32_t* d_counts = nullptr;
  HIP_CHECK(hipMalloc(&d_counts, kRows * sizeof(std::uint32_t)));
  gufo::sampling::LogitCandidate* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, std::size_t{kRows} * stride * sizeof(*d_out)));
  const auto work = k::CandidateWorkAt(scratch, kRows, nullptr);
  std::vector<std::uint32_t> counts;
  std::vector<gufo::sampling::LogitCandidate> out;
  // Twice: the first launch must leave its workspace zeroed.
  for (int pass = 0; pass < 2; ++pass) {
    k::LogitCandidates(d_logits, kRows, kVocab, top, d_extra, n_extra, work,
                       d_counts, d_out, nullptr);
    counts = Host(d_counts, kRows);
    out = Host(d_out, std::size_t{kRows} * stride);
  }
  for (std::uint32_t r = 0; r < kRows; ++r) {
    const auto x = row(r);
    const std::string name = "candidates row " + std::to_string(r);
    if (r == 4) {
      Require(counts[r] == k::kCandidateOverflow, name + " did not overflow");
      continue;
    }
    std::vector<std::uint32_t> finite;
    for (std::uint32_t i = 0; i < kVocab; ++i) {
      if (std::isfinite(x[i])) {
        finite.push_back(i);
      }
    }
    const std::size_t want_count = std::min<std::size_t>(top, finite.size());
    std::ranges::partial_sort(
        finite, finite.begin() + want_count,
        [&](auto a, auto b) { return x[a] == x[b] ? a < b : x[a] > x[b]; });
    std::vector<std::uint32_t> want(finite.begin(),
                                    finite.begin() + want_count);
    Require(counts[r] == want_count, name + " count");
    std::vector<std::uint32_t> got;
    for (std::uint32_t i = 0; i < counts[r]; ++i) {
      const auto& c = out[std::size_t{r} * stride + i];
      Require(std::memcmp(&c.value, &x[c.token], sizeof(float)) == 0,
              name + " value");
      got.push_back(c.token);
    }
    std::ranges::sort(want);
    std::ranges::sort(got);
    Require(got == want, name + " top set");
    for (std::size_t i = 0; i < extra.size(); ++i) {
      const auto& c = out[std::size_t{r} * stride + top + i];
      Require(c.token == extra[i] &&
                  std::memcmp(&c.value, &x[extra[i]], sizeof(float)) == 0,
              name + " extra");
    }
  }

  // Penalized tokens are in `extra`, as FinishCycle arranges.
  using gufo::sampling::SamplerState;
  using gufo::sampling::SamplingConfig;
  std::vector<SamplingConfig> configs(4);
  configs[0].temperature = 1.0F;
  configs[0].top_k = 64;
  configs[0].top_p = 0.95F;
  configs[1].temperature = 0.7F;
  configs[1].top_k = 40;
  configs[1].min_p = 0.05F;
  configs[2] = configs[0];
  configs[2].repeat_penalty = 1.3F;
  configs[2].presence_penalty = 0.5F;
  configs[3] = configs[2];
  configs[3].top_k = 5;
  configs[3].min_keep = 2;
  for (std::size_t n = 0; n < configs.size(); ++n) {
    SamplerState sampler(configs[n]);
    if (configs[n].penalties_enabled()) {
      for (const std::uint32_t token : extra) {
        sampler.Accept(static_cast<gufo::sampling::TokenId>(token));
      }
    }
    for (std::uint32_t r = 0; r < kRows; ++r) {
      if (r == 3 || r == 4) {
        continue;
      }
      std::vector<gufo::sampling::LogitCandidate> candidates;
      for (std::uint32_t i = 0; i < counts[r]; ++i) {
        const auto& c = out[std::size_t{r} * stride + i];
        if (!std::ranges::binary_search(extra, c.token)) {
          candidates.push_back(c);
        }
      }
      for (std::size_t i = 0; i < extra.size(); ++i) {
        candidates.push_back(out[std::size_t{r} * stride + top + i]);
      }
      const auto full = sampler.Distribution(row(r));
      const auto compact = sampler.DistributionFromCandidates(candidates);
      const auto a = full.entries();
      const auto b = compact.entries();
      bool same = a.size() == b.size();
      for (std::size_t i = 0; same && i < a.size(); ++i) {
        same = a[i].token == b[i].token && a[i].value == b[i].value;
      }
      Require(same, "candidate distribution config " + std::to_string(n) +
                        " row " + std::to_string(r));
    }
  }
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_counts));
  HIP_CHECK(hipFree(scratch));
  HIP_CHECK(hipFree(d_extra));
  HIP_CHECK(hipFree(d_logits));
}

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "SKIP: no HIP device\n";
    return 77;
  }
  return gemma4_test::Run([] {
    std::mt19937 rng(7);
    CheckQkvPost(256, false, rng);
    CheckQkvPost(512, true, rng);
    CheckNorms(rng);
    CheckElementwise();
    CheckFusedQuantize(rng);
    CheckLogitCandidates(rng);
  });
}
