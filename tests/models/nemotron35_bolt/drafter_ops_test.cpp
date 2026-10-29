// Checks the LiLiCorr drafter's HIP kernels against direct double CPU
// evaluations of llama.cpp's formulas (src/models/dflash.cpp): the BF16
// weight-streaming product, the block-local dynamic convolution, NeoX RoPE,
// ring attention with sinks and a sliding window across the ring's wrap, the
// candidate top-8 with log-sum-exp, the candidate features, node attention
// and the lattice scores.

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "src/models/nemotron35_bolt/kernels/rocm/drafter_kernels.hpp"

namespace k = gufo::models::nemotron35_bolt::rocm;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++failures;
  }
}

void Hip(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP error in %s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

template<class T>
T* Device(const std::vector<T>& host) {
  T* d = nullptr;
  Hip(hipMalloc(&d, std::max<std::size_t>(1, host.size()) * sizeof(T)),
      "malloc");
  if (!host.empty())
    Hip(hipMemcpy(d, host.data(), host.size() * sizeof(T),
                  hipMemcpyHostToDevice),
        "upload");
  return d;
}

template<class T>
std::vector<T> Host(const T* d, std::size_t n) {
  std::vector<T> out(n);
  Hip(hipMemcpy(out.data(), d, n * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return out;
}

void Free(std::initializer_list<void*> pointers) {
  for (void* p : pointers)
    Hip(hipFree(p), "free");
}

std::vector<float> Random(std::size_t n, std::mt19937& rng, float lo = -1.0F,
                          float hi = 1.0F) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (float& x : v)
    x = dist(rng);
  return v;
}

/// Max |a - b| relative to max |b|.
double RelErr(const std::vector<float>& a, const std::vector<float>& b) {
  double err = 0.0;
  double scale = 1e-12;
  for (std::size_t i = 0; i < a.size(); ++i) {
    err = std::max(err, std::abs(static_cast<double>(a[i]) - b[i]));
    scale = std::max(scale, std::abs(static_cast<double>(b[i])));
  }
  return err / scale;
}

std::uint16_t ToBf16Bits(float x) {
  std::uint32_t u = 0;
  std::memcpy(&u, &x, 4);
  u += 0x7FFFU + ((u >> 16) & 1U);  // round to nearest even
  return static_cast<std::uint16_t>(u >> 16);
}

float FromBf16Bits(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float x = 0;
  std::memcpy(&x, &u, 4);
  return x;
}

float HalfRound(float x) {
  return __half2float(__float2half(x));
}

void TestGemvBf16(std::mt19937& rng) {
  constexpr int kCols = 4096;
  for (int rows : {256, 1024, 4096}) {
    for (int tokens : {1, 3, 8}) {
      const auto wf = Random(static_cast<std::size_t>(rows) * kCols, rng);
      std::vector<std::uint16_t> w(wf.size());
      for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = ToBf16Bits(wf[i]);
      const auto x = Random(static_cast<std::size_t>(tokens) * kCols, rng);
      std::vector<float> want(static_cast<std::size_t>(tokens) * rows);
      for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < rows; ++r) {
          double acc = 0;
          for (int c = 0; c < kCols; ++c)
            acc += double(FromBf16Bits(w[std::size_t(r) * kCols + c])) *
                   x[std::size_t(t) * kCols + c];
          want[std::size_t(t) * rows + r] = float(acc);
        }
      auto* d_w = Device(w);
      auto* d_x = Device(x);
      auto* d_out = Device(std::vector<float>(want.size()));
      Hip(k::GemvBf16(reinterpret_cast<const __hip_bfloat16*>(d_w), d_x, d_out,
                      rows, kCols, tokens, nullptr),
          "bf16 gemv");
      Check(RelErr(Host(d_out, want.size()), want) < 1e-5,
            "bf16 gemv " + std::to_string(rows) + "x" + std::to_string(tokens));
      Free({d_w, d_x, d_out});
    }
  }
}

void TestConv(std::mt19937& rng) {
  constexpr int kTokens = 16;  // two blocks
  constexpr int kBlock = 8;
  constexpr int kHidden = 64;
  constexpr int kGroup = 16;
  constexpr int kKernel = 2;
  constexpr int kGroups = kHidden / kGroup;
  const auto x = Random(kTokens * kHidden, rng);
  const auto dyn = Random(kTokens * 2 * kKernel * kGroups, rng);
  const auto base = Random(2 * kKernel * kHidden, rng);
  auto* d_x = Device(x);
  auto* d_dyn = Device(dyn);
  auto* d_base = Device(base);
  auto* d_out = Device(std::vector<float>(x.size()));
  for (int side = 0; side < 2; ++side) {
    std::vector<float> want(x.size());
    for (int t = 0; t < kTokens; ++t)
      for (int c = 0; c < kHidden; ++c) {
        double acc = 0;
        for (int tap = 0; tap < kKernel && tap <= t % kBlock; ++tap) {
          const int k = side * kKernel + tap;
          const double w =
              double(
                  dyn[t * 2 * kKernel * kGroups + k * kGroups + c / kGroup]) +
              base[k * kHidden + c];
          acc += w * x[(t - tap) * kHidden + c];
        }
        want[t * kHidden + c] = float(acc);
      }
    Hip(k::DflashConv(d_x, d_dyn, d_base, d_out, kTokens, kBlock, kHidden,
                      kGroup, kKernel, side, nullptr),
        "dflash conv");
    Check(RelErr(Host(d_out, want.size()), want) < 1e-6,
          "dflash conv side " + std::to_string(side));
  }
  Free({d_x, d_dyn, d_base, d_out});
}

/// NeoX rotation of rows of `dim` at position `pos`.
void RopeRef(float* row, int dim, double pos, double theta) {
  const int half = dim / 2;
  for (int i = 0; i < half; ++i) {
    const double angle = pos * std::pow(theta, -2.0 * i / dim);
    const double a = row[i];
    const double b = row[i + half];
    row[i] = float(a * std::cos(angle) - b * std::sin(angle));
    row[i + half] = float(a * std::sin(angle) + b * std::cos(angle));
  }
}

void TestRope(std::mt19937& rng) {
  constexpr int kTokens = 3;
  constexpr int kHeads = 4;
  constexpr int kDim = 128;
  constexpr int kFirst = 1000;
  auto x = Random(kTokens * kHeads * kDim, rng);
  auto want = x;
  for (int t = 0; t < kTokens; ++t)
    for (int h = 0; h < kHeads; ++h)
      RopeRef(&want[(t * kHeads + h) * kDim], kDim, kFirst + t, 10000.0);
  auto* d_x = Device(x);
  auto* d_first = Device(std::vector<int>{kFirst});
  Hip(k::RopeNeox(d_x, d_first, kTokens, kHeads, kDim, 10000.0F, nullptr),
      "rope");
  // The device evaluates the angle in float, as ggml does.
  Check(RelErr(Host(d_x, x.size()), want) < 5e-4, "neox rope");
  Free({d_x, d_first});
}

void TestAttention(std::mt19937& rng) {
  constexpr int kHeads = 32;
  constexpr int kKvHeads = 2;
  constexpr int kDim = 128;
  constexpr int kKv = kKvHeads * kDim;
  constexpr int kRing = 64;
  constexpr int kWindow = 40;
  constexpr int kTokens = 8;
  constexpr int kFirst = 100;  // positions 44..107 occupy the ring, wrapped
  const float scale = 1.0F / std::sqrt(float(kDim));
  // Context rows for positions kFirst - kRing + kTokens .. kFirst - 1.
  constexpr int kContext = kRing - kTokens;
  const auto ck = Random(kContext * kKv, rng);
  const auto cv = Random(kContext * kKv, rng);
  const auto bk = Random(kTokens * kKv, rng);
  const auto bv = Random(kTokens * kKv, rng);
  const auto q = Random(kTokens * kHeads * kDim, rng);
  const auto sinks = Random(kHeads, rng, -2.0F, 2.0F);
  auto* d_kr = Device(std::vector<__half>(kRing * kKv));
  auto* d_vr = Device(std::vector<__half>(kRing * kKv));
  auto* d_ck = Device(ck);
  auto* d_cv = Device(cv);
  auto* d_bk = Device(bk);
  auto* d_bv = Device(bv);
  auto* d_q = Device(q);
  auto* d_sinks = Device(sinks);
  auto* d_out = Device(std::vector<float>(q.size()));
  auto* d_ctx_first = Device(std::vector<int>{kFirst - kContext});
  auto* d_first = Device(std::vector<int>{kFirst});
  Hip(k::RingAppend(d_ck, d_cv, d_kr, d_vr, d_ctx_first, kContext, kKv, kRing,
                    nullptr),
      "ring context");
  Hip(k::RingAppend(d_bk, d_bv, d_kr, d_vr, d_first, kTokens, kKv, kRing,
                    nullptr),
      "ring block");
  Hip(k::DrafterAttention(d_q, d_kr, d_vr, d_sinks, d_out, d_first, kTokens,
                          kHeads, kKvHeads, kRing, kWindow, scale, nullptr),
      "drafter attention");
  auto key = [&](int pos, int kvh, int d, bool value) {
    const int j = pos - (kFirst - kContext);
    const auto& src = j < kContext ? (value ? cv : ck) : (value ? bv : bk);
    const int row = j < kContext ? j : j - kContext;
    return HalfRound(src[row * kKv + kvh * kDim + d]);
  };
  std::vector<float> want(q.size());
  for (int t = 0; t < kTokens; ++t) {
    const int pos = kFirst + t;
    const int lo = std::max(0, pos - kWindow + 1);
    const int hi = kFirst + kTokens - 1;
    for (int h = 0; h < kHeads; ++h) {
      const int kvh = h / (kHeads / kKvHeads);
      std::vector<double> s;
      double m = sinks[h];
      for (int p = lo; p <= hi; ++p) {
        double dot = 0;
        for (int d = 0; d < kDim; ++d)
          dot += double(q[(t * kHeads + h) * kDim + d]) * key(p, kvh, d, false);
        s.push_back(dot * scale);
        m = std::max(m, s.back());
      }
      double z = std::exp(sinks[h] - m);
      for (double v : s)
        z += std::exp(v - m);
      for (int d = 0; d < kDim; ++d) {
        double acc = 0;
        for (int p = lo; p <= hi; ++p)
          acc += std::exp(s[p - lo] - m) * key(p, kvh, d, true);
        want[(t * kHeads + h) * kDim + d] = float(acc / z);
      }
    }
  }
  Check(RelErr(Host(d_out, want.size()), want) < 1e-5,
        "drafter attention with sinks over a wrapped ring");
  Free({d_kr, d_vr, d_ck, d_cv, d_bk, d_bv, d_q, d_sinks, d_out, d_ctx_first,
        d_first});
}

void TestTopK(std::mt19937& rng) {
  constexpr int kRows = 3;
  constexpr int kVocab = 5000;
  auto logits = Random(kRows * kVocab, rng, -8.0F, 8.0F);
  // Ties in the top 8 resolve to the lower token id.
  logits[1 * kVocab + 4321] = 20.0F;
  logits[1 * kVocab + 1234] = 20.0F;
  auto* d_logits = Device(logits);
  auto* d_ids = Device(std::vector<std::int32_t>(kRows * 8));
  auto* d_values = Device(std::vector<float>(kRows * 8));
  auto* d_lse = Device(std::vector<float>(kRows));
  Hip(k::TopK8(d_logits, kRows, kVocab, d_ids, d_values, d_lse, nullptr),
      "top-8");
  const auto ids = Host(d_ids, kRows * 8);
  const auto values = Host(d_values, kRows * 8);
  const auto lse = Host(d_lse, kRows);
  bool ids_ok = true;
  bool values_ok = true;
  std::vector<float> want_lse(kRows);
  for (int r = 0; r < kRows; ++r) {
    std::vector<int> order(kVocab);
    std::iota(order.begin(), order.end(), 0);
    const float* row = &logits[r * kVocab];
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return row[a] > row[b]; });
    for (int j = 0; j < 8; ++j) {
      ids_ok = ids_ok && ids[r * 8 + j] == order[j];
      values_ok = values_ok && values[r * 8 + j] == row[order[j]];
    }
    double m = *std::max_element(row, row + kVocab);
    double z = 0;
    for (int i = 0; i < kVocab; ++i)
      z += std::exp(row[i] - m);
    want_lse[r] = float(m + std::log(z));
  }
  Check(ids_ok && values_ok, "top-8 candidates in descending order");
  Check(RelErr(lse, want_lse) < 1e-6, "top-8 log-sum-exp");
  Free({d_logits, d_ids, d_values, d_lse});
}

void TestLiliCorr(std::mt19937& rng) {
  constexpr int kSlots = 7;
  constexpr int kTop = 8;
  constexpr int kNodes = kSlots * kTop;
  constexpr int kWidth = 1024;
  constexpr int kHeads = 8;
  constexpr int kDim = 128;
  // Features.
  std::vector<float> values(kNodes);
  std::vector<float> lse(kSlots);
  for (int s = 0; s < kSlots; ++s) {
    auto v = Random(kTop, rng, -6.0F, 6.0F);
    std::sort(v.begin(), v.end(), std::greater<>());
    std::copy(v.begin(), v.end(), values.begin() + s * kTop);
    lse[s] = v[0] + 0.7F;
  }
  const auto fw = Random(5, rng);
  const auto fb = Random(5, rng);
  std::vector<float> want(kNodes * 5);
  for (int n = 0; n < kNodes; ++n) {
    const int s = n / kTop;
    const int r = n % kTop;
    const double lp = double(values[n]) - lse[s];
    const double lp0 = double(values[s * kTop]) - lse[s];
    const double f[5] = {lp, std::exp(lp), lp - lp0, r / 7.0,
                         r == 0 ? 1.0 : 0.0};
    double mean = 0;
    for (double v : f)
      mean += v / 5;
    double var = 0;
    for (double v : f)
      var += (v - mean) * (v - mean) / 5;
    for (int i = 0; i < 5; ++i)
      want[n * 5 + i] =
          float((f[i] - mean) / std::sqrt(var + 1e-5) * fw[i] + fb[i]);
  }
  auto* d_values = Device(values);
  auto* d_lse = Device(lse);
  auto* d_fw = Device(fw);
  auto* d_fb = Device(fb);
  auto* d_feat = Device(std::vector<float>(want.size()));
  Hip(k::LiliCorrFeatures(d_values, d_lse, d_fw, d_fb, d_feat, kSlots, nullptr),
      "features");
  Check(RelErr(Host(d_feat, want.size()), want) < 1e-5,
        "lilicorr candidate features");
  Free({d_values, d_lse, d_fw, d_fb, d_feat});

  // Node attention with a per-head additive bias.
  const auto qkv = Random(kNodes * 3 * kWidth, rng);
  const auto bias = Random(kHeads * kNodes * kNodes, rng, -3.0F, 3.0F);
  const float scale = 1.0F / std::sqrt(float(kDim));
  std::vector<float> att(kNodes * kWidth);
  for (int h = 0; h < kHeads; ++h)
    for (int qi = 0; qi < kNodes; ++qi) {
      std::vector<double> p(kNodes);
      double m = -1e300;
      for (int j = 0; j < kNodes; ++j) {
        double dot = 0;
        for (int d = 0; d < kDim; ++d)
          dot += double(qkv[qi * 3 * kWidth + h * kDim + d]) *
                 qkv[j * 3 * kWidth + kWidth + h * kDim + d];
        p[j] = dot * scale + bias[(h * kNodes + qi) * kNodes + j];
        m = std::max(m, p[j]);
      }
      double z = 0;
      for (double& v : p) {
        v = std::exp(v - m);
        z += v;
      }
      for (int d = 0; d < kDim; ++d) {
        double acc = 0;
        for (int j = 0; j < kNodes; ++j)
          acc += p[j] * qkv[j * 3 * kWidth + 2 * kWidth + h * kDim + d];
        att[qi * kWidth + h * kDim + d] = float(acc / z);
      }
    }
  auto* d_qkv = Device(qkv);
  auto* d_bias = Device(bias);
  auto* d_att = Device(std::vector<float>(att.size()));
  Hip(k::NodeAttention(d_qkv, d_bias, d_att, kNodes, kHeads, scale, nullptr),
      "node attention");
  Check(RelErr(Host(d_att, att.size()), att) < 1e-5, "lilicorr node attention");
  Free({d_qkv, d_bias, d_att});

  // Normalized factors and lattice scores.
  auto in = Random(kNodes * kWidth, rng);
  auto out = Random(kNodes * kWidth, rng);
  auto anchor = Random(kWidth, rng);
  const auto in_b = Random(kWidth, rng);
  const auto out_b = Random(kWidth, rng);
  const auto anchor_b = Random(kWidth, rng);
  auto normalize = [&](std::vector<float>& x, const std::vector<float>& b,
                       int rows) {
    for (int r = 0; r < rows; ++r) {
      double ss = 0;
      for (int c = 0; c < kWidth; ++c) {
        x[r * kWidth + c] += b[c];
        ss += double(x[r * kWidth + c]) * x[r * kWidth + c];
      }
      const double inv = 1.0 / std::max(std::sqrt(ss), 1e-4);
      for (int c = 0; c < kWidth; ++c)
        x[r * kWidth + c] = float(x[r * kWidth + c] * inv);
    }
  };
  auto* d_in = Device(in);
  auto* d_out = Device(out);
  auto* d_anchor = Device(anchor);
  auto* d_in_b = Device(in_b);
  auto* d_out_b = Device(out_b);
  auto* d_anchor_b = Device(anchor_b);
  normalize(in, in_b, kNodes);
  normalize(out, out_b, kNodes);
  normalize(anchor, anchor_b, 1);
  Hip(k::BiasL2Norm(d_in, d_in_b, kNodes, kWidth, 1e-4F, nullptr), "l2");
  Hip(k::BiasL2Norm(d_out, d_out_b, kNodes, kWidth, 1e-4F, nullptr), "l2");
  Hip(k::BiasL2Norm(d_anchor, d_anchor_b, 1, kWidth, 1e-4F, nullptr), "l2");
  Check(RelErr(Host(d_in, in.size()), in) < 1e-5, "lilicorr l2 norm");
  std::vector<float> scores(kTop + (kSlots - 1) * kTop * kTop);
  auto dot = [&](const float* a, const float* b) {
    double acc = 0;
    for (int c = 0; c < kWidth; ++c)
      acc += double(a[c]) * b[c];
    return acc;
  };
  for (int j = 0; j < kTop; ++j)
    scores[j] = float(8.0 * dot(&in[j * kWidth], anchor.data()));
  for (int s = 1; s < kSlots; ++s)
    for (int prev = 0; prev < kTop; ++prev)
      for (int next = 0; next < kTop; ++next)
        scores[kTop + (s - 1) * kTop * kTop + prev * kTop + next] =
            float(8.0 * dot(&in[(s * kTop + next) * kWidth],
                            &out[((s - 1) * kTop + prev) * kWidth]));
  auto* d_scores = Device(std::vector<float>(scores.size()));
  Hip(k::LatticeScores(d_in, d_out, d_anchor, d_scores, kSlots, kWidth, 8.0F,
                       nullptr),
      "lattice");
  Check(RelErr(Host(d_scores, scores.size()), scores) < 1e-5,
        "lilicorr lattice scores");
  Free({d_in, d_out, d_anchor, d_in_b, d_out_b, d_anchor_b, d_scores});
}

}  // namespace

int main() {
  std::mt19937 rng(4321);
  TestGemvBf16(rng);
  TestConv(rng);
  TestRope(rng);
  TestAttention(rng);
  TestTopK(rng);
  TestLiliCorr(rng);
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all drafter kernel checks passed\n");
  return 0;
}
