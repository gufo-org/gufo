// CPU checks of gufo::npu::host: vectorized paths against scalar formulas,
// attention against a naive reference, and many back-to-back ParallelFor
// rounds (a worker late for one round must never count toward the next).

#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "src/core/npu/host_ops.hpp"

namespace host = gufo::npu::host;

namespace {

int failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

double MaxRel(const std::vector<float>& a, const std::vector<double>& b) {
  double worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    worst = std::max(worst, std::fabs(a[i] - b[i]) / (std::fabs(b[i]) + 1e-6));
  return worst;
}

}  // namespace

int main() {
  std::mt19937 rng(11);
  std::normal_distribution<float> normal(0.0f, 2.0f);

  // ParallelFor: every index exactly once, over many rounds and two callers.
  {
    std::atomic<bool> ok{true};
    auto rounds = [&] {
      for (int round = 0; round < 2000; ++round) {
        const std::size_t n = 1 + std::size_t(round % 97);
        std::vector<std::atomic<int>> hits(n);
        host::ParallelFor(n, 1, [&](std::size_t b, std::size_t e) {
          for (std::size_t i = b; i < e; ++i)
            hits[i].fetch_add(1);
        });
        for (auto& h : hits)
          if (h.load() != 1)
            ok = false;
      }
    };
    std::thread other(rounds);
    rounds();
    other.join();
    Check(ok, "ParallelFor covers each index once");
  }

  // Exponential and activations: vector lanes against std:: formulas.
  {
    std::vector<float> x(1003);
    for (float& v : x)
      v = normal(rng) * 4;
    std::vector<double> ref(x.size());
    double sum_ref = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
      sum_ref += ref[i] = std::exp(double(x[i]) - 1.5);
    std::vector<float> e = x;
    float sum = 0;
    host::ExpInPlace(e.data(), e.size(), 1.5f, &sum);
    Check(MaxRel(e, ref) < 1e-5, "ExpInPlace");
    Check(std::fabs(sum - sum_ref) / sum_ref < 1e-5, "ExpInPlace sum");
    for (auto act : {host::Activation::kSilu, host::Activation::kGeluTanh,
                     host::Activation::kGeluErf}) {
      std::vector<float> y = x;
      host::ActivateRows(y.data(), 1, y.size(), nullptr, act);
      std::vector<double> want(x.size());
      for (std::size_t i = 0; i < x.size(); ++i)
        want[i] = host::Activate(act, x[i]);
      double worst = 0;
      for (std::size_t i = 0; i < x.size(); ++i)
        worst = std::max(worst, std::fabs(y[i] - want[i]));
      Check(worst < 1e-5, "ActivateRows");
    }
  }

  // bf16 rounding: vector and scalar agree, ties round to even.
  {
    std::vector<float> x(77);
    for (float& v : x)
      v = normal(rng);
    x[0] = std::bit_cast<float>(0x3f808000u);  // tie: rounds down to even
    x[1] = std::bit_cast<float>(0x3f818000u);  // tie: rounds up to even
    std::vector<std::uint16_t> v(x.size());
    host::FloatToBf16(x.data(), v.data(), x.size());
    bool same = true;
    for (std::size_t i = 0; i < x.size(); ++i)
      same &= v[i] == host::FloatToBf16(x[i]);
    Check(same, "FloatToBf16 vector == scalar");
    Check(v[0] == 0x3f80 && v[1] == 0x3f82, "FloatToBf16 ties to even");
    std::vector<float> back(x.size());
    host::Bf16ToFloat(v.data(), back.data(), v.size());
    Check(back[0] == 1.0f, "Bf16ToFloat");
  }

  // bfp16 blocks: the strided batch encoder matches the single-block one.
  {
    std::vector<float> x(8 * 40);
    for (float& v : x)
      v = normal(rng) * (normal(rng) > 0 ? 1e3f : 1e-3f);
    std::fill(x.begin(), x.begin() + 8, 0.0f);  // an all-zero block
    std::vector<std::uint8_t> one(9 * 40), many(9 * 40);
    for (int i = 0; i < 40; ++i)
      host::EncodeBfp16Block(x.data() + 8 * i, one.data() + 9 * i);
    host::EncodeBfp16Blocks(x.data(), 40, many.data(), 9);
    Check(one == many, "EncodeBfp16Blocks == EncodeBfp16Block");
  }

  // Attention: grouped heads and a sliding window against a naive loop.
  {
    const std::size_t rows = 37, heads = 4, kv = 2, dim = 24, window = 5;
    std::vector<float> q(rows * heads * dim), k(rows * kv * dim),
        v(rows * kv * dim), out(rows * heads * dim);
    for (auto* t : {&q, &k, &v})
      for (float& x : *t)
        x = normal(rng) * 0.3f;
    host::Attention({.q = q.data(),
                     .q_stride = heads * dim,
                     .k = k.data(),
                     .k_stride = kv * dim,
                     .v = v.data(),
                     .v_stride = kv * dim,
                     .out = out.data(),
                     .out_stride = heads * dim,
                     .rows = rows,
                     .heads = heads,
                     .kv_heads = kv,
                     .dim = dim,
                     .dim_v = dim,
                     .scale = 0.5f},
                    [&](std::size_t r, std::size_t* b, std::size_t* e) {
                      *b = r > window ? r - window : 0;
                      *e = std::min(rows, r + window + 1);
                    });
    std::vector<double> ref(out.size());
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t kh = h / (heads / kv);
        const std::size_t b = r > window ? r - window : 0,
                          e = std::min(rows, r + window + 1);
        std::vector<double> s(e - b);
        double top = -1e300, total = 0;
        for (std::size_t t = b; t < e; ++t) {
          double d = 0;
          for (std::size_t i = 0; i < dim; ++i)
            d += double(q[(r * heads + h) * dim + i]) *
                 k[(t * kv + kh) * dim + i];
          s[t - b] = d * 0.5;
          top = std::max(top, s[t - b]);
        }
        for (double& x : s)
          total += x = std::exp(x - top);
        for (std::size_t i = 0; i < dim; ++i) {
          double acc = 0;
          for (std::size_t t = b; t < e; ++t)
            acc += s[t - b] / total * v[(t * kv + kh) * dim + i];
          ref[(r * heads + h) * dim + i] = acc;
        }
      }
    double worst = 0;
    for (std::size_t i = 0; i < out.size(); ++i)
      worst = std::max(worst, std::fabs(out[i] - ref[i]));
    Check(worst < 1e-5, "Attention");
  }

  // RMS norm and rotary embedding with a row stride.
  {
    const std::size_t rows = 5, heads = 3, dim = 8, stride = 40;
    std::vector<float> x(rows * stride);
    for (float& v : x)
      v = normal(rng);
    std::vector<float> y = x;
    std::vector<std::int32_t> pos{0, 1, 2, 7, 100};
    host::RopeNeox(y.data(), rows, stride, heads, dim, dim, pos.data(), 1e4f);
    bool ok = true;
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t h = 0; h < heads; ++h)
        for (std::size_t i = 0; i < dim / 2; ++i) {
          const double a = pos[r] * std::pow(1e4, -double(2 * i) / double(dim));
          const float* p = &x[r * stride + h * dim];
          const double x0 = p[i], x1 = p[i + dim / 2];
          const float* o = &y[r * stride + h * dim];
          ok &= std::fabs(o[i] - (x0 * std::cos(a) - x1 * std::sin(a))) < 1e-4;
          ok &= std::fabs(o[i + dim / 2] -
                          (x1 * std::cos(a) + x0 * std::sin(a))) < 1e-4;
        }
    for (std::size_t i = heads * dim; i < stride; ++i)
      ok &= y[i] == x[i];  // past the heads: untouched
    Check(ok, "RopeNeox");
    std::vector<float> w(stride, 2.0f), n(rows * stride);
    host::RmsNormRows(x.data(), n.data(), rows, stride, w.data(), 1e-6f);
    double ss = 0;
    for (std::size_t i = 0; i < stride; ++i)
      ss += double(x[i]) * x[i];
    Check(std::fabs(n[3] - x[3] / std::sqrt(ss / stride + 1e-6) * 2) < 1e-5,
          "RmsNormRows");
  }

  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
