// On-device check of gufo::npu::Qwen3Decoder: random Qwen3 weights, a KV
// cache written as prefill would, then decode tokens compared with a double
// reference on the same Q8_0 weights; prints the time per token. Skips (77)
// without an NPU. Usage: npu_qwen3dec_device_test <qwen3dec dir> [tokens]

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "src/core/npu/qwen3_decoder.hpp"

namespace npu = gufo::npu;
using D = npu::Qwen3Decoder;

namespace {

constexpr int kQ = 2048, kKv = 1024;

std::uint16_t Bf16(float f) {
  std::uint32_t u = std::bit_cast<std::uint32_t>(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return std::uint16_t(u >> 16);
}
float F(std::uint16_t h) {
  return std::bit_cast<float>(std::uint32_t(h) << 16);
}

// The packer's Q8_0, dequantized (the reference sees the NPU's weights).
std::vector<float> Dequant(const std::vector<std::uint16_t>& w) {
  std::vector<float> out(w.size());
  for (std::size_t b = 0; b < w.size(); b += 32) {
    float amax = 0;
    for (int i = 0; i < 32; ++i)
      amax = std::max(amax, std::fabs(F(w[b + i])));
    const float d = F(Bf16(amax / 127.0f)), inv = d > 0 ? 1.0f / d : 0.0f;
    for (int i = 0; i < 32; ++i)
      out[b + i] =
          std::clamp(std::nearbyint(F(w[b + i]) * inv), -127.0f, 127.0f) * d;
  }
  return out;
}

struct Mat {
  int n, k;
  std::vector<std::uint16_t> bf;
  std::vector<float> q;
};

Mat Random(std::mt19937& rng, int n, int k, float scale) {
  std::normal_distribution<float> nd(0.0f, scale);
  Mat m{n, k, std::vector<std::uint16_t>(std::size_t(n) * k), {}};
  for (auto& v : m.bf)
    v = Bf16(nd(rng));
  m.q = Dequant(m.bf);
  return m;
}

std::vector<double> MatVec(const Mat& m, const std::vector<double>& x) {
  std::vector<double> y(std::size_t(m.n), 0.0);
  for (int o = 0; o < m.n; ++o) {
    double a = 0;
    const float* w = m.q.data() + std::size_t(o) * m.k;
    for (int i = 0; i < m.k; ++i)
      a += w[i] * x[std::size_t(i)];
    y[std::size_t(o)] = a;
  }
  return y;
}

void Rms(std::vector<double>& x, std::size_t off, int n, const float* w) {
  double ss = 0;
  for (int i = 0; i < n; ++i)
    ss += x[off + i] * x[off + i];
  const double s = 1.0 / std::sqrt(ss / n + 1e-6);
  for (int i = 0; i < n; ++i)
    x[off + i] *= s * w[i];
}

void Rope(std::vector<double>& x, std::size_t off, int pos) {
  for (int i = 0; i < 64; ++i) {
    const double a = pos * std::pow(1e6, -double(2 * i) / 128.0);
    const double x0 = x[off + i], x1 = x[off + i + 64];
    x[off + i] = x0 * std::cos(a) - x1 * std::sin(a);
    x[off + i + 64] = x1 * std::cos(a) + x0 * std::sin(a);
  }
}

double RelErr(const std::vector<float>& a, const std::vector<double>& b) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    num += (a[i] - b[i]) * (a[i] - b[i]);
    den += b[i] * b[i];
  }
  return std::sqrt(num / den);
}

// Random weights for a stack of `layers` x hidden H (FFN I) with `heads`
// heads of `head_rows`; checks `steps` tokens after P cached positions against
// a double reference (with the same Q8_0 weights), then times `timed` tokens.
int CheckStack(D& dec, int H, int I, int layers, int head_rows, int heads,
               int P, int timed, std::mt19937& rng) {
  dec.SetRopeTheta(1e6);
  struct L {
    Mat q, k, v, o, gate, up, down;
    std::vector<float> n1, n2, qn, kn;
  };
  std::vector<L> ws;
  std::uniform_real_distribution<float> norm(0.8f, 1.2f);
  auto vec = [&](int n) {
    std::vector<float> v(static_cast<std::size_t>(n));
    for (float& x : v)
      x = norm(rng);
    return v;
  };
  for (int l = 0; l < layers; ++l) {
    L w{Random(rng, kQ, H, 0.02f),
        Random(rng, kKv, H, 0.02f),
        Random(rng, kKv, H, 0.02f),
        Random(rng, H, kQ, 0.02f),
        Random(rng, I, H, 0.02f),
        Random(rng, I, H, 0.02f),
        Random(rng, H, I, 0.01f),
        vec(H),
        vec(H),
        vec(128),
        vec(128)};
    dec.SetLayer(l, {w.q.bf.data(), w.k.bf.data(), w.v.bf.data(), w.o.bf.data(),
                     w.gate.bf.data(), w.up.bf.data(), w.down.bf.data(),
                     w.n1.data(), w.n2.data(), w.qn.data(), w.kn.data()});
    ws.push_back(std::move(w));
  }
  std::vector<Mat> hs;
  const std::vector<float> fnorm = vec(H);
  for (int i = 0; i < heads; ++i) {
    hs.push_back(Random(rng, head_rows, H, 0.02f));
    dec.SetHead(i, hs.back().bf.data(), head_rows,
                i == 0 ? fnorm.data() : nullptr);
  }

  // A cache of P earlier positions (random, as prefill would have stored).
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<std::vector<std::vector<double>>> K(
      static_cast<std::size_t>(layers));
  std::vector<std::vector<std::vector<double>>> V(
      static_cast<std::size_t>(layers));
  for (int l = 0; l < layers; ++l)
    for (int p = 0; p < P; ++p) {
      std::vector<float> k(kKv), v(kKv);
      for (float& x : k)
        x = F(Bf16(nd(rng)));
      for (float& x : v)
        x = F(Bf16(nd(rng)));
      dec.WriteKv(l, p, 1, k.data(), kKv, v.data(), kKv);
      K[std::size_t(l)].emplace_back(k.begin(), k.end());
      V[std::size_t(l)].emplace_back(v.begin(), v.end());
    }

  int failures = 0;
  for (int step = 0; step < 3; ++step) {
    const int pos = P + step, hi = step % heads;
    std::vector<float> row(static_cast<std::size_t>(H));
    for (float& x : row)
      x = nd(rng);
    dec.SelectHead(hi);
    dec.Start(row.data(), pos);
    dec.Wait();
    // Reference.
    std::vector<double> h(row.begin(), row.end());
    for (int l = 0; l < layers; ++l) {
      const L& w = ws[std::size_t(l)];
      std::vector<double> x = h;
      Rms(x, 0, H, w.n1.data());
      std::vector<double> q = MatVec(w.q, x), k = MatVec(w.k, x),
                          v = MatVec(w.v, x);
      for (int hh = 0; hh < 16; ++hh) {
        Rms(q, std::size_t(hh) * 128, 128, w.qn.data());
        Rope(q, std::size_t(hh) * 128, pos);
      }
      for (int hh = 0; hh < 8; ++hh) {
        Rms(k, std::size_t(hh) * 128, 128, w.kn.data());
        Rope(k, std::size_t(hh) * 128, pos);
      }
      // The NPU stores bf16 entries.
      std::vector<double> kb(k.size()), vb(v.size());
      for (std::size_t i = 0; i < k.size(); ++i) {
        kb[i] = F(Bf16(float(k[i])));
        vb[i] = F(Bf16(float(v[i])));
      }
      K[std::size_t(l)].push_back(kb);
      V[std::size_t(l)].push_back(vb);
      std::vector<double> att(kQ, 0.0);
      const int n = int(K[std::size_t(l)].size());
      for (int hh = 0; hh < 16; ++hh) {
        const int kh = hh / 2;
        std::vector<double> s(static_cast<std::size_t>(n));
        double top = -1e300, total = 0;
        for (int t = 0; t < n; ++t) {
          double d = 0;
          for (int i = 0; i < 128; ++i)
            d += q[std::size_t(hh) * 128 + i] *
                 K[std::size_t(l)][std::size_t(t)][std::size_t(kh) * 128 + i];
          s[std::size_t(t)] = d / std::sqrt(128.0);
          top = std::max(top, s[std::size_t(t)]);
        }
        for (double& e : s)
          total += e = std::exp(e - top);
        for (int t = 0; t < n; ++t)
          for (int i = 0; i < 128; ++i)
            att[std::size_t(hh) * 128 + i] +=
                s[std::size_t(t)] / total *
                V[std::size_t(l)][std::size_t(t)][std::size_t(kh) * 128 + i];
      }
      const std::vector<double> o = MatVec(w.o, att);
      for (int i = 0; i < H; ++i)
        h[std::size_t(i)] += o[std::size_t(i)];
      x = h;
      Rms(x, 0, H, w.n2.data());
      const std::vector<double> g = MatVec(w.gate, x), u = MatVec(w.up, x);
      std::vector<double> a(static_cast<std::size_t>(I));
      for (int i = 0; i < I; ++i)
        a[std::size_t(i)] = g[std::size_t(i)] /
                            (1 + std::exp(-g[std::size_t(i)])) *
                            u[std::size_t(i)];
      const std::vector<double> dn = MatVec(w.down, a);
      for (int i = 0; i < H; ++i)
        h[std::size_t(i)] += dn[std::size_t(i)];
    }
    std::vector<float> hidden(static_cast<std::size_t>(H));
    dec.Hidden(hidden.data());
    const double he = RelErr(hidden, h);
    std::vector<double> x = h;
    Rms(x, 0, H, fnorm.data());
    const std::vector<double> ref = MatVec(hs[std::size_t(hi)], x);
    const auto lg = dec.logits();
    const std::vector<float> got(lg.begin(), lg.end());
    const double le = RelErr(got, ref);
    int arg_n = 0, arg_r = 0;
    for (int i = 1; i < head_rows; ++i) {
      if (got[std::size_t(i)] > got[std::size_t(arg_n)])
        arg_n = i;
      if (ref[std::size_t(i)] > ref[std::size_t(arg_r)])
        arg_r = i;
    }
    std::printf(
        "%d x %d, head %d, pos %d: hidden rel err %.2e, logits rel err %.2e, "
        "argmax %d/%d\n",
        layers, H, hi, pos, he, le, arg_n, arg_r);
    if (!(he < 5e-2 && le < 5e-2))
      ++failures;
  }
  // Timing at growing positions.
  std::vector<float> row(static_cast<std::size_t>(H), 0.01f);
  const int capacity = dec.capacity();
  const auto t0 = std::chrono::steady_clock::now();
  int done = 0;
  for (int i = 0; i < timed && P + 3 + i < capacity; ++i, ++done) {
    dec.Start(row.data(), P + 3 + i);
    dec.Wait();
  }
  const double s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count() /
      std::max(done, 1);
  std::printf("%d x %d decode: %.2f ms per token (head %d rows)\n", layers, H,
              s * 1e3, head_rows);
  return failures;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf(
        "usage: npu_qwen3dec_device_test <qwen3dec dir> [tokens] [cached]\n");
    return 2;
  }
  if (npu::ProbeDevices().empty()) {
    std::printf("SKIP: no NPU\n");
    return 77;
  }
  npu::Device device;
  try {
    device = npu::Device::Open();
  } catch (const npu::Error& e) {
    std::printf("SKIP: %s\n", e.what());
    return 77;
  }
  const int timed = argc > 2 ? std::atoi(argv[2]) : 20;
  const int P = argc > 3 ? std::atoi(argv[3]) : 37;
  std::mt19937 rng(5);
  auto talker = D::Open(device, argv[1]);
  int failures = CheckStack(talker, D::kHidden, D::kFfn, D::kLayers, 3072, 1, P,
                            timed, rng);
  // The code predictor on the talker's program: 5 x 1024, 15 heads.
  auto predictor = D::OpenPredictor(talker, argv[1]);
  failures += CheckStack(predictor, 1024, 3072, 5, 2048, 15, 2, 10, rng);
  // The talker again after the predictor ran on the same context.
  {
    std::vector<float> row(D::kHidden, 0.01f);
    talker.Start(row.data(), P + 3 + timed);
    talker.Wait();
  }
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
