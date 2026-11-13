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

constexpr int kH = D::kHidden, kI = D::kFfn, kHead = 3072, kQ = 2048,
              kKv = 1024;

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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: npu_qwen3dec_device_test <qwen3dec dir> [tokens]\n");
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
  auto dec = D::Open(device, argv[1]);
  dec.SetRopeTheta(1e6);
  std::mt19937 rng(5);
  struct L {
    Mat q, k, v, o, gate, up, down;
    std::vector<float> n1, n2, qn, kn;
  };
  std::vector<L> layers;
  std::uniform_real_distribution<float> norm(0.8f, 1.2f);
  auto vec = [&](int n) {
    std::vector<float> v(static_cast<std::size_t>(n));
    for (float& x : v)
      x = norm(rng);
    return v;
  };
  for (int l = 0; l < D::kLayers; ++l) {
    L w{Random(rng, kQ, kH, 0.02f),
        Random(rng, kKv, kH, 0.02f),
        Random(rng, kKv, kH, 0.02f),
        Random(rng, kH, kQ, 0.02f),
        Random(rng, kI, kH, 0.02f),
        Random(rng, kI, kH, 0.02f),
        Random(rng, kH, kI, 0.01f),
        vec(kH),
        vec(kH),
        vec(128),
        vec(128)};
    dec.SetLayer(l, {w.q.bf.data(), w.k.bf.data(), w.v.bf.data(), w.o.bf.data(),
                     w.gate.bf.data(), w.up.bf.data(), w.down.bf.data(),
                     w.n1.data(), w.n2.data(), w.qn.data(), w.kn.data()});
    layers.push_back(std::move(w));
  }
  const Mat head = Random(rng, kHead, kH, 0.02f);
  const std::vector<float> fnorm = vec(kH);
  dec.SetHead(head.bf.data(), kHead, fnorm.data());

  // A cache of P earlier positions (random, as prefill would have stored).
  const int P = 37;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<std::vector<std::vector<double>>> K(D::kLayers), V(D::kLayers);
  for (int l = 0; l < D::kLayers; ++l)
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
    const int pos = P + step;
    std::vector<float> row(kH);
    for (float& x : row)
      x = nd(rng);
    dec.Start(row.data(), pos);
    dec.Wait();
    // Reference.
    std::vector<double> h(row.begin(), row.end());
    for (int l = 0; l < D::kLayers; ++l) {
      const L& w = layers[std::size_t(l)];
      std::vector<double> x = h;
      Rms(x, 0, kH, w.n1.data());
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
      for (int i = 0; i < kH; ++i)
        h[std::size_t(i)] += o[std::size_t(i)];
      x = h;
      Rms(x, 0, kH, w.n2.data());
      const std::vector<double> g = MatVec(w.gate, x), u = MatVec(w.up, x);
      std::vector<double> a(static_cast<std::size_t>(kI));
      for (int i = 0; i < kI; ++i)
        a[std::size_t(i)] = g[std::size_t(i)] /
                            (1 + std::exp(-g[std::size_t(i)])) *
                            u[std::size_t(i)];
      const std::vector<double> dn = MatVec(w.down, a);
      for (int i = 0; i < kH; ++i)
        h[std::size_t(i)] += dn[std::size_t(i)];
    }
    std::vector<float> hidden(kH);
    dec.Hidden(hidden.data());
    const double he = RelErr(hidden, h);
    std::vector<double> x = h;
    Rms(x, 0, kH, fnorm.data());
    const std::vector<double> ref = MatVec(head, x);
    const auto lg = dec.logits();
    const std::vector<float> got(lg.begin(), lg.end());
    const double le = RelErr(got, ref);
    int arg_n = 0, arg_r = 0;
    for (int i = 1; i < kHead; ++i) {
      if (got[std::size_t(i)] > got[std::size_t(arg_n)])
        arg_n = i;
      if (ref[std::size_t(i)] > ref[std::size_t(arg_r)])
        arg_r = i;
    }
    std::printf(
        "pos %d: hidden rel err %.2e, logits rel err %.2e, argmax %d/%d\n", pos,
        he, le, arg_n, arg_r);
    if (!(he < 5e-2 && le < 5e-2))
      ++failures;
  }
  // Timing at growing positions.
  std::vector<float> row(kH, 0.01f);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < timed; ++i) {
    dec.Start(row.data(), P + 3 + i);
    dec.Wait();
  }
  const double s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count() /
      timed;
  std::printf("decode: %.2f ms per token (head %d rows)\n", s * 1e3, kHead);
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
