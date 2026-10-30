#include "src/models/d1/hip/ops.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/models/lfm3/hip/ops.hpp"
void Check(hipError_t s) {
  if (s != hipSuccess)
    throw std::runtime_error(hipGetErrorString(s));
}
struct Device {
  float* p{};
  explicit Device(std::size_t n) { Check(hipMalloc(&p, n * 4)); }
  explicit Device(const std::vector<float>& v) : Device(v.size()) {
    Check(hipMemcpy(p, v.data(), v.size() * 4, hipMemcpyHostToDevice));
  }
  ~Device() { (void)hipFree(p); }
  std::vector<float> Read(std::size_t n) {
    std::vector<float> v(n);
    Check(hipMemcpy(v.data(), p, n * 4, hipMemcpyDeviceToHost));
    return v;
  }
};
int main() {
  try {
    for (int n : {1, 7, 15, 17, 33, 65}) {
      constexpr int heads = 16, kh = 4, dim = 128;
      std::vector<float> q(n * heads * dim), k(n * kh * dim), v(k.size()),
          gate(n * heads);
      for (std::size_t i = 0; i < q.size(); ++i)
        q[i] = std::sin(i * .17) * .2;
      for (std::size_t i = 0; i < k.size(); ++i) {
        k[i] = std::cos(i * .13) * .3;
        v[i] = std::sin(i * .09);
      }
      for (std::size_t i = 0; i < gate.size(); ++i)
        gate[i] = std::cos(i * .07);
      Device dq(q), dk(k), dv(v), dg(gate), out(q.size());
      gufo::models::lfm3::gpu::BidirectionalAttention(
          dq.p, dk.p, dv.p, dg.p, out.p, n, heads, kh, nullptr);
      auto actual = out.Read(q.size());
      double error = 0;
      for (int row = 0; row < n; ++row)
        for (int h = 0; h < heads; ++h) {
          std::vector<double> scores(n);
          double sum = 0;
          for (int t = 0; t < n; ++t) {
            double dot = 0;
            for (int d = 0; d < dim; ++d)
              dot += double(q[(row * heads + h) * dim + d]) *
                     k[(t * kh + h / 4) * dim + d];
            scores[t] = std::exp(dot / std::sqrt(dim));
            sum += scores[t];
          }
          for (int d = 0; d < dim; ++d) {
            double ref = 0;
            for (int t = 0; t < n; ++t)
              ref += scores[t] / sum * v[(t * kh + h / 4) * dim + d];
            ref /= 1 + std::exp(-double(gate[row * heads + h]));
            error = std::max(
                error, std::abs(ref - actual[(row * heads + h) * dim + d]));
          }
        }
      if (error > 2e-6)
        throw std::runtime_error(
            "bidirectional attention differs from independent FP64 formula");
      std::cout << "attention n=" << n << " max_error=" << error << '\n';
    }
    constexpr int h = 4, w = 6, hidden = 7;
    auto index = [](int y, int x) {
      return (y / 2 * (w / 2) + x / 2) * 4 + y % 2 * 2 + x % 2;
    };
    std::vector<float> p(h * w * hidden * 3), weights(hidden * 9);
    for (std::size_t i = 0; i < p.size(); ++i)
      p[i] = std::sin(i * .13);
    for (std::size_t i = 0; i < weights.size(); ++i)
      weights[i] = std::cos(i * .17);
    Device dp(p), dw(weights), out(h * w * hidden);
    gufo::models::d1::gpu::SpatialConv(dp.p, dw.p, out.p, h, w, hidden,
                                       nullptr);
    auto actual = out.Read(h * w * hidden);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        for (int c = 0; c < hidden; ++c) {
          double ref = 0;
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
              if (y + dy >= 0 && y + dy < h && x + dx >= 0 && x + dx < w) {
                const auto i = index(y + dy, x + dx) * 3 * hidden + c;
                ref += double(float(p[i] * p[i + 2 * hidden])) *
                       weights[c * 9 + (dy + 1) * 3 + dx + 1];
              }
          ref *= p[index(y, x) * 3 * hidden + hidden + c];
          if (std::abs(ref - actual[index(y, x) * hidden + c]) > 2e-6)
            throw std::runtime_error(
                "spatial gated convolution differs from FP64 formula");
        }
    std::vector<float> rope(h * w * 8 * 64);
    for (std::size_t i = 0; i < rope.size(); ++i)
      rope[i] = std::sin(i * .19);
    Device dr(rope);
    gufo::models::d1::gpu::AxialRope(dr.p, h, w, 8, nullptr);
    actual = dr.Read(rope.size());
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        for (int head = 0; head < 8; ++head)
          for (int d = 0; d < 32; ++d) {
            const int at = (index(y, x) * 8 + head) * 64 + d;
            const double angle =
                (d < 16 ? y : x) * std::pow(1e6, -double(d % 16) / 16);
            const double a =
                rope[at] * std::cos(angle) - rope[at + 32] * std::sin(angle);
            const double b =
                rope[at + 32] * std::cos(angle) + rope[at] * std::sin(angle);
            if (std::max(std::abs(a - actual[at]),
                         std::abs(b - actual[at + 32])) > 2e-6)
              throw std::runtime_error("axial RoPE differs from FP64 formula");
          }
    std::cout << "PASS d1 spatial convolution and axial RoPE\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
