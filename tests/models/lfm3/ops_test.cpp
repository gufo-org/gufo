#include "src/models/lfm3/hip/ops.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
namespace g = gufo::models::lfm3::gpu;
namespace {
void Check(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
struct Buffer {
  float* p{};
  explicit Buffer(std::size_t n) { Check(hipMalloc(&p, n * 4)); }
  explicit Buffer(const std::vector<float>& x) : Buffer(x.size()) {
    Check(hipMemcpy(p, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  }
  ~Buffer() { (void)hipFree(p); }
  std::vector<float> Read(std::size_t n) const {
    std::vector<float> x(n);
    Check(hipMemcpy(x.data(), p, n * 4, hipMemcpyDeviceToHost));
    return x;
  }
};
std::vector<float> Data(std::size_t n, float scale = 0.1F) {
  std::vector<float> x(n);
  for (std::size_t i = 0; i < n; ++i)
    x[i] = scale * std::sin(float(i) * 0.73F);
  return x;
}
void Near(const std::vector<float>& got, const std::vector<float>& want,
          float limit) {
  if (got.size() != want.size())
    throw std::runtime_error("size mismatch");
  for (std::size_t i = 0; i < got.size(); ++i)
    if (!std::isfinite(got[i]) || std::abs(got[i] - want[i]) > limit)
      throw std::runtime_error("numerical mismatch at " + std::to_string(i) +
                               ": " + std::to_string(got[i]) + " vs " +
                               std::to_string(want[i]));
}
void Projection(int n) {
  constexpr int M = 37, K = 2048;
  std::vector<gufo::quant::block_q8_0> weights(M * K / 32);
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i].d = 0x2400 + 32 * (i % 10);
    for (int j = 0; j < 32; ++j)
      weights[i].qs[j] = int((i * 17 + j * 37) % 255) - 127;
  }
  auto x = Data(n * K, 0.07F);
  std::vector<float> expected(n * M);
  for (int row = 0; row < n; ++row)
    for (int m = 0; m < M; ++m) {
      double sum = 0;
      for (int k = 0; k < K; ++k) {
        const auto& block = weights[m * (K / 32) + k / 32];
        sum += double(gufo::quant::Fp16ToFloat(block.d)) * block.qs[k % 32] *
               x[row * K + k];
      }
      expected[row * M + m] = sum;
    }
  auto bytes = weights.size() * sizeof(weights[0]);
  Buffer w((bytes + 3) / 4), dx(x), out(n * M),
      quant((g::DenseWorkspaceBytes(n, K) + 3) / 4);
  Check(hipMemcpy(w.p, weights.data(), bytes, hipMemcpyHostToDevice));
  gufo::models::lfm3::Tensor tensor{w.p, gufo::core::GgmlType::kQ8_0, K, M,
                                    bytes};
  g::Dense(tensor, dx.p, out.p, n, nullptr, {quant.p});
  Check(hipDeviceSynchronize());
  Near(out.Read(n * M), expected, 0.0001F);
}
void Conv(int n) {
  constexpr int H = 37, L = 7;
  auto projection = Data(n * 3 * H), weight = Data(H * L, 0.3F),
       history = Data(H * (L - 1));
  auto expected_history = history;
  std::vector<float> expected(n * H);
  for (int c = 0; c < H; ++c) {
    std::vector<double> seq;
    for (int j = 0; j < L - 1; ++j)
      seq.push_back(history[c * (L - 1) + j]);
    for (int t = 0; t < n; ++t) {
      seq.push_back(
          float(projection[t * 3 * H + c] * projection[t * 3 * H + 2 * H + c]));
      double sum = 0;
      for (int j = 0; j < L; ++j)
        sum += double(weight[c * L + j]) * seq[t + j];
      expected[t * H + c] = sum * projection[t * 3 * H + H + c];
    }
    for (int j = 0; j < L - 1; ++j)
      expected_history[c * (L - 1) + j] = seq[n + j];
  }
  Buffer p(projection), w(weight), h(history), y(n * H);
  g::ShortConv(p.p, w.p, h.p, y.p, n, H, L, nullptr);
  Check(hipDeviceSynchronize());
  Near(y.Read(n * H), expected, 1e-7F);
  Near(h.Read(history.size()), expected_history, 0);
}
void Rotary(int n) {
  constexpr int D = 128, H = 3, P = 12001;
  auto x = Data(n * H * D), expected = x;
  std::vector<float> factors(D / 2, 1);
  for (int i = 32; i < 64; ++i)
    factors[i] = 1e30F;
  // Full-head pairing: the second operand of pair 0 is dimension 64,
  // while pairs 32..63 (both halves) remain unchanged.
  for (int row = 0; row < n; ++row)
    for (int h = 0; h < H; ++h)
      for (int i = 0; i < D / 2; ++i) {
        auto base = (row * H + h) * D;
        double angle =
            float((P + row) * std::pow(1000000.0F, -2.0F * i / D) / factors[i]);
        expected[base + i] =
            x[base + i] * std::cos(angle) - x[base + i + 64] * std::sin(angle);
        expected[base + i + 64] =
            x[base + i + 64] * std::cos(angle) + x[base + i] * std::sin(angle);
      }
  Buffer b(x), f(factors), frequencies(D / 2);
  g::RopeFrequencies(frequencies.p, D, 1000000, nullptr);
  g::Rope(b.p, frequencies.p, f.p, n, H, D, P, nullptr);
  Check(hipDeviceSynchronize());
  Near(b.Read(x.size()), expected, 2e-5F);
}
void Attention(int n, int window, int P = 259) {
  constexpr int H = 4, KH = 2, D = 128;
  const int capacity = window ? window : P + n, stride = KH * D;
  auto q = Data(n * H * D, 0.7F), k = Data(n * stride, 0.5F),
       v = Data(n * stride, 0.9F), gates = Data(n * H, 0.7F);
  auto ck = Data(capacity * stride, 0.6F), cv = Data(capacity * stride, 0.8F);
  std::vector<float> expected(n * H * D);
  for (int row = 0; row < n; ++row)
    for (int h = 0; h < H; ++h) {
      const int last = P + row,
                first = window ? std::max(0, last + 1 - window) : 0,
                kh = h / (H / KH);
      std::vector<double> scores;
      double max = -INFINITY;
      for (int t = first; t <= last; ++t) {
        const auto* key = t >= P ? k.data() + (t - P) * stride
                                 : ck.data() + (t % capacity) * stride;
        double sum = 0;
        for (int d = 0; d < D; ++d)
          sum += double(q[(row * H + h) * D + d]) * key[kh * D + d];
        scores.push_back(sum / std::sqrt(D));
        max = std::max(max, scores.back());
      }
      double den = 0;
      for (auto& s : scores) {
        s = std::exp(s - max);
        den += s;
      }
      for (int d = 0; d < D; ++d) {
        double sum = 0;
        for (int t = first; t <= last; ++t) {
          const auto* value = t >= P ? v.data() + (t - P) * stride
                                     : cv.data() + (t % capacity) * stride;
          sum += scores[t - first] / den * value[kh * D + d];
        }
        expected[(row * H + h) * D + d] =
            sum / (1 + std::exp(-double(gates[row * H + h])));
      }
    }
  Buffer dq(q), dk(k), dv(v), dg(gates), dck(ck), dcv(cv), out(expected.size()),
      scratch(g::AttentionScratchBytes(H, D, P + n, n) / 4);
  g::Attention(dq.p, dk.p, dv.p, dg.p, dck.p, dcv.p, out.p, n, H, KH, D, P,
               capacity, window, nullptr, scratch.p);
  g::StoreKV(dk.p, dv.p, dck.p, dcv.p, n, stride, P, capacity, nullptr);
  Check(hipDeviceSynchronize());
  Near(out.Read(expected.size()), expected, 2e-6F);
  for (int row = 0; row < n; ++row)
    for (int d = 0; d < stride; ++d) {
      ck[((P + row) % capacity) * stride + d] = k[row * stride + d];
      cv[((P + row) % capacity) * stride + d] = v[row * stride + d];
    }
  Near(dck.Read(ck.size()), ck, 0);
  Near(dcv.Read(cv.size()), cv, 0);
}
}  // namespace
int main() {
  try {
    for (int n : {1, 8, 9, 32, 33, 128, 512}) {
      std::cerr << "conv " << n << "\n";
      Projection(n);
      Conv(n);
      std::cerr << "rope " << n << "\n";
      Rotary(n);
      std::cerr << "full " << n << "\n";
      Attention(n, 0);
      std::cerr << "swa " << n << "\n";
      Attention(n, 256);
    }
    // Long decode histories span several key tiles per block.
    std::cerr << "decode 4999\n";
    Attention(1, 0, 4999);
    Attention(1, 256, 4999);
    // Prompts over long histories split their keys across blocks.
    std::cerr << "prompt 17000\n";
    Attention(40, 0, 17000);
    Attention(40, 256, 17000);
    std::cout << "PASS LFM3 convolution, RoPE, gated GQA and sliding ring "
                 "against independent formulas\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
