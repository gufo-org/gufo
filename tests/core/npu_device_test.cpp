// On-device check of the NPU runtime: Q8_0 products through gufo::npu::Q8Gemm
// (the q8gemm design compiled at build time) against a double reference, one
// call and one batch of shapes. Exits 77 (skipped) without a usable NPU.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "src/core/npu/npu.hpp"
#include "src/core/npu/q8_gemm.hpp"

namespace {

struct Matrix {
  int n, k;
  std::vector<std::int8_t> codes;
  std::vector<float> scales;
};

Matrix Random(int n, int k, std::mt19937& rng) {
  std::normal_distribution<float> nd;
  Matrix m{n, k, std::vector<std::int8_t>(std::size_t(n) * k),
           std::vector<float>(std::size_t(n) * k / 32)};
  for (auto& c : m.codes)
    c = std::int8_t(
        std::lround(std::fmax(-127.0f, std::fmin(127.0f, nd(rng) * 40.0f))));
  for (auto& d : m.scales)
    d = float(_Float16(0.001f * (1.0f + std::fabs(nd(rng)))));
  return m;
}

// Largest error relative to the largest reference value.
double Check(const Matrix& m, const std::vector<float>& x, int rows,
             const std::vector<float>& y) {
  double worst = 0, top = 0;
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < m.n; ++j) {
      double ref = 0;
      for (int c = 0; c < m.k; ++c)
        ref += double(x[std::size_t(r) * m.k + c]) *
               m.codes[std::size_t(j) * m.k + c] *
               m.scales[std::size_t(j) * (m.k / 32) + c / 32];
      worst = std::fmax(worst, std::fabs(ref - y[std::size_t(r) * m.n + j]));
      top = std::fmax(top, std::fabs(ref));
    }
  return worst / top;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2)
    return 2;
  namespace npu = gufo::npu;
  if (npu::ProbeDevices().empty()) {
    std::puts("SKIP: no XDNA NPU");
    return 77;
  }
  npu::Device device;
  try {
    device = npu::Device::Open();
  } catch (const npu::Error& e) {
    std::printf("SKIP: %s\n", e.what());
    return 77;
  }
  auto gemm = npu::Q8Gemm::Open(device, argv[1], std::size_t(64) << 20);
  std::mt19937 rng(7);
  std::normal_distribution<float> nd;
  struct Shape {
    int n, k, rows;
  };
  const Shape shapes[] = {{1024, 1024, 12}, {300, 2816, 5}, {5000, 512, 40}};
  std::vector<Matrix> ms;
  std::vector<int> ids;
  std::vector<std::vector<float>> xs, ys;
  for (const auto& s : shapes) {
    ms.push_back(Random(s.n, s.k, rng));
    ids.push_back(gemm.Upload(s.n, s.k, ms.back().codes, ms.back().scales));
    xs.emplace_back(std::size_t(s.rows) * s.k);
    for (auto& v : xs.back())
      v = nd(rng);
    ys.emplace_back(std::size_t(s.rows) * s.n);
  }
  int failures = 0;
  // one product per call, then all three in one batch
  for (int batch = 0; batch < 2; ++batch) {
    if (batch) {
      std::vector<npu::Q8Gemm::Call> calls;
      for (std::size_t i = 0; i < ms.size(); ++i)
        calls.push_back({ids[i], xs[i].data(), shapes[i].rows, ys[i].data()});
      gemm.Run(calls);
    } else {
      for (std::size_t i = 0; i < ms.size(); ++i)
        gemm.Run(ids[i], xs[i].data(), shapes[i].rows, ys[i].data());
    }
    for (std::size_t i = 0; i < ms.size(); ++i) {
      const double err = Check(ms[i], xs[i], shapes[i].rows, ys[i]);
      std::printf("%s n %d k %d rows %d: relative error %.2e\n",
                  batch ? "batch" : "call", shapes[i].n, shapes[i].k,
                  shapes[i].rows, err);
      failures += err > 2e-4;
    }
  }
  return failures ? 1 : 0;
}
