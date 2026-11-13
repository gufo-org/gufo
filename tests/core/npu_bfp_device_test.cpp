// On-device check of gufo::npu::BfpGemm: bf16 x bfp16 products against a
// double reference over shapes that pad rows, inputs and outputs, then the
// timing of prefill-sized products. Skips (77) without an NPU.
// Usage: npu_bfp_device_test <bfpgemm dir>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "src/core/npu/bfp_gemm.hpp"

namespace npu = gufo::npu;

namespace {

double RelativeError(const std::vector<float>& y,
                     const std::vector<double>& ref) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    num += (y[i] - ref[i]) * (y[i] - ref[i]);
    den += ref[i] * ref[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: npu_bfp_device_test <bfpgemm dir>\n");
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
  auto gemm = npu::BfpGemm::Open(device, argv[1], std::size_t(512) << 20);
  if (argc == 6 && std::string(argv[2]) == "bench") {
    // bench n k rows: the time of one product shape, nothing else.
    const int n = std::atoi(argv[3]), k = std::atoi(argv[4]),
              rows = std::atoi(argv[5]);
    std::vector<float> w(std::size_t(n) * k, 0.01f),
        x(std::size_t(rows) * k, 1.0f), y(std::size_t(rows) * n);
    const int id = gemm.Upload(n, k, w);
    for (int i = 0; i < 3; ++i)
      gemm.Run(id, x.data(), rows, y.data());
    const int reps = 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
      gemm.Run(id, x.data(), rows, y.data());
    const double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() /
        reps;
    std::printf("bench n %d k %d rows %d: %.3f ms, %.2f TFLOPS\n", n, k, rows,
                s * 1e3, 2.0 * n * k * rows / s / 1e12);
    return 0;
  }
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  int failures = 0;
  struct Shape {
    int n, k, rows;
  };
  for (const Shape s :
       {Shape{1024, 512, 512}, Shape{1000, 600, 37}, Shape{3000, 2048, 700},
        Shape{512, 6144, 1100}, Shape{2048, 512, 2349}, Shape{1500, 700, 200},
        Shape{3000, 2048, 128}}) {
    std::vector<float> w(std::size_t(s.n) * s.k), x(std::size_t(s.rows) * s.k);
    for (float& v : w)
      v = normal(rng) * 0.05f;
    for (float& v : x)
      v = normal(rng);
    const int id = gemm.Upload(s.n, s.k, w);
    std::vector<float> y(std::size_t(s.rows) * s.n, -1.0f);
    for (int rep = 0; rep < 5; ++rep)  // short shapes reach bfpgemm_s
      gemm.Run(id, x.data(), s.rows, y.data());
    std::vector<double> ref(y.size());
    for (int r = 0; r < s.rows; ++r)
      for (int o = 0; o < s.n; ++o) {
        double acc = 0;
        for (int i = 0; i < s.k; ++i)
          acc +=
              double(x[std::size_t(r) * s.k + i]) * w[std::size_t(o) * s.k + i];
        ref[std::size_t(r) * s.n + o] = acc;
      }
    const double err = RelativeError(y, ref);
    std::printf("n %d k %d rows %d: relative error %.2e\n", s.n, s.k, s.rows,
                err);
    if (!(err < 2e-2))
      ++failures;
  }
  // A batch: one job over same-shaped calls of consecutive matrices.
  {
    const int k = 256;
    std::vector<int> ids, ns{300, 512, 1000, 450, 77, 1024},
        rows{100, 128, 7, 33, 64, 120};
    std::vector<std::vector<float>> ws, xs, ys;
    for (std::size_t i = 0; i < ns.size(); ++i) {
      ws.emplace_back(std::size_t(ns[i]) * k);
      xs.emplace_back(std::size_t(rows[i]) * k);
      for (float& v : ws.back())
        v = normal(rng) * 0.05f;
      for (float& v : xs.back())
        v = normal(rng);
      ids.push_back(gemm.Upload(ns[i], k, ws.back()));
      ys.emplace_back(std::size_t(rows[i]) * ns[i]);
    }
    std::vector<npu::BfpGemm::Call> calls;
    for (std::size_t i = 0; i < ns.size(); ++i)
      calls.push_back({ids[i], xs[i].data(), rows[i], ys[i].data()});
    for (int rep = 0; rep < 5; ++rep)
      gemm.Run(calls);
    double worst = 0;
    for (std::size_t c = 0; c < ns.size(); ++c) {
      std::vector<double> ref(ys[c].size());
      for (int r = 0; r < rows[c]; ++r)
        for (int o = 0; o < ns[c]; ++o) {
          double acc = 0;
          for (int i = 0; i < k; ++i)
            acc += double(xs[c][std::size_t(r) * k + i]) *
                   ws[c][std::size_t(o) * k + i];
          ref[std::size_t(r) * ns[c] + o] = acc;
        }
      worst = std::max(worst, RelativeError(ys[c], ref));
    }
    std::printf("batch of %zu calls: relative error %.2e\n", ns.size(), worst);
    if (!(worst < 2e-2))
      ++failures;
  }
  // Switching programs: 128-row bands for short calls, 512-row bands for
  // long ones, alternately.
  {
    std::vector<float> w(std::size_t(1024) * 512, 0.01f),
        x(std::size_t(600) * 512, 1.0f), y(std::size_t(600) * 1024);
    const int id = gemm.Upload(1024, 512, w);
    for (int i = 0; i < 3; ++i) {
      gemm.Run(id, x.data(), 64, y.data());
      gemm.Run(id, x.data(), 600, y.data());
    }
    auto time = [&](int rows_a, int rows_b) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 10; ++i) {
        gemm.Run(id, x.data(), rows_a, y.data());
        gemm.Run(id, x.data(), rows_b, y.data());
      }
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0)
                 .count() /
             20;
    };
    std::printf(
        "time 64-row runs %.3f ms, 600-row runs %.3f ms, alternating %.3f ms\n",
        time(64, 64), time(600, 600), time(64, 600));
  }
  // Timing: prefill-sized products (Qwen3 1.7B layer shapes).
  struct Timed {
    int n, k, rows;
  };
  for (const Timed t :
       {Timed{1024, 512, 8}, Timed{4096, 512, 8}, Timed{512, 2048, 8},
        Timed{4096, 2048, 512}, Timed{12288, 2048, 512}, Timed{2048, 6144, 512},
        Timed{12288, 2048, 2048}}) {
    std::vector<float> w(std::size_t(t.n) * t.k, 0.01f),
        x(std::size_t(t.rows) * t.k, 1.0f), y(std::size_t(t.rows) * t.n);
    const int id = gemm.Upload(t.n, t.k, w);
    gemm.Run(id, x.data(), t.rows, y.data());
    const int reps = 10;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
      gemm.Run(id, x.data(), t.rows, y.data());
    const double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() /
        reps;
    std::printf("time n %d k %d rows %d: %.3f ms, %.2f TFLOPS\n", t.n, t.k,
                t.rows, s * 1e3, 2.0 * t.n * t.k * t.rows / s / 1e12);
  }
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
