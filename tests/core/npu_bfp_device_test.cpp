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
        Shape{512, 6144, 1100}, Shape{2048, 512, 2349}}) {
    std::vector<float> w(std::size_t(s.n) * s.k), x(std::size_t(s.rows) * s.k);
    for (float& v : w)
      v = normal(rng) * 0.05f;
    for (float& v : x)
      v = normal(rng);
    const int id = gemm.Upload(s.n, s.k, w);
    std::vector<float> y(std::size_t(s.rows) * s.n, -1.0f);
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
