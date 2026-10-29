#define ENGINE_ENABLE_HIP 1
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/lfm3/hip/q8_prefill.hpp"
using namespace gufo::models::lfm3::gpu;
void Check(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
struct Buffer {
  void* p{};
  explicit Buffer(std::size_t n) { Check(hipMalloc(&p, n)); }
  ~Buffer() {
    if (p)
      (void)hipFree(p);
  }
};
struct Q8Block {
  __half d;
  signed char qs[32];
};
constexpr int N = 512;
template<int BM, int BN, int WM, int WN>
void Run(const void* w, SplitActivations x, float* y, int M, int K) {
  SplitQ8Gemm<BM, BN, WM, WN>
      <<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM), 256>>>(w, x, y, N, M, K);
}
void Shape(int M, int K) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> x(N * K);
  for (auto& v : x)
    v = dist(gen);
  // One outlier per row makes the row maximum dominate.
  for (int t = 0; t < N; ++t)
    x[t * K + (t * 7) % K] *= 40;
  std::vector<Q8Block> weights(std::size_t(M) * K / 32);
  for (auto& b : weights) {
    b.d = __float2half(0.001F + std::abs(dist(gen)) * 0.1F);
    for (auto& q : b.qs)
      q = int(gen() % 255) - 127;
  }
  std::size_t weight_bytes = weights.size() * sizeof(weights[0]);
  Buffer dw(4 * weight_bytes), dx(x.size() * 4), dy(N * M * 4),
      split(SplitActivationBytes(N, K));
  for (int r = 0; r < 4; ++r)
    Check(hipMemcpy(static_cast<char*>(dw.p) + r * weight_bytes, weights.data(),
                    weight_bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(dx.p, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  const auto activations = SplitActivations::In(split.p, N, K);
  SplitRows<<<N, 256>>>(static_cast<const float*>(dx.p), activations, K);
  using Fn = void (*)(const void*, SplitActivations, float*, int, int);
  struct Case {
    const char* name;
    Fn run;
  };
  Case cases[] = {{"64x64-4x2", Run<64, 64, 4, 2>},
                  {"128x64-4x2", Run<128, 64, 4, 2>},
                  {"128x128-2x4", Run<128, 128, 2, 4>},
                  {"128x128-4x2", Run<128, 128, 4, 2>},
                  {"256x64-8x1", Run<256, 64, 8, 1>}};
  for (auto c : cases) {
    c.run(dw.p, activations, static_cast<float*>(dy.p), M, K);
    Check(hipDeviceSynchronize());
    std::vector<float> y(N * M);
    Check(hipMemcpy(y.data(), dy.p, y.size() * 4, hipMemcpyDeviceToHost));
    double error = 0, magnitude = 0;
    for (int sample = 0; sample < 397; ++sample) {
      int row = (sample * 173) % M, token = (sample * 37) % N;
      double expected = 0;
      for (int k = 0; k < K; ++k) {
        auto& b = weights[row * (K / 32) + k / 32];
        expected += double(__half2float(b.d)) * b.qs[k % 32] * x[token * K + k];
      }
      error = std::max(error, std::abs(expected - y[token * M + row]));
      magnitude = std::max(magnitude, std::abs(expected));
    }
    if (error > 1e-5 * magnitude)
      throw std::runtime_error(std::string(c.name) + " relative error " +
                               std::to_string(error / magnitude));
    hipEvent_t start, end;
    Check(hipEventCreate(&start));
    Check(hipEventCreate(&end));
    std::vector<float> times;
    for (int rep = 0; rep < 7; ++rep) {
      Check(hipEventRecord(start));
      for (int r = 0; r < 8; ++r)
        c.run(static_cast<char*>(dw.p) + (r % 4) * weight_bytes, activations,
              static_cast<float*>(dy.p), M, K);
      Check(hipEventRecord(end));
      Check(hipEventSynchronize(end));
      float ms;
      Check(hipEventElapsedTime(&ms, start, end));
      times.push_back(ms * 1000 / 8);
    }
    std::sort(times.begin(), times.end());
    std::cout << M << "x" << K << " " << c.name << " us=" << times[3]
              << " effective_tops=" << 2.0 * N * M * K / times[3] / 1e6
              << " relative_error=" << error / magnitude << std::endl;
    Check(hipEventDestroy(start));
    Check(hipEventDestroy(end));
  }
}
int main() {
  try {
    Shape(7680, 2048);
    Shape(2048, 7680);
    Shape(6144, 2048);
    Shape(2048, 2048);
    Shape(512, 2048);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
