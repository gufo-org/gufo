// Times one decode HC mixer (diagnostic): the five-launch sequence
// (HcCombineQ8_1, Q8_0 down projection, SiluScaleQ8_1, Q8_0 up projection,
// HcMixEpilogue) against HcMixQ8Fused, both replayed from hipGraphs as decode
// runs them. Each graph holds `mixers` consecutive mixers whose weights
// rotate through `copies` sets (~7 MB each), well past the 32 MiB MALL.
//
// hc_mix_bench [mixers] [copies] [replays]
//
// Prints the median over alternating rounds of each graph's median
// microseconds per mixer for 1, 2, 4 and 8 tokens.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kStreams = 4;
constexpr std::uint32_t kRank = 320;
constexpr std::uint32_t kDim = kHidden * kStreams;
constexpr std::uint32_t kMaxTokens = 8;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

void* Device(std::size_t bytes) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, bytes), "hipMalloc");
  CheckHip(hipMemset(p, 0, bytes), "hipMemset");
  return p;
}

void* Q8Weights(std::size_t blocks, std::uint32_t seed) {
  std::vector<std::uint8_t> host(blocks * 34);
  for (std::size_t b = 0; b < blocks; ++b) {
    const __half d =
        __float2half_rn(static_cast<float>(1 + Next(&seed) % 13) / 2048.0F);
    std::memcpy(host.data() + b * 34, &d, sizeof(d));
    for (std::size_t j = 0; j < 32; ++j)
      host[b * 34 + 2 + j] =
          static_cast<std::uint8_t>(static_cast<int>(Next(&seed) % 255) - 127);
  }
  void* p = Device(host.size());
  CheckHip(hipMemcpy(p, host.data(), host.size(), hipMemcpyHostToDevice),
           "weights");
  return p;
}

float* Values(std::size_t count, std::uint32_t seed, float scale,
              float offset = 0.0F) {
  std::vector<float> host(count);
  for (float& v : host)
    v = offset +
        scale *
            static_cast<float>(static_cast<int>(Next(&seed) & 0xFFFF) - 32768) /
            32768.0F;
  auto* p = static_cast<float*>(Device(count * sizeof(float)));
  CheckHip(
      hipMemcpy(p, host.data(), count * sizeof(float), hipMemcpyHostToDevice),
      "values");
  return p;
}

double MedianMicros(hipGraphExec_t exec, hipStream_t stream, int replays,
                    int mixers) {
  hipEvent_t start, stop;
  CheckHip(hipEventCreate(&start), "event");
  CheckHip(hipEventCreate(&stop), "event");
  for (int i = 0; i < 3; ++i)
    CheckHip(hipGraphLaunch(exec, stream), "warm replay");
  std::vector<double> times;
  for (int i = 0; i < replays; ++i) {
    CheckHip(hipEventRecord(start, stream), "record");
    CheckHip(hipGraphLaunch(exec, stream), "replay");
    CheckHip(hipEventRecord(stop, stream), "record");
    CheckHip(hipEventSynchronize(stop), "sync");
    float ms = 0.0F;
    CheckHip(hipEventElapsedTime(&ms, start, stop), "elapsed");
    times.push_back(1000.0 * ms / mixers);
  }
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

hipGraphExec_t Capture(hipStream_t stream, const std::function<void()>& body) {
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
           "capture");
  body();
  CheckHip(hipStreamEndCapture(stream, &graph), "capture end");
  CheckHip(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
           "instantiate");
  (void)hipGraphDestroy(graph);
  return exec;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int mixers = argc > 1 ? std::atoi(argv[1]) : 48;
    const int copies = argc > 2 ? std::atoi(argv[2]) : 16;
    const int replays = argc > 3 ? std::atoi(argv[3]) : 21;
    hipStream_t stream = nullptr;
    CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    std::vector<void*> down(copies), up(copies);
    for (int i = 0; i < copies; ++i) {
      down[i] = Q8Weights(std::size_t{kRank} * kDim / 32, 0x1000U + i);
      up[i] = Q8Weights(std::size_t{kDim} * kRank / 32, 0x2000U + i);
    }
    const std::uint32_t parts = q::HcInjectParts(kHidden);
    const auto lo_cols =
        static_cast<std::uint32_t>(qfn_mmq_q8_1_bytes(1, kRank) / 36 * 32);
    float* res = Values(std::size_t{kMaxTokens} * kDim, 1, 1.0F);
    float* block_out = Values(std::size_t{kMaxTokens} * kHidden, 2, 0.01F);
    float* inject = Values(std::size_t{kMaxTokens} * kStreams * parts, 3, 0.0F);
    float* gamma = Values(kDim, 4, 0.25F, 1.0F);
    float* inject_w = Values(std::size_t{kStreams} * kDim, 5, 0.02F);
    auto* xn = static_cast<float*>(Device(sizeof(float) * kMaxTokens * kDim));
    void* xn_q8 = Device(qfn_mmq_q8_1_bytes(kMaxTokens, kDim));
    auto* lo = static_cast<float*>(Device(sizeof(float) * kMaxTokens * kRank));
    void* lo_q8 = Device(qfn_mmq_q8_1_bytes(kMaxTokens, kRank));
    auto* gate = static_cast<float*>(Device(sizeof(float) * kMaxTokens * kDim));
    auto* mixed =
        static_cast<float*>(Device(sizeof(float) * kMaxTokens * kHidden));
    void* mixed_q8 = Device(qfn_mmq_q8_1_bytes(kMaxTokens, kHidden));
    // The mixers chain through one inject buffer, as a decode step does;
    // block_out stays tiny so the residual stays bounded across replays.
    void* state = Device(q::HcMixQ8FusedStateBytes());
    std::printf("tokens  five-launch_us  fused_us  speedup\n");
    for (const std::uint32_t n : {1U, 2U, 4U, 8U}) {
      const auto old_exec = Capture(stream, [&] {
        for (int m = 0; m < mixers; ++m) {
          const int c = m % copies;
          q::HcCombineQ8_1(res, block_out, inject, parts, gamma, xn, xn_q8, n,
                           kHidden, kStreams, 1e-6F, stream);
          qfn_mmq_q8_0_dense_vec_preq(down[c], nullptr, xn_q8, lo, kRank,
                                      static_cast<int>(n), kDim, stream);
          q::SiluScaleQ8_1(lo, 1.0F / kStreams, n, kRank, lo_cols, lo_q8,
                           stream);
          qfn_mmq_q8_0_dense_vec_preq(up[c], nullptr, lo_q8, gate, kDim,
                                      static_cast<int>(n), kRank, stream);
          q::HcMixEpilogue(xn, gate, inject_w, mixed, inject, n, kHidden,
                           kStreams, stream, mixed_q8);
        }
      });
      const auto fused_exec = Capture(stream, [&] {
        for (int m = 0; m < mixers; ++m) {
          const int c = m % copies;
          const q::HcFusedMix mix{down[c],        up[c],    res,      block_out,
                                  inject,         parts,    gamma,    xn,
                                  xn_q8,          lo,       lo_q8,    lo_cols,
                                  inject_w,       mixed,    mixed_q8, inject,
                                  kHidden,        kStreams, kRank,    1e-6F,
                                  1.0F / kStreams};
          if (!q::HcMixQ8Fused(state, mix, n, stream))
            throw std::runtime_error("fused mixer rejected");
        }
      });
      // Alternating rounds share whatever else loads the memory system.
      std::vector<double> old_rounds, fused_rounds;
      for (int round = 0; round < 7; ++round) {
        old_rounds.push_back(MedianMicros(old_exec, stream, replays, mixers));
        fused_rounds.push_back(
            MedianMicros(fused_exec, stream, replays, mixers));
      }
      std::sort(old_rounds.begin(), old_rounds.end());
      std::sort(fused_rounds.begin(), fused_rounds.end());
      const double old_us = old_rounds[old_rounds.size() / 2];
      const double fused_us = fused_rounds[fused_rounds.size() / 2];
      std::printf("%6u  %14.2f  %8.2f  %7.3f\n", n, old_us, fused_us,
                  old_us / fused_us);
      (void)hipGraphExecDestroy(old_exec);
      (void)hipGraphExecDestroy(fused_exec);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
