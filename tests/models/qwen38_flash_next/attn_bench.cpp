// Times the prefill WMMA attention on synthetic caches and selections
// (diagnostic): one prefill chunk at several depths, sparse selections of
// 512 four-token blocks per query that neighbouring queries mostly share.
//
// attn_bench [tokens] [reps] [start positions...]
//
// A start position of 0 runs the dense causal window. Each line prints an
// FNV-1a hash of the output after the first call, so kernel rewrites can be
// checked for identical results.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kSelected = 512;

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

template<typename T>
T* Upload(const std::vector<T>& host) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "hipMemcpy");
  return static_cast<T*>(p);
}

std::vector<float> Values(std::size_t count, std::uint32_t seed, float scale) {
  std::vector<float> host(count);
  for (auto& v : host)
    v = scale * (static_cast<float>(Next(&seed) % 2001) / 1000.0F - 1.0F);
  return host;
}

std::uint64_t Hash(const float* device, std::size_t count) {
  std::vector<std::uint8_t> host(count * sizeof(float));
  CheckHip(hipMemcpy(host.data(), device, host.size(), hipMemcpyDeviceToHost),
           "hipMemcpy");
  std::uint64_t h = 1469598103934665603ULL;
  for (std::uint8_t c : host)
    h = (h ^ c) * 1099511628211ULL;
  return h;
}

}  // namespace

int main(int argc, char** argv) try {
  const std::uint32_t tokens = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
  std::vector<std::uint32_t> starts;
  for (int i = 3; i < argc; ++i)
    starts.push_back(static_cast<std::uint32_t>(std::atoi(argv[i])));
  if (starts.empty())
    starts = {0, 8192, 32768, 65536, 118000};
  hipEvent_t start;
  hipEvent_t stop;
  CheckHip(hipEventCreate(&start), "hipEventCreate");
  CheckHip(hipEventCreate(&stop), "hipEventCreate");
  for (const std::uint32_t start_pos : starts) {
    const std::uint32_t n_kv = start_pos + tokens;
    const std::size_t q_count = std::size_t{tokens} * kHeads * kDim;
    const std::size_t kv_count = std::size_t{n_kv} * kKvHeads * kDim;
    float* d_q = Upload(Values(q_count, 0x1234ABCDU, 4.0F));
    float* d_gate = Upload(Values(q_count, 0x0BADF00DU, 3.0F));
    std::vector<__half> kh(kv_count);
    std::vector<__half> vh(kv_count);
    {
      std::uint32_t s = 0xBADC0FFEU;
      for (std::size_t i = 0; i < kv_count; ++i) {
        kh[i] =
            __float2half(static_cast<float>(Next(&s) % 2001) / 1000.0F - 1.0F);
        vh[i] =
            __float2half(static_cast<float>(Next(&s) % 2001) / 1000.0F - 1.0F);
      }
    }
    __half* d_k = Upload(kh);
    __half* d_v = Upload(vh);
    // Sixteen-query groups share a base selection; each query swaps a fifth
    // of it, so four packed queries select a union of roughly 1.6x 512.
    const std::uint32_t blocks = (n_kv + kRatio - 1) / kRatio;
    const std::uint32_t mask_words = (blocks + 31) / 32;
    std::uint32_t* d_mask = nullptr;
    if (start_pos != 0) {
      std::vector<std::uint32_t> mask(std::size_t{tokens} * mask_words);
      std::vector<std::uint32_t> base;
      std::uint32_t s = 0xDEADBEEFU;
      for (std::uint32_t t = 0; t < tokens; ++t) {
        const std::uint32_t complete = (start_pos + t + 1) / kRatio;
        if (t % 16 == 0) {
          base.clear();
          for (std::uint32_t j = 0; j < kSelected; ++j)
            base.push_back(Next(&s) % complete);
        }
        std::uint32_t* row = mask.data() + std::size_t{t} * mask_words;
        for (std::uint32_t b : base) {
          if (Next(&s) % 5 == 0)
            b = Next(&s) % complete;
          row[b / 32] |= 1U << (b % 32);
        }
      }
      d_mask = Upload(mask);
    }
    float* d_out = Upload(std::vector<float>(q_count));
    const auto run = [&] {
      if (!q::WmmaCausalAttention(d_q, d_gate, d_k, d_v, d_mask, mask_words,
                                  d_out, tokens, start_pos, kHeads, kKvHeads,
                                  kDim, kRatio, nullptr))
        throw std::runtime_error("attention rejected the geometry");
    };
    run();
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
    const std::uint64_t hash = Hash(d_out, q_count);
    run();
    std::vector<float> ms(reps);
    for (int i = 0; i < reps; ++i) {
      CheckHip(hipEventRecord(start, nullptr), "hipEventRecord");
      run();
      CheckHip(hipEventRecord(stop, nullptr), "hipEventRecord");
      CheckHip(hipEventSynchronize(stop), "hipEventSynchronize");
      CheckHip(hipEventElapsedTime(&ms[i], start, stop), "hipEventElapsedTime");
    }
    std::sort(ms.begin(), ms.end());
    std::printf("start %6u tokens %u %s  %8.3f ms  hash %016llx\n", start_pos,
                tokens, d_mask ? "sparse" : "dense ", ms[reps / 2],
                static_cast<unsigned long long>(hash));
    for (void* p : {static_cast<void*>(d_q), static_cast<void*>(d_gate),
                    static_cast<void*>(d_k), static_cast<void*>(d_v),
                    static_cast<void*>(d_mask), static_cast<void*>(d_out)})
      if (p != nullptr)
        CheckHip(hipFree(p), "hipFree");
  }
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
