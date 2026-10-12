#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::size_t kTokens = 4096;
constexpr std::size_t kUsed = 10;
constexpr std::size_t kExperts = 512;
constexpr std::size_t kHidden = 2560;
constexpr std::size_t kFf = 640;
constexpr int kWarmup = 3;
constexpr int kIters = 20;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template<typename T>
T* Upload(const std::vector<T>& v) {
  T* d = nullptr;
  Check(hipMalloc(&d, v.size() * sizeof(T) + 4096), "malloc");
  Check(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice),
        "upload");
  return d;
}

std::vector<std::uint8_t> RandomBytes(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<std::uint8_t> v(n);
  for (auto& b : v)
    b = static_cast<std::uint8_t>(rng());
  return v;
}

double TimeMs(const std::function<bool()>& fn) {
  for (int i = 0; i < kWarmup; ++i)
    if (!fn())
      throw std::runtime_error("launch rejected");
  Check(hipDeviceSynchronize(), "warmup");
  hipEvent_t a, b;
  Check(hipEventCreate(&a), "ev");
  Check(hipEventCreate(&b), "ev");
  Check(hipEventRecord(a, nullptr), "rec");
  for (int i = 0; i < kIters; ++i)
    fn();
  Check(hipEventRecord(b, nullptr), "rec");
  Check(hipEventSynchronize(b), "sync");
  float ms = 0;
  Check(hipEventElapsedTime(&ms, a, b), "elapsed");
  return ms / kIters;
}

std::uint64_t Hash(const void* device, std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  Check(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost), "hash");
  std::uint64_t h = 1469598103934665603ULL;
  for (std::uint8_t b : host) {
    h ^= b;
    h *= 1099511628211ULL;
  }
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  const double skew = argc > 1 ? std::atof(argv[1]) : 0.55;
  // Zipf-like expert popularity matched to the measured real-prompt fill.
  std::vector<double> weight(kExperts);
  for (std::size_t e = 0; e < kExperts; ++e)
    weight[e] = 1.0 / std::pow(static_cast<double>(e) + 8.0, skew);
  std::mt19937 rng(12345);
  std::discrete_distribution<int> pick(weight.begin(), weight.end());
  std::vector<std::int32_t> ids(kTokens * kUsed);
  std::vector<std::uint32_t> counts(kExperts, 0);
  for (std::size_t t = 0; t < kTokens; ++t) {
    std::vector<int> chosen;
    while (chosen.size() < kUsed) {
      const int e = pick(rng);
      if (std::find(chosen.begin(), chosen.end(), e) == chosen.end())
        chosen.push_back(e);
    }
    for (std::size_t j = 0; j < kUsed; ++j) {
      ids[t * kUsed + j] = chosen[j];
      ++counts[chosen[j]];
    }
  }
  const std::size_t slots = kTokens * kUsed;
  std::uint32_t max_bucket = 0;
  std::uint64_t tiles48 = 0, tiles128 = 0;
  for (std::uint32_t c : counts) {
    max_bucket = std::max(max_bucket, c);
    tiles48 += ((c + 15u) / 16u * 16u + 47u) / 48u;
    tiles128 += (c + 127u) / 128u;
  }
  std::printf(
      "routing: slots=%zu max_bucket=%u tiles48=%llu (fill %.2f) "
      "tiles128=%llu (fill %.2f)\n",
      slots, max_bucket, static_cast<unsigned long long>(tiles48),
      static_cast<double>(slots) / (tiles48 * 48.0),
      static_cast<unsigned long long>(tiles128),
      static_cast<double>(slots) / (tiles128 * 128.0));

  // IQ3_S gate/up (110 bytes per 256 weights) and IQ4_NL down (18 per 32).
  const std::size_t gate_row = kHidden / 256 * 110;
  const std::size_t down_row = kFf / 32 * 18;
  auto* d_gate = Upload(RandomBytes(kExperts * kFf * gate_row, 1));
  auto* d_up = Upload(RandomBytes(kExperts * kFf * gate_row, 2));
  auto* d_down = Upload(RandomBytes(kExperts * kHidden * down_row, 3));

  std::vector<__half> xh(kTokens * kHidden);
  std::mt19937 xr(7);
  for (auto& v : xh)
    v = __float2half((static_cast<int>(xr() & 0xFFFF) - 32768) / 65536.0F);
  std::vector<__half> uh(slots * kFf);
  for (auto& v : uh)
    v = __float2half((static_cast<int>(xr() & 0xFFFF) - 32768) / 65536.0F);
  __half* d_x = Upload(xh);
  __half* d_u = Upload(uh);
  __half* d_up_out = nullptr;
  __half* d_down_out = nullptr;
  Check(hipMalloc(&d_up_out, (slots + 8192) * kFf * sizeof(__half)), "malloc");
  Check(hipMalloc(&d_down_out, (slots + 8192) * kHidden * sizeof(__half)),
        "malloc");
  Check(hipMemset(d_up_out, 0, (slots + 8192) * kFf * sizeof(__half)),
        "memset");
  Check(hipMemset(d_down_out, 0, (slots + 8192) * kHidden * sizeof(__half)),
        "memset");

  const std::size_t compact = q::RoutedCompactRows(slots, kExperts);
  std::int32_t* d_ids = Upload(ids);
  std::uint32_t* d_counts = Upload(counts);
  std::int32_t* d_bounds = Upload(std::vector<std::int32_t>(kExperts + 1, 0));
  std::int32_t* d_cursors = Upload(std::vector<std::int32_t>(kExperts, 0));
  std::int32_t* d_rows_token = Upload(std::vector<std::int32_t>(compact, 0));
  std::int32_t* d_rows_slot = Upload(std::vector<std::int32_t>(compact, 0));
  q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rows_token,
                   d_rows_slot, kTokens, kUsed, kExperts, nullptr);
  Check(hipDeviceSynchronize(), "compact");

  std::vector<std::int32_t> t128, t48;
  for (std::size_t e = 0; e < kExperts; ++e) {
    for (std::uint32_t j = 0; j < (counts[e] + 127u) / 128u; ++j)
      t128.push_back(static_cast<std::int32_t>(e | (j << 16)));
    const std::uint32_t padded = (counts[e] + 15u) / 16u * 16u;
    for (std::uint32_t j = 0; j < (padded + 47u) / 48u; ++j)
      t48.push_back(static_cast<std::int32_t>(e | (j << 16)));
  }
  std::int32_t* d_t128 = Upload(t128);
  std::int32_t* d_t48 = Upload(t48);

  const double flop_pair = 2.0 * 2.0 * slots * kFf * kHidden;
  const double flop_down = 2.0 * slots * kHidden * kFf;

  const double pair_ms = TimeMs([&] {
    return q::RoutedGatedF16Gemm(
        d_gate, d_up, q::WeightType::kIQ3_S, d_x, d_t128,
        static_cast<std::uint32_t>(t128.size()), 128, d_bounds, d_rows_token,
        d_rows_slot, d_up_out, kFf, kHidden, nullptr);
  });
  std::printf(
      "pair  IQ3_S gate+up : %7.3f ms/call  %5.1f TFLOPS (real rows)  "
      "hash %016llx\n",
      pair_ms, flop_pair / (pair_ms * 1e9),
      static_cast<unsigned long long>(
          Hash(d_up_out, slots * kFf * sizeof(__half))));

  const double down_ms = TimeMs([&] {
    return q::RoutedF16Gemm(d_down, q::WeightType::kIQ4_NL, d_u, d_t128,
                            static_cast<std::uint32_t>(t128.size()), 128,
                            d_bounds, d_rows_slot, d_rows_slot, nullptr,
                            nullptr, d_down_out, kHidden, kFf, nullptr);
  });
  std::printf(
      "down  IQ4_NL (128) : %7.3f ms/call  %5.1f TFLOPS (real rows)  "
      "hash %016llx\n",
      down_ms, flop_down / (down_ms * 1e9),
      static_cast<unsigned long long>(
          Hash(d_down_out, slots * kHidden * sizeof(__half))));

  const double down48_ms = TimeMs([&] {
    return q::RoutedF16Gemm(d_down, q::WeightType::kIQ4_NL, d_u, d_t48,
                            static_cast<std::uint32_t>(t48.size()), 48,
                            d_bounds, d_rows_slot, d_rows_slot, nullptr,
                            nullptr, d_down_out, kHidden, kFf, nullptr);
  });
  std::printf("down  IQ4_NL (48)  : %7.3f ms/call  %5.1f TFLOPS (real rows)\n",
              down48_ms, flop_down / (down48_ms * 1e9));
  return 0;
}
