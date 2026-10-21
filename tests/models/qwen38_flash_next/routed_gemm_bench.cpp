// Times the wide-batch routed expert GEMMs of one Flash-Next layer: the
// paired Q4_K gate/up projection (128-token tiles, SwiGLU, F16 out) and the
// Q5_1 down projection (64-token tiles), with the executor's tile maps over
// a skewed top-10 routing of 512 experts, or the per-expert counts of each
// line of a counts file (one layer per line; tokens = sum / 10). Prints the
// median time, the streamed weight bandwidth of the active experts and
// FNV-1a hashes of both outputs, then the totals over all layers.
//
// routed_gemm_bench [reps] [tokens | counts-file]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

constexpr std::uint32_t kExperts = 512;
constexpr std::uint32_t kUsed = 10;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kFf = 640;
constexpr std::size_t kQ4KBytes = 144;  // per 256 elements
constexpr std::size_t kQ51Bytes = 24;   // per 32 elements

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

template<typename T>
T* Alloc(std::size_t count) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, count * sizeof(T)), "hipMalloc");
  CheckHip(hipMemset(p, 0, count * sizeof(T)), "hipMemset");
  return static_cast<T*>(p);
}

template<typename T>
std::uint64_t Hash(const T* device, std::size_t count) {
  std::vector<std::uint8_t> host(count * sizeof(T));
  CheckHip(hipMemcpy(host.data(), device, host.size(), hipMemcpyDeviceToHost),
           "hipMemcpy");
  std::uint64_t h = 1469598103934665603ULL;
  for (std::uint8_t c : host)
    h = (h ^ c) * 1099511628211ULL;
  return h;
}

std::uint16_t HalfBits(float v) {
  const __half h = __float2half(v);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

/// Random quantized blocks whose F16 scale fields hold small finite values.
std::vector<std::uint8_t> MakeBlocks(std::size_t blocks, std::size_t bytes,
                                     std::uint32_t seed) {
  std::vector<std::uint8_t> w(blocks * bytes);
  for (std::size_t i = 0; i < w.size(); i += 4) {
    const std::uint32_t r = Next(&seed);
    std::memcpy(&w[i], &r, std::min<std::size_t>(4, w.size() - i));
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::uint16_t d = HalfBits(0.01F + 0.02F * (Next(&seed) % 97) / 97);
    const std::uint16_t m = HalfBits(0.005F + 0.01F * (Next(&seed) % 89) / 89);
    std::memcpy(&w[b * bytes], &d, 2);
    std::memcpy(&w[b * bytes + 2], &m, 2);
  }
  return w;
}

template<typename F>
float Median(int reps, F&& run) {
  hipEvent_t a;
  hipEvent_t b;
  CheckHip(hipEventCreate(&a), "event");
  CheckHip(hipEventCreate(&b), "event");
  std::vector<float> ms;
  for (int r = 0; r < reps; ++r) {
    CheckHip(hipEventRecord(a), "record");
    run();
    CheckHip(hipEventRecord(b), "record");
    CheckHip(hipEventSynchronize(b), "sync");
    float t = 0.0F;
    CheckHip(hipEventElapsedTime(&t, a, b), "elapsed");
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

/// Synthetic skewed routing: expert e has weight 1 / (rank + 16).
std::vector<std::uint32_t> SkewedCounts(std::uint32_t tokens) {
  std::uint32_t seed = 0x9E3779B9U;
  std::vector<std::uint32_t> rank(kExperts);
  std::iota(rank.begin(), rank.end(), 0U);
  for (std::uint32_t i = kExperts - 1; i > 0; --i)
    std::swap(rank[i], rank[Next(&seed) % (i + 1)]);
  std::vector<double> cdf(kExperts);
  double sum = 0.0;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    sum += 1.0 / (rank[e] + 16.0);
    cdf[e] = sum;
  }
  std::vector<std::uint32_t> counts(kExperts, 0);
  std::vector<std::uint32_t> row;
  for (std::uint32_t t = 0; t < tokens; ++t) {
    row.clear();
    while (row.size() < kUsed) {
      const double x = sum * (Next(&seed) / 4294967296.0);
      const auto e = static_cast<std::uint32_t>(
          std::lower_bound(cdf.begin(), cdf.end(), x) - cdf.begin());
      if (std::find(row.begin(), row.end(), e) == row.end())
        row.push_back(e);
    }
    for (std::uint32_t e : row)
      ++counts[e];
  }
  return counts;
}

struct Result {
  float pair_ms;
  float down_ms;
  double pair_gb;
  double down_gb;
};

Result RunLayer(const std::vector<std::uint32_t>& counts, int reps,
                const void* d_gate, const void* d_up, const void* d_down) {
  const std::uint32_t slots = std::accumulate(counts.begin(), counts.end(), 0U);
  const std::uint32_t tokens = slots / kUsed;
  // Expert e takes the next counts[e] tokens cyclically, so every token
  // receives kUsed distinct experts.
  std::vector<std::int32_t> ids(slots);
  std::vector<std::uint32_t> filled(tokens, 0);
  std::uint32_t cursor = 0;
  std::uint32_t active = 0;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    active += counts[e] != 0;
    for (std::uint32_t i = 0; i < counts[e]; ++i) {
      const std::uint32_t t = cursor++ % tokens;
      ids[std::size_t{t} * kUsed + filled[t]++] = static_cast<std::int32_t>(e);
    }
  }
  // The executor's maps: 64-token down tiles, then 128-token pair tiles;
  // below 1,024 tokens one 16- or 48-token map serves separate gate, up and
  // down projections.
  const bool wide = tokens >= 1024;
  const std::uint32_t narrow_rows = slots >= 16 * kExperts ? 48 : 16;
  std::vector<std::int32_t> tiles;
  const auto append = [&](std::uint32_t rows) {
    for (std::uint32_t e = 0; e < kExperts; ++e)
      for (std::uint32_t j = 0; j < (counts[e] + rows - 1) / rows; ++j)
        tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
  };
  append(wide ? 64 : narrow_rows);
  const auto down_tiles = static_cast<std::uint32_t>(tiles.size());
  if (wide)
    append(128);
  const auto pair_tiles = static_cast<std::uint32_t>(tiles.size()) - down_tiles;
  std::printf("tokens %u  active %3u  down tiles %3u  pair tiles %3u  ", tokens,
              active, down_tiles, pair_tiles);

  const std::size_t compact = q::RoutedCompactRows(slots, kExperts);
  auto* d_ids = Upload(ids);
  auto* d_counts = Upload(counts);
  auto* d_tiles = Upload(tiles);
  auto* d_bounds = Alloc<std::int32_t>(kExperts + 1);
  auto* d_cursors = Alloc<std::int32_t>(kExperts);
  auto* d_rows_token = Alloc<std::int32_t>(compact);
  auto* d_rows_slot = Alloc<std::int32_t>(compact);
  q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rows_token,
                   d_rows_slot, tokens, kUsed, kExperts, nullptr);

  std::uint32_t seed = 0x12345678U;
  std::vector<__half> x(std::size_t{tokens} * kHidden);
  for (auto& v : x)
    v = __float2half(
        static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
        1000.0F);
  auto* d_x = Upload(x);
  auto* d_mid = Alloc<__half>(std::size_t{slots} * kFf);
  auto* d_gate_out = Alloc<float>(wide ? 1 : std::size_t{slots} * kFf);
  auto* d_out = Alloc<__half>(std::size_t{slots} * kHidden);
  CheckHip(hipDeviceSynchronize(), "setup");

  const auto pair = [&] {
    if (!wide) {
      if (!q::RoutedF16Gemm(d_gate, q::WeightType::kQ4_K, d_x, d_tiles,
                            down_tiles, narrow_rows, d_bounds, d_rows_token,
                            d_rows_slot, nullptr, d_gate_out, nullptr, kFf,
                            kHidden, nullptr) ||
          !q::RoutedF16Gemm(d_up, q::WeightType::kQ4_K, d_x, d_tiles,
                            down_tiles, narrow_rows, d_bounds, d_rows_token,
                            d_rows_slot, d_gate_out, nullptr, d_mid, kFf,
                            kHidden, nullptr))
        throw std::runtime_error("gate/up GEMM rejected");
      return;
    }
    if (!q::RoutedGatedF16Gemm(d_gate, d_up, q::WeightType::kQ4_K, d_x,
                               d_tiles + down_tiles, pair_tiles, d_bounds,
                               d_rows_token, d_rows_slot, d_mid, kFf, kHidden,
                               nullptr))
      throw std::runtime_error("gated GEMM rejected");
  };
  const auto down = [&] {
    if (!q::RoutedF16Gemm(d_down, q::WeightType::kQ5_1, d_mid, d_tiles,
                          down_tiles, wide ? 64 : narrow_rows, d_bounds,
                          d_rows_slot, d_rows_slot, nullptr, nullptr, d_out,
                          kHidden, kFf, nullptr))
      throw std::runtime_error("down GEMM rejected");
  };
  pair();
  down();
  const float pair_ms = Median(reps, pair);
  const float down_ms = Median(reps, down);
  const double pair_gb = 2.0 * active * kFf * kHidden / 256 * kQ4KBytes / 1e9;
  const double down_gb = 1.0 * active * kHidden * kFf / 32 * kQ51Bytes / 1e9;
  std::printf(
      "gate/up %6.3f ms %5.1f GB/s %016llx  down %6.3f ms %5.1f GB/s "
      "%016llx\n",
      pair_ms, pair_gb / pair_ms * 1e3,
      static_cast<unsigned long long>(Hash(d_mid, std::size_t{slots} * kFf)),
      down_ms, down_gb / down_ms * 1e3,
      static_cast<unsigned long long>(
          Hash(d_out, std::size_t{slots} * kHidden)));
  for (void* p : std::initializer_list<void*>{
           d_ids, d_counts, d_tiles, d_bounds, d_cursors, d_rows_token,
           d_rows_slot, d_x, d_mid, d_gate_out, d_out})
    CheckHip(hipFree(p), "hipFree");
  return {pair_ms, down_ms, pair_gb, down_gb};
}

}  // namespace

int main(int argc, char** argv) try {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 10;
  std::vector<std::vector<std::uint32_t>> layers;
  const std::string source = argc > 2 ? argv[2] : "2048";
  if (!source.empty() && std::all_of(source.begin(), source.end(), [](char c) {
        return c >= '0' && c <= '9';
      })) {
    layers.push_back(
        SkewedCounts(static_cast<std::uint32_t>(std::stoul(source))));
  } else {
    std::ifstream in(source);
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream fields(line);
      std::vector<std::uint32_t> counts;
      std::uint32_t n;
      while (fields >> n)
        counts.push_back(n);
      if (counts.size() == kExperts)
        layers.push_back(std::move(counts));
    }
    if (layers.empty())
      throw std::runtime_error("no 512-expert count lines in " + source);
  }
  const std::size_t gate_blocks = std::size_t{kExperts} * kFf * kHidden / 256;
  auto* d_gate = Upload(MakeBlocks(gate_blocks, kQ4KBytes, 1U));
  auto* d_up = Upload(MakeBlocks(gate_blocks, kQ4KBytes, 2U));
  const std::size_t down_blocks = std::size_t{kExperts} * kHidden * kFf / 32;
  auto* d_down = Upload(MakeBlocks(down_blocks, kQ51Bytes, 3U));
  Result total{};
  for (const auto& counts : layers) {
    const Result r = RunLayer(counts, reps, d_gate, d_up, d_down);
    total.pair_ms += r.pair_ms;
    total.down_ms += r.down_ms;
    total.pair_gb += r.pair_gb;
    total.down_gb += r.down_gb;
  }
  std::printf(
      "total %zu layers: gate/up %.2f ms (%.1f GB/s)  down %.2f ms "
      "(%.1f GB/s)\n",
      layers.size(), total.pair_ms, total.pair_gb / total.pair_ms * 1e3,
      total.down_ms, total.down_gb / total.down_ms * 1e3);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
