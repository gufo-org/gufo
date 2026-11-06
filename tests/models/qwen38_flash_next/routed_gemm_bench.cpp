// Times the wide-batch routed expert GEMMs of one Flash-Next layer: the
// paired gate/up projection (SwiGLU, F16 out) and the down projection over
// the executor's 128-token tile map for a skewed
// top-10 routing of 512 experts, or the per-expert counts of each line of a
// counts file (one layer per line; tokens = sum / 10). Prints the median
// time, the useful TFLOPS, the streamed weight bandwidth of the active
// experts and FNV-1a hashes of both outputs, then the totals over all layers.
//
// routed_gemm_bench [reps] [tokens | counts-file] [gate type] [down type]
//   gate type: q4_k (default) | q5_k; down type: q5_1 (default) | q8_0
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
q::WeightType gate_type = q::WeightType::kQ4_K;
q::WeightType down_type = q::WeightType::kQ5_1;

/// Bytes per quantized block and elements per block.
std::size_t BlockBytes(q::WeightType type) {
  return type == q::WeightType::kQ4_K   ? 144
         : type == q::WeightType::kQ5_K ? 176
         : type == q::WeightType::kQ5_1 ? 24
                                        : 34;
}
std::size_t BlockElems(q::WeightType type) {
  return type == q::WeightType::kQ4_K || type == q::WeightType::kQ5_K ? 256
                                                                      : 32;
}
q::WeightType ParseType(const std::string& name) {
  if (name == "q4_k")
    return q::WeightType::kQ4_K;
  if (name == "q5_k")
    return q::WeightType::kQ5_K;
  if (name == "q5_1")
    return q::WeightType::kQ5_1;
  if (name == "q8_0")
    return q::WeightType::kQ8_0;
  throw std::runtime_error("unknown weight type " + name);
}

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

/// Random quantized blocks whose F16 scale fields hold model-like values
/// (weights near 0.02), so neither output overflows F16.
std::vector<std::uint8_t> MakeBlocks(q::WeightType type, std::size_t elems,
                                     std::uint32_t seed) {
  const std::size_t bytes = BlockBytes(type);
  const std::size_t blocks = elems / BlockElems(type);
  const bool k_quant = BlockElems(type) == 256;
  std::vector<std::uint8_t> w(blocks * bytes);
  for (std::size_t i = 0; i < w.size(); i += 4) {
    const std::uint32_t r = Next(&seed);
    std::memcpy(&w[i], &r, std::min<std::size_t>(4, w.size() - i));
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    const float d = (k_quant ? 4e-4F : 1.5e-3F) *
                    (1.0F + 0.5F * static_cast<float>(Next(&seed) % 97) / 97);
    const std::uint16_t dh = HalfBits(d);
    std::memcpy(&w[b * bytes], &dh, 2);
    if (type == q::WeightType::kQ4_K || type == q::WeightType::kQ5_K ||
        type == q::WeightType::kQ5_1) {
      const std::uint16_t mh =
          HalfBits(type == q::WeightType::kQ5_1 ? -16.0F * d : 0.5F * d);
      std::memcpy(&w[b * bytes + 2], &mh, 2);
    }
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
  double pair_tflop;
  double down_tflop;
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
  // The executor's maps: one of 128-token tiles for the paired gate/up and
  // the down projections; below 1,024 tokens one 16- or 48-token map serves
  // separate gate, up and down projections.
  const bool wide = tokens >= 1024;
  const std::uint32_t narrow_rows = slots >= 16 * kExperts ? 48 : 16;
  std::vector<std::int32_t> tiles;
  const auto append = [&](std::uint32_t rows) {
    for (std::uint32_t e = 0; e < kExperts; ++e)
      for (std::uint32_t j = 0; j < (counts[e] + rows - 1) / rows; ++j)
        tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
  };
  const std::uint32_t tile_rows = wide ? 128 : narrow_rows;
  append(tile_rows);
  const auto n_tiles = static_cast<std::uint32_t>(tiles.size());
  std::printf("tokens %u  active %3u  tiles %4u  ", tokens, active, n_tiles);

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
      if (!q::RoutedF16Gemm(d_gate, gate_type, d_x, d_tiles, n_tiles, tile_rows,
                            d_bounds, d_rows_token, d_rows_slot, nullptr,
                            d_gate_out, nullptr, kFf, kHidden, nullptr) ||
          !q::RoutedF16Gemm(d_up, gate_type, d_x, d_tiles, n_tiles, tile_rows,
                            d_bounds, d_rows_token, d_rows_slot, d_gate_out,
                            nullptr, d_mid, kFf, kHidden, nullptr))
        throw std::runtime_error("gate/up GEMM rejected");
      return;
    }
    if (!q::RoutedGatedF16Gemm(d_gate, d_up, gate_type, d_x, d_tiles, n_tiles,
                               d_bounds, d_rows_token, d_rows_slot, d_mid, kFf,
                               kHidden, nullptr))
      throw std::runtime_error("gated GEMM rejected");
  };
  const auto down = [&] {
    if (!q::RoutedF16Gemm(d_down, down_type, d_mid, d_tiles, n_tiles, tile_rows,
                          d_bounds, d_rows_slot, d_rows_slot, nullptr, nullptr,
                          d_out, kHidden, kFf, nullptr))
      throw std::runtime_error("down GEMM rejected");
  };
  pair();
  down();
  const float pair_ms = Median(reps, pair);
  const float down_ms = Median(reps, down);
  const double pair_gb = 2.0 * active * kFf * kHidden / BlockElems(gate_type) *
                         BlockBytes(gate_type) / 1e9;
  const double down_gb = 1.0 * active * kHidden * kFf / BlockElems(down_type) *
                         BlockBytes(down_type) / 1e9;
  const double pair_tflop = 4.0 * slots * kFf * kHidden / 1e12;
  const double down_tflop = 2.0 * slots * kHidden * kFf / 1e12;
  std::printf(
      "gate/up %6.3f ms %4.1f TFLOPS %5.1f GB/s %016llx  down %6.3f ms %4.1f "
      "TFLOPS %5.1f GB/s %016llx\n",
      pair_ms, pair_tflop / pair_ms * 1e3, pair_gb / pair_ms * 1e3,
      static_cast<unsigned long long>(Hash(d_mid, std::size_t{slots} * kFf)),
      down_ms, down_tflop / down_ms * 1e3, down_gb / down_ms * 1e3,
      static_cast<unsigned long long>(
          Hash(d_out, std::size_t{slots} * kHidden)));
  for (void* p : std::initializer_list<void*>{
           d_ids, d_counts, d_tiles, d_bounds, d_cursors, d_rows_token,
           d_rows_slot, d_x, d_mid, d_gate_out, d_out})
    CheckHip(hipFree(p), "hipFree");
  return {pair_ms, down_ms, pair_gb, down_gb, pair_tflop, down_tflop};
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
  if (argc > 3)
    gate_type = ParseType(argv[3]);
  if (argc > 4)
    down_type = ParseType(argv[4]);
  const std::size_t elems = std::size_t{kExperts} * kFf * kHidden;
  auto* d_gate = Upload(MakeBlocks(gate_type, elems, 1U));
  auto* d_up = Upload(MakeBlocks(gate_type, elems, 2U));
  auto* d_down = Upload(MakeBlocks(down_type, elems, 3U));
  Result total{};
  for (const auto& counts : layers) {
    const Result r = RunLayer(counts, reps, d_gate, d_up, d_down);
    total.pair_ms += r.pair_ms;
    total.down_ms += r.down_ms;
    total.pair_gb += r.pair_gb;
    total.down_gb += r.down_gb;
    total.pair_tflop += r.pair_tflop;
    total.down_tflop += r.down_tflop;
  }
  std::printf(
      "total %zu layers: gate/up %.2f ms (%.1f TFLOPS, %.1f GB/s)  down %.2f "
      "ms (%.1f TFLOPS, %.1f GB/s)\n",
      layers.size(), total.pair_ms, total.pair_tflop / total.pair_ms * 1e3,
      total.pair_gb / total.pair_ms * 1e3, total.down_ms,
      total.down_tflop / total.down_ms * 1e3,
      total.down_gb / total.down_ms * 1e3);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
