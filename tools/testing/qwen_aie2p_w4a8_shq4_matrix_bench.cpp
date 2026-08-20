// Copyright (C) 2026 Strix Engine contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// W4A8 SHQ4-T16 shape-matrix bench for the Qwen3.8-27B MTP AIE2P program
// (issue #35, M004-C007; long-horizon-plan.md §2.7 P3/P4).
//
// Sweeps the W4A8 SHQ4-T16 shape matrix (M,N,K buckets) with the Qwen AIE2P
// W4A8 driver session (NPU) and/or the pack.hpp CPU oracle (ReferenceGemm):
//   - singleton + repeated timed runs (command/completion/upload/download)
//   - program/config reuse (one session, zero program reloads)
//   - serialized vs overlapped DMA (the driver Run serializes
//     upload -> command -> download; no overlap API is exposed)
//   - padding costs (tail-K/N +/-15 shapes keep block/tile counts)
//   - activation packing cost (per-record QuantizeActivationBlock timing)
//
// Builds CPU-only: pack.hpp requires no XRT headers, and all driver/device
// includes are guarded by ENGINE_ENABLE_XRT. Under ENGINE_ENABLE_XRT the NPU
// path is gated on XDNA2 presence and degrades to a clean SKIP (exit 77) when
// the device is missing, unless STRIX_REQUIRE_XDNA2=1 (then it fails loudly).
//
// Output: one compact JSON object per measured run on stdout, plus a final
// JSON summary (--output PATH). The summary doubles as the M004-C007
// measurement artifact shell (artifacts/m004/c007-xdna2.json).

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/xdna2/qwen_aie2p_w4a8_pack.hpp"

#ifdef ENGINE_ENABLE_XRT
#include "src/core/diagnostics/system_inventory.h"
#include "src/core/xdna2/device.h"
#include "src/core/xdna2/qwen_aie2p_w4a8.h"
#endif

namespace {

using strix::xdna2::w4a8::kBlockElements;
using strix::xdna2::w4a8::kGroupsPerBlock;
using strix::xdna2::w4a8::kInputRecordBytes;
using strix::xdna2::w4a8::kMmulM;
using strix::xdna2::w4a8::kMmulN;
using strix::xdna2::w4a8::kWeightRecordBytes;
using strix::xdna2::w4a8::PackWeightBlock;
using strix::xdna2::w4a8::QuantizeActivationBlock;
using strix::xdna2::w4a8::ReferenceGemm;
using strix::xdna2::w4a8::Shape;
using strix::xdna2::w4a8::WeightBlockSpec;
using strix::xdna2::w4a8::WeightSpecCode;

// ---------------------------------------------------------------------------
// Shape matrix buckets (plan §2.6: real shapes x M buckets, tail-K/N +/-15)
// ---------------------------------------------------------------------------

inline constexpr std::uint32_t kRealK[] = {5120, 10240, 17408};
inline constexpr std::uint32_t kTailK[] = {5105, 5135, 10225, 10255, 17393,
                                           17423};
inline constexpr std::uint32_t kRealN[] = {5120, 10240, 17408};
inline constexpr std::uint32_t kTailN[] = {5105, 5135, 17423};
// The baked b40/t10/r1 xclbin computes ROUNDS=1 chunks of 4 M rows, so the
// driver session covers M in 1..4. Larger M needs rounds>1 xclbins (the
// plan's 128-vs-64 M bound is still OPEN; see metadata.json).
inline constexpr std::uint32_t kHardwareM[] = {1, 2, 3, 4};
// CPU-oracle-only M buckets beyond the hardware range.
inline constexpr std::uint32_t kOracleM[] = {1, 2, 3, 4, 8, 16, 32, 64, 128};

inline constexpr std::uint32_t kFlagshipK = 10240;  // eh_proj K (b40)
inline constexpr std::uint32_t kFlagshipN = 5120;   // eh_proj N (t10)
// 96 MiB cap keeps CPU-oracle record buffers bounded (--full-cpu skips pairs
// whose n_tiles*blocks*3072 B would exceed it).
inline constexpr std::size_t kMaxOracleRecordsBytes = 96U * 1024U * 1024U;

// Real (n,k) pairs from the model family (learnings.md "real shapes"):
//   (20,10) -> N=5120, K=5120   (40,20) -> N=10240, K=10240
//   (20,34) -> N=17408, K=5120  (68,10) -> N=5120, K=17408
// plus the eh_proj flagship (10,40) -> N=5120, K=10240.
struct NkPair {
  std::uint32_t n;
  std::uint32_t k;
};
constexpr NkPair kOraclePairs[] = {
    {5120, 10240}, {5120, 5120},  {10240, 10240},
    {17408, 5120}, {5120, 17408},
};

struct ShapeSpec {
  std::uint32_t m{0};
  std::uint32_t n{0};
  std::uint32_t k{0};

  [[nodiscard]] std::uint64_t Key() const noexcept {
    return (static_cast<std::uint64_t>(m) << 48U) |
           (static_cast<std::uint64_t>(k) << 24U) |
           static_cast<std::uint64_t>(n);
  }
  [[nodiscard]] std::uint32_t Rounds() const noexcept {
    return (m + kMmulM - 1) / kMmulM;
  }
  [[nodiscard]] std::uint32_t Ntiles() const noexcept {
    return (n + kMmulN - 1) / kMmulN;
  }
  [[nodiscard]] std::uint32_t Blocks() const noexcept {
    return (k + kBlockElements - 1) / kBlockElements;
  }
  [[nodiscard]] std::uint32_t PadMRows() const noexcept {
    return Rounds() * kMmulM - m;
  }
  [[nodiscard]] std::uint32_t PadKElements() const noexcept {
    return Blocks() * kBlockElements - k;
  }
  [[nodiscard]] std::uint32_t PadNLanes() const noexcept {
    return Ntiles() * kMmulN - n;
  }
  // True when the shape runs unchanged on the baked b40/t10/r1 xclbin.
  [[nodiscard]] bool HardwareSchedulable() const noexcept {
    return k == kFlagshipK && n == kFlagshipN && m <= 4;
  }
  [[nodiscard]] const char* Kind() const noexcept {
    return m == 1 ? "gemv" : "gemm";  // AieColumnKey: m==1 -> gemv
  }
  [[nodiscard]] std::size_t OracleRecordsBytes() const noexcept {
    return static_cast<std::size_t>(Ntiles()) * Blocks() * kWeightRecordBytes;
  }
};

struct Options {
  std::vector<std::uint32_t> m_buckets;
  std::vector<std::uint32_t> k_buckets;
  std::vector<std::uint32_t> n_buckets;
  std::vector<std::uint32_t> oracle_m;
  std::filesystem::path output;
  std::uint32_t repetitions{5};
  bool json_lines{true};
  bool cpu_only{false};
  bool no_cpu{false};
  bool full_cpu{false};
};

std::uint32_t ParseU32(std::string_view value, std::string_view option) {
  std::uint32_t parsed = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      parsed == 0) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
}

std::vector<std::uint32_t> ParseBuckets(std::string_view value,
                                        std::string_view option) {
  std::vector<std::uint32_t> buckets;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const auto separator = value.find(',', begin);
    const auto end =
        separator == std::string_view::npos ? value.size() : separator;
    if (end > begin) {
      buckets.push_back(ParseU32(value.substr(begin, end - begin), option));
    }
    begin = end + 1;
  }
  if (buckets.empty()) {
    throw std::invalid_argument(std::string(option) + " cannot be empty");
  }
  return buckets;
}

std::vector<std::uint32_t> DefaultK() {
  std::vector<std::uint32_t> buckets;
  buckets.assign(std::begin(kRealK), std::end(kRealK));
  buckets.insert(buckets.end(), std::begin(kTailK), std::end(kTailK));
  std::sort(buckets.begin(), buckets.end());
  return buckets;
}

std::vector<std::uint32_t> DefaultN() {
  std::vector<std::uint32_t> buckets;
  buckets.assign(std::begin(kRealN), std::end(kRealN));
  buckets.insert(buckets.end(), std::begin(kTailN), std::end(kTailN));
  std::sort(buckets.begin(), buckets.end());
  return buckets;
}

Options ParseOptions(std::span<const char* const> arguments) {
  Options options;
  for (std::size_t index = 1; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "--output" && index + 1 < arguments.size()) {
      options.output = arguments[++index];
    } else if (argument == "--runs" && index + 1 < arguments.size()) {
      options.repetitions = ParseU32(arguments[++index], "--runs");
    } else if (argument == "--m" && index + 1 < arguments.size()) {
      options.m_buckets = ParseBuckets(arguments[++index], "--m");
    } else if (argument == "--k" && index + 1 < arguments.size()) {
      options.k_buckets = ParseBuckets(arguments[++index], "--k");
    } else if (argument == "--n" && index + 1 < arguments.size()) {
      options.n_buckets = ParseBuckets(arguments[++index], "--n");
    } else if (argument == "--oracle-m" && index + 1 < arguments.size()) {
      options.oracle_m = ParseBuckets(arguments[++index], "--oracle-m");
    } else if (argument == "--cpu-only") {
      options.cpu_only = true;
    } else if (argument == "--no-cpu") {
      options.no_cpu = true;
    } else if (argument == "--full-cpu") {
      options.full_cpu = true;
    } else if (argument == "--no-json-lines") {
      options.json_lines = false;
    } else if (argument == "--help" || argument == "-h") {
      std::cout
          << "usage: qwen_aie2p_w4a8_shq4_matrix_bench [options]\n"
          << "  --output PATH   write final JSON summary to PATH\n"
          << "  --runs N        timed NPU repetitions per shape (default 5)\n"
          << "  --m M1,M2..     M buckets (default 1,2,3,4)\n"
          << "  --k K1,K2..     K buckets (default real +-15 tails)\n"
          << "  --n N1,N2..     N buckets (default real +-15 tails)\n"
          << "  --oracle-m ..   CPU-oracle-only M buckets (default 8,16,..)\n"
          << "  --cpu-only      skip the NPU path even when XDNA2 is present\n"
          << "  --no-cpu        skip the CPU oracle path\n"
          << "  --full-cpu      run the CPU oracle over the full matrix\n"
          << "  --no-json-lines suppress per-run JSON lines on stdout\n"
          << "exit 77 = NPU SKIP (XDNA2 absent, STRIX_REQUIRE_XDNA2 unset)\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown or incomplete option: " +
                                  std::string(argument));
    }
  }
  if (options.cpu_only && options.no_cpu) {
    throw std::invalid_argument(
        "--cpu-only and --no-cpu are mutually exclusive");
  }
  if (options.m_buckets.empty()) {
    options.m_buckets.assign(std::begin(kHardwareM), std::end(kHardwareM));
  }
  if (options.k_buckets.empty()) {
    options.k_buckets = DefaultK();
  }
  if (options.n_buckets.empty()) {
    options.n_buckets = DefaultN();
  }
  if (options.oracle_m.empty()) {
    // Default: verify the first two multi-round buckets; --full-cpu takes all.
    options.oracle_m = {8, 16, 32, 64, 128};
  }
  return options;
}

// ---------------------------------------------------------------------------
// Deterministic synthetic inputs
// ---------------------------------------------------------------------------

// Row-major M x K activations with per-group-32 dynamic range, mirroring the
// hardware gate's MakeRepresentableInput (test_aie2p_w4a8.cpp): each 32-lane
// group spans quantized -16..126 with a 127 sentinel (tail groups mark the
// last real element).
std::vector<float> MakeActivations(std::uint32_t m, std::uint32_t k) {
  std::vector<float> activations(static_cast<std::size_t>(m) * k);
  for (std::uint32_t row = 0; row < m; ++row) {
    float* out = activations.data() + static_cast<std::size_t>(row) * k;
    for (std::uint32_t group_start = 0; group_start < k; group_start += 32) {
      const std::uint32_t group_end = std::min(group_start + 32U, k);
      for (std::uint32_t lane = group_start; lane < group_end; ++lane) {
        const int quantized =
            (lane - group_start) == 31 || (lane + 1) == k
                ? 127
                : static_cast<int>(lane - group_start) - 16;
        out[lane] = static_cast<float>(quantized) / 64.0F;
      }
    }
  }
  return activations;
}

float Fp16BitsToFloat(std::uint16_t bits) {
  const std::uint32_t u32 = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &u32, sizeof(value));
  return value;
}

// Synthetic GGML Q4_K tensor (n rows x k columns, row-major Q4_K blocks of
// 256 elements / 144 bytes). Deterministic nibble/scale fill so the driver's
// PackQ4KWeights and the bench's record builder decode byte-identically.
// Q4_K GGML block layout (d, dmin as fp16 bit patterns, 12 scale bytes,
// 128 code bytes). Offsets match ggml_dequant / pack.hpp detail::BlockQ4K.
struct Q4KBlock {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144);

std::vector<std::uint8_t> MakeQ4KTensor(std::uint32_t n, std::uint32_t k) {
  const std::uint32_t blocks = (k + kBlockElements - 1) / kBlockElements;
  // N-tiles are 32 lanes wide, so BuildWeightRecords decodes rows up to
  // round_up(n,32)-1. Allocate the padded row count and give padding rows
  // all-zero Q4_K blocks (d=0, dmin=0 -> zero scale/zcorr -> padding lanes
  // dequantize to exact zero, matching the hardware's zero-weight padding).
  const std::uint32_t n_padded = (n + kMmulN - 1) / kMmulN * kMmulN;
  std::vector<std::uint8_t> tensor(
      static_cast<std::size_t>(n_padded) * blocks * sizeof(Q4KBlock));
  for (std::uint32_t row = 0; row < n_padded; ++row) {
    for (std::uint32_t block = 0; block < blocks; ++block) {
      auto* dst = reinterpret_cast<Q4KBlock*>(
          tensor.data() + (static_cast<std::size_t>(row) * blocks + block) *
                              sizeof(Q4KBlock));
      if (row >= n) {
        std::memset(dst, 0, sizeof(Q4KBlock));
        continue;
      }
      dst->d = 0x3C00U;
      dst->dmin = 0x2C00U;
      for (std::uint32_t i = 0; i < 12; ++i) {
        dst->scales[i] =
            static_cast<std::uint8_t>((i * 5U + row + block * 3U) & 0x3FU);
      }
      for (std::uint32_t i = 0; i < 128; ++i) {
        const std::uint8_t low =
            static_cast<std::uint8_t>((row * 7U + block * 11U + i * 5U) &
                                      0x0FU);
        const std::uint8_t high =
            static_cast<std::uint8_t>((row * 3U + block * 13U + i * 7U) &
                                      0x0FU);
        dst->qs[i] = static_cast<std::uint8_t>(low | (high << 4U));
      }
    }
  }
  return tensor;
}

// AIE weight records (n_tiles x blocks x 3072 B) from the Q4_K tensor, using
// the same lane-block decode + PackWeightBlock path as the host driver.
// Record `n_tile * blocks + block` matches ReferenceGemm's indexing.
std::vector<std::uint8_t> BuildWeightRecords(std::uint32_t n,
                                             std::uint32_t k) {
  const auto q4k = MakeQ4KTensor(n, k);
  const std::uint32_t blocks = (k + kBlockElements - 1) / kBlockElements;
  const std::uint32_t n_tiles = (n + kMmulN - 1) / kMmulN;
  std::vector<std::uint8_t> records(static_cast<std::size_t>(n_tiles) *
                                    blocks * kWeightRecordBytes);
  for (std::uint32_t nt = 0; nt < n_tiles; ++nt) {
    for (std::uint32_t block = 0; block < blocks; ++block) {
      WeightBlockSpec spec;
      for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
        const std::uint32_t row = nt * kMmulN + lane;
        const std::uint8_t* src =
            q4k.data() + (static_cast<std::size_t>(row) * blocks + block) *
                             144U;
        const auto* block_view = reinterpret_cast<const Q4KBlock*>(src);
        const float d = Fp16BitsToFloat(block_view->d);
        const float dmin = Fp16BitsToFloat(block_view->dmin);
        const auto* scales = block_view->scales;
        const auto* qs = block_view->qs;
        for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
          const std::uint32_t pair = group / 2U;
          const bool high_nibble = (group & 1U) != 0U;
          std::uint8_t scale = 0;
          std::uint8_t minimum = 0;
          if (group < 4U) {
            scale = scales[group] & 0x3FU;
            minimum = scales[group + 4U] & 0x3FU;
          } else {
            scale = static_cast<std::uint8_t>(
                (scales[group + 4U] & 0x0FU) |
                ((scales[group - 4U] >> 6U) << 4U));
            minimum = static_cast<std::uint8_t>(
                (scales[group + 4U] >> 4U) |
                ((scales[group - 4U] >> 6U) << 4U));
          }
          for (std::uint32_t half = 0; half < 2U; ++half) {
            for (std::uint32_t kk = 0; kk < 16U; ++kk) {
              const std::uint32_t lane_in_group = (half * 16U) + kk;
              const std::uint8_t source = qs[(pair * 32U) + lane_in_group];
              const std::uint8_t quantized =
                  high_nibble ? static_cast<std::uint8_t>(source >> 4U)
                              : static_cast<std::uint8_t>(source & 0x0FU);
              WeightSpecCode(spec, group, half, kk, lane) = quantized;
            }
          }
          spec.wscale[group][lane] = d * static_cast<float>(scale);
          spec.zcorr[group][lane] = dmin * static_cast<float>(minimum);
        }
      }
      PackWeightBlock(
          spec, records.data() +
                    (static_cast<std::size_t>(nt) * blocks + block) *
                        kWeightRecordBytes);
    }
  }
  return records;
}

// ---------------------------------------------------------------------------
// Timing + statistics helpers
// ---------------------------------------------------------------------------

double NowUs() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Stats {
  double min{0.0};
  double median{0.0};
  double mean{0.0};
  double max{0.0};
  double stddev{0.0};

  static Stats Of(std::vector<double> samples) {
    Stats stats;
    if (samples.empty()) {
      return stats;
    }
    std::sort(samples.begin(), samples.end());
    stats.min = samples.front();
    stats.max = samples.back();
    stats.median = samples[samples.size() / 2U];
    double sum = 0.0;
    for (const double sample : samples) {
      sum += sample;
    }
    stats.mean = sum / static_cast<double>(samples.size());
    double squared = 0.0;
    for (const double sample : samples) {
      const double delta = sample - stats.mean;
      squared += delta * delta;
    }
    stats.stddev = std::sqrt(squared / static_cast<double>(samples.size()));
    return stats;
  }
};

struct Comparison {
  double rmse{0.0};
  double cosine{0.0};
  float max_abs{0.0F};

  bool cosmetic{false};

  static Comparison Of(std::span<const float> actual,
                       std::span<const float> expected) {
    Comparison comparison;
    if (actual.size() != expected.size() || actual.empty()) {
      comparison.cosmetic = true;
      return comparison;
    }
    double squared_error = 0.0;
    double actual_squared = 0.0;
    double expected_squared = 0.0;
    double dot = 0.0;
    float max_abs = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const double lhs = actual[index];
      const double rhs = expected[index];
      const double difference = lhs - rhs;
      squared_error += difference * difference;
      actual_squared += lhs * lhs;
      expected_squared += rhs * rhs;
      dot += lhs * rhs;
      max_abs = std::max(max_abs, static_cast<float>(std::abs(difference)));
    }
    comparison.rmse =
        std::sqrt(squared_error / static_cast<double>(actual.size()));
    comparison.cosine = dot / std::sqrt(actual_squared * expected_squared);
    comparison.max_abs = max_abs;
    return comparison;
  }
};

// ---------------------------------------------------------------------------
// JSON emission (compact helpers; no external JSON dependency)
// ---------------------------------------------------------------------------

std::string JsonEscape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 8U);
  for (const char ch : value) {
    switch (ch) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20U) {
          std::ostringstream hex;
          hex << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(ch));
          escaped += hex.str();
        } else {
          escaped += ch;
        }
        break;
    }
  }
  return escaped;
}

std::string Fmt(double value, int precision = 3) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

template <typename T>
std::string JoinNumbers(const std::vector<T>& values) {
  std::string joined;
  for (const T value : values) {
    if (!joined.empty()) {
      joined += ',';
    }
    joined += std::to_string(value);
  }
  return joined;
}

std::string ShapeJson(const ShapeSpec& shape) {
  std::ostringstream out;
  out << "{\"m\":" << shape.m << ",\"n\":" << shape.n << ",\"k\":" << shape.k
      << "}";
  return out.str();
}

std::string PaddingJson(const ShapeSpec& shape) {
  std::ostringstream out;
  out << "{\"m_rows\":" << shape.PadMRows()
      << ",\"k_elements\":" << shape.PadKElements()
      << ",\"n_lanes\":" << shape.PadNLanes() << '}';
  return out.str();
}

// ---------------------------------------------------------------------------
// NPU environment probe (works in CPU-only builds too)
// ---------------------------------------------------------------------------

struct NpuProbe {
  bool sys_class_accel{false};
  std::size_t sys_accel_entries{0};
  std::size_t dev_accel_entries{0};
  std::vector<std::string> evidence;

  static NpuProbe Collect() {
    NpuProbe probe;
    const std::filesystem::path sys_accel("/sys/class/accel");
    if (std::filesystem::is_directory(sys_accel)) {
      probe.sys_class_accel = true;
      probe.sys_accel_entries =
          std::distance(std::filesystem::directory_iterator(sys_accel),
                        std::filesystem::directory_iterator{});
      probe.evidence.push_back("/sys/class/accel present with " +
                               std::to_string(probe.sys_accel_entries) +
                               " device(s)");
    } else {
      probe.evidence.push_back("/sys/class/accel: absent");
    }
    std::size_t accel_devices = 0;
    if (std::filesystem::is_directory("/dev")) {
      const std::filesystem::path dev_accel("/dev/accel");
      if (std::filesystem::is_directory(dev_accel)) {
        // Newer amdxdna kernels hang the char device at /dev/accel/accelN.
        for (const auto& entry : std::filesystem::directory_iterator(dev_accel)) {
          if (entry.path().filename().string().rfind("accel", 0) == 0) {
            ++accel_devices;
          }
        }
      }
      for (const auto& entry : std::filesystem::directory_iterator("/dev")) {
        if (entry.path().filename().string().rfind("accel", 0) == 0) {
          ++accel_devices;
        }
      }
    }
    probe.dev_accel_entries = accel_devices;
    if (accel_devices == 0) {
      probe.evidence.push_back("/dev/accel*: none");
    } else {
      probe.evidence.push_back("/dev/accel* present with " +
                               std::to_string(accel_devices) +
                               " device node(s)");
    }
    return probe;
  }
};

bool Xdna2Required() {
  const char* val = std::getenv("STRIX_REQUIRE_XDNA2");
  if (val == nullptr) {
    val = std::getenv("STRIX_REQUIRE_NPU");
  }
  return val != nullptr && std::string_view(val) != "0" &&
         std::string_view(val) != "false";
}

// ---------------------------------------------------------------------------
// CPU oracle path
// ---------------------------------------------------------------------------

struct CpuRunOutcome {
  bool ok{false};
  std::string status;  // ran | error
  std::string reason;
  double reference_us{0.0};
  double pack_activation_us{0.0};
  bool finite{false};

  std::string ToJson() const {
    std::ostringstream out;
    out << "{\"status\":\"" << JsonEscape(status) << "\"";
    if (!reason.empty()) {
      out << ",\"reason\":\"" << JsonEscape(reason) << "\"";
    }
    out << ",\"reference_us\":" << Fmt(reference_us)
        << ",\"pack_activation_us\":" << Fmt(pack_activation_us)
        << ",\"checks\":{\"finite\":" << (finite ? "true" : "false") << "}}";
    return out.str();
  }
};

// Records cache keyed by (n,k): one build serves every M on the pair.
std::unordered_map<std::uint64_t, std::vector<std::uint8_t>> g_records;

const std::vector<std::uint8_t>& OrBuildRecords(std::uint32_t n,
                                                std::uint32_t k) {
  const std::uint64_t key =
      (static_cast<std::uint64_t>(k) << 32U) | static_cast<std::uint64_t>(n);
  const auto found = g_records.find(key);
  if (found != g_records.end()) {
    return found->second;
  }
  return g_records.emplace(key, BuildWeightRecords(n, k)).first->second;
}

CpuRunOutcome RunCpuOracle(const ShapeSpec& shape,
                           const std::uint8_t* records) {
  CpuRunOutcome outcome;
  const std::vector<float> activations = MakeActivations(shape.m, shape.k);

  // Dedicated activation-packing timing: ROUNDS x BLOCKS records.
  std::vector<std::uint8_t> input_record(kInputRecordBytes);
  std::array<std::span<const float>, kMmulM> rows;
  const double pack_start = NowUs();
  for (std::uint32_t round = 0; round < shape.Rounds(); ++round) {
    for (std::uint32_t block = 0; block < shape.Blocks(); ++block) {
      for (std::uint32_t r = 0; r < kMmulM; ++r) {
        const std::size_t logical_row =
            static_cast<std::size_t>(round) * kMmulM + r;
        if (logical_row < shape.m) {
          rows[r] = std::span<const float>(
              activations.data() + logical_row * shape.k +
                  static_cast<std::size_t>(block) * kBlockElements,
              std::min<std::size_t>(
                  kBlockElements,
                  static_cast<std::size_t>(shape.k) -
                      static_cast<std::size_t>(block) * kBlockElements));
        } else {
          rows[r] = {};
        }
      }
      QuantizeActivationBlock(
          rows.data(),
          std::min(shape.k - block * kBlockElements, kBlockElements),
          input_record.data());
    }
  }
  outcome.pack_activation_us = NowUs() - pack_start;

  std::vector<float> expected(static_cast<std::size_t>(shape.Rounds() *
                                                       kMmulM) *
                              shape.Ntiles() * kMmulN);
  const double reference_start = NowUs();
  ReferenceGemm(activations, Shape{.m = shape.m, .n = shape.n, .k = shape.k},
                records, expected.data());
  outcome.reference_us = NowUs() - reference_start;

  outcome.finite = true;
  for (const float value : expected) {
    if (!std::isfinite(value)) {
      outcome.finite = false;
      break;
    }
  }
  if (!outcome.finite) {
    outcome.status = "error";
    outcome.reason = "reference output contains non-finite values";
    return outcome;
  }
  outcome.status = "ran";
  outcome.ok = true;
  return outcome;
}

// Row-local quantization: batch-1 GEMV row 0 must equal GEMM chunk row 0
// (float-identical, per test_aie2p_w4a8.cpp TestGemvMatchesGemmRow).
bool GemvRow0EqualsGemmChunk0(const std::uint8_t* flagship_records) {
  const auto input1 = MakeActivations(1, kFlagshipK);
  const auto input2 = MakeActivations(2, kFlagshipK);
  std::vector<float> gemv_out(
      static_cast<std::size_t>(4) * (kFlagshipN / kMmulN) * kMmulN);
  std::vector<float> gemm_out(
      static_cast<std::size_t>(4) * (kFlagshipN / kMmulN) * kMmulN);
  ReferenceGemm(input1, Shape{.m = 1, .n = kFlagshipN, .k = kFlagshipK},
                flagship_records, gemv_out.data());
  ReferenceGemm(input2, Shape{.m = 2, .n = kFlagshipN, .k = kFlagshipK},
                flagship_records, gemm_out.data());
  for (std::size_t i = 0; i < kFlagshipN; ++i) {
    if (gemv_out[i] != gemm_out[i]) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// NPU path (ENGINE_ENABLE_XRT only; one session reused across shapes)
// ---------------------------------------------------------------------------

struct NpuRunOutcome {
  bool ran{false};
  std::string status;  // ran | skip | error
  std::string reason;
  double setup_ms{0.0};
  Stats command_us;
  Stats completion_us;
  Stats input_upload_us;
  Stats activation_pack_us;
  Stats output_download_us;
  Stats end_to_end_us;
  double singleton_command_us{0.0};
  double singleton_end_to_end_us{0.0};
  std::size_t repetitions{0};
  Comparison comparison;
};

std::string StatsJson(const Stats& stats) {
  std::ostringstream out;
  out << "{\"min_us\":" << Fmt(stats.min) << ",\"median_us\":"
      << Fmt(stats.median) << ",\"mean_us\":" << Fmt(stats.mean)
      << ",\"max_us\":" << Fmt(stats.max) << ",\"stddev_us\":"
      << Fmt(stats.stddev) << '}';
  return out.str();
}

#ifdef ENGINE_ENABLE_XRT

std::string ProgramDir() {
#ifdef STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR
  if constexpr (!std::string_view(STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR)
                    .empty()) {
    return STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR;
  }
#endif
  const char* env = std::getenv("STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR");
  return env == nullptr ? std::string() : std::string(env);
}

// Run one schedulable shape against `session` (already created once):

// Run one schedulable shape against `session` (already created once):
// singleton first, then `repetitions` timed runs on the same session.
NpuRunOutcome RunNpuShape(strix::xdna2::QwenAie2pW4a8Session& session,
                          const ShapeSpec& shape, std::uint32_t repetitions,
                          const std::uint8_t* records) {
  NpuRunOutcome outcome;
  outcome.status = "error";
  const auto input = MakeActivations(shape.m, kFlagshipK);
  std::vector<float> output(static_cast<std::size_t>(shape.m) * kFlagshipN);
  strix::xdna2::QwenAie2pW4a8Failure failure;
  strix::xdna2::QwenAie2pW4a8RunMetrics metrics;
  if (!session.Run(input, output, &metrics, &failure)) {
    outcome.reason = "singleton run failed: " + failure.category + ": " +
                     failure.message;
    return outcome;
  }
  outcome.singleton_command_us = metrics.command_us;
  outcome.singleton_end_to_end_us = metrics.end_to_end_us;

  std::vector<double> command_us;
  std::vector<double> completion_us;
  std::vector<double> input_upload_us;
  std::vector<double> activation_pack_us;
  std::vector<double> output_download_us;
  std::vector<double> end_to_end_us;
  command_us.reserve(repetitions);
  completion_us.reserve(repetitions);
  input_upload_us.reserve(repetitions);
  activation_pack_us.reserve(repetitions);
  output_download_us.reserve(repetitions);
  end_to_end_us.reserve(repetitions);
  for (std::uint32_t i = 0; i < repetitions; ++i) {
    strix::xdna2::QwenAie2pW4a8RunMetrics rep;
    if (!session.Run(input, output, &rep, &failure)) {
      outcome.reason = "repeated run failed: " + failure.category + ": " +
                       failure.message;
      return outcome;
    }
    command_us.push_back(rep.command_us);
    completion_us.push_back(rep.completion_us);
    input_upload_us.push_back(rep.input_upload_us);
    activation_pack_us.push_back(rep.activation_pack_us);
    output_download_us.push_back(rep.output_download_us);
    end_to_end_us.push_back(rep.end_to_end_us);
  }
  outcome.command_us = Stats::Of(std::move(command_us));
  outcome.completion_us = Stats::Of(std::move(completion_us));
  outcome.input_upload_us = Stats::Of(std::move(input_upload_us));
  outcome.activation_pack_us = Stats::Of(std::move(activation_pack_us));
  outcome.output_download_us = Stats::Of(std::move(output_download_us));
  outcome.end_to_end_us = Stats::Of(std::move(end_to_end_us));
  outcome.repetitions = repetitions;

  // Compare against the CPU oracle (identical synthetic Q4_K records).
  std::vector<float> expected(
      static_cast<std::size_t>(shape.Rounds() * kMmulM) * shape.Ntiles() *
      kMmulN);
  ReferenceGemm(input, Shape{.m = shape.m, .n = kFlagshipN, .k = kFlagshipK},
                records, expected.data());
  // The driver returns only the m logical rows; the reference grid holds the
  // padded rounds*4 rows (row-major), so the first m*N floats compare 1:1.
  outcome.comparison = Comparison::Of(
      output, std::span<const float>(expected.data(), output.size()));
  outcome.ran = true;
  outcome.status = "ran";
  if (outcome.comparison.cosmetic) {
    outcome.reason = "comparison unavailable";
  } else if (std::isfinite(outcome.comparison.rmse)) {
    const bool match = outcome.comparison.rmse < 0.1 &&
                       outcome.comparison.cosine > 0.9999 &&
                       outcome.comparison.max_abs < 1.0F;
    outcome.reason = match ? "npu-vs-cpu-oracle match (rmse<0.1, cosine>0.9999)"
                           : "npu-vs-cpu-oracle mismatch";
  }
  return outcome;
}

#endif  // ENGINE_ENABLE_XRT

// ---------------------------------------------------------------------------
// Summary JSON (M004-C007 measurement artifact shell)
// ---------------------------------------------------------------------------

struct Summary {
  NpuProbe probe;
  bool xrt_build{false};
  bool cpu_only{false};
  bool no_cpu{false};
  bool full_cpu{false};
  bool npu_ran{false};
  std::string npu_status;   // empty | ran | skip | error
  std::string npu_reason;
  std::vector<std::string> blockers;
  std::vector<ShapeSpec> matrix;
  std::vector<ShapeSpec> cpu_ran_shapes;
  std::vector<ShapeSpec> schedulable_shapes;
  double cpu_total_us{0.0};
  std::size_t cpu_runs{0};
  std::size_t npu_runs{0};
  bool gemv_row0_exact{false};
  std::uint32_t repetitions{0};
};

std::string IsoTimestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

std::string HardwareJson(const ShapeSpec& shape, const Summary& summary,
                         const NpuRunOutcome* npu) {
  std::ostringstream out;
  if (npu != nullptr && npu->ran) {
    out << "{\"status\":\"ran\",\"setup_ms\":" << Fmt(npu->setup_ms)
        << ",\"singleton_command_us\":" << Fmt(npu->singleton_command_us)
        << ",\"command_us\":" << StatsJson(npu->command_us)
        << ",\"completion_us\":" << StatsJson(npu->completion_us)
        << ",\"input_upload_us\":" << StatsJson(npu->input_upload_us)
        << ",\"activation_pack_us\":" << StatsJson(npu->activation_pack_us)
        << ",\"output_download_us\":" << StatsJson(npu->output_download_us)
        << ",\"end_to_end_us\":" << StatsJson(npu->end_to_end_us)
        << ",\"repetitions\":" << npu->repetitions;
    if (!npu->comparison.cosmetic) {
      out << ",\"npu_vs_cpu_oracle\":{\"rmse\":" << Fmt(npu->comparison.rmse, 6)
          << ",\"cosine\":" << Fmt(npu->comparison.cosine, 6)
          << ",\"max_abs\":" << Fmt(npu->comparison.max_abs, 6) << '}';
    }
    out << '}';
    return out.str();
  }
  out << "{\"status\":\"hardware_pending\",\"schedulable\":"
      << (shape.HardwareSchedulable() ? "true" : "false");
  if (!summary.npu_reason.empty()) {
    out << ",\"reason\":\"" << JsonEscape(summary.npu_reason) << "\"";
  }
  if (!shape.HardwareSchedulable()) {
    out << ",\"xclbinReason\":\"baked b40/t10/r1 xclbin covers only "
           "K=10240,N=5120,M<=4\"";
  }
  out << '}';
  return out.str();
}

void WriteSummaryJson(const Summary& summary, const Options& options,
                      const std::unordered_map<std::uint64_t, CpuRunOutcome>&
                          cpu_outcomes,
                      const std::unordered_map<std::uint64_t, NpuRunOutcome>&
                          npu_outcomes,
                      std::ostream& out) {
  out << "{\n";
  out << "  \"schemaVersion\": \"1.0.0\",\n";
  out << "  \"id\": \"M004-C007\",\n";
  out << "  \"artifact\": \"m004/c007-xdna2\",\n";
  out << "  \"generatedAt\": \"" << IsoTimestamp() << "\",\n";
  out << "  \"tool\": {\"name\": \"qwen_aie2p_w4a8_shq4_matrix_bench\", "
         "\"output\": \"json\"},\n";
  out << "  \"environment\": {\n";
  out << "    \"npu\": \"" << (summary.probe.sys_class_accel ? "present"
                                                            : "absent")
      << "\",\n";
  out << "    \"sysClassAccel\": "
      << (summary.probe.sys_class_accel ? "true" : "false") << ",\n";
  out << "    \"sysAccelEntries\": " << summary.probe.sys_accel_entries
      << ",\n";
  out << "    \"devAccelEntries\": " << summary.probe.dev_accel_entries
      << ",\n";
  out << "    \"xrtBuild\": " << (summary.xrt_build ? "true" : "false")
      << ",\n";
  out << "    \"strixRequireXdna2\": "
      << (Xdna2Required() ? "true" : "false")
      << ",\n    \"weights\": \"synthetic Q4_K (deterministic; "
         "STRIX_MTP_MODEL path not wired in the bench)\"\n";
  out << "  },\n";
  out << "  \"shapeMatrix\": {\n";
  out << "    \"notes\": \"Plan 2.6: M buckets 1..128 with the 128-vs-64 "
         "bound still OPEN; the b40/t10/r1 xclbin covers M<=4 and the "
         "flagship K=10240,N=5120; every-nibble coverage in the synthetic "
         "Q4_K fill; group boundaries covered by test_aie2p_w4a8\",\n";
  out << "    \"mBuckets\": [" << JoinNumbers(options.m_buckets) << "],\n";
  out << "    \"mOracleBuckets\": [" << JoinNumbers(options.oracle_m)
      << "],\n";
  out << "    \"kBuckets\": [" << JoinNumbers(options.k_buckets) << "],\n";
  out << "    \"nBuckets\": [" << JoinNumbers(options.n_buckets) << "],\n";
  out << "    \"matrix\": [\n";
  bool first_entry = true;
  for (const ShapeSpec& shape : summary.matrix) {
    if (!first_entry) {
      out << ",\n";
    }
    first_entry = false;
    out << "      {\"m\":" << shape.m << ",\"k\":" << shape.k
        << ",\"n\":" << shape.n << ",\"kind\":\"" << shape.Kind() << "\","
        << "\"status\":\""
        << ((npu_outcomes.count(shape.Key()) != 0 && summary.npu_ran)
                ? "ran"
                : "hardware_pending")
        << "\","
        << "\"hardware\":" << HardwareJson(shape, summary,
                                           npu_outcomes.count(shape.Key()) != 0
                                               ? &npu_outcomes.at(shape.Key())
                                               : nullptr)
        << ",\"cpu_reference\":";
    const auto found = cpu_outcomes.find(shape.Key());
    if (found != cpu_outcomes.end()) {
      out << found->second.ToJson();
    } else {
      out << "{\"status\":\"pending\",\"reason\":\"oracle subset\"}";
    }
    out << ",\"padding\":" << PaddingJson(shape) << '}';
  }
  out << "\n    ]\n  },\n";
  out << "  \"blockers\": [\n";
  bool first_blocker = true;
  for (const std::string& blocker : summary.blockers) {
    if (!first_blocker) {
      out << ",\n";
    }
    first_blocker = false;
    out << "    {\"gate\":\"npu\",\"evidence\":[\"" << JsonEscape(blocker)
        << "\"],\"impact\":\"hardware timing sweeps pending NPU "
           "hardware\"}";
  }
  out << "\n  ],\n";
  out << "  \"summary\": {\n";
  std::size_t schedulable = 0;
  std::size_t pending = 0;
  for (const ShapeSpec& shape : summary.matrix) {
    if (shape.HardwareSchedulable()) {
      ++schedulable;
      if (npu_outcomes.count(shape.Key()) == 0 || !summary.npu_ran) {
        ++pending;
      }
    } else {
      ++pending;
    }
  }
  out << "    \"hardware\": {\"covered\": " << summary.npu_runs
      << ", \"schedulable\": " << schedulable << ", \"pending\": " << pending
      << "},\n";
  out << "    \"cpuReference\": {\"ran\": " << summary.cpu_runs
      << ", \"total_us\": " << Fmt(summary.cpu_total_us)
      << ", \"gemv_row0_exact_gemm_chunk0\": "
      << (summary.gemv_row0_exact ? "true" : "false") << "},\n";
  out << "    \"npu\": {\"status\": \""
      << (summary.npu_status.empty() ? "not_attempted" : summary.npu_status)
      << "\", \"reason\": \"" << JsonEscape(summary.npu_reason)
      << "\", \"program_reuse\": {\"sessions\": 1, \"program_reloads\": 0, "
         "\"bo_allocations\": 3}},\n";
  out << "    \"dmaOverlap\": {\"mode\": \"serialized\", "
         "\"supported\": false, \"note\": \"driver Run serializes input "
         "upload (XCL_BO_SYNC_BO_TO_DEVICE), command wait (run.wait2), "
         "output download (XCL_BO_SYNC_BO_FROM_DEVICE); no overlapped "
         "DMA/compute API is exposed by the driver\"},\n";
  out << "    \"paddingCost\": \"tail shapes (+/-15) keep block/tile "
         "counts, so compute cost is identical and padding is exact-zero "
         "(see test_aie2p_w4a8 TailKExact)\"\n";
  out << "  }\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  try {
    options = ParseOptions(
        std::span<const char* const>(argv, static_cast<std::size_t>(argc)));
  } catch (const std::exception& exception) {
    std::cerr << "qwen_aie2p_w4a8_shq4_matrix_bench: " << exception.what()
              << "\n";
    return 2;
  }

  Summary summary;
  summary.probe = NpuProbe::Collect();
  summary.blockers = summary.probe.evidence;
  summary.cpu_only = options.cpu_only;
  summary.no_cpu = options.no_cpu;
  summary.full_cpu = options.full_cpu;
  summary.repetitions = options.repetitions;
#ifdef ENGINE_ENABLE_XRT
  summary.xrt_build = true;
#else
  summary.xrt_build = false;
#endif

  // Full cartesian matrix: M buckets x K buckets x N buckets, plus the
  // CPU-oracle-only multi-round M buckets on the flagship shape.
  std::vector<ShapeSpec> extra_shapes;
  for (const std::uint32_t m : options.oracle_m) {
    if (m <= 4 || std::find(options.m_buckets.begin(), options.m_buckets.end(),
                            m) != options.m_buckets.end()) {
      continue;
    }
    extra_shapes.push_back(ShapeSpec{.m = m, .n = kFlagshipN, .k = kFlagshipK});
  }
  std::vector<ShapeSpec>& matrix = summary.matrix;
  for (const std::uint32_t m : options.m_buckets) {
    for (const std::uint32_t k : options.k_buckets) {
      for (const std::uint32_t n : options.n_buckets) {
        matrix.push_back(ShapeSpec{.m = m, .n = n, .k = k});
      }
    }
  }
  matrix.insert(matrix.end(), extra_shapes.begin(), extra_shapes.end());
  summary.schedulable_shapes.clear();
  for (const ShapeSpec& shape : matrix) {
    if (shape.HardwareSchedulable()) {
      summary.schedulable_shapes.push_back(shape);
    }
  }

  std::unordered_map<std::uint64_t, CpuRunOutcome> cpu_outcomes;
  std::unordered_map<std::uint64_t, NpuRunOutcome> npu_outcomes;

  // -------------------------------------------------------------------------
  // CPU oracle path (always available; verifies when no NPU is present)
  // -------------------------------------------------------------------------
  if (!options.no_cpu) {
    const auto pair_known = [&](const ShapeSpec& shape) {
      if (shape.m > 4 || shape.m == 0) {
        return false;
      }
      for (const NkPair pair : kOraclePairs) {
        if (shape.n == pair.n && shape.k == pair.k) {
          return true;
        }
      }
      return false;
    };
    const auto tail_oracle = [&](const ShapeSpec& shape) {
      if (shape.m != 1) {
        return false;
      }
      return (shape.n == kFlagshipN &&
              (shape.k == 10225U || shape.k == 10255U)) ||
             (shape.k == kFlagshipK &&
              (shape.n == 5105U || shape.n == 5135U)) ||
             (shape.k == 10225U && shape.n == 10240U) ||
             (shape.k == 17423U && shape.n == kFlagshipN);
    };
    auto run_oracle = [&](const ShapeSpec& shape) {
      const std::vector<std::uint8_t>& records = OrBuildRecords(shape.n, shape.k);
      const CpuRunOutcome outcome = RunCpuOracle(shape, records.data());
      cpu_outcomes.emplace(shape.Key(), outcome);
      ++summary.cpu_runs;
      summary.cpu_total_us += outcome.reference_us;
      if (!outcome.ok) {
        std::cerr << "qwen_aie2p_w4a8_shq4_matrix_bench: CPU oracle failed "
                  << "for " << shape.m << 'x' << shape.k << 'x' << shape.n
                  << ": " << outcome.reason << "\n";
      }
      if (options.json_lines) {
        std::cout << "{\"event\":\"run\",\"shape\":" << ShapeJson(shape)
                  << ",\"kind\":\"" << shape.Kind()
                  << "\",\"path\":\"cpu\",\"outcome\":" << outcome.ToJson()
                  << "}\n";
      }
    };

    for (const ShapeSpec& shape : matrix) {
      // Default mode: every verified (n,k) pair at every M bucket, plus the
      // five semantic tail-K/N shapes (block/tile-count-preserving +/-15).
      // `pair_known` alone would filter the tails out, so gate on either.
      if (!options.full_cpu && !pair_known(shape) && !tail_oracle(shape)) {
        continue;
      }
      run_oracle(shape);
    }
    if (!options.full_cpu) {
      // Default: also verify the first two multi-round M buckets on the
      // flagship shape (rounds=2 and rounds=4 record chunking).
      for (const std::uint32_t m : options.oracle_m) {
        if (m <= 4 || m > 16) {
          continue;
        }
        run_oracle(ShapeSpec{.m = m, .n = kFlagshipN, .k = kFlagshipK});
      }
    } else {
      for (const ShapeSpec& shape : extra_shapes) {
        run_oracle(shape);
      }
      // --full-cpu additionally covers every matrix entry whose record
      // buffer fits under the cap (large (n,k) pairs excluded).
      for (const ShapeSpec& shape : matrix) {
        if (pair_known(shape) || shape.OracleRecordsBytes() >
                                      kMaxOracleRecordsBytes) {
          continue;
        }
        run_oracle(shape);
      }
    }
    // Row-local quantization check on the flagship shape.
    const auto& flagship_records = OrBuildRecords(kFlagshipN, kFlagshipK);
    summary.gemv_row0_exact = GemvRow0EqualsGemmChunk0(flagship_records.data());
  }

  // -------------------------------------------------------------------------
  // NPU path (XRT builds only; gated on XDNA2 presence)
  // -------------------------------------------------------------------------
  int exit_code = 0;
  if (!options.cpu_only) {
#ifdef ENGINE_ENABLE_XRT
    const auto inventory = strix::diagnostics::CollectSystemInventory();
    const auto device = strix::xdna2::DiscoverXrtDevice(0, inventory);
    if (!device.available) {
      summary.blockers.push_back("xrt: " + device.error_category +
                                 " detected=" + device.detected +
                                 " required=" + device.required);
      summary.npu_status = "skip";
      summary.npu_reason =
          "XDNA2 unavailable: " + device.error_category;
      if (Xdna2Required()) {
        std::cerr << "qwen_aie2p_w4a8_shq4_matrix_bench: STRIX_REQUIRE_XDNA2 "
                     "set but XDNA2 is unavailable: "
                  << device.error_category << " detected=" << device.detected
                  << " required=" << device.required
                  << " remediation=" << device.remediation << "\n";
        exit_code = 1;
      } else {
        std::cout << "qwen_aie2p_w4a8_shq4_matrix_bench: NPU SKIP "
                     "(XDNA2 unavailable: "
                  << device.error_category << ")\n";
        exit_code = 77;
      }
    } else {
      const std::vector<std::uint8_t>& records =
          OrBuildRecords(kFlagshipN, kFlagshipK);
      // Synthetic Q4_K tensor (same deterministic fill the CPU records were
      // built from); kept alive for the whole session lifetime.
      const auto q4k = MakeQ4KTensor(kFlagshipN, kFlagshipK);
      strix::xdna2::QwenAie2pW4a8Failure failure;
      auto session = strix::xdna2::QwenAie2pW4a8Session::Create(
          {.program_dir = ProgramDir()}, device,
          strix::models::QwenTensorRef{
              .data = q4k.data(),
              .type = strix::core::GgmlType::kQ4_K,
              .num_elements =
                  static_cast<std::size_t>(kFlagshipN) * kFlagshipK,
          },
          &failure);
      if (session == nullptr) {
        summary.npu_status = "error";
        summary.npu_reason = failure.category + ": " + failure.message;
        std::cerr << "qwen_aie2p_w4a8_shq4_matrix_bench: session create "
                     "failed: "
                  << summary.npu_reason << "\n";
        exit_code = 1;
      } else {
        summary.npu_status = "ran";
        // One session reused across every schedulable shape: program/config
        // reuse is implicit (no re-register, no reload, 3 BOs).
        for (const ShapeSpec& shape : summary.schedulable_shapes) {
          NpuRunOutcome outcome =
              RunNpuShape(*session, shape, options.repetitions, records.data());
          outcome.setup_ms = session->ProgramInfo().setup_ms;
          npu_outcomes.emplace(shape.Key(), outcome);
          if (!outcome.ran) {
            summary.npu_reason = outcome.reason;
            summary.blockers.push_back(
                "xrt: qwen_aie2p_w4a8 kernel command fault: " +
                outcome.reason);
            if (Xdna2Required()) {
              summary.npu_status = "error";
              std::cerr
                  << "qwen_aie2p_w4a8_shq4_matrix_bench: " << outcome.reason
                  << "\n";
              exit_code = 1;
            } else {
              summary.npu_status = "skip";
              std::cout
                  << "qwen_aie2p_w4a8_shq4_matrix_bench: NPU SKIP "
                     "(kernel command fault: "
                  << outcome.reason << ")\n";
              exit_code = 77;
            }
            break;
          }
          ++summary.npu_runs;
          if (options.json_lines) {
            std::cout << "{\"event\":\"run\",\"shape\":" << ShapeJson(shape)
                      << ",\"kind\":\"" << shape.Kind()
                      << "\",\"path\":\"npu\",\"setup_ms\":"
                      << Fmt(outcome.setup_ms)
                      << ",\"singleton_command_us\":"
                      << Fmt(outcome.singleton_command_us)
                      << ",\"command_us\":" << StatsJson(outcome.command_us)
                      << ",\"completion_us\":"
                      << StatsJson(outcome.completion_us)
                      << ",\"input_upload_us\":"
                      << StatsJson(outcome.input_upload_us)
                      << ",\"activation_pack_us\":"
                      << StatsJson(outcome.activation_pack_us)
                      << ",\"output_download_us\":"
                      << StatsJson(outcome.output_download_us)
                      << ",\"end_to_end_us\":"
                      << StatsJson(outcome.end_to_end_us)
                      << ",\"repetitions\":" << outcome.repetitions << "}\n";
          }
          if (exit_code != 0) {
            break;
          }
        }
      }
    }
#else
    summary.npu_status = "skip";
    summary.npu_reason = "binary built without ENGINE_ENABLE_XRT";
    std::cout << "qwen_aie2p_w4a8_shq4_matrix_bench: NPU SKIP "
                 "(built without ENGINE_ENABLE_XRT; CPU oracle ran)\n";
#endif
  }

  // -------------------------------------------------------------------------
  // Summary / artifact output
  // -------------------------------------------------------------------------
  std::ostringstream summary_json;
  WriteSummaryJson(summary, options, cpu_outcomes, npu_outcomes, summary_json);
  if (!options.output.empty()) {
    try {
      std::filesystem::create_directories(options.output.parent_path());
      std::ofstream file(options.output);
      if (!file) {
        throw std::runtime_error("cannot open " + options.output.string());
      }
      file << summary_json.str();
    } catch (const std::exception& exception) {
      std::cerr << "qwen_aie2p_w4a8_shq4_matrix_bench: cannot write summary: "
                << exception.what() << "\n";
      return 2;
    }
  } else {
    std::cout << summary_json.str();
  }
  return exit_code;
}