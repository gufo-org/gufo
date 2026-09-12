// opt-q4kxl: equivalence tests for every quantization format the Unsloth
// Qwen3.8-27B UD-Q4_K_XL shard uses.
//
// The shard mixes Q5_K, IQ4_XS, Q4_K, Q6_K, IQ4_NL, Q3_K and Q8_0, and all of
// them are now decoded on the GPU straight from their packed form. Two device
// routes consume that decode and both are checked here against the CPU oracles
// in ggml_dequant.cpp:
//
//   * the decode GEMV (LaunchQ8KBlockGEMV -> QuantWarpBlockDot), exact fp32
//   * the prefill WMMA GEMM (LaunchBatchedQuantGEMMPreQuantized ->
//     WKQuantA8BlockedWmmaGEMMKernel), which additionally quantizes the
//     activation to Q8_1, so it is checked against a CPU model of the same
//     W-quant x Q8-activation arithmetic rather than against exact fp32.
//
// Weight bytes are filled from a deterministic PRNG rather than from hand-set
// patterns: a wrong nibble order, scale-pair unpack or sign-mask index is only
// reliably caught when every field takes many distinct values.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace {

constexpr std::size_t kK = 1024;

struct FormatCase {
  gufo::core::GgmlType type;
  const char* name;
};

const FormatCase kFormats[] = {
    {gufo::core::GgmlType::kQ4_K, "Q4_K"},
    {gufo::core::GgmlType::kQ5_K, "Q5_K"},
    {gufo::core::GgmlType::kQ6_K, "Q6_K"},
    {gufo::core::GgmlType::kQ3_K, "Q3_K"},
    {gufo::core::GgmlType::kIQ4_NL, "IQ4_NL"},
    {gufo::core::GgmlType::kIQ4_XS, "IQ4_XS"},
    {gufo::core::GgmlType::kIQ3_S, "IQ3_S"},
    {gufo::core::GgmlType::kQ8_0, "Q8_0"},
};

std::uint32_t NextRandom(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

/// Fills a weight buffer with pseudo-random bytes. Half-precision scale fields
/// are overwritten with well-conditioned values so the comparison is not
/// dominated by denormals or infinities drawn from random bit patterns.
std::vector<std::uint8_t> MakeWeights(gufo::core::GgmlType type,
                                      std::size_t rows, std::size_t k,
                                      std::uint32_t seed) {
  const std::size_t row_bytes = gufo::quant::QuantizedRowBytes(type, k);
  std::vector<std::uint8_t> bytes(rows * row_bytes);
  std::uint32_t state = seed;
  for (auto& byte : bytes) {
    byte = static_cast<std::uint8_t>(NextRandom(state) & 0xFFU);
  }

  // fp16 values in [0.5, 1.0) keep every product comfortably in range.
  auto tame_half = [&](std::uint8_t* at) {
    const std::uint16_t bits =
        static_cast<std::uint16_t>(0x3800U | (NextRandom(state) & 0x03FFU));
    std::memcpy(at, &bits, sizeof(bits));
  };

  const std::size_t block_bytes = gufo::quant::QuantizedRowBytes(
      type, gufo::quant::QuantizedBlockElements(type));
  for (std::size_t off = 0; off < bytes.size(); off += block_bytes) {
    std::uint8_t* block = bytes.data() + off;
    switch (type) {
      case gufo::core::GgmlType::kQ4_K:
      case gufo::core::GgmlType::kQ5_K:
        tame_half(block);      // d
        tame_half(block + 2);  // dmin
        break;
      case gufo::core::GgmlType::kQ6_K:
        tame_half(block + 208);  // d is last
        break;
      case gufo::core::GgmlType::kQ3_K:
        tame_half(block + 108);  // d is last
        break;
      case gufo::core::GgmlType::kIQ4_NL:
      case gufo::core::GgmlType::kIQ4_XS:
      case gufo::core::GgmlType::kIQ3_S:
      case gufo::core::GgmlType::kQ8_0:
        tame_half(block);  // d is first
        break;
      default:
        break;
    }
  }
  return bytes;
}

float CpuDot(gufo::core::GgmlType type, const void* row,
             std::span<const float> x, std::size_t k) {
  switch (type) {
    case gufo::core::GgmlType::kQ4_K:
      return gufo::quant::DotProductQ4_K(row, x, k);
    case gufo::core::GgmlType::kQ5_K:
      return gufo::quant::DotProductQ5_K(row, x, k);
    case gufo::core::GgmlType::kQ6_K:
      return gufo::quant::DotProductQ6_K(row, x, k);
    case gufo::core::GgmlType::kQ3_K:
      return gufo::quant::DotProductQ3_K(row, x, k);
    case gufo::core::GgmlType::kIQ4_NL:
      return gufo::quant::DotProductIQ4_NL(row, x, k);
    case gufo::core::GgmlType::kIQ4_XS:
      return gufo::quant::DotProductIQ4_XS(row, x, k);
    case gufo::core::GgmlType::kIQ3_S:
      return gufo::quant::DotProductIQ3_S(row, x, k);
    default:
      return gufo::quant::DotProductQ8_0(row, x, k);
  }
}

void Dequantize(gufo::core::GgmlType type, const void* row, float* out,
                std::size_t k) {
  switch (type) {
    case gufo::core::GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(row, out, k);
      return;
    case gufo::core::GgmlType::kQ5_K:
      gufo::quant::DequantizeQ5_K(row, out, k);
      return;
    case gufo::core::GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(row, out, k);
      return;
    case gufo::core::GgmlType::kQ3_K:
      gufo::quant::DequantizeQ3_K(row, out, k);
      return;
    case gufo::core::GgmlType::kIQ4_NL:
      gufo::quant::DequantizeIQ4_NL(row, out, k);
      return;
    case gufo::core::GgmlType::kIQ4_XS:
      gufo::quant::DequantizeIQ4_XS(row, out, k);
      return;
    case gufo::core::GgmlType::kIQ3_S:
      gufo::quant::DequantizeIQ3_S(row, out, k);
      return;
    default:
      gufo::quant::DequantizeQ8_0(row, out, k);
      return;
  }
}

bool g_failed = false;

void Report(const char* what, const char* format, double worst,
            double tolerance) {
  const bool ok = worst <= tolerance;
  std::cout << (ok ? "[ OK ] " : "[FAIL] ") << what << " " << format
            << " worst relative error " << worst << " (tolerance " << tolerance
            << ")\n";
  if (!ok) {
    g_failed = true;
  }
}

/// The decode GEMV dequantizes to fp32 and accumulates in fp32, so it must
/// track the CPU oracle to rounding.
void TestDecodeGemv(const FormatCase& format) {
  constexpr std::size_t kM = 64;
  const auto weights = MakeWeights(format.type, kM, kK, 0x5EED1234U);

  std::vector<float> x(kK);
  std::uint32_t state = 0xA5A5A5A5U;
  for (auto& value : x) {
    value = static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }

  void* d_w = nullptr;
  void* d_x = nullptr;
  void* d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_w, weights.size()));
  HIP_CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, kM * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_w, weights.data(), weights.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  gufo::hip::LaunchQ8KBlockGEMV(d_w, format.type, static_cast<float*>(d_x),
                                static_cast<float*>(d_y), kM, kK, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> gpu(kM);
  HIP_CHECK(
      hipMemcpy(gpu.data(), d_y, kM * sizeof(float), hipMemcpyDeviceToHost));

  const std::size_t row_bytes = gufo::quant::QuantizedRowBytes(format.type, kK);
  std::vector<float> row_values(kK);
  double worst = 0.0;
  for (std::size_t row = 0; row < kM; ++row) {
    const float expected =
        CpuDot(format.type, weights.data() + (row * row_bytes), x, kK);
    // Normalize by the magnitude of the terms being summed, not by the result.
    // These dots are random-signed and cancel heavily, so |result| understates
    // the conditioning by orders of magnitude and would make the threshold a
    // measure of luck. Against sum|w_j x_j| the tolerance is a genuine bound on
    // per-term error, which is what actually distinguishes fp32 accumulation
    // noise from a mis-decoded field.
    Dequantize(format.type, weights.data() + (row * row_bytes),
               row_values.data(), kK);
    double magnitude = 0.0;
    for (std::size_t j = 0; j < kK; ++j) {
      magnitude += std::abs(static_cast<double>(row_values[j]) *
                            static_cast<double>(x[j]));
    }
    const double scale = std::max(1e-6, magnitude);
    worst = std::max(
        worst, std::abs(static_cast<double>(gpu[row] - expected)) / scale);
  }
  Report("decode GEMV", format.name, worst, 1e-6);

  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

/// The prefill GEMM quantizes the activation to Q8_1 per 32 elements, so the
/// reference must apply the same activation quantization before comparing.
void TestPrefillGemm(const FormatCase& format, std::size_t batch) {
  constexpr std::size_t kM = 128;
  const auto weights = MakeWeights(format.type, kM, kK, 0x1BADB002U);

  std::vector<float> x(batch * kK);
  std::uint32_t state = 0xC0FFEEU;
  for (auto& value : x) {
    value = static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }

  // CPU model of the device arithmetic: quantize each 32-element activation
  // block to int8 exactly as QuantizeActivationToQ8_1Kernel does, then take the
  // dot with the dequantized weight row.
  std::vector<float> x_requantized(x.size());
  for (std::size_t token = 0; token < batch; ++token) {
    for (std::size_t block = 0; block < kK / 32; ++block) {
      float max_abs = 0.0F;
      for (std::size_t j = 0; j < 32; ++j) {
        max_abs =
            std::max(max_abs, std::abs(x[(token * kK) + (block * 32) + j]));
      }
      const float d = max_abs / 127.0F;
      const float inverse = (d != 0.0F) ? (1.0F / d) : 0.0F;
      for (std::size_t j = 0; j < 32; ++j) {
        const std::size_t at = (token * kK) + (block * 32) + j;
        x_requantized[at] =
            d *
            static_cast<float>(static_cast<int>(std::round(x[at] * inverse)));
      }
    }
  }

  void* d_w = nullptr;
  void* d_x = nullptr;
  void* d_q8 = nullptr;
  void* d_y = nullptr;
  const std::size_t act_bytes = gufo::hip::QuantizedActivationBytes(batch, kK);
  HIP_CHECK(hipMalloc(&d_w, weights.size()));
  HIP_CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q8, act_bytes));
  HIP_CHECK(hipMalloc(&d_y, batch * kM * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_w, weights.data(), weights.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_y, 0, batch * kM * sizeof(float)));

  gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(
      static_cast<const float*>(d_x), d_q8, batch, kK, nullptr);
  gufo::hip::LaunchBatchedQuantGEMMPreQuantized(
      format.type, d_w, d_q8, static_cast<float*>(d_y), batch, kM, kK, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> gpu(batch * kM);
  HIP_CHECK(hipMemcpy(gpu.data(), d_y, gpu.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  const std::size_t row_bytes = gufo::quant::QuantizedRowBytes(format.type, kK);
  std::vector<float> row_values(kK);
  double worst = 0.0;
  for (std::size_t row = 0; row < kM; ++row) {
    Dequantize(format.type, weights.data() + (row * row_bytes),
               row_values.data(), kK);
    for (std::size_t token = 0; token < batch; ++token) {
      double expected = 0.0;
      double magnitude = 0.0;
      for (std::size_t j = 0; j < kK; ++j) {
        const double term =
            static_cast<double>(row_values[j]) *
            static_cast<double>(x_requantized[(token * kK) + j]);
        expected += term;
        magnitude += std::abs(term);
      }
      // See TestDecodeGemv: normalizing by sum|terms| rather than by |result|
      // keeps the threshold a bound on per-term error instead of a bound on how
      // much the sum happened to cancel.
      const double scale = std::max(1e-6, magnitude);
      worst = std::max(
          worst,
          std::abs(static_cast<double>(gpu[(token * kM) + row]) - expected) /
              scale);
    }
  }
  Report(batch <= 8 ? "prefill GEMM (small batch)" : "prefill GEMM",
         format.name, worst, 1e-6);

  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_q8));
  HIP_CHECK(hipFree(d_y));
}

/// The small-batch draft path must be BIT-EXACT with the single-token decode
/// GEMV, not merely close: speculative verification accepts a drafted token
/// only if the verifier reproduces exactly what the unspeculated decode would
/// have emitted, so any reassociation at batch > 1 silently changes which
/// tokens are accepted.
void TestSmallBatchExactness(const FormatCase& format, std::size_t batch,
                             std::size_t rows = 64, std::size_t columns = kK) {
  const auto weights = MakeWeights(format.type, rows, columns, 0x2468ACE0U);

  std::vector<float> x(batch * columns);
  std::uint32_t state = 0x13579BDFU;
  for (auto& value : x) {
    value = static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }

  void* d_w = nullptr;
  void* d_x = nullptr;
  void* d_batched = nullptr;
  void* d_single = nullptr;
  HIP_CHECK(hipMalloc(&d_w, weights.size()));
  HIP_CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_batched, batch * rows * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_single, rows * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_w, weights.data(), weights.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  gufo::hip::LaunchBatchedQuantGEMMFp32(
      format.type, d_w, static_cast<const float*>(d_x),
      static_cast<float*>(d_batched), batch, rows, columns, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> batched(batch * rows);
  HIP_CHECK(hipMemcpy(batched.data(), d_batched, batched.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  std::size_t mismatches = 0;
  std::vector<float> single(rows);
  for (std::size_t token = 0; token < batch; ++token) {
    gufo::hip::LaunchQ8KBlockGEMV(
        d_w, format.type, static_cast<const float*>(d_x) + (token * columns),
        static_cast<float*>(d_single), rows, columns, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(single.data(), d_single, rows * sizeof(float),
                        hipMemcpyDeviceToHost));
    for (std::size_t row = 0; row < rows; ++row) {
      if (batched[(token * rows) + row] != single[row]) {
        ++mismatches;
      }
    }
  }

  std::cout << (mismatches == 0 ? "[ OK ] " : "[FAIL] ") << "batch " << batch
            << " bit-exact vs decode GEMV " << format.name << " mismatches "
            << mismatches << " of " << (batch * rows) << "\n";
  if (mismatches != 0) {
    g_failed = true;
  }

  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_batched));
  HIP_CHECK(hipFree(d_single));
}

}  // namespace

int main() {
  for (const auto& format : kFormats) {
    TestDecodeGemv(format);
  }
  for (const auto& format : kFormats) {
    TestPrefillGemm(format, 4);
    TestPrefillGemm(format, 128);
    // 128 is narrower and 288 wider than the 256-token macro tile the wide
    // route launches, so between them they cover a partially and a fully
    // populated token block plus a ragged tail.
    TestPrefillGemm(format, 288);
  }
  for (const auto& format : kFormats) {
    if (format.type == gufo::core::GgmlType::kQ8_0) {
      continue;  // Q8_0 keeps its own long-standing exact kernels.
    }
    for (const std::size_t batch :
         {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4},
          std::size_t{5}, std::size_t{6}, std::size_t{7}, std::size_t{8}}) {
      TestSmallBatchExactness(format, batch);
    }
  }
  TestSmallBatchExactness({gufo::core::GgmlType::kQ4_K, "Q4_K"}, 8, 1280, 5120);
  if (g_failed) {
    std::cerr << "q4kxl quant equivalence FAILED\n";
    return 1;
  }
  std::cout << "q4kxl quant equivalence passed\n";
  return 0;
}

#else
int main() {
  return 0;
}
#endif
