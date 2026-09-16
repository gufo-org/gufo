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
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

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
void TestPrefillGemm(const FormatCase& format, std::size_t batch,
                     std::size_t kM = 128) {
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
    const float unit =
        static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                           32768) /
        32768.0F;
    value = std::ldexp(unit, static_cast<int>(NextRandom(state) % 12U) - 6);
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
      const float actual = batched[(token * rows) + row];
      if (!std::isfinite(actual) || !std::isfinite(single[row]) ||
          std::bit_cast<std::uint32_t>(actual) !=
              std::bit_cast<std::uint32_t>(single[row])) {
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

// AR's fused FFN and verification must produce identical finite activations.
// Check separate and packed gate/up rows, including mixed quantization
// formats, two distinct inputs and incomplete row groups.
void TestFusedSwiGLU(gufo::core::GgmlType gate_type,
                     gufo::core::GgmlType up_type, std::size_t rows,
                     std::size_t columns) {
  using gufo::test::DeviceBuffer;
  constexpr std::size_t batch = 2;
  const DeviceBuffer<std::uint8_t> gate(
      MakeWeights(gate_type, rows, columns, 0x5EED1234U));
  const DeviceBuffer<std::uint8_t> up(
      MakeWeights(up_type, rows, columns, 0xA5A5A5A5U));
  std::vector<float> x(batch * columns);
  std::uint32_t state = 0x13579BDFU;
  for (auto& value : x) {
    value = static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                               32768) *
            0.000001F;
  }
  const DeviceBuffer<float> input(x);
  DeviceBuffer<float> fused(batch * rows), gate_out(batch * rows),
      up_out(batch * rows), split(batch * rows);
  HIP_CHECK(hipMemset(fused.data(), 0xFF, batch * rows * sizeof(float)));
  HIP_CHECK(hipMemset(split.data(), 0xFF, batch * rows * sizeof(float)));
  for (std::size_t token = 0; token < batch; ++token) {
    gufo::hip::LaunchFusedSwiGLUGEMV(gate.data(), gate_type, up.data(), up_type,
                                     input.data() + token * columns,
                                     fused.data() + token * rows, rows, columns,
                                     nullptr);
  }
  gufo::hip::LaunchBatchedQuantGEMMFp32(gate_type, gate.data(), input.data(),
                                        gate_out.data(), batch, rows, columns,
                                        nullptr);
  gufo::hip::LaunchBatchedQuantGEMMFp32(up_type, up.data(), input.data(),
                                        up_out.data(), batch, rows, columns,
                                        nullptr);
  gufo::hip::LaunchBatchedSwiGLUActivation(gate_out.data(), up_out.data(),
                                           split.data(), nullptr, batch * rows,
                                           nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  const auto actual = fused.CopyToHost();
  const auto expected = split.CopyToHost();
  const auto gate_values = gate_out.CopyToHost();
  const auto up_values = up_out.CopyToHost();
  std::vector<float> packed_values(2 * batch * rows);
  for (std::size_t token = 0; token < batch; ++token) {
    std::copy_n(gate_values.data() + token * rows, rows,
                packed_values.data() + 2 * token * rows);
    std::copy_n(up_values.data() + token * rows, rows,
                packed_values.data() + (2 * token + 1) * rows);
  }
  const DeviceBuffer<float> packed_input(packed_values);
  DeviceBuffer<float> packed_output(batch * rows);
  HIP_CHECK(
      hipMemset(packed_output.data(), 0xFF, batch * rows * sizeof(float)));
  gufo::hip::LaunchPackedSwiGLUActivation(
      packed_input.data(), packed_output.data(), batch, rows, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  const auto packed = packed_output.CopyToHost();
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
        !std::isfinite(packed[index]) ||
        std::bit_cast<std::uint32_t>(actual[index]) !=
            std::bit_cast<std::uint32_t>(expected[index]) ||
        std::bit_cast<std::uint32_t>(packed[index]) !=
            std::bit_cast<std::uint32_t>(expected[index])) {
      ++mismatches;
    }
  }
  std::cout << (mismatches == 0 ? "[ OK ] " : "[FAIL] ")
            << "fused/packed SwiGLU vs verification "
            << gufo::core::ToString(gate_type) << "/"
            << gufo::core::ToString(up_type) << " " << rows << "x" << columns
            << ": " << mismatches << " of " << actual.size() << " differ\n";
  g_failed |= mismatches != 0;
}

// Compare against independently decoded weights and FP64 dot products, not
// against the quantized activation that the implementation itself produces.
// Every format shares this test; large and ragged tiles exercise both stores.
void TestFp16Prefill(const FormatCase& format, std::size_t rows,
                     std::size_t batch) {
  using gufo::test::DeviceBuffer;
  const auto weights = MakeWeights(format.type, rows, kK, 0x821314U);
  std::uint32_t state = 0x726135U;
  const auto value = [&] {
    return std::bit_cast<float>((NextRandom(state) & 0x807FFFFFU) |
                                0x3C000000U);
  };
  std::vector<float> x(batch * kK), gate(batch * rows);
  for (float& item : x)
    item = value();
  for (float& item : gate)
    item = value() * 32.0F;
  DeviceBuffer<std::uint8_t> d_weights(weights);
  DeviceBuffer<float> d_x(x), d_gate(gate), d_half_y(batch * rows),
      d_a8_y(batch * rows);
  DeviceBuffer<std::uint16_t> d_half_x(batch * kK);
  DeviceBuffer<std::uint8_t> d_q8(
      gufo::hip::QuantizedActivationBytes(batch, kK));
  gufo::hip::LaunchFloatToFp16(d_x.data(), d_half_x.data(), x.size(), nullptr);
  gufo::hip::LaunchBatchedQuantGEMMFp16(format.type, d_weights.data(),
                                        d_half_x.data(), d_half_y.data(), batch,
                                        rows, kK, nullptr);
  gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(d_x.data(), d_q8.data(),
                                                  batch, kK, nullptr);
  gufo::hip::LaunchBatchedQuantGEMMPreQuantized(format.type, d_weights.data(),
                                                d_q8.data(), d_a8_y.data(),
                                                batch, rows, kK, nullptr);
  const auto actual = d_half_y.CopyToHost();
  const auto a8 = d_a8_y.CopyToHost();
  bool passed = std::ranges::all_of(
      actual, [](float item) { return std::isfinite(item); });
  double half_error = 0.0, a8_error = 0.0, half_max = 0.0, a8_max = 0.0;
  std::vector<float> decoded(kK);
  const std::size_t row_bytes = gufo::quant::QuantizedRowBytes(format.type, kK);
  for (std::size_t sample = 0; sample < 256; ++sample) {
    const std::size_t row = (sample * 137 + 17) % rows;
    const std::size_t token = (sample * 61 + 7) % batch;
    Dequantize(format.type, weights.data() + row * row_bytes, decoded.data(),
               kK);
    double expected = 0.0;
    for (std::size_t j = 0; j < kK; ++j)
      expected += static_cast<double>(decoded[j]) * x[token * kK + j];
    const double error_half = actual[token * rows + row] - expected;
    const double error_a8 = a8[token * rows + row] - expected;
    half_error += error_half * error_half;
    a8_error += error_a8 * error_a8;
    half_max = std::max(half_max, std::abs(error_half));
    a8_max = std::max(a8_max, std::abs(error_a8));
  }
  passed &= half_error < a8_error * 0.25 && half_max < a8_max;

  // The residual must be added after the complete dot product, including
  // when the output reuses the live residual buffer.
  DeviceBuffer<float> d_residual(gate), d_residual_expected(batch * rows);
  gufo::hip::LaunchBatchedResidualAdd(d_gate.data(), d_half_y.data(),
                                      d_residual_expected.data(), batch, rows,
                                      nullptr);
  gufo::hip::LaunchBatchedQuantGEMMResidualFp16(
      format.type, d_weights.data(), d_half_x.data(), d_residual.data(), batch,
      rows, kK, nullptr);
  const auto residual = d_residual.CopyToHost();
  const auto residual_expected = d_residual_expected.CopyToHost();
  passed &= std::memcmp(residual.data(), residual_expected.data(),
                        residual.size() * sizeof(float)) == 0;

  if (rows >= 4096) {
    // This is the production alias epoch: the up buffer holds the half
    // activation after the up projection, while the norm input stays live.
    DeviceBuffer<float> d_activation(batch * rows), d_up_storage(batch * rows);
    DeviceBuffer<std::uint16_t> d_expected(batch * rows);
    gufo::hip::LaunchBatchedSwiGLUActivation(d_gate.data(), d_half_y.data(),
                                             d_activation.data(), nullptr,
                                             batch * rows, nullptr);
    gufo::hip::LaunchFloatToFp16(d_activation.data(), d_expected.data(),
                                 batch * rows, nullptr);
    gufo::hip::LaunchBatchedQuantGEMMSwiGLUFp16(
        format.type, d_weights.data(), d_half_x.data(), d_gate.data(),
        d_up_storage.data(), batch, rows, kK, nullptr);
    std::vector<std::uint16_t> fused(batch * rows);
    HIP_CHECK(hipMemcpy(fused.data(), d_up_storage.data(),
                        fused.size() * sizeof(std::uint16_t),
                        hipMemcpyDeviceToHost));
    passed &= fused == d_expected.CopyToHost();
    passed &= std::ranges::all_of(
        fused, [](std::uint16_t bits) { return (bits & 0x7C00U) != 0x7C00U; });

    // Independent gate and up matrices catch swapped fragments. Compare the
    // paired kernel with two standalone projections and the activation.
    // Synthetic block scales are much larger than model weights; bound both
    // projections so their product remains representable in FP16.
    auto paired_input = x;
    for (float& item : paired_input)
      item *= 0.03125F;
    d_x.CopyFrom(paired_input);
    gufo::hip::LaunchFloatToFp16(d_x.data(), d_half_x.data(), x.size(),
                                 nullptr);
    gufo::hip::LaunchBatchedQuantGEMMFp16(format.type, d_weights.data(),
                                          d_half_x.data(), d_half_y.data(),
                                          batch, rows, kK, nullptr);
    DeviceBuffer<std::uint8_t> d_gate_weights(
        MakeWeights(format.type, rows, kK, 0x173841U));
    gufo::hip::LaunchBatchedQuantGEMMFp16(format.type, d_gate_weights.data(),
                                          d_half_x.data(), d_gate.data(), batch,
                                          rows, kK, nullptr);
    gufo::hip::LaunchBatchedSwiGLUActivation(d_gate.data(), d_half_y.data(),
                                             d_activation.data(), nullptr,
                                             batch * rows, nullptr);
    gufo::hip::LaunchFloatToFp16(d_activation.data(), d_expected.data(),
                                 batch * rows, nullptr);
    const bool paired = gufo::hip::TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
        format.type, format.type, d_gate_weights.data(), d_weights.data(),
        d_half_x.data(), d_up_storage.data(), batch, rows, kK, nullptr);
    const bool supported = format.type == gufo::core::GgmlType::kQ4_K ||
                           format.type == gufo::core::GgmlType::kQ5_K ||
                           format.type == gufo::core::GgmlType::kQ6_K ||
                           format.type == gufo::core::GgmlType::kIQ4_XS;
    passed &= paired == supported;
    if (paired) {
      HIP_CHECK(hipMemcpy(fused.data(), d_up_storage.data(),
                          fused.size() * sizeof(std::uint16_t),
                          hipMemcpyDeviceToHost));
      passed &= fused == d_expected.CopyToHost();
      passed &= std::ranges::all_of(fused, [](std::uint16_t bits) {
        return (bits & 0x7C00U) != 0x7C00U;
      });
    }
  }
  std::cout << (passed ? "[ OK ] " : "[FAIL] ") << "FP16 prefill "
            << format.name << " " << rows << "x" << kK << " batch=" << batch
            << " FP64 RMSE ratio=" << std::sqrt(half_error / a8_error) << '\n';
  g_failed |= !passed;
}

void TestFp16MixedPair(gufo::core::GgmlType gate_type,
                        gufo::core::GgmlType up_type, std::size_t rows,
                        std::size_t batch) {
  using gufo::test::DeviceBuffer;
  constexpr std::size_t kK = 768;
  std::uint32_t state = 0x493578U;
  std::vector<float> input(batch * kK);
  // Match the bounded inputs used by the same-format fusion test: synthetic
  // quantization scales are much larger than the checkpoint's scales.
  for (float& item : input)
    item = std::bit_cast<float>((NextRandom(state) & 0x807FFFFFU) |
                                0x3C000000U) *
           0.03125F;
  DeviceBuffer<std::uint8_t> gate_weights(
      MakeWeights(gate_type, rows, kK, 0x173841U));
  DeviceBuffer<std::uint8_t> up_weights(
      MakeWeights(up_type, rows, kK, 0x639482U));
  DeviceBuffer<float> x(input), gate(batch * rows), up(batch * rows),
      activation(batch * rows), output(batch * rows);
  DeviceBuffer<std::uint16_t> half_x(input.size()), expected(batch * rows);
  gufo::hip::LaunchFloatToFp16(x.data(), half_x.data(), input.size(), nullptr);
  gufo::hip::LaunchBatchedQuantGEMMFp16(
      gate_type, gate_weights.data(), half_x.data(), gate.data(), batch, rows,
      kK, nullptr);
  gufo::hip::LaunchBatchedQuantGEMMFp16(
      up_type, up_weights.data(), half_x.data(), up.data(), batch, rows, kK,
      nullptr);
  gufo::hip::LaunchBatchedSwiGLUActivation(
      gate.data(), up.data(), activation.data(), nullptr, batch * rows,
      nullptr);
  gufo::hip::LaunchFloatToFp16(activation.data(), expected.data(), batch * rows,
                               nullptr);
  bool passed = gufo::hip::TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
      gate_type, up_type, gate_weights.data(), up_weights.data(), half_x.data(),
      output.data(), batch, rows, kK, nullptr);
  std::vector<std::uint16_t> actual(batch * rows);
  HIP_CHECK(hipMemcpy(actual.data(), output.data(),
                      actual.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  const auto reference = expected.CopyToHost();
  std::size_t mismatches = 0, nonfinite = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    mismatches += actual[i] != reference[i];
    nonfinite += (actual[i] & 0x7C00U) == 0x7C00U ||
                 (reference[i] & 0x7C00U) == 0x7C00U;
  }
  passed &= mismatches == 0 && nonfinite == 0;
  std::cout << (passed ? "[ OK ] " : "[FAIL] ") << "FP16 mixed gate/up "
            << static_cast<int>(gate_type) << "/" << static_cast<int>(up_type)
            << " rows=" << rows << " batch=" << batch
            << " mismatches=" << mismatches << " nonfinite=" << nonfinite
            << '\n';
  g_failed |= !passed;
}

void TestFp16Norm() {
  using gufo::test::DeviceBuffer;
  // A full chunk exposes rare FP16 ties affected by contracting the first
  // two squared values in a different order.
  constexpr std::size_t dim = 5120, batch = 2048, elements = batch * dim;
  std::uint32_t state = 0x764203U;
  const auto value = [&] {
    return std::bit_cast<float>((NextRandom(state) & 0x807FFFFFU) |
                                0x3F000000U);
  };
  std::vector<float> input(elements), residual(elements), weight(dim);
  for (float& item : input)
    item = value();
  for (float& item : residual)
    item = value();
  for (float& item : weight)
    item = value();
  DeviceBuffer<float> d_input(input), d_residual(residual), d_weight(weight),
      d_sum(elements), d_norm(elements);
  DeviceBuffer<std::uint16_t> d_expected(elements), d_actual(elements);
  for (bool add_residual : {false, true}) {
    d_input.CopyFrom(input);
    if (add_residual)
      gufo::hip::LaunchBatchedResidualAdd(d_input.data(), d_residual.data(),
                                          d_sum.data(), batch, dim, nullptr);
    else
      d_sum.CopyFrom(input);
    gufo::hip::LaunchBatchedRMSNorm(d_sum.data(), d_weight.data(),
                                    d_norm.data(), nullptr, batch, dim, 1e-6F,
                                    nullptr);
    gufo::hip::LaunchFloatToFp16(d_norm.data(), d_expected.data(), elements,
                                 nullptr);
    gufo::hip::LaunchBatchedRMSNormFp16(
        d_input.data(), add_residual ? d_residual.data() : nullptr,
        d_weight.data(), add_residual ? d_input.data() : nullptr,
        d_actual.data(), batch, dim, 1e-6F, nullptr);
    bool passed = d_actual.CopyToHost() == d_expected.CopyToHost();
    if (add_residual) {
      const auto actual_sum = d_input.CopyToHost();
      const auto expected_sum = d_sum.CopyToHost();
      passed &= std::memcmp(actual_sum.data(), expected_sum.data(),
                            elements * sizeof(float)) == 0;
    }
    std::cout << (passed ? "[ OK ] " : "[FAIL] ")
              << "FP16 norm/residual rounding, residual=" << add_residual
              << '\n';
    g_failed |= !passed;
  }
}

}  // namespace

int main() {
  TestFp16Norm();
  for (const auto& format : kFormats) {
    TestFp16Prefill(format, 4097, 257);
    if (format.type == gufo::core::GgmlType::kQ4_K ||
        format.type == gufo::core::GgmlType::kQ5_K) {
      TestFp16Prefill(format, 4096, 256);
    }
  }
  TestFp16Prefill({gufo::core::GgmlType::kQ8_0, "Q8_0"}, 48, 129);
  TestFp16Prefill({gufo::core::GgmlType::kQ6_K, "Q6_K"}, 1057, 129);
  TestFp16Prefill({gufo::core::GgmlType::kIQ4_NL, "IQ4_NL"}, 48, 129);
  TestFp16Prefill({gufo::core::GgmlType::kIQ4_XS, "IQ4_XS"}, 1057, 129);
  // Attention K/V projections use the larger tile at long prefill widths.
  // An incomplete final token tile must preserve the same FP32 dot order.
  for (const auto& format : {FormatCase{gufo::core::GgmlType::kQ4_K, "Q4_K"},
                             FormatCase{gufo::core::GgmlType::kQ5_K, "Q5_K"},
                             FormatCase{gufo::core::GgmlType::kQ6_K, "Q6_K"},
                             FormatCase{gufo::core::GgmlType::kQ8_0, "Q8_0"}}) {
    TestFp16Prefill(format, 1024, 1025);
  }
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
    if (format.type == gufo::core::GgmlType::kQ4_K ||
        format.type == gufo::core::GgmlType::kQ5_K ||
        format.type == gufo::core::GgmlType::kQ6_K ||
        format.type == gufo::core::GgmlType::kQ8_0) {
      // Native wave64: partial token tiles and a partial output-row tile.
      TestPrefillGemm(format, 129, 1057);
    }
    if (format.type == gufo::core::GgmlType::kIQ4_XS) {
      TestPrefillGemm(format, 129, 1024);
    }
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
  // Production IQ4 verification shapes cover the two lane partials, grouped
  // requests and the longer down projection with independent token inputs.
  const FormatCase iq4{gufo::core::GgmlType::kIQ4_XS, "IQ4_XS"};
  TestSmallBatchExactness(iq4, 12, 34816, 5120);
  TestSmallBatchExactness(iq4, 28, 34816, 5120);
  TestSmallBatchExactness(iq4, 16, 5120, 17408);
  TestSmallBatchExactness(iq4, 42, 5120, 17408);
  using Type = gufo::core::GgmlType;
  const std::pair<Type, Type> fused_formats[] = {
      {Type::kQ4_K, Type::kQ4_K},     {Type::kQ5_K, Type::kQ5_K},
      {Type::kIQ4_XS, Type::kIQ4_XS}, {Type::kQ4_K, Type::kQ5_K},
      {Type::kIQ4_XS, Type::kQ4_K},   {Type::kIQ4_XS, Type::kQ5_K},
      {Type::kQ4_K, Type::kIQ4_XS}};
  for (const auto& [gate, up] : fused_formats) {
    TestFusedSwiGLU(gate, up, 66, 768);
    if (gate != up) {
      // Different byte strides, full tiles, and both row and token tails.
      TestFp16MixedPair(gate, up, 256, 256);
      TestFp16MixedPair(gate, up, 263, 259);
    }
  }
  // Additional FP16 pairs cover the remaining FFN layers in this checkpoint.
  // Keep complete tiles and independently ragged row/token tails.
  for (const auto& [gate, up] :
       {std::pair{Type::kQ6_K, Type::kQ5_K},
        std::pair{Type::kIQ4_XS, Type::kQ3_K},
        std::pair{Type::kQ3_K, Type::kIQ4_XS},
        std::pair{Type::kIQ4_NL, Type::kIQ4_XS},
        std::pair{Type::kIQ4_NL, Type::kQ5_K},
        std::pair{Type::kQ5_K, Type::kIQ4_NL}}) {
    TestFp16MixedPair(gate, up, 256, 256);
    TestFp16MixedPair(gate, up, 263, 259);
  }
  TestFusedSwiGLU(Type::kQ4_K, Type::kQ5_K, 65, 768);
  TestFusedSwiGLU(Type::kQ8_0, Type::kQ8_0, 17408, 5120);
  TestFusedSwiGLU(Type::kQ6_K, Type::kQ6_K, 17408, 5120);
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
