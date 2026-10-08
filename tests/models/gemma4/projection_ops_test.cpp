// Tripwire over the shared projection kernels at Gemma 4 shapes. Decode
// (one-row GEMV) and verification (small-batch FP32 GEMM) must agree bit for
// bit at every width, so greedy speculation reproduces autoregressive
// decoding; both must match the CPU dequantized dot product, and the W8A8
// prefill GEMM must stay within its activation-rounding envelope. A change in
// the Qwen-owned kernels that breaks any of this fails here.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/gemma4/kernels/rocm/gemv.hpp"
#include "src/models/gemma4/kernels/rocm/wmma_gemv.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "tests/models/gemma4/check.hpp"

using gemma4_test::Require;
using gufo::core::GgmlType;

namespace {

struct Format {
  GgmlType type;
  std::optional<gufo::models::gemma4::rocm::GemvFormat> gemv;
  std::size_t block;
  std::size_t bytes;
  std::vector<std::size_t> half_offsets;  ///< fp16 scale fields per block
  const char* name;
};

std::uint16_t Half(float x) {
  const _Float16 h = static_cast<_Float16>(x);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

/// Random blocks with bounded, finite scales.
std::vector<std::uint8_t> RandomMatrix(const Format& f, std::size_t rows,
                                       std::size_t cols, std::mt19937& rng) {
  const std::size_t blocks = rows * cols / f.block;
  std::vector<std::uint8_t> data(blocks * f.bytes + 4096, 0);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> scale(0.0005F, 0.004F);
  for (std::size_t i = 0; i < blocks * f.bytes; ++i) {
    data[i] = static_cast<std::uint8_t>(byte(rng));
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    for (std::size_t off : f.half_offsets) {
      const std::uint16_t h = Half(scale(rng));
      std::memcpy(&data[b * f.bytes + off], &h, 2);
    }
  }
  return data;
}

template<class T>
T* Device(const T* host, std::size_t n) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, n * sizeof(T)));
  HIP_CHECK(hipMemcpy(ptr, host, n * sizeof(T), hipMemcpyHostToDevice));
  return ptr;
}

void CheckShape(const Format& f, std::size_t m, std::size_t k,
                std::mt19937& rng) {
  const std::string name =
      std::string(f.name) + " " + std::to_string(m) + "x" + std::to_string(k);
  const auto w = RandomMatrix(f, m, k, rng);
  constexpr std::size_t kRows = 16;
  constexpr std::size_t kPrefillRows = 40;
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(kPrefillRows * k);
  for (float& v : x)
    v = normal(rng);
  auto* dw = Device(w.data(), w.size());
  float* dx = Device(x.data(), x.size());
  float* dy = nullptr;
  HIP_CHECK(hipMalloc(&dy, kPrefillRows * m * sizeof(float)));

  // Decode reference rows.
  std::vector<float> gemv(kRows * m);
  for (std::size_t r = 0; r < kRows; ++r) {
    gufo::hip::LaunchGEMV(dw, f.type, dx + r * k, dy + r * m, m, k, nullptr);
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(gemv.data(), dy, gemv.size() * 4, hipMemcpyDeviceToHost));

  // Every verification width reproduces the decode rows bit for bit,
  // including the double-stage pass at one row (drafter-mode decode).
  std::vector<float> batch(kRows * m);
  HIP_CHECK(hipMemset(dy, 0xFF, m * sizeof(float)));
  if (gufo::hip::LaunchKQuantSmallBatchDoubleStage(f.type, dw, dx, dy, 1, m, k,
                                                   nullptr)) {
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(batch.data(), dy, m * 4, hipMemcpyDeviceToHost));
    Require(std::memcmp(batch.data(), gemv.data(), m * 4) == 0,
            name + ": one-row double stage differs from decode GEMV");
  }
  for (std::size_t width = 2; width <= 16; ++width) {
    HIP_CHECK(hipMemset(dy, 0xFF, width * m * sizeof(float)));
    gufo::hip::LaunchBatchedQuantGEMMFp32(f.type, dw, dx, dy, width, m, k,
                                          nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(
        hipMemcpy(batch.data(), dy, width * m * 4, hipMemcpyDeviceToHost));
    Require(std::memcmp(batch.data(), gemv.data(), width * m * 4) == 0,
            name + ": width " + std::to_string(width) +
                " differs from decode GEMV");
    // Q8_0 past eight rows in one weight pass.
    if (f.type == GgmlType::kQ8_0 && width > 8) {
      HIP_CHECK(hipMemset(dy, 0xFF, width * m * sizeof(float)));
      Require(
          gufo::hip::LaunchQ8_0SmallBatchWide(dw, dx, dy, width, m, k, nullptr),
          name + ": wide Q8_0 small batch rejected");
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(
          hipMemcpy(batch.data(), dy, width * m * 4, hipMemcpyDeviceToHost));
      Require(std::memcmp(batch.data(), gemv.data(), width * m * 4) == 0,
              name + ": wide Q8_0 width " + std::to_string(width) +
                  " differs from decode GEMV");
    }
    // The double-stage configuration verification uses for Gemma shapes.
    {
      HIP_CHECK(hipMemset(dy, 0xFF, width * m * sizeof(float)));
      if (gufo::hip::LaunchKQuantSmallBatchDoubleStage(f.type, dw, dx, dy,
                                                       width, m, k, nullptr)) {
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(
            hipMemcpy(batch.data(), dy, width * m * 4, hipMemcpyDeviceToHost));
        Require(std::memcmp(batch.data(), gemv.data(), width * m * 4) == 0,
                name + ": double-stage width " + std::to_string(width) +
                    " differs from decode GEMV");
      }
    }
  }

  // Decode against an FP64 dot of the CPU-dequantized row, on a row sample;
  // the error is bounded relative to sum |w x| (FP32 accumulation).
  const std::size_t encoded_row = k / f.block * f.bytes;
  std::vector<float> row(k);
  double worst = 0.0;
  for (std::size_t o = 0; o < m; o += std::max<std::size_t>(1, m / 97)) {
    gufo::quant::Dequantize(f.type, w.data() + o * encoded_row, row.data(), k);
    for (std::size_t r = 0; r < 2; ++r) {
      double want = 0.0, magnitude = 0.0;
      for (std::size_t c = 0; c < k; ++c) {
        const double p = double{row[c]} * x[r * k + c];
        want += p;
        magnitude += std::fabs(p);
      }
      worst = std::max(worst, std::fabs(gemv[r * m + o] - want) / magnitude);
    }
  }
  Require(worst < 2e-6, name + ": decode error " + std::to_string(worst));

  // The Gemma autoregressive GEMV matches the FP64 dot (256-multiple widths).
  if (f.gemv && k % 256 == 0) {
    namespace g4k = gufo::models::gemma4::rocm;
    std::vector<float> one(2 * m);
    for (std::size_t r = 0; r < 2; ++r) {
      Require(g4k::LaunchKQuantGemv(*f.gemv, dw, dx + r * k, dy + r * m,
                                    static_cast<std::uint32_t>(m),
                                    static_cast<std::uint32_t>(k), nullptr),
              name + ": Gemma GEMV rejected the shape");
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(one.data(), dy, one.size() * 4, hipMemcpyDeviceToHost));
    double gworst = 0.0;
    for (std::size_t o = 0; o < m; o += std::max<std::size_t>(1, m / 97)) {
      gufo::quant::Dequantize(f.type, w.data() + o * encoded_row, row.data(),
                              k);
      for (std::size_t r = 0; r < 2; ++r) {
        double want = 0.0, magnitude = 0.0;
        for (std::size_t c = 0; c < k; ++c) {
          const double p = double{row[c]} * x[r * k + c];
          want += p;
          magnitude += std::fabs(p);
        }
        gworst = std::max(gworst, std::fabs(one[r * m + o] - want) / magnitude);
      }
    }
    Require(gworst < 2e-6,
            name + ": Gemma GEMV error " + std::to_string(gworst));

    // Cold-weight bandwidth, rotating copies beyond the 32 MiB MALL.
    const std::size_t bytes = m * encoded_row;
    const std::size_t copies = std::max<std::size_t>(2, (256u << 20) / bytes);
    std::vector<std::uint8_t*> rot;
    for (std::size_t c = 0; c < copies; ++c) {
      rot.push_back(Device(w.data(), w.size()));
    }
    const auto time = [&](auto&& launch) {
      for (int warm = 0; warm < 2; ++warm)
        launch(rot[warm % copies]);
      HIP_CHECK(hipDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      const int iters = 40;
      for (int i = 0; i < iters; ++i)
        launch(rot[i % copies]);
      HIP_CHECK(hipDeviceSynchronize());
      return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                           t0)
                 .count() /
             iters;
    };
    const double gemma = time([&](std::uint8_t* wp) {
      (void)g4k::LaunchKQuantGemv(*f.gemv, wp, dx, dy,
                                  static_cast<std::uint32_t>(m),
                                  static_cast<std::uint32_t>(k), nullptr);
    });
    const double qwen = time([&](std::uint8_t* wp) {
      gufo::hip::LaunchGEMV(wp, f.type, dx, dy, m, k, nullptr);
    });
    const double qwen1 = time([&](std::uint8_t* wp) {
      gufo::hip::LaunchBatchedQuantGEMMFp32(f.type, wp, dx, dy, 1, m, k,
                                            nullptr);
    });
    const double qwen5 = time([&](std::uint8_t* wp) {
      gufo::hip::LaunchBatchedQuantGEMMFp32(f.type, wp, dx, dy, 5, m, k,
                                            nullptr);
    });
    const double staged5 = time([&](std::uint8_t* wp) {
      (void)gufo::hip::LaunchKQuantSmallBatchDoubleStage(f.type, wp, dx, dy, 5,
                                                         m, k, nullptr);
    });
    std::cout << name << ": one row: Gemma GEMV " << gemma * 1e6
              << " us, shared GEMV " << qwen * 1e6 << " us, shared small-batch "
              << qwen1 * 1e6 << " us; five rows: shared small-batch "
              << qwen5 * 1e6 << " us, double-stage " << staged5 * 1e6 << " us ("
              << bytes / 1e6 << " MB)\n";
    for (auto* ptr : rot)
      HIP_CHECK(hipFree(ptr));
  }

  // Prefill W8A8 stays within Q8_1 activation rounding.
  void* dq = nullptr;
  HIP_CHECK(
      hipMalloc(&dq, gufo::hip::QuantizedActivationBytes(kPrefillRows, k)));
  gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(dx, dq, kPrefillRows, k,
                                                  nullptr);
  gufo::hip::LaunchBatchedQuantGEMMPreQuantized(f.type, dw, dq, dy,
                                                kPrefillRows, m, k, nullptr);
  std::vector<float> prefill(kPrefillRows * m);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(
      hipMemcpy(prefill.data(), dy, prefill.size() * 4, hipMemcpyDeviceToHost));
  double err2 = 0.0, ref2 = 0.0;
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t o = 0; o < m; ++o) {
      const double d = prefill[r * m + o] - gemv[r * m + o];
      err2 += d * d;
      ref2 += double{gemv[r * m + o]} * gemv[r * m + o];
      Require(std::isfinite(prefill[r * m + o]), name + ": non-finite prefill");
    }
  }
  const double rel = std::sqrt(err2 / ref2);
  Require(rel < 2e-2, name + ": prefill relative error " + std::to_string(rel));
  HIP_CHECK(hipFree(dq));
  HIP_CHECK(hipFree(dw));
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dy));
}

/// Binary16 weights (the F16 and BF16 layers of the UD-Q8_K_XL quants): the
/// Gemma small-batch GEMV rounds every width like one row and matches the
/// FP64 dot; the binary16 WMMA prefill GEMM stays within activation
/// rounding.
void CheckHalfShape(std::size_t m, std::size_t k, std::mt19937& rng) {
  namespace g4k = gufo::models::gemma4::rocm;
  const std::string name = "F16 " + std::to_string(m) + "x" + std::to_string(k);
  constexpr std::size_t kRows = g4k::kMaxHalfGemvRows;
  constexpr std::size_t kPrefillRows = 40;
  std::normal_distribution<float> weight(0.0F, 0.02F);
  std::vector<std::uint16_t> w(m * k);
  for (auto& v : w) {
    v = Half(weight(rng));
  }
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(kPrefillRows * k);
  for (float& v : x)
    v = normal(rng);
  auto* dw = Device(w.data(), w.size());
  float* dx = Device(x.data(), x.size());
  float* dy = nullptr;
  HIP_CHECK(hipMalloc(&dy, kPrefillRows * m * sizeof(float)));
  const auto gemv = [&](const std::uint16_t* wp, std::size_t rows,
                        const float* xp, float* yp) {
    Require(g4k::LaunchHalfGemv(wp, xp, yp, static_cast<std::uint32_t>(rows),
                                static_cast<std::uint32_t>(m),
                                static_cast<std::uint32_t>(k), nullptr),
            name + ": binary16 GEMV rejected the shape");
  };

  std::vector<float> one(kRows * m);
  for (std::size_t r = 0; r < kRows; ++r) {
    gemv(dw, 1, dx + r * k, dy + r * m);
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(one.data(), dy, one.size() * 4, hipMemcpyDeviceToHost));
  std::vector<float> batch(kRows * m);
  for (std::size_t width = 2; width <= kRows; ++width) {
    HIP_CHECK(hipMemset(dy, 0xFF, width * m * sizeof(float)));
    gemv(dw, width, dx, dy);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(
        hipMemcpy(batch.data(), dy, width * m * 4, hipMemcpyDeviceToHost));
    Require(
        std::memcmp(batch.data(), one.data(), width * m * 4) == 0,
        name + ": width " + std::to_string(width) + " differs from one row");
  }

  std::vector<float> row(k);
  double worst = 0.0;
  for (std::size_t o = 0; o < m; o += std::max<std::size_t>(1, m / 97)) {
    gufo::quant::Dequantize(GgmlType::kF16, w.data() + o * k, row.data(), k);
    for (std::size_t r = 0; r < 2; ++r) {
      double want = 0.0, magnitude = 0.0;
      for (std::size_t c = 0; c < k; ++c) {
        const double p = double{row[c]} * x[r * k + c];
        want += p;
        magnitude += std::fabs(p);
      }
      worst = std::max(worst, std::fabs(one[r * m + o] - want) / magnitude);
    }
  }
  Require(worst < 2e-6, name + ": GEMV error " + std::to_string(worst));

  // Prefill over binary16 activation rows.
  std::vector<__half> xh(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    xh[i] = __float2half(x[i]);
  }
  auto* dxh = Device(xh.data(), xh.size());
  Require(gufo::models::qwen38_flash_next::rocm::UnquantizedF16Gemm(
              dw, dxh, dy, kPrefillRows, m, k, nullptr),
          name + ": binary16 prefill GEMM rejected the shape");
  std::vector<float> prefill(kPrefillRows * m);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(
      hipMemcpy(prefill.data(), dy, prefill.size() * 4, hipMemcpyDeviceToHost));
  double err2 = 0.0, ref2 = 0.0;
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t o = 0; o < m; ++o) {
      const double d = prefill[r * m + o] - one[r * m + o];
      err2 += d * d;
      ref2 += double{one[r * m + o]} * one[r * m + o];
    }
  }
  const double rel = std::sqrt(err2 / ref2);
  Require(rel < 2e-3, name + ": prefill relative error " + std::to_string(rel));

  // Cold-weight bandwidth, rotating copies beyond the 32 MiB MALL.
  const std::size_t bytes = m * k * 2;
  const std::size_t copies = std::max<std::size_t>(2, (256u << 20) / bytes);
  std::vector<std::uint16_t*> rot;
  for (std::size_t c = 0; c < copies; ++c) {
    rot.push_back(Device(w.data(), w.size()));
  }
  const auto time = [&](std::size_t rows) {
    for (int warm = 0; warm < 2; ++warm)
      gemv(rot[warm % copies], rows, dx, dy);
    HIP_CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    const int iters = 40;
    for (int i = 0; i < iters; ++i)
      gemv(rot[i % copies], rows, dx, dy);
    HIP_CHECK(hipDeviceSynchronize());
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count() /
           iters;
  };
  const double t1 = time(1);
  const double t5 = time(5);
  const double t16 = time(16);
  std::cout << name << ": binary16 GEMV one row " << t1 * 1e6 << " us ("
            << bytes / t1 / 1e9 << " GB/s), five rows " << t5 * 1e6
            << " us, sixteen rows " << t16 * 1e6 << " us (" << bytes / 1e6
            << " MB)\n";
  for (auto* ptr : rot)
    HIP_CHECK(hipFree(ptr));
  HIP_CHECK(hipFree(dxh));
  HIP_CHECK(hipFree(dw));
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dy));
}

/// The WMMA projections: every width from 1 to 16 rounds a row identically
/// (a row's output never depends on the other rows), and each output stays
/// within FP32 rounding of the FP64 dot of the dequantized row. Activation
/// sub-blocks span zeros, tiny values and magnitudes past the binary16
/// range, which the per-sub-block power-of-two scale must absorb.
void CheckWmmaShape(const Format& f, std::uint32_t m, std::uint32_t k,
                    std::mt19937& rng) {
  namespace g4k = gufo::models::gemma4::rocm;
  const std::string name = std::string(f.name) + " WMMA " + std::to_string(m) +
                           "x" + std::to_string(k);
  constexpr std::uint32_t kRows = g4k::kWmmaGemvRows;
  const auto w = RandomMatrix(f, m, k, rng);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::uniform_int_distribution<int> kind(0, 9);
  std::vector<float> x(std::size_t{kRows} * k);
  for (std::size_t b = 0; b < x.size() / 32; ++b) {
    const int c = kind(rng);
    const float scale = c == 0 ? 0.0F : c == 1 ? 1e-6F : c == 2 ? 3e4F : 1.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      x[b * 32 + j] = scale * normal(rng);
    }
  }
  auto* dw = Device(w.data(), w.size());
  float* dx = Device(x.data(), x.size());
  float* dy = nullptr;
  void* scratch = nullptr;
  HIP_CHECK(hipMalloc(&dy, std::size_t{kRows} * m * sizeof(float)));
  HIP_CHECK(hipMalloc(&scratch, g4k::WmmaGemvScratchBytes(k)));
  std::vector<float> full(std::size_t{kRows} * m);
  Require(
      g4k::LaunchWmmaGemv(f.type, dw, dx, dy, kRows, m, k, scratch, nullptr),
      name + " rejected");
  HIP_CHECK(hipMemcpy(full.data(), dy, full.size() * 4, hipMemcpyDeviceToHost));
  std::vector<float> part(full.size());
  for (std::uint32_t width = 1; width < kRows; ++width) {
    // Each width starts at a different row, so neighbours change too.
    const std::uint32_t first = (width * 5) % (kRows - width + 1);
    HIP_CHECK(hipMemset(dy, 0xFF, std::size_t{width} * m * sizeof(float)));
    Require(g4k::LaunchWmmaGemv(f.type, dw, dx + std::size_t{first} * k, dy,
                                width, m, k, scratch, nullptr),
            name + " rejected");
    HIP_CHECK(hipMemcpy(part.data(), dy, std::size_t{width} * m * 4,
                        hipMemcpyDeviceToHost));
    Require(std::memcmp(part.data(), full.data() + std::size_t{first} * m,
                        std::size_t{width} * m * 4) == 0,
            name + ": width " + std::to_string(width) + " differs");
  }
  const std::size_t row_bytes = k / f.block * f.bytes;
  std::vector<float> row(k);
  double worst = 0.0;
  for (std::uint32_t o = 0; o < m; o += std::max<std::uint32_t>(1, m / 61)) {
    gufo::quant::Dequantize(f.type, w.data() + o * row_bytes, row.data(), k);
    for (std::uint32_t r = 0; r < kRows; r += 5) {
      double want = 0.0, magnitude = 0.0;
      for (std::uint32_t c = 0; c < k; ++c) {
        const double p = double{row[c]} * x[std::size_t{r} * k + c];
        want += p;
        magnitude += std::fabs(p);
      }
      const float got = full[std::size_t{r} * m + o];
      Require(std::isfinite(got), name + ": non-finite output");
      worst = std::max(worst, std::fabs(got - want) / magnitude);
    }
  }
  std::cout << name << ": width-invariant, error " << worst << "\n";
  Require(worst < 1e-6, name + ": error " + std::to_string(worst));
  HIP_CHECK(hipFree(dw));
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dy));
  HIP_CHECK(hipFree(scratch));
}

/// The load-time BF16 -> binary16 rewrite: exact inside the binary16 range,
/// round to nearest below it, and every value past it counted.
void CheckBf16Conversion(std::mt19937& rng) {
  std::vector<std::uint16_t> bf16;
  std::uniform_int_distribution<int> bits(0, 0xFFFF);
  for (int i = 0; i < 100000; ++i) {
    bf16.push_back(static_cast<std::uint16_t>(bits(rng)));
  }
  // 65280 (largest BF16 below the binary16 limit), 65536, and tiny values.
  for (std::uint16_t v : {0x477FU, 0xC77FU, 0x4780U, 0x3780U, 0x3700U, 0x0001U,
                          0x0000U, 0x8000U}) {
    bf16.push_back(v);
  }
  std::size_t outside = 0;
  std::vector<std::uint16_t> want(bf16.size());
  for (std::size_t i = 0; i < bf16.size(); ++i) {
    const std::uint32_t word = static_cast<std::uint32_t>(bf16[i]) << 16U;
    float f;
    std::memcpy(&f, &word, 4);
    const _Float16 h = static_cast<_Float16>(f);
    std::memcpy(&want[i], &h, 2);
    if (!std::isfinite(static_cast<float>(h))) {
      ++outside;
    } else if (std::fabs(f) >= 0x1p-17F || f == 0.0F) {
      Require(static_cast<float>(h) == f,
              "BF16 value " + std::to_string(f) + " is not a binary16 value");
    }
  }
  auto* data = Device(bf16.data(), bf16.size());
  std::uint32_t* count = nullptr;
  HIP_CHECK(hipMalloc(&count, sizeof(std::uint32_t)));
  HIP_CHECK(hipMemset(count, 0, sizeof(std::uint32_t)));
  gufo::models::gemma4::rocm::ConvertBf16ToHalf(data, bf16.size(), count,
                                                nullptr);
  std::vector<std::uint16_t> got(bf16.size());
  std::uint32_t counted = 0;
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(got.data(), data, got.size() * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&counted, count, 4, hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < bf16.size(); ++i) {
    _Float16 a, b;
    std::memcpy(&a, &got[i], 2);
    std::memcpy(&b, &want[i], 2);
    const bool nan = a != a && b != b;
    Require(nan || got[i] == want[i],
            "BF16 conversion differs at element " + std::to_string(i));
  }
  Require(counted == outside, "BF16 conversion counted " +
                                  std::to_string(counted) +
                                  " values outside "
                                  "binary16, expected " +
                                  std::to_string(outside));
  HIP_CHECK(hipFree(data));
  HIP_CHECK(hipFree(count));
}

}  // namespace

/// The drafter head repack: Q4_K rows reproduce the Q8_0 values within the
/// error of a 16-level fit per 32 values.
void CheckRepack(std::mt19937& rng) {
  constexpr std::size_t kRows = 257;
  constexpr std::size_t kCols = 1024;
  const Format q8{GgmlType::kQ8_0, std::nullopt, 32, 34, {0}, "Q8_0"};
  auto src = RandomMatrix(q8, kRows, kCols, rng);
  // Gaussian-like values, as in trained weights, instead of uniform bytes.
  std::normal_distribution<float> normal(0.0F, 30.0F);
  for (std::size_t b = 0; b < kRows * kCols / 32; ++b) {
    for (std::size_t i = 0; i < 32; ++i) {
      src[b * 34 + 2 + i] = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(std::clamp(normal(rng), -127.0F, 127.0F)));
    }
  }
  const std::size_t q4_bytes = kRows * kCols / 256 * 144;
  std::uint8_t* dsrc = Device(src.data(), src.size());
  std::uint8_t* ddst = nullptr;
  HIP_CHECK(hipMalloc(&ddst, q4_bytes));
  gufo::models::gemma4::rocm::RepackQ8_0AsQ4K(dsrc, ddst, kRows, kCols,
                                              nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<std::uint8_t> dst(q4_bytes);
  HIP_CHECK(hipMemcpy(dst.data(), ddst, q4_bytes, hipMemcpyDeviceToHost));
  std::vector<float> want(kCols), got(kCols);
  double error = 0.0, energy = 0.0;
  for (std::size_t r = 0; r < kRows; ++r) {
    gufo::quant::Dequantize(GgmlType::kQ8_0, src.data() + r * kCols / 32 * 34,
                            want.data(), kCols);
    gufo::quant::Dequantize(GgmlType::kQ4_K, dst.data() + r * kCols / 256 * 144,
                            got.data(), kCols);
    for (std::size_t c = 0; c < kCols; ++c) {
      error += (double{got[c]} - want[c]) * (double{got[c]} - want[c]);
      energy += double{want[c]} * want[c];
    }
  }
  const double relative = std::sqrt(error / energy);
  std::cout << "draft head repack: relative RMS error " << relative << "\n";
  Require(relative < 0.08,
          "Q8_0 -> Q4_K repack error " + std::to_string(relative));
  HIP_CHECK(hipFree(dsrc));
  HIP_CHECK(hipFree(ddst));
}

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "SKIP: no HIP device\n";
    return 77;
  }
  return gemma4_test::Run([] {
    std::mt19937 rng(11);
    const Format formats[] = {
        {GgmlType::kQ4_K,
         gufo::models::gemma4::rocm::GemvFormat::kQ4_K,
         256,
         144,
         {0, 2},
         "Q4_K"},
        {GgmlType::kQ5_K,
         gufo::models::gemma4::rocm::GemvFormat::kQ5_K,
         256,
         176,
         {0, 2},
         "Q5_K"},
        {GgmlType::kQ6_K,
         gufo::models::gemma4::rocm::GemvFormat::kQ6_K,
         256,
         210,
         {208},
         "Q6_K"},
        {GgmlType::kQ8_0,
         gufo::models::gemma4::rocm::GemvFormat::kQ8_0,
         32,
         34,
         {0},
         "Q8_0"},
        {GgmlType::kQ4_0,
         gufo::models::gemma4::rocm::GemvFormat::kQ4_0,
         32,
         18,
         {0},
         "Q4_0"},
    };
    // (M, K) of every target projection, plus the drafter's widths.
    const std::pair<std::size_t, std::size_t> shapes[] = {
        {8192, 5376},  {4096, 5376},  {16384, 5376}, {2048, 5376},
        {5376, 8192},  {5376, 16384}, {21504, 5376}, {5376, 21504},
        {1024, 10752}, {8192, 1024},  {5376, 1024}};
    for (const auto& f : formats) {
      for (const auto& [m, k] : shapes) {
        CheckShape(f, m, k, rng);
      }
    }
    // The 26B-A4B keeps attention, the dense MLP and the tied head in Q8_0
    // in every Unsloth quant; its drafter reads 1024-wide rows.
    const std::pair<std::size_t, std::size_t> moe_shapes[] = {
        {4096, 2816}, {2048, 2816}, {8192, 2816}, {1024, 2816},
        {2816, 4096}, {2816, 8192}, {2112, 2816}, {2816, 2112},
        {1024, 5632}, {2816, 1024}, {4096, 1024}, {1024, 4096}};
    for (const auto& [m, k] : moe_shapes) {
      CheckShape(formats[3], m, k, rng);
    }
    // Binary16 projections of the UD-Q8_K_XL quants, fused where the loader
    // fuses them: 31B q, k, [q | k] of a global layer, [gate | up], down;
    // 26B [q | k], attention output, [gate | up], down.
    const std::pair<std::size_t, std::size_t> half_shapes[] = {
        {8192, 5376}, {4096, 5376}, {18432, 5376}, {43008, 5376}, {5376, 21504},
        {9216, 2816}, {2816, 8192}, {4224, 2816},  {2816, 2112}};
    for (const auto& [m, k] : half_shapes) {
      CheckHalfShape(m, k, rng);
    }
    // WMMA projections at the 31B's Q4_K shapes (fused [q | k] of a global
    // layer and [gate | up]), a partial row block and the split reduction.
    const std::pair<std::uint32_t, std::uint32_t> wmma_shapes[] = {
        {8192, 5376},  {4096, 5376}, {18432, 5376},
        {43008, 5376}, {5376, 8192}, {5376, 16384},
        {5376, 21504}, {4100, 5376}, {4100, 21504}};
    for (const auto& [m, k] : wmma_shapes) {
      CheckWmmaShape(formats[0], m, k, rng);
    }
    // Its Q5_K / Q6_K layers (V, down, the Q5_K vocabulary head's width) and
    // the QAT target's Q4_0, with partial row blocks.
    CheckWmmaShape(formats[1], 43008, 5376, rng);
    CheckWmmaShape(formats[1], 4100, 21504, rng);
    CheckWmmaShape(formats[2], 4096, 5376, rng);
    CheckWmmaShape(formats[2], 5376, 21504, rng);
    CheckWmmaShape(formats[2], 4100, 5376, rng);
    CheckWmmaShape(formats[4], 43008, 5376, rng);
    CheckWmmaShape(formats[4], 4100, 21504, rng);
    CheckBf16Conversion(rng);
    CheckRepack(rng);
  });
}
