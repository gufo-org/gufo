#include "staging/models/glm5_next/kernels/rocm/w8a8_dense.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "staging/models/glm5_next/kernels/rocm/gemv.hpp"
#include "staging/tests/models/glm5_next/hip_test.hpp"

// Quality contract for the opt-in W8A8 dense tier: against the scalar
// GemvRows reference the Q8_1-quantized activations stay inside the same
// 1e-4 relative NMSE budget as the F16 DenseWmma tier (observed worst
// 1.5e-5: per-K32 int8 activation scale is the only loss; weights decode
// exactly), the tier must reproduce itself bit-for-bit across launches and
// may never emit a non-finite output. Block scales mirror dense_wmma_test
// (pinned to 1.0) so the comparison measures quantization, not binary16
// range.
namespace q = gufo::models::glm5_next::rocm;
namespace t = gufo::tests::glm5_next;
using gufo::core::GgmlType;

namespace {

constexpr std::size_t kQ5KBytes = 176U;
constexpr std::size_t kQ6KBytes = 210U;
constexpr std::size_t kQ8Bytes = 34U;

bool Check(const char* what, double worst, double limit) {
  const bool ok = worst <= limit;
  std::printf("%-34s worst %.3e limit %.0e %s\n", what, worst, limit,
              ok ? "OK" : "FAIL");
  return ok;
}

/// rows * (cols / 256) random Q5_K super-blocks with the fp16 scale and min
/// pinned to 1.0 (same fixture as dense_wmma_test).
std::vector<std::uint8_t> MakeQ5K(std::uint32_t rows, std::uint32_t cols,
                                  std::uint32_t seed) {
  const std::size_t row_bytes = cols / 256U * kQ5KBytes;
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) * row_bytes);
  std::uint32_t rng = seed;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>(t::NextRandom(&rng) & 0xFFU);
  }
  for (std::size_t b = 0; b < bytes.size() / kQ5KBytes; ++b) {
    std::uint8_t* blk = bytes.data() + b * kQ5KBytes;
    blk[0] = 0x00U;
    blk[1] = 0x3CU;
    blk[2] = 0x00U;
    blk[3] = 0x3CU;
  }
  return bytes;
}

/// rows * (cols / 256) random Q6_K blocks with the head fp16 pinned to 1.0
/// and the int8 sub-block scales masked into [0, 15].
std::vector<std::uint8_t> MakeQ6K(std::uint32_t rows, std::uint32_t cols,
                                  std::uint32_t seed) {
  const std::size_t row_bytes = cols / 256U * kQ6KBytes;
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) * row_bytes);
  std::uint32_t rng = seed;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>(t::NextRandom(&rng) & 0xFFU);
  }
  for (std::size_t b = 0; b < bytes.size() / kQ6KBytes; ++b) {
    std::uint8_t* blk = bytes.data() + b * kQ6KBytes;
    blk[208] = 0x00U;
    blk[209] = 0x3CU;
    for (int j = 0; j < 16; ++j) {
      blk[192 + j] = static_cast<std::uint8_t>(blk[192 + j] & 0xFU);
    }
  }
  return bytes;
}

/// rows * (cols / 32) random Q8_0 blocks with the fp16 scale pinned to 1.0.
std::vector<std::uint8_t> MakeQ8(std::uint32_t rows, std::uint32_t cols,
                                 std::uint32_t seed) {
  const std::size_t row_bytes = cols / 32U * kQ8Bytes;
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) * row_bytes);
  std::uint32_t rng = seed;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>(t::NextRandom(&rng) & 0xFFU);
  }
  for (std::size_t b = 0; b < bytes.size() / kQ8Bytes; ++b) {
    std::uint8_t* blk = bytes.data() + b * kQ8Bytes;
    blk[0] = 0x00U;
    blk[1] = 0x3CU;
  }
  return bytes;
}

double RelNmse(const std::vector<float>& ref, const std::vector<float>& got) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return num / den;
}

bool BitsDiffer(const std::vector<float>& a, const std::vector<float>& b) {
  std::size_t diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::memcpy(&x, &a[i], sizeof(x));
    std::memcpy(&y, &b[i], sizeof(y));
    diff += x != y;
  }
  return diff != 0;
}

/// One shape and quant: the W8A8 tier vs the scalar GemvRows reference.
bool TestShape(GgmlType type, std::uint32_t rows, std::uint32_t cols,
               std::uint32_t tokens) {
  const char* label = type == GgmlType::kQ8_0
                          ? "Q8_0"
                          : type == GgmlType::kQ6_K ? "Q6_K" : "Q5_K";
  std::printf("%s rows %u cols %u tokens %u\n", label, rows, cols, tokens);
  const std::vector<std::uint8_t> w =
      type == GgmlType::kQ8_0
          ? MakeQ8(rows, cols, 0x13579BDFU)
          : type == GgmlType::kQ6_K ? MakeQ6K(rows, cols, 0x13579BDFU)
                                    : MakeQ5K(rows, cols, 0x13579BDFU);
  const std::size_t row_bytes =
      type == GgmlType::kQ8_0
          ? cols / 32U * kQ8Bytes
          : type == GgmlType::kQ6_K ? cols / 256U * kQ6KBytes
                                    : cols / 256U * kQ5KBytes;
  const auto x =
      t::MakeValues(static_cast<std::size_t>(tokens) * cols, 0xfeedfaceU, 8.0F);
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload w");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);

  const std::size_t scratch_size = q::Glm5W8A8ScratchBytes(tokens, cols);
  const std::size_t out_size = static_cast<std::size_t>(tokens) * rows;
  t::HipBuffer<std::uint8_t> d_scratch(scratch_size);
  t::HipBuffer<float> d_ref(out_size);
  const q::GemvType gemv_type = type == GgmlType::kQ8_0
                                    ? q::GemvType::kQ8_0
                                    : type == GgmlType::kQ6_K
                                          ? q::GemvType::kQ6_K
                                          : q::GemvType::kQ5_K;
  q::GemvRows(d_w.get(), gemv_type, tokens, rows, cols, row_bytes, d_x.get(),
              cols, d_ref.get(), rows, nullptr);
  t::HipBuffer<float> d_got(out_size);
  t::HipBuffer<float> d_got2(out_size);
  q::Glm5W8A8Dense(d_w.get(), type, d_x.get(), d_scratch.get(), d_got.get(),
                   tokens, rows, cols, nullptr);
  q::Glm5W8A8Dense(d_w.get(), type, d_x.get(), d_scratch.get(), d_got2.get(),
                   tokens, rows, cols, nullptr);
  const auto ref = t::Download(&d_ref, out_size);
  const auto got = t::Download(&d_got, out_size);
  const auto again = t::Download(&d_got2, out_size);

  std::size_t nonfinite = 0;
  for (const float v : got) {
    nonfinite += std::isfinite(v) ? 0U : 1U;
  }
  bool ok = Check("w8a8 finite", static_cast<double>(nonfinite), 0.0);
  ok &= Check("w8a8 vs scalar NMSE", RelNmse(ref, got), 1e-4);
  ok &= Check("deterministic replay", BitsDiffer(got, again) ? 1.0 : 0.0, 0.0);
  return ok;
}

}  // namespace

int main() {
  bool ok = q::Glm5W8A8Supported(GgmlType::kQ5_K) &&
            q::Glm5W8A8Supported(GgmlType::kQ6_K) &&
            q::Glm5W8A8Supported(GgmlType::kQ8_0) &&
            !q::Glm5W8A8Supported(GgmlType::kIQ1_M);
  std::printf("Q5_K/Q6_K/Q8_0 support: %s\n", ok ? "OK" : "FAIL");
  try {
    hipDeviceProp_t props{};
    t::CheckHip(hipGetDeviceProperties(&props, 0), "device");
    std::printf("device %s\n", props.name);
    ok &= TestShape(GgmlType::kQ5_K, 144U, 1024U, 16U);
    ok &= TestShape(GgmlType::kQ5_K, 144U, 1024U, 40U);
    ok &= TestShape(GgmlType::kQ5_K, 300U, 2048U, 96U);
    ok &= TestShape(GgmlType::kQ5_K, 4096U, 1024U, 128U);
    ok &= TestShape(GgmlType::kQ6_K, 144U, 1024U, 16U);
    ok &= TestShape(GgmlType::kQ6_K, 144U, 1024U, 40U);
    ok &= TestShape(GgmlType::kQ6_K, 300U, 2048U, 96U);
    ok &= TestShape(GgmlType::kQ6_K, 4096U, 2048U, 128U);
    ok &= TestShape(GgmlType::kQ8_0, 144U, 1024U, 16U);
    ok &= TestShape(GgmlType::kQ8_0, 144U, 1024U, 40U);
    ok &= TestShape(GgmlType::kQ8_0, 300U, 2048U, 96U);
    ok &= TestShape(GgmlType::kQ8_0, 4096U, 1024U, 128U);
    ok &= TestShape(GgmlType::kQ8_0, 3584U, 512U, 256U);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "exception: %s\n", error.what());
    return 1;
  }
  return ok ? 0 : 1;
}
