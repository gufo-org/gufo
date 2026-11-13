#include "staging/models/glm5_next/kernels/rocm/dense_wmma.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#include "staging/models/glm5_next/kernels/rocm/gemv.hpp"
#include "staging/tests/models/glm5_next/hip_test.hpp"

// Quality contract for the WMMA dense tier over Q5_K and Q8_0: against the
// scalar GemvRows reference (bit-verified against the CPU dequantizer by
// rocm_gemv_test) the wmma tier must stay inside a tight relative NMSE
// budget (f16 fragments change the accumulation order, not the decoded
// values), must reproduce itself bit-for-bit across launches, and may never
// emit a non-finite output. The block scale (and Q5_K min) are pinned to 1.0
// (fp16 halves at the head of each block) so the decoded factors stay
// inside binary16: with random d the scalar tier is still finite where the
// wmma decode is not, and the comparison would measure f16 range, not
// summation order.
namespace q = gufo::models::glm5_next::rocm;
namespace t = gufo::tests::glm5_next;

namespace {

constexpr std::size_t kQ5KBytes = 176U;
constexpr std::size_t kQ8Bytes = 34U;

bool Check(const char* what, double worst, double limit) {
  const bool ok = worst <= limit;
  std::printf("%-34s worst %.3e limit %.0e %s\n", what, worst, limit,
              ok ? "OK" : "FAIL");
  return ok;
}

/// rows * (cols / 256) random Q5_K super-blocks with the fp16 scale and min
/// pinned to 1.0; the scale/min nibbles and the quant bits stay random.
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

/// rows * (cols / 32) random Q8_0 blocks with the fp16 scale pinned to 1.0;
/// the signed magnitude bytes stay random and decode exactly in binary16.
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

/// One shape and quant: dense W[x] tier vs the scalar GemvRows reference.
bool TestShape(q::GemvType type, std::uint32_t rows, std::uint32_t cols,
               std::uint32_t tokens) {
  const bool q8 = type == q::GemvType::kQ8_0;
  std::printf("%s rows %u cols %u tokens %u\n", q8 ? "Q8_0" : "Q5_K", rows,
              cols, tokens);
  const std::vector<std::uint8_t> w =
      q8 ? MakeQ8(rows, cols, 0x13579BDFU) : MakeQ5K(rows, cols, 0x13579BDFU);
  const std::size_t row_bytes = q8 ? cols / 32U * kQ8Bytes
                                   : cols / 256U * kQ5KBytes;
  const auto x =
      t::MakeValues(static_cast<std::size_t>(tokens) * cols, 0xfeedfaceU, 8.0F);
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload w");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);

  const std::size_t out_size = static_cast<std::size_t>(tokens) * rows;
  t::HipBuffer<float> d_ref(out_size);
  q::GemvRows(d_w.get(), type, tokens, rows, cols, row_bytes, d_x.get(), cols,
              d_ref.get(), rows, nullptr);
  t::HipBuffer<float> d_got(out_size);
  t::HipBuffer<float> d_got2(out_size);
  q::DenseWmma(d_w.get(), type, rows, cols, d_x.get(), cols, d_got.get(), rows,
               tokens, nullptr);
  q::DenseWmma(d_w.get(), type, rows, cols, d_x.get(), cols, d_got2.get(), rows,
               tokens, nullptr);
  const auto ref = t::Download(&d_ref, out_size);
  const auto got = t::Download(&d_got, out_size);
  const auto again = t::Download(&d_got2, out_size);

  std::size_t nonfinite = 0;
  for (const float v : got) {
    nonfinite += std::isfinite(v) ? 0U : 1U;
  }
  bool ok = Check("dense finite", static_cast<double>(nonfinite), 0.0);
  ok &= Check("dense vs scalar NMSE", RelNmse(ref, got), 1e-4);
  ok &= Check("deterministic replay", BitsDiffer(got, again) ? 1.0 : 0.0, 0.0);
  return ok;
}

}  // namespace

int main() {
  bool ok = q::DenseWmmaSupports(q::GemvType::kQ5_K) &&
            q::DenseWmmaSupports(q::GemvType::kQ8_0) &&
            !q::DenseWmmaSupports(q::GemvType::kIQ1_M);
  std::printf("Q5_K/Q8_0 support: %s\n", ok ? "OK" : "FAIL");
  try {
    hipDeviceProp_t props{};
    t::CheckHip(hipGetDeviceProperties(&props, 0), "device");
    std::printf("device %s\n", props.name);
    ok &= TestShape(q::GemvType::kQ5_K, 144U, 1024U, 32U);
    ok &= TestShape(q::GemvType::kQ5_K, 144U, 1024U, 40U);
    ok &= TestShape(q::GemvType::kQ5_K, 4096U, 1024U, 128U);
    ok &= TestShape(q::GemvType::kQ5_K, 300U, 2048U, 96U);
    ok &= TestShape(q::GemvType::kQ8_0, 144U, 1024U, 32U);
    ok &= TestShape(q::GemvType::kQ8_0, 144U, 1024U, 40U);
    ok &= TestShape(q::GemvType::kQ8_0, 4096U, 1024U, 128U);
    ok &= TestShape(q::GemvType::kQ8_0, 300U, 2048U, 96U);
    ok &= TestShape(q::GemvType::kQ8_0, 3584U, 512U, 256U);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "exception: %s\n", error.what());
    return 1;
  }
  return ok ? 0 : 1;
}
