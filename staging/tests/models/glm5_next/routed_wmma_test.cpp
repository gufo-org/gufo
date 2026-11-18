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
#include "staging/models/glm5_next/kernels/rocm/ops.hpp"
#include "staging/models/glm5_next/kernels/rocm/routed_compact.hpp"
#include "staging/models/glm5_next/kernels/rocm/routed_wmma.hpp"
#include "staging/tests/models/glm5_next/hip_test.hpp"

// Quality contract for the WMMA routed tier: against the scalar
// GemvRoutedTiles (+ SwigluClamp for gate/up) reference (already
// bit-verified against the CPU dequantizer by rocm_gemv_test) the wmma tier
// must stay inside a tight relative NMSE budget (f16 fragments change the
// accumulation order, not the decoded values), must reproduce itself
// bit-for-bit across launches, and may never emit a non-finite activation.
// The block scales are pinned to 1.0 at each codec's scale offsets (see
// MakeRows) so the decoded factors stay inside binary16: with random d the
// scalar tier is still finite where the wmma decode is not, and the
// comparison would measure f16 range, not summation order.
namespace q = gufo::models::glm5_next::rocm;
namespace t = gufo::tests::glm5_next;

namespace {

constexpr std::uint32_t kCols = 1024;
constexpr std::uint32_t kRows = 144;  // 128-row block + a 16-row tail
constexpr std::uint32_t kExperts = 9;
constexpr std::uint32_t kTokens = 50;
constexpr std::uint32_t kUsed = 8;
constexpr float kClamp = 7.0F;
constexpr std::size_t kIq1MBytes = 56U;
constexpr std::size_t kIq2XxsBytes = 66U;
constexpr std::size_t kIq3XxsBytes = 98U;
constexpr std::size_t kQ2KBytes = 84U;
constexpr std::size_t kQ3KBytes = 110U;

bool Check(const char* what, double worst, double limit) {
  const bool ok = worst <= limit;
  std::printf("%-34s worst %.3e limit %.0e %s\n", what, worst, limit,
              ok ? "OK" : "FAIL");
  return ok;
}

/// rows * row_bytes of random codes over 256-element blocks with the block
/// scale pinned to 1.0: IQ1_M fixes the four top nibbles of the scale
/// words (bytes 49/51/53/55) to the F16 pattern 0x3C00; IQ2_XXS/IQ3_XXS
/// store d as fp16 at bytes 0..1 of each block; Q2_K stores d/dmin as fp16
/// at bytes 80..81/82..83; Q3_K stores d as fp16 at bytes 108..109. The
/// remaining bytes stay random codes / per-group scale bits.
std::vector<std::uint8_t> MakeRows(std::uint32_t rows, std::size_t block_bytes,
                                    std::uint32_t seed) {
  const std::size_t row_bytes = (kCols / 256U) * block_bytes;
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) * row_bytes);
  std::uint32_t rng = seed;
  for (std::size_t r = 0; r < rows; ++r) {
    std::uint8_t* row = bytes.data() + r * row_bytes;
    for (std::size_t i = 0; i < row_bytes; ++i) {
      row[i] = static_cast<std::uint8_t>(t::NextRandom(&rng) & 0xFFU);
    }
    for (std::size_t b = 0; b < kCols / 256U; ++b) {
      std::uint8_t* blk = row + b * block_bytes;
      if (block_bytes == kIq1MBytes) {
        blk[49] &= 0x0FU;
        blk[51] &= 0x0FU;
        blk[53] = static_cast<std::uint8_t>((blk[53] & 0x0FU) | 0xC0U);
        blk[55] = static_cast<std::uint8_t>((blk[55] & 0x0FU) | 0x30U);
      } else if (block_bytes == kQ2KBytes) {
        blk[80] = 0x00U;
        blk[81] = 0x3CU;
        blk[82] = 0x00U;
        blk[83] = 0x3CU;
      } else if (block_bytes == kQ3KBytes) {
        blk[108] = 0x00U;
        blk[109] = 0x3CU;
      } else {
        blk[0] = 0x00U;
        blk[1] = 0x3CU;
      }
    }
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

/// Fused gate/up + swiglu over IQ1_M, IQ2_XXS or Q2_K stacks against two
/// scalar GemvRoutedTiles + the clamp pass (the Ffn fallback).
bool TestFusedGateUp(const char* label, q::GemvType type,
                     std::size_t block_bytes) {
  std::printf("gate/up %s\n", label);
  const std::uint32_t slots = kTokens * kUsed;
  const std::vector<std::uint8_t> gate =
      MakeRows(kExperts * kRows, block_bytes, 0x13579BDFU);
  const std::vector<std::uint8_t> up =
      MakeRows(kExperts * kRows, block_bytes, 0x2468ACE0U);
  std::vector<std::int32_t> ids(slots);
  std::uint32_t rng = 0x9e3779b9U;
  for (std::int32_t& id : ids) {
    id = static_cast<std::int32_t>(t::NextRandom(&rng) % kExperts);
  }
  const auto x = t::MakeValues(static_cast<std::size_t>(kTokens) * kCols,
                               0xfeedfaceU, 8.0F);

  t::HipBuffer<std::uint8_t> d_gate(gate.size());
  t::HipBuffer<std::uint8_t> d_up(up.size());
  t::CheckHip(hipMemcpy(d_gate.get(), gate.data(), gate.size(),
                        hipMemcpyHostToDevice),
              "upload gate");
  t::CheckHip(hipMemcpy(d_up.get(), up.data(), up.size(),
                        hipMemcpyHostToDevice),
              "upload up");
  t::HipBuffer<std::int32_t> d_ids(slots);
  t::CheckHip(hipMemcpy(d_ids.get(), ids.data(), ids.size() *
                                                 sizeof(std::int32_t),
                        hipMemcpyHostToDevice),
              "upload ids");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);

  constexpr std::uint32_t kTileRows = 16U;
  constexpr std::uint32_t kWideRows = 32U;
  const std::uint32_t cap = q::RoutedTileCapacity(slots, kExperts, kTileRows);
  const std::uint32_t wide_cap =
      q::RoutedTileCapacity(slots, kExperts, kWideRows);
  t::HipBuffer<std::uint32_t> d_counts(kExperts);
  t::HipBuffer<std::int32_t> d_pad(kExperts + 1);
  t::HipBuffer<std::int32_t> d_cur(kExperts);
  t::HipBuffer<std::int32_t> d_slot(q::RoutedCompactRows(slots, kExperts));
  t::HipBuffer<std::int32_t> d_tiles(cap);
  t::HipBuffer<std::int32_t> d_tiles32(wide_cap);
  q::ExpertCounts(d_ids.get(), slots, kExperts, d_counts.get(), nullptr);
  q::BuildRoutedTiles(d_counts.get(), kExperts, kTileRows, cap, d_tiles.get(),
                      kWideRows, wide_cap, d_tiles32.get(), nullptr);
  q::RoutedCompact(d_ids.get(), d_counts.get(), d_pad.get(), d_cur.get(),
                   d_slot.get(), kTokens, kUsed, kExperts, nullptr);

  // The production entry converts the hidden-state rows once for the tier.
  t::HipBuffer<__half> d_xh(x.size());
  t::HipBuffer<std::uint32_t> d_ovf(1);
  t::CheckHip(hipMemset(d_ovf.get(), 0, sizeof(std::uint32_t)), "zero ovf");
  q::ConvertF32ToHalf(d_x.get(), d_xh.get(), x.size(), d_ovf.get(), nullptr);
  std::uint32_t ovf = 1U;
  t::CheckHip(hipMemcpy(&ovf, d_ovf.get(), sizeof(ovf), hipMemcpyDeviceToHost),
              "download ovf");
  bool ok = Check("x-half overflow", static_cast<double>(ovf), 0.0);

  // Scalar reference: the Ffn fallback (two routed GEMVs + the clamp pass).
  const std::size_t out_rows = static_cast<std::size_t>(slots) * kRows;
  t::HipBuffer<float> d_g(out_rows);
  t::HipBuffer<float> d_u(out_rows);
  t::HipBuffer<float> d_ref(out_rows);
  q::GemvRoutedTiles(d_gate.get(), type, gate.size() / kExperts, d_tiles.get(),
                     cap, d_pad.get(), d_slot.get(), kUsed, kRows, kCols,
                     d_x.get(), kCols, d_g.get(), nullptr);
  q::GemvRoutedTiles(d_up.get(), type, up.size() / kExperts, d_tiles.get(), cap,
                     d_pad.get(), d_slot.get(), kUsed, kRows, kCols, d_x.get(),
                     kCols, d_u.get(), nullptr);
  q::SwigluClamp(d_g.get(), d_u.get(), d_ref.get(),
                 static_cast<std::uint32_t>(slots), kRows, kRows, kRows, kRows,
                 kClamp, kClamp, nullptr);

  t::HipBuffer<float> d_act(out_rows);
  t::HipBuffer<float> d_act2(out_rows);
  q::RoutedGateUpWmma(type, d_gate.get(), d_up.get(), gate.size() / kExperts,
                      gate.size(), d_xh.get(), kCols, d_tiles32.get(),
                      wide_cap, d_pad.get(), d_slot.get(), kUsed, kRows, kCols,
                      kClamp, d_act.get(), nullptr);
  q::RoutedGateUpWmma(type, d_gate.get(), d_up.get(), up.size() / kExperts,
                      up.size(), d_xh.get(), kCols, d_tiles32.get(), wide_cap,
                      d_pad.get(), d_slot.get(), kUsed, kRows, kCols, kClamp,
                      d_act2.get(), nullptr);
  const auto ref = t::Download(&d_ref, out_rows);
  const auto got = t::Download(&d_act, out_rows);
  const auto again = t::Download(&d_act2, out_rows);

  std::size_t nonfinite = 0;
  for (const float v : got) {
    nonfinite += std::isfinite(v) ? 0U : 1U;
  }
  ok &= Check("fused finite", static_cast<double>(nonfinite), 0.0);
  ok &= Check("fused vs scalar NMSE", RelNmse(ref, got), 1e-4);
  ok &= Check("deterministic replay",
              BitsDiffer(got, again) ? 1.0 : 0.0, 0.0);
  return ok;
}

/// Down projection over IQ3_XXS/Q3_K stacks against the scalar bucketed
/// down GEMV (slots_per_row = 1: every slot carries its own activation row).
bool TestRoutedDown(const char* label, q::GemvType type,
                    std::size_t block_bytes) {
  std::printf("down %s\n", label);
  const std::uint32_t slots = kTokens * kUsed;
  const std::vector<std::uint8_t> down =
      MakeRows(kExperts * kRows, block_bytes, 0x5eed1234U);
  std::vector<std::int32_t> ids(slots);
  std::uint32_t rng = 0x9e3779b9U;
  for (std::int32_t& id : ids) {
    id = static_cast<std::int32_t>(t::NextRandom(&rng) % kExperts);
  }
  // One activation row per slot (the intermediate moe_inter_).
  const auto x = t::MakeValues(static_cast<std::size_t>(slots) * kCols,
                               0xfeedfaceU, 8.0F);

  t::HipBuffer<std::uint8_t> d_down(down.size());
  t::CheckHip(hipMemcpy(d_down.get(), down.data(), down.size(),
                        hipMemcpyHostToDevice),
              "upload down");
  t::HipBuffer<std::int32_t> d_ids(slots);
  t::CheckHip(hipMemcpy(d_ids.get(), ids.data(), ids.size() *
                                                 sizeof(std::int32_t),
                        hipMemcpyHostToDevice),
              "upload ids");
  auto dx = t::HipBuffer<float>(x.size());
  t::Upload(&dx, x);
  // The MakeValues range is far inside f16; the production entry feeds the
  // clamped-range intermediates through the same ConvertF32ToHalf guard.
  t::HipBuffer<__half> d_xh(x.size());
  t::HipBuffer<std::uint32_t> d_ovf(1);
  t::CheckHip(hipMemset(d_ovf.get(), 0, sizeof(std::uint32_t)), "zero ovf");
  q::ConvertF32ToHalf(dx.get(), d_xh.get(), x.size(), d_ovf.get(), nullptr);
  std::uint32_t ovf = 1U;
  t::CheckHip(hipMemcpy(&ovf, d_ovf.get(), sizeof(ovf), hipMemcpyDeviceToHost),
              "download ovf");
  bool ok = Check("x-half overflow", static_cast<double>(ovf), 0.0);

  constexpr std::uint32_t kTileRows = 16U;
  constexpr std::uint32_t kWideRows = 32U;
  const std::uint32_t cap = q::RoutedTileCapacity(slots, kExperts, kTileRows);
  const std::uint32_t wide_cap =
      q::RoutedTileCapacity(slots, kExperts, kWideRows);
  t::HipBuffer<std::uint32_t> d_counts(kExperts);
  t::HipBuffer<std::int32_t> d_pad(kExperts + 1);
  t::HipBuffer<std::int32_t> d_cur(kExperts);
  t::HipBuffer<std::int32_t> d_slot(q::RoutedCompactRows(slots, kExperts));
  t::HipBuffer<std::int32_t> d_tiles(cap);
  t::HipBuffer<std::int32_t> d_tiles32(wide_cap);
  q::ExpertCounts(d_ids.get(), slots, kExperts, d_counts.get(), nullptr);
  q::BuildRoutedTiles(d_counts.get(), kExperts, kTileRows, cap, d_tiles.get(),
                      kWideRows, wide_cap, d_tiles32.get(), nullptr);
  q::RoutedCompact(d_ids.get(), d_counts.get(), d_pad.get(), d_cur.get(),
                   d_slot.get(), kTokens, kUsed, kExperts, nullptr);

  const std::size_t out_rows = static_cast<std::size_t>(slots) * kRows;
  t::HipBuffer<float> d_ref(out_rows);
  q::GemvRoutedTiles(d_down.get(), type, down.size() / kExperts,
                     d_tiles.get(), cap, d_pad.get(), d_slot.get(), 1U, kRows,
                     kCols, dx.get(), kCols, d_ref.get(), nullptr);

  t::HipBuffer<float> d_act(out_rows);
  t::HipBuffer<float> d_act2(out_rows);
  q::RoutedDownWmma(type, d_down.get(), down.size() / kExperts,
                    down.size(), d_xh.get(), kCols, d_tiles32.get(), wide_cap,
                    d_pad.get(), d_slot.get(), kRows, kCols, d_act.get(),
                    nullptr);
  q::RoutedDownWmma(type, d_down.get(), down.size() / kExperts,
                    down.size(), d_xh.get(), kCols, d_tiles32.get(), wide_cap,
                    d_pad.get(), d_slot.get(), kRows, kCols, d_act2.get(),
                    nullptr);
  const auto ref = t::Download(&d_ref, out_rows);
  const auto got = t::Download(&d_act, out_rows);
  const auto again = t::Download(&d_act2, out_rows);

  std::size_t nonfinite = 0;
  for (const float v : got) {
    nonfinite += std::isfinite(v) ? 0U : 1U;
  }
  ok &= Check("down finite", static_cast<double>(nonfinite), 0.0);
  ok &= Check("down vs scalar NMSE", RelNmse(ref, got), 1e-4);
  ok &= Check("deterministic replay",
              BitsDiffer(got, again) ? 1.0 : 0.0, 0.0);
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  try {
    hipDeviceProp_t props{};
    t::CheckHip(hipGetDeviceProperties(&props, 0), "device");
    std::printf("device %s\n", props.name);
    ok &= TestFusedGateUp("IQ1_M", q::GemvType::kIQ1_M, kIq1MBytes);
    ok &= TestFusedGateUp("IQ2_XXS", q::GemvType::kIQ2_XXS, kIq2XxsBytes);
    ok &= TestFusedGateUp("Q2_K", q::GemvType::kQ2_K, kQ2KBytes);
    ok &= TestRoutedDown("IQ3_XXS", q::GemvType::kIQ3_XXS, kIq3XxsBytes);
    ok &= TestRoutedDown("Q3_K", q::GemvType::kQ3_K, kQ3KBytes);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "exception: %s\n", error.what());
    return 1;
  }
  return ok ? 0 : 1;
}
