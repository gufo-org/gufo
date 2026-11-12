#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "staging/models/glm5_next/kernels/rocm/gemv.hpp"
#include "staging/models/glm5_next/kernels/rocm/routed_compact.hpp"
#include "staging/tests/models/glm5_next/hip_test.hpp"

// GEMV tier checks for glm5_next. Synthetic random-byte matrices (with the
// fp16 scale fields patched to finite values) go through every kernel tier;
// the references are the canonical gufo_core CPU dequantizers
// (gufo::quant::DequantizeRow) plus double-precision dots. GemvRows output
// must be bit-identical to the matching Gemv call by construction.
namespace q = gufo::models::glm5_next::rocm;
namespace t = gufo::tests::glm5_next;
namespace core = gufo::core;

namespace {

constexpr std::uint32_t kCols = 1024;
constexpr std::uint32_t kRows = 40;

struct TypeSpec {
  q::GemvType type;
  core::GgmlType core_type;
  const char* name;
  std::size_t row_bytes;              // per row at kCols
  const std::uint16_t* half_offsets;  // fp16 scale fields within one
                                      // super-block, 0xFFFF-terminated
};

constexpr std::size_t Super(std::size_t bytes) {
  return (kCols / 256U) * bytes;
}

constexpr std::uint16_t kEnd = 0xFFFFU;
constexpr std::uint16_t kOffQ8[] = {0, kEnd};
constexpr std::uint16_t kOffQ4K[] = {0, 2, kEnd};
constexpr std::uint16_t kOffQ5K[] = {0, 2, kEnd};
constexpr std::uint16_t kOffQ6K[] = {208, kEnd};
constexpr std::uint16_t kOffQ2K[] = {80, 82, kEnd};
constexpr std::uint16_t kOffQ3K[] = {108, kEnd};
constexpr std::uint16_t kOffD[] = {0, kEnd};

const TypeSpec kTypes[] = {
    {q::GemvType::kQ8_0, core::GgmlType::kQ8_0, "q8_0", (kCols / 32U) * 34U,
     kOffQ8},
    {q::GemvType::kQ4_K, core::GgmlType::kQ4_K, "q4_k", Super(144), kOffQ4K},
    {q::GemvType::kQ5_K, core::GgmlType::kQ5_K, "q5_k", Super(176), kOffQ5K},
    {q::GemvType::kQ6_K, core::GgmlType::kQ6_K, "q6_k", Super(210), kOffQ6K},
    {q::GemvType::kQ2_K, core::GgmlType::kQ2_K, "q2_k", Super(84), kOffQ2K},
    {q::GemvType::kQ3_K, core::GgmlType::kQ3_K, "q3_k", Super(110), kOffQ3K},
    {q::GemvType::kIQ2_XXS, core::GgmlType::kIQ2_XXS, "iq2_xxs", Super(66),
     kOffD},
    {q::GemvType::kIQ1_M, core::GgmlType::kIQ1_M, "iq1_m", Super(56), nullptr},
    {q::GemvType::kIQ3_XXS, core::GgmlType::kIQ3_XXS, "iq3_xxs", Super(98),
     kOffD},
    {q::GemvType::kIQ4_XS, core::GgmlType::kIQ4_XS, "iq4_xs", Super(136),
     kOffD},
    {q::GemvType::kF32, core::GgmlType::kF32, "f32", kCols * 4U, nullptr},
    {q::GemvType::kF16, core::GgmlType::kF16, "f16", kCols * 2U, nullptr},
};

std::vector<std::uint8_t> MakeMatrix(const TypeSpec& spec, std::uint32_t rows,
                                     std::uint32_t seed) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) *
                                  spec.row_bytes);
  if (spec.type == q::GemvType::kF32) {
    const auto values =
        t::MakeValues(static_cast<std::size_t>(rows) * kCols, seed, 1.0F);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
  }
  if (spec.type == q::GemvType::kF16) {
    std::uint32_t rng = seed;
    for (std::size_t i = 0; i < bytes.size(); i += 2) {
      t::PutLittle16(bytes.data() + i, t::FiniteHalf(&rng));
    }
    return bytes;
  }
  std::uint32_t rng = seed;
  for (std::size_t r = 0; r < rows; ++r) {
    std::uint8_t* row = bytes.data() + r * spec.row_bytes;
    for (std::size_t i = 0; i < spec.row_bytes; ++i) {
      row[i] = static_cast<std::uint8_t>(t::NextRandom(&rng) & 0xFFU);
    }
    const std::size_t blocks = kCols / 256U;
    if (spec.type == q::GemvType::kIQ1_M) {
      // d is packed into the top nibbles of the four scale words at byte 48;
      // clearing bit 14 of the fourth word forces a finite exponent.
      for (std::size_t b = 0; b < blocks; ++b) {
        row[b * 56U + 55U] &= 0xBFU;
      }
      continue;
    }
    for (std::size_t b = 0; b < blocks; ++b) {
      t::PatchScaleFields(row + b * (spec.row_bytes / blocks),
                          spec.half_offsets, &rng);
    }
  }
  return bytes;
}

std::vector<float> CpuRow(const TypeSpec& spec, const std::uint8_t* row) {
  std::vector<float> values(kCols);
  if (!gufo::quant::DequantizeRow(spec.core_type, row, values.data(), kCols)) {
    std::fprintf(stderr, "CPU dequant rejected %s\n", spec.name);
    std::abort();
  }
  return values;
}

bool Check(const char* what, double worst, double limit) {
  const bool ok = worst <= limit;
  std::printf("%-34s worst %.3e limit %.0e %s\n", what, worst, limit,
              ok ? "OK" : "FAIL");
  return ok;
}

bool TestEmbed(const TypeSpec& spec, const std::vector<std::uint8_t>& w) {
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload weights");
  t::HipBuffer<float> d_out(kCols);
  bool ok = true;
  const std::uint32_t probes[] = {0U, 17U, kRows - 1U};
  for (const std::uint32_t row : probes) {
    q::EmbedRow(d_w.get(), spec.type, row, kCols, d_out.get(), nullptr);
    const auto got = t::Download(&d_out, kCols);
    const auto ref =
        CpuRow(spec, w.data() + static_cast<std::size_t>(row) * spec.row_bytes);
    ok &= Check((std::string("embed ") + spec.name).c_str(),
                t::WorstRelative(ref, got), 1e-6);
  }
  return ok;
}

bool TestGemv(const TypeSpec& spec, const std::vector<std::uint8_t>& w) {
  const auto x = t::MakeValues(kCols, 0x11112222U, 1.0F);
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload weights");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_out(kRows);
  q::Gemv(d_w.get(), spec.type, kRows, kCols, spec.row_bytes, d_x.get(),
          d_out.get(), nullptr);
  const auto got = t::Download(&d_out, kRows);
  std::vector<float> ref(kRows);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    const auto row =
        CpuRow(spec, w.data() + static_cast<std::size_t>(r) * spec.row_bytes);
    double acc = 0.0;
    for (std::uint32_t k = 0; k < kCols; ++k) {
      acc += static_cast<double>(row[k]) * x[k];
    }
    ref[r] = static_cast<float>(acc);
  }
  return Check((std::string("gemv ") + spec.name).c_str(),
               t::WorstRelative(ref, got), 2e-3);
}

// Each per-token dot must be bit-identical to the single-token kernel.
bool TestRowsBits(const TypeSpec& spec) {
  constexpr std::uint32_t kTokens = 3;
  const auto w = MakeMatrix(
      spec, kRows, 0x5a5a0003U + static_cast<std::uint32_t>(spec.row_bytes));
  const auto x = t::MakeValues(static_cast<std::size_t>(kTokens) * kCols,
                               0x77771234U, 1.0F);
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload weights");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_multi(static_cast<std::size_t>(kTokens) * kRows);
  t::HipBuffer<float> d_single(kRows);
  q::GemvRows(d_w.get(), spec.type, kTokens, kRows, kCols, spec.row_bytes,
              d_x.get(), kCols, d_multi.get(), kRows, nullptr);
  const auto multi =
      t::Download(&d_multi, static_cast<std::size_t>(kTokens) * kRows);
  bool ok = true;
  for (std::uint32_t o = 0; o < kTokens; ++o) {
    q::Gemv(d_w.get(), spec.type, kRows, kCols, spec.row_bytes,
            d_x.get() + static_cast<std::size_t>(o) * kCols, d_single.get(),
            nullptr);
    const auto single = t::Download(&d_single, kRows);
    double diff = 0.0;
    for (std::uint32_t r = 0; r < kRows; ++r) {
      diff = std::max(diff,
                      static_cast<double>(
                          std::bit_cast<std::int32_t>(
                              multi[static_cast<std::size_t>(o) * kRows + r]) !=
                          std::bit_cast<std::int32_t>(single[r])));
    }
    ok &= Check((std::string("gemv_rows ") + spec.name).c_str(), diff, 0.0);
  }
  return ok;
}

// Stacking the per-head GEMV loop into one grouped launch must reproduce
// the per-head calls bit-for-bit.
bool TestRowsGroupedBits(const TypeSpec& spec) {
  constexpr std::uint32_t kGroups = 3U;
  constexpr std::uint32_t kTokens = 2U;
  const auto w = MakeMatrix(
      spec, kGroups * kRows,
      0x9e370007U + static_cast<std::uint32_t>(spec.row_bytes));
  const auto x =
      t::MakeValues(static_cast<std::size_t>(kTokens) * kGroups * kCols,
                    0x44556677U, 1.0F);
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload weights");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  const std::size_t total = static_cast<std::size_t>(kTokens) * kGroups * kRows;
  t::HipBuffer<float> d_multi(total);
  q::GemvRowsGrouped(d_w.get(), spec.type, kTokens, kGroups, kRows, kCols,
                     spec.row_bytes, d_x.get(), kGroups * kCols, kCols,
                     d_multi.get(), kGroups * kRows, nullptr);
  const auto multi = t::Download(&d_multi, total);
  bool ok = true;
  for (std::uint32_t hd = 0; hd < kGroups; ++hd) {
    t::HipBuffer<float> d_single(total);
    q::GemvRows(d_w.get() + static_cast<std::size_t>(hd) * kRows *
                                   spec.row_bytes,
                spec.type, kTokens, kRows, kCols, spec.row_bytes,
                d_x.get() + static_cast<std::size_t>(hd) * kCols,
                kGroups * kCols, d_single.get(), kGroups * kRows, nullptr);
    const auto single = t::Download(&d_single, total);
    double diff = 0.0;
    for (std::uint32_t o = 0; o < kTokens; ++o) {
      for (std::uint32_t r = 0; r < kRows; ++r) {
        const std::size_t base =
            static_cast<std::size_t>(o) * kGroups * kRows;
        diff = std::max(
            diff, static_cast<double>(
                      std::bit_cast<std::int32_t>(multi[base + hd * kRows +
                                                        r]) !=
                      std::bit_cast<std::int32_t>(single[base + r])));
      }
    }
    ok &=
        Check((std::string("gemv_rows_grouped ") + spec.name).c_str(), diff,
              0.0);
  }
  return ok;
}

// Grouped expert decode: random ids over a stacked matrix against the CPU
// dot of each selected expert row.
bool TestGrouped(q::GemvType type, std::size_t expert_stride,
                 const std::vector<std::uint8_t>& stack) {
  const TypeSpec* spec = nullptr;
  for (const auto& candidate : kTypes) {
    if (candidate.type == type) {
      spec = &candidate;
    }
  }
  constexpr std::uint32_t kExperts = 4;
  constexpr std::uint32_t kStackRows = 6;
  constexpr std::uint32_t kUsed = 17;
  const auto x =
      t::MakeValues(static_cast<std::size_t>(4) * kCols, 0x0badf00dU, 1.0F);
  std::vector<std::int32_t> ids(kUsed);
  std::uint32_t rng = 0x1234abcdU;
  for (std::int32_t& id : ids) {
    id = static_cast<std::int32_t>(t::NextRandom(&rng) % kExperts);
  }
  t::HipBuffer<std::uint8_t> d_w(stack.size());
  t::CheckHip(
      hipMemcpy(d_w.get(), stack.data(), stack.size(), hipMemcpyHostToDevice),
      "upload stack");
  t::HipBuffer<std::int32_t> d_ids(kUsed);
  t::CheckHip(
      hipMemcpy(d_ids.get(), ids.data(), ids.size() * sizeof(std::int32_t),
                hipMemcpyHostToDevice),
      "upload ids");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_out(static_cast<std::size_t>(kUsed) * kStackRows);
  constexpr std::uint32_t kSlotsPerRow = 5;
  q::GemvGrouped(d_w.get(), type, expert_stride, d_ids.get(), kUsed,
                 kSlotsPerRow, kStackRows, kCols, d_x.get(), kCols, d_out.get(),
                 nullptr);
  const auto got =
      t::Download(&d_out, static_cast<std::size_t>(kUsed) * kStackRows);
  std::vector<float> ref(got.size());
  for (std::uint32_t s = 0; s < kUsed; ++s) {
    const std::uint8_t* expert =
        stack.data() + static_cast<std::size_t>(ids[s]) * expert_stride;
    const float* xr =
        x.data() + static_cast<std::size_t>(s / kSlotsPerRow) * kCols;
    for (std::uint32_t r = 0; r < kStackRows; ++r) {
      const auto row =
          CpuRow(*spec, expert + static_cast<std::size_t>(r) * spec->row_bytes);
      double acc = 0.0;
      for (std::uint32_t k = 0; k < kCols; ++k) {
        acc += static_cast<double>(row[k]) * xr[k];
      }
      ref[static_cast<std::size_t>(s) * kStackRows + r] =
          static_cast<float>(acc);
    }
  }
  const std::string label = std::string("grouped ") + spec->name;
  return Check(label.c_str(), t::WorstRelative(ref, got), 2e-3);
}

// Bucketed tile-tier parity: over the same random routing (tokens x used
// slots over stacked experts), the slot-order GemvGrouped and the
// compacted 16-row bucket tier (ExpertCounts / BuildRoutedTiles /
// RoutedCompact + GemvRoutedTiles) must produce bit-identical output for
// both Ffn layouts: gate/up sharing one hidden row per token
// (slots_per_row = used) and down with one activation row per slot
// (slots_per_row = 1).
bool TestRouted(q::GemvType type, std::size_t expert_stride,
                const std::vector<std::uint8_t>& stack,
                std::uint32_t slots_per_row) {
  constexpr std::uint32_t kExperts = 9;
  constexpr std::uint32_t kStackRows = 6;
  constexpr std::uint32_t kTokens = 33;
  constexpr std::uint32_t kUsed = 8;
  const std::uint32_t slots = kTokens * kUsed;
  std::vector<std::int32_t> ids(slots);
  std::uint32_t rng = 0x9e3779b9U + slots_per_row;
  for (std::int32_t& id : ids) {
    id = static_cast<std::int32_t>(t::NextRandom(&rng) % kExperts);
  }
  const std::size_t x_rows = slots_per_row == 1U ? slots : kTokens;
  const auto x = t::MakeValues(x_rows * kCols, 0xfeedfaceU, 1.0F);
  t::HipBuffer<std::uint8_t> d_w(stack.size());
  t::CheckHip(
      hipMemcpy(d_w.get(), stack.data(), stack.size(), hipMemcpyHostToDevice),
      "upload stack");
  t::HipBuffer<std::int32_t> d_ids(slots);
  t::CheckHip(
      hipMemcpy(d_ids.get(), ids.data(), ids.size() * sizeof(std::int32_t),
                hipMemcpyHostToDevice),
      "upload ids");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_ref(static_cast<std::size_t>(slots) * kStackRows);
  q::GemvGrouped(d_w.get(), type, expert_stride, d_ids.get(), slots,
                 slots_per_row, kStackRows, kCols, d_x.get(), kCols,
                 d_ref.get(), nullptr);
  constexpr std::uint32_t kTileRows = 16U;
  const std::uint32_t cap =
      q::RoutedTileCapacity(slots, kExperts, kTileRows);
  t::HipBuffer<std::uint32_t> d_counts(kExperts);
  t::HipBuffer<std::int32_t> d_pad(kExperts + 1);
  t::HipBuffer<std::int32_t> d_cur(kExperts);
  t::HipBuffer<std::int32_t> d_slot(
      q::RoutedCompactRows(slots, kExperts));
  t::HipBuffer<std::int32_t> d_tiles(cap);
  q::ExpertCounts(d_ids.get(), slots, kExperts, d_counts.get(), nullptr);
  q::BuildRoutedTiles(d_counts.get(), kExperts, kTileRows, cap, d_tiles.get(),
                      kTileRows, cap, d_tiles.get(), nullptr);
  q::RoutedCompact(d_ids.get(), d_counts.get(), d_pad.get(), d_cur.get(),
                   d_slot.get(), kTokens, kUsed, kExperts, nullptr);
  t::HipBuffer<float> d_out(static_cast<std::size_t>(slots) * kStackRows);
  q::GemvRoutedTiles(d_w.get(), type, expert_stride, d_tiles.get(), cap,
                     d_pad.get(), d_slot.get(), slots_per_row, kStackRows,
                     kCols, d_x.get(), kCols, d_out.get(), nullptr);
  const auto ref =
      t::Download(&d_ref, static_cast<std::size_t>(slots) * kStackRows);
  const auto got =
      t::Download(&d_out, static_cast<std::size_t>(slots) * kStackRows);
  std::size_t diff = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::memcpy(&a, &ref[i], sizeof(a));
    std::memcpy(&b, &got[i], sizeof(b));
    diff += a != b;
  }
  const TypeSpec* spec = nullptr;
  for (const auto& candidate : kTypes) {
    if (candidate.type == type) {
      spec = &candidate;
    }
  }
  const std::string label = std::string("routed==grouped ") + spec->name +
                            (slots_per_row == 1U ? " down" : " gate");
  return Check(label.c_str(), static_cast<double>(diff), 0.0);
}

bool TestF32ToHalf() {
  bool ok = true;
  // Values exactly representable in binary16 round-trip through the kernel
  // conversion and the F16 embed path unchanged.
  constexpr std::size_t kCount = 1024;
  std::vector<float> exact(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    exact[i] = static_cast<float>(static_cast<int>(i) - 512) * 0.125F;
  }
  exact[0] = -65.5F;
  t::HipBuffer<float> d_src(kCount);
  t::Upload(&d_src, exact);
  t::HipBuffer<__half> d_dst(kCount);
  t::HipBuffer<std::uint32_t> d_overflow(1);
  t::CheckHip(hipMemset(d_overflow.get(), 0, sizeof(std::uint32_t)),
              "clear overflow");
  q::ConvertF32ToHalf(d_src.get(), d_dst.get(), kCount, d_overflow.get(),
                      nullptr);
  std::uint32_t overflow = 0;
  t::CheckHip(hipMemcpy(&overflow, d_overflow.get(), sizeof(overflow),
                        hipMemcpyDeviceToHost),
              "download overflow");
  ok &=
      Check("f32_to_half overflow(clean)", static_cast<double>(overflow), 0.0);
  t::HipBuffer<float> d_back(kCount);
  q::EmbedRow(d_dst.get(), q::GemvType::kF16, 0,
              static_cast<std::uint32_t>(kCount), d_back.get(), nullptr);
  const auto back = t::Download(&d_back, kCount);
  ok &=
      Check("f32_to_half roundtrip(exact)", t::WorstRelative(exact, back), 0.0);
  // An out-of-range value saturates the overflow counter.
  std::vector<float> big(2, 70000.0F);
  big[1] = 1.5F;
  t::Upload(&d_src, big);
  t::CheckHip(hipMemset(d_overflow.get(), 0, sizeof(std::uint32_t)),
              "clear overflow");
  q::ConvertF32ToHalf(d_src.get(), d_dst.get(), big.size(), d_overflow.get(),
                      nullptr);
  t::CheckHip(hipMemcpy(&overflow, d_overflow.get(), sizeof(overflow),
                        hipMemcpyDeviceToHost),
              "download overflow");
  ok &=
      Check("f32_to_half overflow(count)", static_cast<double>(overflow), 1.0);
  ok &= (overflow == 1U);
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  try {
    hipDeviceProp_t props{};
    t::CheckHip(hipGetDeviceProperties(&props, 0), "device");
    std::printf("device %s\n", props.name);
    for (const auto& spec : kTypes) {
      const auto w =
          MakeMatrix(spec, kRows,
                     0xC0FFEE01U + static_cast<std::uint32_t>(spec.row_bytes));
      ok &= TestEmbed(spec, w);
      ok &= TestGemv(spec, w);
    }
    for (const auto& spec : kTypes) {
      if (spec.type == q::GemvType::kF32 || spec.type == q::GemvType::kF16) {
        continue;
      }
      ok &= TestRowsBits(spec);
      ok &= TestRowsGroupedBits(spec);
    }
    // Grouped tier over stacked IQ + low-bit matrices and F32.
    for (q::GemvType type :
         {q::GemvType::kIQ2_XXS, q::GemvType::kIQ1_M, q::GemvType::kIQ3_XXS,
          q::GemvType::kIQ4_XS, q::GemvType::kQ2_K, q::GemvType::kQ3_K,
          q::GemvType::kF32}) {
      const TypeSpec* spec = nullptr;
      for (const auto& candidate : kTypes) {
        if (candidate.type == type) {
          spec = &candidate;
        }
      }
      const auto w = MakeMatrix(*spec, 6, 0x35792468U);
      // Replicate the 6-row matrix into 4 experts.
      std::vector<std::uint8_t> stack(w.size() * 4);
      for (std::uint32_t e = 0; e < 4; ++e) {
        std::memcpy(stack.data() + static_cast<std::size_t>(e) * w.size(),
                    w.data(), w.size());
      }
      ok &= TestGrouped(type, w.size(), stack);
    }
    // Routed tier: 9 replicated experts and a wide random routing, both
    // Ffn layouts bit-identical to the grouped tier.
    for (q::GemvType type : {q::GemvType::kIQ2_XXS, q::GemvType::kIQ1_M,
                             q::GemvType::kIQ3_XXS, q::GemvType::kIQ4_XS,
                             q::GemvType::kQ2_K, q::GemvType::kQ3_K}) {
      const TypeSpec* spec = nullptr;
      for (const auto& candidate : kTypes) {
        if (candidate.type == type) {
          spec = &candidate;
        }
      }
      const auto w = MakeMatrix(*spec, 6, 0x24681357U);
      std::vector<std::uint8_t> stack(w.size() * 9);
      for (std::uint32_t e = 0; e < 9; ++e) {
        std::memcpy(stack.data() + static_cast<std::size_t>(e) * w.size(),
                    w.data(), w.size());
      }
      ok &= TestRouted(type, w.size(), stack, 8U);
      ok &= TestRouted(type, w.size(), stack, 1U);
    }
    ok &= TestF32ToHalf();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "exception: %s\n", error.what());
    return 1;
  }
  return ok ? 0 : 1;
}
