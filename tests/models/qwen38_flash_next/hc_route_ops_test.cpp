// A prefill row must not depend on the pass that computes it. A prompt
// prefilled in one wide pass and the same prompt restored from a checkpoint
// and finished by a short pass reach each layer's HC combines and mixers
// through different kernels: from 96 rows the attention combine also writes
// the FFN mixer's inject partials, the mixer projections are fused
// (HcDownF16Gemm, HcMixF16Gemm) and the FFN combine writes the next
// attention mixer's partials; below 96 rows the separate W8A8 / SiLU / F16
// GEMM / epilogue kernels do. One layer's chain over model shapes (hidden
// 2560, four streams, rank 320, top-10 routed rows) runs once over a wide
// reference pass and again over short passes of 1, 7, 9 and 95 rows and a
// second wide pass of 96, at the head, middle and tail of the reference.
// Every row of every intermediate the next operator reads must match bit
// for bit. The reference outputs' hashes are printed so a change of the
// wide route's arithmetic is visible across builds.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kStreams = 4;
constexpr std::uint32_t kRank = 320;
constexpr std::uint32_t kUsed = 10;
constexpr std::size_t kDim = std::size_t{kStreams} * kHidden;
constexpr float kEps = 1e-6F;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }
  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

/// Uniform values in [offset - scale, offset + scale).
std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale, float offset = 0.0F) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = offset +
            scale *
                static_cast<float>(
                    static_cast<int>(NextRandom(&seed) & 0xFFFFU) - 32768) /
                32768.0F;
  }
  return values;
}

std::vector<__half> ToHalf(const std::vector<float>& values) {
  std::vector<__half> half(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    half[i] = __float2half_rn(values[i]);
  return half;
}

/// Q8_0 rows [rows][cols] with small F16 scales.
std::vector<std::uint8_t> MakeQ8_0(std::size_t rows, std::size_t cols,
                                   std::uint32_t seed) {
  std::vector<std::uint8_t> weights(rows * cols / 32 * 34);
  for (std::size_t block = 0; block < weights.size() / 34; ++block) {
    const __half scale = __float2half_rn(
        static_cast<float>(1 + NextRandom(&seed) % 13) / 8192.0F);
    std::memcpy(weights.data() + block * 34, &scale, sizeof(scale));
    for (std::size_t j = 0; j < 32; ++j) {
      weights[block * 34 + 2 + j] = static_cast<std::uint8_t>(
          static_cast<int>(NextRandom(&seed) % 255) - 127);
    }
  }
  return weights;
}

template<typename T>
void UploadRows(HipBuffer<T>* dst, const std::vector<T>& src, std::size_t first,
                std::size_t count) {
  CheckHip(hipMemcpy(dst->get(), src.data() + first, count * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
}

template<typename T>
std::vector<std::uint8_t> Bytes(HipBuffer<T>* buffer, std::size_t count) {
  std::vector<std::uint8_t> bytes(count * sizeof(T));
  CheckHip(hipMemcpy(bytes.data(), buffer->get(), bytes.size(),
                     hipMemcpyDeviceToHost),
           "download");
  return bytes;
}

/// Row t's 32-wide blocks of a tiled Q8 buffer (K = cols): each block's
/// 32 codes followed by its scale, so rows of passes with different tile
/// offsets compare directly.
std::vector<std::uint8_t> TiledRow(const std::vector<std::uint8_t>& tiled,
                                   std::size_t t, std::size_t cols) {
  const std::size_t blocks = cols / 32;
  std::vector<std::uint8_t> row(blocks * 36);
  for (std::size_t kb = 0; kb < blocks; ++kb) {
    const std::uint8_t* tile = tiled.data() + ((t / 16) * blocks + kb) * 576;
    for (std::size_t j = 0; j < 32; ++j)
      row[kb * 36 + j] = tile[(j / 16) * 256 + (t % 16) * 16 + j % 16];
    std::memcpy(row.data() + kb * 36 + 32, tile + 512 + (t % 16) * 4, 4);
  }
  return row;
}

std::uint64_t Fnv1a(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t hash = 0xCBF29CE484222325ULL;
  for (const std::uint8_t b : bytes) {
    hash ^= b;
    hash *= 0x100000001B3ULL;
  }
  return hash;
}

/// One layer's inputs over the reference rows, host side.
struct Inputs {
  explicit Inputs(std::uint32_t n)
      : res(MakeValues(std::size_t{n} * kDim, 0x7E570001U + n, 1.0F)),
        attn_out(MakeValues(std::size_t{n} * kHidden, 0x7E570002U, 0.3F)),
        // The attention mixer's partials: logits of order one.
        inject(MakeValues(
            std::size_t{n} * kStreams * q::HcInjectPartsVec4(kHidden),
            0x7E570003U, 0.5F)),
        experts(ToHalf(
            MakeValues(std::size_t{n} * kUsed * kHidden, 0x7E570004U, 1.0F))),
        weights(MakeValues(std::size_t{n} * kUsed, 0x7E570005U, 0.05F, 0.1F)),
        shared(MakeValues(std::size_t{n} * kHidden, 0x7E570006U, 0.5F)),
        gate(MakeValues(n, 0x7E570007U, 2.0F)) {}
  std::vector<float> res, attn_out, inject;
  std::vector<__half> experts;
  std::vector<float> weights, shared, gate;
};

/// Layer constants (device).
struct Weights {
  Weights()
      : down(kRank * kDim / 32 * 34),
        up(kDim * kRank / 32 * 34),
        gamma_ffn(kDim),
        gamma_attn(kDim),
        inject_ffn(kStreams * kDim),
        inject_attn(kStreams * kDim) {
    const auto d = MakeQ8_0(kRank, kDim, 0x3E1C0001U);
    const auto u = MakeQ8_0(kDim, kRank, 0x3E1C0002U);
    UploadRows(&down, d, 0, d.size());
    UploadRows(&up, u, 0, u.size());
    UploadRows(&gamma_ffn, MakeValues(kDim, 0x3E1C0003U, 0.25F, 1.0F), 0, kDim);
    UploadRows(&gamma_attn, MakeValues(kDim, 0x3E1C0004U, 0.25F, 1.0F), 0,
               kDim);
    UploadRows(&inject_ffn, MakeValues(kStreams * kDim, 0x3E1C0005U, 0.02F), 0,
               kStreams * kDim);
    UploadRows(&inject_attn, MakeValues(kStreams * kDim, 0x3E1C0006U, 0.02F), 0,
               kStreams * kDim);
  }
  HipBuffer<std::uint8_t> down, up;
  HipBuffer<float> gamma_ffn, gamma_attn, inject_ffn, inject_attn;
};

/// Every intermediate of one pass, downloaded.
struct Pass {
  std::uint32_t first = 0;  ///< reference row of local row 0
  std::uint32_t n = 0;
  std::vector<std::uint8_t> defer_w, ffn_xn, ffn_xq, ffn_lo, ffn_mixed,
      ffn_half, ffn_mq, ffn_inject, res, attn_xn, attn_xq, attn_lo, attn_mixed,
      attn_half, attn_mq, attn_inject;
};

/// The mixer as Executor::HcMix runs it on the F16 route: fused projections
/// from 96 rows, else W8A8 down, SiLU(x / 4), F16 rows, F16 up GEMM and the
/// epilogue (which then also writes the partials unless the combine did).
void Mixer(Weights* w, HipBuffer<__half>* xn, HipBuffer<std::uint8_t>* xq,
           const float* inject_w, bool inject_done, std::uint32_t n,
           HipBuffer<__half>* lo, HipBuffer<float>* lo_f32,
           HipBuffer<float>* gate, HipBuffer<float>* mixed,
           HipBuffer<__half>* mixed_half, HipBuffer<std::uint8_t>* mixed_q8,
           HipBuffer<float>* inject) {
  const std::size_t lo_count = std::size_t{n} * kRank;
  if (n >= 96) {
    if (!q::HcDownF16Gemm(w->down.get(), xq->get(), lo->get(), n, nullptr) ||
        !q::HcMixF16Gemm(w->up.get(), lo->get(), xn->get(),
                         inject_done ? nullptr : inject_w, mixed->get(),
                         mixed_half->get(), mixed_q8->get(), inject->get(), n,
                         kHidden, kRank, nullptr))
      throw std::runtime_error("fused HC mixer rejected");
    return;
  }
  if (!q::W8A8Gemm(w->down.get(), xq->get(), lo_f32->get(), n, kRank, kDim,
                   nullptr))
    throw std::runtime_error("W8A8 HC down rejected");
  q::SiluScale(lo_f32->get(), 1.0F / kStreams, lo_count, nullptr);
  q::NarrowActivations(lo_f32->get(), lo->get(), false, lo_count, nullptr);
  if (!q::DenseF16Gemm(w->up.get(), lo->get(), gate->get(), n, kDim, kRank,
                       nullptr))
    throw std::runtime_error("F16 HC up rejected");
  q::HcMixEpilogueVec4F16(xn->get(), gate->get(), inject_w, mixed->get(),
                          mixed_half->get(), mixed_q8->get(), inject->get(), n,
                          kHidden, nullptr);
}

/// Attention combine (deferring), FFN mixer, fused MoE FFN combine (owing
/// the deferred update) and the next attention mixer over reference rows
/// [first, first + n), as Executor::Combine / HcMix order them.
Pass RunPass(const Inputs& in, Weights* w, std::uint32_t first,
             std::uint32_t n) {
  const std::uint32_t parts = q::HcInjectPartsVec4(kHidden);
  const bool wide = n >= 96;
  const std::size_t rows = std::size_t{n} * kDim;
  const std::size_t inject_count = std::size_t{n} * kStreams * parts;
  HipBuffer<float> res(rows), attn_out(std::size_t{n} * kHidden);
  HipBuffer<float> inject(inject_count), defer_w(std::size_t{n} * kStreams);
  HipBuffer<__half> experts(std::size_t{n} * kUsed * kHidden);
  HipBuffer<float> weights(std::size_t{n} * kUsed);
  HipBuffer<float> shared(std::size_t{n} * kHidden), gate_logit(n);
  UploadRows(&res, in.res, std::size_t{first} * kDim, rows);
  UploadRows(&attn_out, in.attn_out, std::size_t{first} * kHidden,
             std::size_t{n} * kHidden);
  UploadRows(&inject, in.inject, std::size_t{first} * kStreams * parts,
             inject_count);
  UploadRows(&experts, in.experts, std::size_t{first} * kUsed * kHidden,
             std::size_t{n} * kUsed * kHidden);
  UploadRows(&weights, in.weights, std::size_t{first} * kUsed,
             std::size_t{n} * kUsed);
  UploadRows(&shared, in.shared, std::size_t{first} * kHidden,
             std::size_t{n} * kHidden);
  UploadRows(&gate_logit, in.gate, first, n);

  const std::size_t xq_bytes = q::Q8TiledBytes(n, kDim);
  const std::size_t mq_bytes = q::Q8TiledBytes(n, kHidden);
  HipBuffer<__half> xn(rows), lo(std::size_t{n} * kRank);
  HipBuffer<std::uint8_t> xq(xq_bytes), mq(mq_bytes);
  HipBuffer<float> lo_f32(std::size_t{n} * kRank), gate(rows);
  HipBuffer<float> mixed(std::size_t{n} * kHidden);
  HipBuffer<__half> mixed_half(std::size_t{n} * kHidden);
  CheckHip(hipMemset(xq.get(), 0, xq_bytes), "clear Q8");
  CheckHip(hipMemset(mq.get(), 0, mq_bytes), "clear Q8");

  Pass p;
  p.first = first;
  p.n = n;
  // The attention combine defers the residual update to the FFN combine;
  // a wide pass also writes the FFN mixer's partials over the ones it read.
  if (!q::HcCombineF16(res.get(), attn_out.get(), inject.get(), parts,
                       w->gamma_ffn.get(), xn.get(), xq.get(), n, kHidden,
                       kStreams, kEps, nullptr, defer_w.get(), inject.get(),
                       wide ? w->inject_ffn.get() : nullptr))
    throw std::runtime_error("deferring HC combine rejected");
  Mixer(w, &xn, &xq, w->inject_ffn.get(), wide, n, &lo, &lo_f32, &gate, &mixed,
        &mixed_half, &mq, &inject);
  CheckHip(hipDeviceSynchronize(), "FFN mixer");
  p.defer_w = Bytes(&defer_w, std::size_t{n} * kStreams);
  p.ffn_xn = Bytes(&xn, rows);
  p.ffn_xq = Bytes(&xq, xq_bytes);
  p.ffn_lo = Bytes(&lo, std::size_t{n} * kRank);
  p.ffn_mixed = Bytes(&mixed, std::size_t{n} * kHidden);
  p.ffn_half = Bytes(&mixed_half, std::size_t{n} * kHidden);
  p.ffn_mq = Bytes(&mq, mq_bytes);
  p.ffn_inject = Bytes(&inject, inject_count);

  // The FFN combine with the MoE epilogue folded in settles the deferred
  // attention update; a wide pass also writes the next attention mixer's
  // partials.
  CheckHip(hipMemset(xq.get(), 0, xq_bytes), "clear Q8");
  CheckHip(hipMemset(mq.get(), 0, mq_bytes), "clear Q8");
  if (!q::HcCombineMoeF16(res.get(), experts.get(), weights.get(), shared.get(),
                          gate_logit.get(), 1, kUsed, inject.get(), parts,
                          w->gamma_attn.get(), xn.get(), xq.get(), n, kHidden,
                          kStreams, kEps, nullptr, attn_out.get(),
                          defer_w.get(), inject.get(),
                          wide ? w->inject_attn.get() : nullptr))
    throw std::runtime_error("fused MoE HC combine rejected");
  Mixer(w, &xn, &xq, w->inject_attn.get(), wide, n, &lo, &lo_f32, &gate, &mixed,
        &mixed_half, &mq, &inject);
  CheckHip(hipDeviceSynchronize(), "attention mixer");
  p.res = Bytes(&res, rows);
  p.attn_xn = Bytes(&xn, rows);
  p.attn_xq = Bytes(&xq, xq_bytes);
  p.attn_lo = Bytes(&lo, std::size_t{n} * kRank);
  p.attn_mixed = Bytes(&mixed, std::size_t{n} * kHidden);
  p.attn_half = Bytes(&mixed_half, std::size_t{n} * kHidden);
  p.attn_mq = Bytes(&mq, mq_bytes);
  p.attn_inject = Bytes(&inject, inject_count);
  return p;
}

/// The reference pass's view of one quantity: per-row bytes (`row_bytes`
/// wide) or tiled Q8 rows of width `cols`.
struct Quantity {
  const char* name;
  std::vector<std::uint8_t> Pass::* field;
  std::size_t row_bytes;  ///< 0: tiled Q8 of `cols`
  std::size_t cols;
};

const std::vector<Quantity>& Quantities() {
  const std::size_t parts = q::HcInjectPartsVec4(kHidden);
  static const std::vector<Quantity> list = {
      {"stream weights", &Pass::defer_w, kStreams * sizeof(float), 0},
      {"FFN norm F16", &Pass::ffn_xn, kDim * sizeof(__half), 0},
      {"FFN norm Q8", &Pass::ffn_xq, 0, kDim},
      {"FFN low rank", &Pass::ffn_lo, kRank * sizeof(__half), 0},
      {"FFN mixed F32", &Pass::ffn_mixed, kHidden * sizeof(float), 0},
      {"FFN mixed F16", &Pass::ffn_half, kHidden * sizeof(__half), 0},
      {"FFN mixed Q8", &Pass::ffn_mq, 0, kHidden},
      {"FFN inject partials", &Pass::ffn_inject,
       kStreams * parts * sizeof(float), 0},
      {"residual", &Pass::res, kDim * sizeof(float), 0},
      {"attention norm F16", &Pass::attn_xn, kDim * sizeof(__half), 0},
      {"attention norm Q8", &Pass::attn_xq, 0, kDim},
      {"attention low rank", &Pass::attn_lo, kRank * sizeof(__half), 0},
      {"attention mixed F32", &Pass::attn_mixed, kHidden * sizeof(float), 0},
      {"attention mixed F16", &Pass::attn_half, kHidden * sizeof(__half), 0},
      {"attention mixed Q8", &Pass::attn_mq, 0, kHidden},
      {"attention inject partials", &Pass::attn_inject,
       kStreams * parts * sizeof(float), 0},
  };
  return list;
}

std::vector<std::uint8_t> Row(const Pass& p, const Quantity& qty,
                              std::uint32_t local) {
  const auto& bytes = p.*(qty.field);
  if (qty.row_bytes == 0)
    return TiledRow(bytes, local, qty.cols);
  const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(
                                         std::size_t{local} * qty.row_bytes);
  return {begin, begin + static_cast<std::ptrdiff_t>(qty.row_bytes)};
}

/// Rows of `candidate` that differ from the reference, per quantity.
std::size_t Compare(const Pass& reference, const Pass& candidate) {
  std::size_t failures = 0;
  for (const Quantity& qty : Quantities()) {
    std::size_t differ = 0;
    for (std::uint32_t r = 0; r < candidate.n; ++r)
      differ +=
          Row(reference, qty, candidate.first + r) != Row(candidate, qty, r);
    if (differ != 0) {
      std::cout << "  " << qty.name << ": " << differ << " of " << candidate.n
                << " rows differ\n";
    }
    failures += differ;
  }
  return failures;
}

void PrintHashes(const Pass& reference) {
  for (const Quantity& qty : Quantities()) {
    std::vector<std::uint8_t> all;
    for (std::uint32_t r = 0; r < reference.n; ++r) {
      const auto row = Row(reference, qty, r);
      all.insert(all.end(), row.begin(), row.end());
    }
    std::cout << "  reference " << qty.name << " fnv1a " << std::hex
              << std::setw(16) << std::setfill('0') << Fnv1a(all) << std::dec
              << std::setfill(' ') << '\n';
  }
}

/// The wide pass over `n` rows, and short and wide passes over its head,
/// middle and tail rows.
std::size_t CheckLayerRoutes(std::uint32_t n) {
  const Inputs in(n);
  Weights w;
  const Pass reference = RunPass(in, &w, 0, n);
  std::cout << "HC layer routes, reference pass of " << n << " rows\n";
  PrintHashes(reference);
  std::size_t failures = 0;
  for (const std::uint32_t rows : {1U, 7U, 9U, 95U, 96U}) {
    for (const std::uint32_t first : {0U, (n - rows) / 2, n - rows}) {
      const Pass candidate = RunPass(in, &w, first, rows);
      const std::size_t differ = Compare(reference, candidate);
      std::cout << (differ == 0 ? "  exact" : "  DIFFERS") << ": " << rows
                << " rows from " << first << '\n';
      failures += differ;
    }
  }
  return failures;
}

}  // namespace

int main() {
  try {
    std::size_t failures = 0;
    // 1619 rows end in a partial 128-row tile; 1549 rows also leave a tail
    // of 13 that the W8A8 projections hand to their tail kernel.
    for (const std::uint32_t n : {1619U, 1549U})
      failures += CheckLayerRoutes(n);
    if (failures != 0) {
      std::cerr << "HC prefill rows depend on the pass route (" << failures
                << " row mismatches)\n";
      return 1;
    }
    std::cout << "HC prefill layer routes exact\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
