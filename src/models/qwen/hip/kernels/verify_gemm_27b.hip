#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/kernels/verify_gemm_27b.hpp"

namespace gufo::hip {
namespace {

using core::GgmlType;

// One lane walks its sub-blocks in the same order and the lanes reduce in the
// same butterfly for every block shape, so these launches only change how
// output rows are spread over waves and how much activation staging a block
// shares. Measured on gfx1151 with real UD-Q4_K_XL and DFlash2 Q4_K_M weights
// against the production dispatcher, alternating rounds, median of nine or
// eleven, every candidate bit-identical to production; only gains of 5% or
// more are kept. tools/qwen27b/verify_sweep repeats the measurement for
// another GGUF or GPU. Lanes = 16 runs two sixteen-lane groups per wave; each
// keeps both original lane partials and combines them in the same butterfly,
// so it is exact too. MinWaves is only the occupancy hint of the launch
// bounds.
template<GgmlType T, std::size_t Batch, std::uint32_t Waves, std::size_t Rows,
         std::size_t Tiles, std::size_t TokensPerStep = Batch,
         std::uint32_t MinWaves = 4, std::uint32_t Lanes = 32>
bool Launch(const void* w, const float* x, float* y, std::size_t m,
            std::size_t k, hipStream_t stream) {
  constexpr std::size_t kRowsPerBlock = Waves * (32 / Lanes) * Rows;
  SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, Rows, T, Tiles, MinWaves, 1,
                                      true, TokensPerStep, 32, Lanes>
      <<<static_cast<unsigned>((m + kRowsPerBlock - 1) / kRowsPerBlock),
         Waves * 32, 0, stream>>>(w, x, y, m, k);
  return true;
}

bool Is(std::size_t m, std::size_t k, std::size_t want_m, std::size_t want_k) {
  return m == want_m && k == want_k;
}

// Target UD-Q4_K_XL projections.
bool LaunchQ5K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ5_K;
  const bool ffn_up = Is(m, k, 17408, 5120);
  const bool ffn_down = Is(m, k, 5120, 17408);
  const bool ssm_qkv = Is(m, k, 10240, 5120);
  const bool ssm_gate = Is(m, k, 6144, 5120);
  const bool attn_q = Is(m, k, 12288, 5120);
  const bool out_6144 = Is(m, k, 5120, 6144);  // ssm_out, attn_output
  const bool any =
      ffn_up || ffn_down || ssm_qkv || ssm_gate || attn_q || out_6144;
  switch (batch) {
    case 3:
      if (ffn_down)
        return Launch<T, 3, 16, 4, 1, 1>(w, x, y, m, k, stream);
      if (ssm_qkv)
        return Launch<T, 3, 16, 4, 1>(w, x, y, m, k, stream);
      if (ssm_gate)
        return Launch<T, 3, 8, 4, 1, 1>(w, x, y, m, k, stream);
      if (attn_q)
        return Launch<T, 3, 4, 4, 1, 1>(w, x, y, m, k, stream);
      if (out_6144)
        return Launch<T, 3, 4, 4, 1, 1, 4, 16>(w, x, y, m, k, stream);
      return false;
    case 4:
      if (ffn_up)
        return Launch<T, 4, 16, 4, 1, 1>(w, x, y, m, k, stream);
      if (ffn_down || ssm_qkv || ssm_gate || attn_q)
        return Launch<T, 4, 8, 4, 1, 4, 4, 16>(w, x, y, m, k, stream);
      if (out_6144)
        return Launch<T, 4, 4, 4, 1, 1, 4, 16>(w, x, y, m, k, stream);
      return false;
    case 5:
      if (ssm_gate)
        return Launch<T, 5, 4, 4, 1>(w, x, y, m, k, stream);
      if (ffn_down || out_6144)
        return Launch<T, 5, 16, 4, 2>(w, x, y, m, k, stream);
      return false;
    // 6-8 rows: 16-lane groups, 22-38% faster than the 32-lane shapes on
    // every projection.
    case 6:
      if (any)
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      return false;
    case 7:
      if (any)
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      return false;
    case 8:
      if (any)
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      return false;
    default:
      return false;
  }
}

// Target FFN gate.
bool LaunchIQ4XS(const void* w, const float* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kIQ4_XS;
  if (!Is(m, k, 17408, 5120))
    return false;
  if (batch == 2)
    return Launch<T, 2, 4, 3, 2>(w, x, y, m, k, stream);
  if (batch == 3)
    return Launch<T, 3, 4, 4, 2, 3, 1>(w, x, y, m, k, stream);
  if (batch == 4)
    return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
  if (batch == 5)
    return Launch<T, 5, 4, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
  if (batch == 6)
    return Launch<T, 6, 4, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
  if (batch == 7)
    return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
  if (batch == 8)
    return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
  return false;
}

// Target Q4_K and Q6_K projections of UD-Q4_K_XL:
// real weights rotated through >= 512 MiB of copies so no run reads from the
// Infinity Cache, nine alternating rounds, median, every shape bit-identical
// to the production dispatch. Gains at 3-8 rows: attn_qkv +13-50%,
// attn_gate +21-50%, attn_q +13-48%, ssm_out/attn_output +15-53%; at 2 rows
// only attn_gate (+12%) and ssm_out/attn_output (+8%) changed.
bool LaunchQ4KTarget(const void* w, const float* x, float* y, std::size_t batch,
                     std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ4_K;
  if (Is(m, k, 10240, 5120)) {  // attn_qkv
    switch (batch) {
      case 3:
        return Launch<T, 3, 16, 4, 1, 3, 4, 16>(w, x, y, m, k, stream);
      case 4:
        return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:
        // Eight-wave occupancy spills Q4_K at seven rows; four waves do not.
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 6144, 5120)) {  // attn_gate
    switch (batch) {
      case 2:
        return Launch<T, 2, 8, 4, 2, 2, 4>(w, x, y, m, k, stream);
      case 3:
        return Launch<T, 3, 8, 4, 1, 3, 8, 16>(w, x, y, m, k, stream);
      case 4:
        return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:
        return Launch<T, 5, 8, 4, 2, 5, 4>(w, x, y, m, k, stream);
      case 6:
        return Launch<T, 6, 4, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
      case 7:
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 12288, 5120)) {  // attn_q
    switch (batch) {
      case 3:
        return Launch<T, 3, 4, 4, 1, 3, 4, 16>(w, x, y, m, k, stream);
      case 4:
        return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:
        return Launch<T, 6, 4, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
      case 7:
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  return false;
}

bool LaunchQ6KTarget(const void* w, const float* x, float* y, std::size_t batch,
                     std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ6_K;
  if (!Is(m, k, 5120, 6144))  // ssm_out, attn_output
    return false;
  switch (batch) {
    case 2:
      return Launch<T, 2, 4, 4, 1, 2, 4, 16>(w, x, y, m, k, stream);
    case 3:
      return Launch<T, 3, 16, 4, 1, 1, 4>(w, x, y, m, k, stream);
    case 4:
      return Launch<T, 4, 4, 4, 1, 4, 4, 16>(w, x, y, m, k, stream);
    case 5:
      return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
    case 6:
      return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
    case 7:
      return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
    case 8:
      return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
    default:
      return false;
  }
}

// Output head (shared by verification and the draft) and attention V.
bool LaunchQ6K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ6_K;
  if (LaunchQ6KTarget(w, x, y, batch, m, k, stream))
    return true;
  const bool head = Is(m, k, 248320, 5120);
  const bool attn_v = Is(m, k, 1024, 5120);
  const bool ffn_down = Is(m, k, 5120, 17408);  // DFlash2 draft layers
  switch (batch) {
    case 2:
      if (attn_v)
        return Launch<T, 2, 4, 3, 2>(w, x, y, m, k, stream);
      if (ffn_down)
        return Launch<T, 2, 4, 2, 1, 2, 1>(w, x, y, m, k, stream);
      return false;
    case 3:
      if (attn_v)
        return Launch<T, 3, 4, 4, 1, 1>(w, x, y, m, k, stream);
      if (head)
        return Launch<T, 3, 16, 4, 1>(w, x, y, m, k, stream);
      if (ffn_down)
        return Launch<T, 3, 4, 4, 1, 3, 4, 16>(w, x, y, m, k, stream);
      return false;
    case 4:
      if (attn_v)
        return Launch<T, 4, 4, 4, 1, 1>(w, x, y, m, k, stream);
      // The head -9%, the draft down projection -44%.
      if (head)
        return Launch<T, 4, 8, 3, 1, 4, 4, 16>(w, x, y, m, k, stream);
      if (ffn_down)
        return Launch<T, 4, 8, 2, 1, 4, 4, 16>(w, x, y, m, k, stream);
      return false;
    // The head at 5-8 rows: 15-31% faster, 16-lane groups.
    case 5:
      if (attn_v)
        return Launch<T, 5, 8, 4, 2>(w, x, y, m, k, stream);
      if (head)
        return Launch<T, 5, 4, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      if (ffn_down)
        return Launch<T, 5, 4, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      return false;
    case 6:
      if (head)
        return Launch<T, 6, 16, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
      break;
    case 7:
      if (head)
        return Launch<T, 7, 16, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      break;
    case 8:
      if (head)
        return Launch<T, 8, 16, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      break;
    default:
      return false;
  }
  // attn_k/attn_v at 6-8 rows (measured like LaunchQ6KTarget):
  // +36-42%; 2-5 rows keep the attn_v shapes above.
  if (Is(m, k, 1024, 5120)) {
    if (batch == 6)
      return Launch<T, 6, 8, 4, 2, 6, 4>(w, x, y, m, k, stream);
    if (batch == 7)
      return Launch<T, 7, 8, 4, 2, 7, 4>(w, x, y, m, k, stream);
    return Launch<T, 8, 8, 4, 2, 8, 4>(w, x, y, m, k, stream);
  }
  return false;
}

// DFlash2 Q4_K_M draft.
bool LaunchQ4K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ4_K;
  if (LaunchQ4KTarget(w, x, y, batch, m, k, stream))
    return true;
  const bool ffn_down = Is(m, k, 5120, 17408);
  const bool ffn_gate_up = Is(m, k, 17408, 5120);
  const bool attn_q = Is(m, k, 4096, 5120);
  const bool attn_kv = Is(m, k, 1024, 5120);
  const bool attn_output = Is(m, k, 5120, 4096);
  const bool conv = Is(m, k, 1280, 5120);
  const bool fc = Is(m, k, 5120, 25600);
  // Draft shapes at 2-5 rows; conv, attn_kv, attn_q and attn_output were
  // re-measured with tools/qwen27b/verify_sweep and replaced where it found
  // them 5% or more faster.
  switch (batch) {
    case 2:
      if (attn_q)
        return Launch<T, 2, 8, 4, 1, 2, 8, 16>(w, x, y, m, k, stream);
      if (attn_kv)
        return Launch<T, 2, 8, 4, 2, 2, 4>(w, x, y, m, k, stream);
      if (conv)
        return Launch<T, 2, 16, 4, 2, 2, 4>(w, x, y, m, k, stream);
      return false;
    case 3:
      if (ffn_down)
        return Launch<T, 3, 8, 4, 1, 3, 4, 16>(w, x, y, m, k, stream);
      if (ffn_gate_up)
        return Launch<T, 3, 8, 4, 1, 3, 8, 16>(w, x, y, m, k, stream);
      if (attn_q)
        return Launch<T, 3, 4, 4, 1, 1, 4, 16>(w, x, y, m, k, stream);
      if (attn_kv)
        return Launch<T, 3, 8, 4, 2, 3, 1>(w, x, y, m, k, stream);
      if (attn_output)
        return Launch<T, 3, 8, 3, 1, 3, 4, 16>(w, x, y, m, k, stream);
      if (conv)
        return Launch<T, 3, 8, 4, 2, 3, 4>(w, x, y, m, k, stream);
      if (fc)
        return Launch<T, 3, 8, 4, 1, 1, 4, 16>(w, x, y, m, k, stream);
      return false;
    case 4:
      if (ffn_down)
        return Launch<T, 4, 8, 4, 1, 4, 4, 16>(w, x, y, m, k, stream);
      if (attn_q)
        return Launch<T, 4, 4, 4, 1, 4, 4, 16>(w, x, y, m, k, stream);
      if (attn_kv)
        return Launch<T, 4, 8, 4, 2, 4, 1>(w, x, y, m, k, stream);
      if (attn_output)
        return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      if (conv)
        return Launch<T, 4, 8, 4, 2, 4, 4>(w, x, y, m, k, stream);
      if (fc)
        return Launch<T, 4, 4, 4, 1, 4, 4, 16>(w, x, y, m, k, stream);
      return false;
    case 5:
      if (ffn_down)
        return Launch<T, 5, 4, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      if (ffn_gate_up)
        return Launch<T, 5, 8, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      if (attn_q)
        return Launch<T, 5, 4, 4, 1, 1, 4, 16>(w, x, y, m, k, stream);
      if (attn_kv || conv)
        return Launch<T, 5, 8, 4, 1, 1, 4>(w, x, y, m, k, stream);
      if (attn_output)
        return Launch<T, 5, 8, 4, 1, 1, 4, 16>(w, x, y, m, k, stream);
      if (fc)
        return Launch<T, 5, 4, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      return false;
    default:
      break;
  }
  // 6-8 rows (tools/qwen27b/verify_sweep): the generic dispatch reads these at
  // half speed; +24-55%. ffn_gate_up and ffn_down gain nothing here.
  if (batch < 6 || batch > 8)
    return false;
  if (attn_q || attn_output) {
    if (batch == 6)
      return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
    if (batch == 7)
      return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
    return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
  }
  if (attn_kv || conv) {
    if (batch == 6)
      return Launch<T, 6, 8, 4, 1, 1, 4>(w, x, y, m, k, stream);
    if (batch == 7)
      return Launch<T, 7, 16, 4, 2, 7, 4>(w, x, y, m, k, stream);
    return Launch<T, 8, 8, 4, 1, 1, 4>(w, x, y, m, k, stream);
  }
  return false;
}

// Target FFN projections at 2-8 rows, measured like LaunchQ4KTarget; gains of
// 5% or more over the production path, which includes the shapes above, are
// kept. "packed" is the joined 34816-row gate/up matrix the decode uses when
// both halves share a format; "single" is one 17408-row half; "down" the
// 17408-column down projection. Gains: packed +6-51%, down +10-50%, single
// +8-46% at the listed widths.
template<GgmlType T>
bool LaunchFfnShape(const void* w, const float* x, float* y, std::size_t batch,
                    std::size_t m, std::size_t k, hipStream_t stream) {
  const bool packed = Is(m, k, 34816, 5120);
  const bool single = Is(m, k, 17408, 5120);
  const bool down = Is(m, k, 5120, 17408);
  if (!packed && !single && !down)
    return false;
  // Six to eight rows: four waves of four rows in 16-lane groups.
  const auto wide = [&]() {
    switch (batch) {
      case 6:
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:
        // Q4_K spills at seven rows under the eight-wave occupancy hint.
        if constexpr (T == GgmlType::kQ4_K)
          return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
        else
          return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      default:
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
    }
  };
  if constexpr (T == GgmlType::kIQ4_XS) {
    if (down) {
      switch (batch) {
        case 2:
          return Launch<T, 2, 16, 4, 1, 2, 8, 16>(w, x, y, m, k, stream);
        case 4:
          return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
        case 5:
          return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
        case 8:
          return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
        case 6:
        case 7:
          return wide();
        default:
          return false;
      }
    }
    if (packed) {
      switch (batch) {
        case 2:
          return Launch<T, 2, 16, 4, 1, 2, 8, 16>(w, x, y, m, k, stream);
        case 3:
          return Launch<T, 3, 16, 4, 1, 3, 8, 16>(w, x, y, m, k, stream);
        case 4:
          return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
        case 5:
          return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
        case 8:
          return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
        case 6:
        case 7:
          return wide();
        default:
          return false;
      }
    }
    return false;  // single: the gate shapes in LaunchIQ4XS stay faster
  }
  if constexpr (T == GgmlType::kQ4_K) {
    if (packed) {
      if (batch == 4)
        return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      if (batch == 5)
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      return batch >= 6 && wide();
    }
    if (single) {
      switch (batch) {
        case 4:
          return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
        case 6:
          return Launch<T, 6, 4, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
        case 7:
          return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
        case 8:
          return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
        default:
          return false;
      }
    }
    if (batch == 7)
      return Launch<T, 7, 8, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
    return batch >= 6 && wide();  // down
  }
  if constexpr (T == GgmlType::kQ5_K) {
    if (packed) {
      if (batch == 4)
        return Launch<T, 4, 8, 4, 1, 4, 4, 16>(w, x, y, m, k, stream);
      if (batch == 5)
        return Launch<T, 5, 8, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      return batch >= 6 && wide();
    }
    if (batch != 5)
      return false;  // the existing Q5_K shapes stay faster
    if (single)
      return Launch<T, 5, 4, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
    return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);  // down
  }
  if constexpr (T == GgmlType::kQ6_K) {
    if (packed) {
      switch (batch) {
        case 3:
          return Launch<T, 3, 8, 2, 1, 3, 8, 16>(w, x, y, m, k, stream);
        case 4:
          return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
        case 5:
          return Launch<T, 5, 8, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
        default:
          return batch >= 6 && wide();
      }
    }
    if (single) {
      switch (batch) {
        case 3:
          return Launch<T, 3, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
        case 4:
          return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
        case 5:
          return Launch<T, 5, 8, 3, 1, 5, 4, 16>(w, x, y, m, k, stream);
        case 8:
          return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
        default:
          return batch >= 6 && wide();
      }
    }
    return batch >= 6 && wide();  // down; 2-5 rows keep the draft shapes
  }
  return false;
}

}  // namespace

bool TryLaunchVerifyGemm27(GgmlType type, const void* w, const float* x,
                           float* y, std::size_t batch, std::size_t m,
                           std::size_t k, hipStream_t stream) {
  if (batch < 2 || batch > 8)
    return false;
  switch (type) {
    case GgmlType::kQ4_K:
      if (LaunchFfnShape<GgmlType::kQ4_K>(w, x, y, batch, m, k, stream))
        return true;
      break;
    case GgmlType::kQ5_K:
      if (LaunchFfnShape<GgmlType::kQ5_K>(w, x, y, batch, m, k, stream))
        return true;
      break;
    case GgmlType::kQ6_K:
      if (LaunchFfnShape<GgmlType::kQ6_K>(w, x, y, batch, m, k, stream))
        return true;
      break;
    case GgmlType::kIQ4_XS:
      if (LaunchFfnShape<GgmlType::kIQ4_XS>(w, x, y, batch, m, k, stream))
        return true;
      break;
    default:
      break;
  }
  switch (type) {
    case GgmlType::kQ5_K:
      return LaunchQ5K(w, x, y, batch, m, k, stream);
    case GgmlType::kIQ4_XS:
      return LaunchIQ4XS(w, x, y, batch, m, k, stream);
    case GgmlType::kQ6_K:
      return LaunchQ6K(w, x, y, batch, m, k, stream);
    case GgmlType::kQ4_K:
      return LaunchQ4K(w, x, y, batch, m, k, stream);
    default:
      return false;
  }
}

}  // namespace gufo::hip
