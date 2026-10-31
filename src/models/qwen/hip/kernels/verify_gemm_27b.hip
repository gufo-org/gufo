#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/kernels/verify_gemm_27b.hpp"

namespace gufo::hip {
namespace {

using core::GgmlType;

// One lane walks its sub-blocks in the same order and the lanes reduce in the
// same butterfly for every block shape, so these launches only change how
// output rows are spread over waves and how much activation staging a block
// shares. Lanes = 16 runs two sixteen-lane groups per wave; each keeps both
// original lane partials and combines them in the same butterfly, so it is
// exact too. MinWaves is only the occupancy hint of the launch bounds.
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

#if defined(_WIN32)
// Measured on gfx1151 under Windows with TheRock 10.0.0, the SDK of the Windows
// build, with tools/qwen27b/verify_sweep and the method described below. The
// two toolchains favour different shapes: several Linux shapes lose to the
// generic dispatch here, and 16-lane groups win at four to eight rows. A shape
// is listed only where it beat LaunchBatchedQuantGEMMFp32 by 5% or more,
// summed over the target and draft layers that run it, and a second
// independent sweep reproduced that; the comment gives the first sweep's gain.
bool LaunchQ4K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ4_K;
  if (Is(m, k, 1024, 5120)) {  // attn_k/attn_v
    switch (batch) {
      case 3:  // +25%
        return Launch<T, 3, 8, 4, 2, 3, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +38%
        return Launch<T, 4, 8, 4, 2, 4, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +40%
        return Launch<T, 5, 8, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +47%
        return Launch<T, 6, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 7:  // +47%
        return Launch<T, 7, 16, 4, 2, 7, 4, 32>(w, x, y, m, k, stream);
      case 8:  // +48%
        return Launch<T, 8, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 1280, 5120)) {  // draft conv
    switch (batch) {
      case 2:  // +11%
        return Launch<T, 2, 8, 4, 2, 2, 4, 32>(w, x, y, m, k, stream);
      case 3:  // +17%
        return Launch<T, 3, 8, 4, 2, 3, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +35%
        return Launch<T, 4, 8, 4, 2, 4, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +38%
        return Launch<T, 5, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +50%
        return Launch<T, 6, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 7:  // +47%
        return Launch<T, 7, 16, 4, 2, 7, 4, 32>(w, x, y, m, k, stream);
      case 8:  // +22%
        return Launch<T, 8, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 4096, 5120)) {  // draft attn_q
    switch (batch) {
      case 2:  // +7%
        return Launch<T, 2, 8, 4, 1, 2, 4, 16>(w, x, y, m, k, stream);
      case 3:  // +21%
        return Launch<T, 3, 8, 4, 1, 3, 8, 16>(w, x, y, m, k, stream);
      case 4:  // +34%
        return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +44%
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +51%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +37%
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +55%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 5120, 4096)) {  // draft attn_output
    switch (batch) {
      case 3:  // +14%
        return Launch<T, 3, 8, 3, 1, 3, 4, 16>(w, x, y, m, k, stream);
      case 4:  // +25%
        return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +47%
        return Launch<T, 5, 4, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      case 6:  // +47%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +34%
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +54%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 5120, 17408)) {  // ffn_down
    switch (batch) {
      case 4:  // +13%
        return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +17%
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +21%
        return Launch<T, 6, 8, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +8%
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +34%
        return Launch<T, 8, 8, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 6144, 5120)) {  // attn_gate
    switch (batch) {
      case 3:  // +21%
        return Launch<T, 3, 8, 4, 1, 3, 8, 16>(w, x, y, m, k, stream);
      case 4:  // +24%
        return Launch<T, 4, 4, 4, 1, 4, 4, 16>(w, x, y, m, k, stream);
      case 5:  // +34%
        return Launch<T, 5, 8, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +41%
        return Launch<T, 6, 8, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
      case 7:  // +40%
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +49%
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 10240, 5120)) {  // attn_qkv
    switch (batch) {
      case 3:  // +11%
        return Launch<T, 3, 8, 3, 1, 3, 4, 16>(w, x, y, m, k, stream);
      case 4:  // +23%
        return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +35%
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +44%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +39%
        return Launch<T, 7, 8, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +49%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 12288, 5120)) {  // attn_q
    switch (batch) {
      case 3:  // +13%
        return Launch<T, 3, 8, 4, 2, 3, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +22%
        return Launch<T, 4, 8, 3, 1, 4, 4, 16>(w, x, y, m, k, stream);
      case 5:  // +33%
        return Launch<T, 5, 4, 3, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +40%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +37%
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +43%
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 17408, 5120)) {  // ffn_gate/ffn_up
    switch (batch) {
      case 4:  // +10%
        return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +29%
        return Launch<T, 5, 8, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      case 6:  // +36%
        return Launch<T, 6, 4, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
      case 7:  // +23%
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +20%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 34816, 5120)) {  // joined ffn_gate+ffn_up
    switch (batch) {
      case 4:  // +13%
        return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +35%
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +42%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +27%
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +26%
        return Launch<T, 8, 16, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  return false;
}

bool LaunchQ5K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ5_K;
  if (Is(m, k, 1024, 5120)) {  // attn_k/attn_v
    switch (batch) {
      case 2:  // +18%
        return Launch<T, 2, 16, 4, 2, 2, 4, 32>(w, x, y, m, k, stream);
      case 3:  // +29%
        return Launch<T, 3, 16, 4, 2, 3, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +38%
        return Launch<T, 4, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +50%
        return Launch<T, 5, 16, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +50%
        return Launch<T, 6, 16, 4, 2, 6, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 5120, 6144)) {  // ssm_out/attn_output
    switch (batch) {
      case 3:  // +11%
        return Launch<T, 3, 4, 4, 1, 3, 4, 16>(w, x, y, m, k, stream);
      case 5:  // +18%
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +20%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +33%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +31%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 5120, 17408)) {  // ffn_down
    switch (batch) {
      case 3:  // +11%
        return Launch<T, 3, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +9%
        return Launch<T, 4, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +15%
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +14%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +28%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +42%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 6144, 5120)) {  // attn_gate
    switch (batch) {
      case 2:  // +9%
        return Launch<T, 2, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 3:  // +18%
        return Launch<T, 3, 8, 4, 1, 3, 8, 16>(w, x, y, m, k, stream);
      case 4:  // +12%
        return Launch<T, 4, 8, 4, 1, 4, 4, 16>(w, x, y, m, k, stream);
      case 5:  // +9%
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +13%
        return Launch<T, 6, 8, 3, 1, 6, 4, 16>(w, x, y, m, k, stream);
      case 7:  // +25%
        return Launch<T, 7, 8, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +30%
        return Launch<T, 8, 8, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 10240, 5120)) {  // attn_qkv
    switch (batch) {
      case 3:  // +8%
        return Launch<T, 3, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +19%
        return Launch<T, 4, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +8%
        return Launch<T, 5, 8, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +13%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +27%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +30%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 12288, 5120)) {  // attn_q
    switch (batch) {
      case 3:  // +10%
        return Launch<T, 3, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +20%
        return Launch<T, 4, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +5%
        return Launch<T, 5, 8, 3, 1, 5, 4, 16>(w, x, y, m, k, stream);
      case 6:  // +14%
        return Launch<T, 6, 8, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
      case 7:  // +24%
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +27%
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 17408, 5120)) {  // ffn_gate/ffn_up
    switch (batch) {
      case 4:  // +18%
        return Launch<T, 4, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +32%
        return Launch<T, 6, 8, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
      case 7:  // +25%
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +27%
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 34816, 5120)) {  // joined ffn_gate+ffn_up
    switch (batch) {
      case 4:  // +20%
        return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +15%
        return Launch<T, 5, 8, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      case 6:  // +36%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +30%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +31%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  return false;
}

bool LaunchQ6K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ6_K;
  if (Is(m, k, 1024, 5120)) {  // attn_k/attn_v
    switch (batch) {
      case 3:  // +27%
        return Launch<T, 3, 8, 4, 2, 3, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +30%
        return Launch<T, 4, 8, 4, 2, 4, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +38%
        return Launch<T, 5, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +35%
        return Launch<T, 6, 8, 4, 2, 6, 4, 32>(w, x, y, m, k, stream);
      case 7:  // +40%
        return Launch<T, 7, 8, 4, 2, 7, 4, 32>(w, x, y, m, k, stream);
      case 8:  // +36%
        return Launch<T, 8, 8, 4, 2, 8, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 5120, 6144)) {  // ssm_out/attn_output
    switch (batch) {
      case 2:  // +7%
        return Launch<T, 2, 4, 4, 1, 2, 4, 16>(w, x, y, m, k, stream);
      case 3:  // +20%
        return Launch<T, 3, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +29%
        return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +41%
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +14%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +19%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +48%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 5120, 17408)) {  // ffn_down
    switch (batch) {
      case 2:  // +7%
        return Launch<T, 2, 8, 4, 1, 2, 4, 16>(w, x, y, m, k, stream);
      case 3:  // +18%
        return Launch<T, 3, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +29%
        return Launch<T, 4, 4, 4, 1, 4, 4, 16>(w, x, y, m, k, stream);
      case 5:  // +19%
        return Launch<T, 5, 8, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      case 6:  // +28%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +39%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +44%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 17408, 5120)) {  // ffn_gate/ffn_up
    switch (batch) {
      case 3:  // +15%
        return Launch<T, 3, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +26%
        return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +15%
        return Launch<T, 5, 8, 3, 1, 5, 4, 16>(w, x, y, m, k, stream);
      case 6:  // +24%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +35%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +42%
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 34816, 5120)) {  // joined ffn_gate+ffn_up
    switch (batch) {
      case 3:  // +17%
        return Launch<T, 3, 8, 2, 1, 3, 8, 16>(w, x, y, m, k, stream);
      case 4:  // +30%
        return Launch<T, 4, 8, 4, 2, 4, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +44%
        return Launch<T, 5, 8, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      case 6:  // +49%
        return Launch<T, 6, 4, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
      case 7:  // +41%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +44%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  return false;
}

bool LaunchIQ4XS(const void* w, const float* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kIQ4_XS;
  if (Is(m, k, 5120, 17408)) {  // ffn_down
    switch (batch) {
      case 2:  // +10%
        return Launch<T, 2, 16, 4, 1, 2, 4, 16>(w, x, y, m, k, stream);
      case 4:  // +9%
        return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +10%
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +42%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +29%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +24%
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 6144, 5120)) {  // attn_gate
    switch (batch) {
      case 2:  // +14%
        return Launch<T, 2, 4, 3, 1, 2, 8, 16>(w, x, y, m, k, stream);
      case 3:  // +29%
        return Launch<T, 3, 4, 3, 1, 3, 8, 16>(w, x, y, m, k, stream);
      case 4:  // +35%
        return Launch<T, 4, 8, 4, 2, 4, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +38%
        return Launch<T, 5, 8, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +32%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +44%
        return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      case 8:  // +48%
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 10240, 5120)) {  // attn_qkv
    switch (batch) {
      case 2:  // +8%
        return Launch<T, 2, 4, 2, 1, 2, 8, 16>(w, x, y, m, k, stream);
      case 3:  // +20%
        return Launch<T, 3, 4, 2, 1, 3, 8, 16>(w, x, y, m, k, stream);
      case 4:  // +35%
        return Launch<T, 4, 16, 4, 1, 4, 4, 16>(w, x, y, m, k, stream);
      case 5:  // +39%
        return Launch<T, 5, 16, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      case 6:  // +40%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +45%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +35%
        return Launch<T, 8, 8, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 12288, 5120)) {  // attn_q
    switch (batch) {
      case 2:  // +9%
        return Launch<T, 2, 16, 4, 1, 2, 4, 16>(w, x, y, m, k, stream);
      case 3:  // +21%
        return Launch<T, 3, 16, 4, 1, 3, 4, 16>(w, x, y, m, k, stream);
      case 4:  // +31%
        return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +36%
        return Launch<T, 5, 4, 4, 1, 5, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +36%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +42%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +41%
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 17408, 5120)) {  // ffn_gate/ffn_up
    switch (batch) {
      case 4:  // +6%
        return Launch<T, 4, 4, 3, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +35%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +26%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +30%
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 34816, 5120)) {  // joined ffn_gate+ffn_up
    switch (batch) {
      case 3:  // +7%
        return Launch<T, 3, 16, 4, 1, 3, 8, 16>(w, x, y, m, k, stream);
      case 4:  // +5%
        return Launch<T, 4, 4, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +40%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 7:  // +32%
        return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +34%
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  return false;
}
#else
// Measured on gfx1151 under Linux with ROCm 10.1 (tools/qwen27b/verify_sweep):
// real UD-Q4_K_XL target and DFlash2 Q4_K_M draft weights rotated through
// >= 512 MiB of copies, alternating rounds, median, every shape
// bit-identical to the generic dispatch. A shape is listed only where it beat
// LaunchBatchedQuantGEMMFp32 by 5% or more; the comment gives that gain.
bool LaunchQ4K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ4_K;
  if (Is(m, k, 1024, 5120)) {  // attn_k/attn_v
    switch (batch) {
      case 2:  // +15%
        return Launch<T, 2, 8, 4, 2, 2, 4, 32>(w, x, y, m, k, stream);
      case 3:  // +19%
        return Launch<T, 3, 8, 4, 2, 3, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +14%
        return Launch<T, 4, 8, 4, 2, 4, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +55%
        return Launch<T, 5, 16, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +50%
        return Launch<T, 6, 16, 4, 2, 6, 4, 32>(w, x, y, m, k, stream);
      case 7:  // +41%
        return Launch<T, 7, 16, 4, 2, 7, 4, 32>(w, x, y, m, k, stream);
      case 8:  // +65%
        return Launch<T, 8, 16, 4, 2, 8, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 1280, 5120)) {  // draft conv
    switch (batch) {
      case 2:  // +10%
        return Launch<T, 2, 16, 4, 2, 2, 4, 32>(w, x, y, m, k, stream);
      case 3:  // +16%
        return Launch<T, 3, 16, 4, 2, 3, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +9%
        return Launch<T, 4, 8, 4, 2, 4, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +49%
        return Launch<T, 5, 16, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +40%
        return Launch<T, 6, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 7:  // +38%
        return Launch<T, 7, 16, 4, 2, 7, 4, 32>(w, x, y, m, k, stream);
      case 8:  // +10%
        return Launch<T, 8, 16, 4, 2, 8, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 4096, 5120)) {  // draft attn_q
    switch (batch) {
      case 2:  // +12%
        return Launch<T, 2, 16, 4, 1, 2, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +7%
        return Launch<T, 5, 4, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      case 6:  // +11%
        return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      case 8:  // +28%
        return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 5120, 4096)) {  // draft attn_output
    switch (batch) {
      case 2:  // +13%
        return Launch<T, 2, 8, 4, 2, 2, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +13%
        return Launch<T, 5, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +12%
        return Launch<T, 6, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 8:  // +36%
        return Launch<T, 8, 8, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 5120, 17408)) {  // ffn_down
    switch (batch) {
      case 4:  // +8%
        return Launch<T, 4, 16, 4, 2, 4, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 6144, 5120)) {  // attn_gate
    switch (batch) {
      case 3:  // +9%
        return Launch<T, 3, 8, 4, 2, 3, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +11%
        return Launch<T, 4, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +9%
        return Launch<T, 5, 8, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +12%
        return Launch<T, 6, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 8:  // +5%
        return Launch<T, 8, 8, 4, 2, 8, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 10240, 5120)) {  // attn_qkv
    switch (batch) {
      case 8:  // +8%
        return Launch<T, 8, 16, 4, 2, 8, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 12288, 5120)) {  // attn_q
    switch (batch) {
      case 5:  // +9%
        return Launch<T, 5, 16, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +7%
        return Launch<T, 6, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 8:  // +15%
        return Launch<T, 8, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  return false;
}

bool LaunchQ5K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ5_K;
  if (Is(m, k, 1024, 5120)) {  // attn_k/attn_v
    switch (batch) {
      case 2:  // +10%
        return Launch<T, 2, 16, 4, 2, 2, 4, 32>(w, x, y, m, k, stream);
      case 3:  // +20%
        return Launch<T, 3, 8, 4, 2, 3, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +19%
        return Launch<T, 4, 8, 4, 2, 4, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +48%
        return Launch<T, 5, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +42%
        return Launch<T, 6, 16, 4, 2, 6, 4, 32>(w, x, y, m, k, stream);
      case 7:  // +22%
        return Launch<T, 7, 16, 4, 2, 7, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  return false;
}

bool LaunchQ6K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ6_K;
  if (Is(m, k, 1024, 5120)) {  // attn_k/attn_v
    switch (batch) {
      case 3:  // +24%
        return Launch<T, 3, 8, 4, 2, 3, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +24%
        return Launch<T, 4, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 5120, 6144)) {  // ssm_out/attn_output
    switch (batch) {
      case 2:  // +9%
        return Launch<T, 2, 8, 4, 2, 2, 4, 32>(w, x, y, m, k, stream);
      case 3:  // +10%
        return Launch<T, 3, 8, 2, 1, 3, 8, 16>(w, x, y, m, k, stream);
      case 6:  // +6%
        return Launch<T, 6, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 5120, 17408)) {  // ffn_down
    switch (batch) {
      case 2:  // +9%
        return Launch<T, 2, 16, 4, 2, 2, 4, 32>(w, x, y, m, k, stream);
      case 3:  // +12%
        return Launch<T, 3, 16, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 7:  // +6%
        return Launch<T, 7, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      case 8:  // +13%
        return Launch<T, 8, 16, 4, 2, 8, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  return false;
}

bool LaunchIQ4XS(const void* w, const float* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kIQ4_XS;
  if (Is(m, k, 5120, 17408)) {  // ffn_down
    switch (batch) {
      case 5:  // +40%
        return Launch<T, 5, 16, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 6144, 5120)) {  // attn_gate
    switch (batch) {
      case 2:  // +15%
        return Launch<T, 2, 4, 3, 1, 2, 8, 16>(w, x, y, m, k, stream);
      case 3:  // +27%
        return Launch<T, 3, 4, 3, 1, 3, 8, 16>(w, x, y, m, k, stream);
      case 4:  // +20%
        return Launch<T, 4, 8, 4, 2, 4, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +18%
        return Launch<T, 5, 8, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      case 6:  // +8%
        return Launch<T, 6, 8, 4, 2, 6, 4, 32>(w, x, y, m, k, stream);
      case 7:  // +8%
        return Launch<T, 7, 8, 4, 2, 7, 4, 32>(w, x, y, m, k, stream);
      case 8:  // +10%
        return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 10240, 5120)) {  // attn_qkv
    switch (batch) {
      case 2:  // +9%
        return Launch<T, 2, 16, 4, 1, 2, 4, 16>(w, x, y, m, k, stream);
      case 3:  // +6%
        return Launch<T, 3, 4, 2, 1, 3, 8, 16>(w, x, y, m, k, stream);
      case 4:  // +6%
        return Launch<T, 4, 4, 3, 1, 4, 8, 16>(w, x, y, m, k, stream);
      case 5:  // +6%
        return Launch<T, 5, 8, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 12288, 5120)) {  // attn_q
    switch (batch) {
      case 3:  // +9%
        return Launch<T, 3, 8, 4, 2, 3, 4, 32>(w, x, y, m, k, stream);
      case 4:  // +8%
        return Launch<T, 4, 8, 4, 2, 4, 4, 32>(w, x, y, m, k, stream);
      case 5:  // +5%
        return Launch<T, 5, 8, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      case 8:  // +5%
        return Launch<T, 8, 8, 4, 1, 1, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 17408, 5120)) {  // ffn_gate/ffn_up
    switch (batch) {
      case 5:  // +45%
        return Launch<T, 5, 8, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  if (Is(m, k, 34816, 5120)) {  // joined ffn_gate+ffn_up
    switch (batch) {
      case 3:  // +6%
        return Launch<T, 3, 16, 4, 1, 3, 4, 16>(w, x, y, m, k, stream);
      case 5:  // +50%
        return Launch<T, 5, 8, 4, 2, 5, 4, 32>(w, x, y, m, k, stream);
      default:
        return false;
    }
  }
  return false;
}

#endif  // defined(_WIN32)

}  // namespace

bool TryLaunchVerifyGemm27(GgmlType type, const void* w, const float* x,
                           float* y, std::size_t batch, std::size_t m,
                           std::size_t k, hipStream_t stream) {
  if (batch < 2 || batch > 8)
    return false;
  switch (type) {
    case GgmlType::kQ4_K:
      return LaunchQ4K(w, x, y, batch, m, k, stream);
    case GgmlType::kQ5_K:
      return LaunchQ5K(w, x, y, batch, m, k, stream);
    case GgmlType::kQ6_K:
      return LaunchQ6K(w, x, y, batch, m, k, stream);
    case GgmlType::kIQ4_XS:
      return LaunchIQ4XS(w, x, y, batch, m, k, stream);
    default:
      return false;
  }
}

}  // namespace gufo::hip
