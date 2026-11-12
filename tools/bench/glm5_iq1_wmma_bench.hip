// Standalone WMMA IQ1_M routed-expert GEMM probe for gfx1151 (glm5-next
// MoE tier, PLAN phase 3 item 1).
//
// Pattern-B port (flint moe_prefill RoutedHalfKQuantKernel -> IQ1_M):
// raw 56-byte super-blocks are staged through LDS; each lane decodes its
// row's groups directly into f16 wmma fragments; activations are F16 rows
// read at the same fragment offsets as flint. v1 keeps the epilogue plain
// (no swiglu fusion) and validates against a scalar reference kernel using
// the verbatim glm5 DecodeIq1M8 semantics.
//
// Build: nix develop -c tools/bench/build.sh glm5_iq1_wmma_bench
// Run:   /tmp/glm5_iq1_wmma_bench [--tokens N] [--small]
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "staging/models/glm5_next/kernels/rocm/iq_tables.hip.hpp"

namespace {

using namespace gufo::models::glm5_next::rocm;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kRowsPerBlock = 128;
constexpr int kStageK = 256;  // one IQ1_M super-block
constexpr int kIq1MBytes = 56;
/// Bytes per staged row: 56 code bytes padded to 15 dwords, so the 16
/// rows a fragment read touches sit on distinct bank groups.
constexpr int kRowStride = 60;
constexpr int kActStride = kStageK + 8;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ std::uint32_t U32At(const std::uint8_t* p) {
  std::uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

/// The 4 grid nibbles of one iq1s entry as 8 halves: nibble j sits at bit
/// 4 * (2 * (j & 3) + (j >> 2)) of the GPU-packed word.
__device__ __forceinline__ v16h GridHalves(std::uint32_t gv, float dl,
                                           float delta) {
  v16h out;
  auto* h = reinterpret_cast<__half*>(&out);
  const int db = delta > 0.0F ? 1 : -1;  // * 0.125 folded below
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int nib = static_cast<int>(
        (gv >> (4 * (2 * (j & 3) + (j >> 2)))) & 0xFU);
    h[j] = __float2half(dl * (static_cast<float>(nib - 1) + 0.125F * db));
  }
  return out;
}

/// The 16 elements [k0, k0 + 16) (= two 8-element groups) of the staged
/// IQ1_M super-block row `blk` as one f16 wmma fragment operand.
__device__ __forceinline__ v16h DecodeIq1FragmentHalf(
    const std::uint8_t* __restrict__ blk, int k0) {
  const std::uint32_t scw[4] = {U32At(blk + 48) & 0xFFFFU,
                                U32At(blk + 50) & 0xFFFFU,
                                U32At(blk + 52) & 0xFFFFU,
                                U32At(blk + 54) & 0xFFFFU};
  const std::uint16_t packed = static_cast<std::uint16_t>(
      (scw[0] >> 12) | ((scw[1] >> 8) & 0x00F0U) | ((scw[2] >> 4) & 0x0F00U) |
      (scw[3] & 0xF000U));
  __half_raw d_raw;
  d_raw.x = packed;
  const float d = __half2float(*reinterpret_cast<const __half*>(&d_raw));
  v16h frag;
  auto* halves = reinterpret_cast<__half*>(&frag);
#pragma unroll
  for (int sel = 0; sel < 2; ++sel) {
    const int g = k0 / 8 + sel;  // group index 0..31
    const int ib = g >> 2;
    const int l = g & 3;
    const std::uint8_t qs = blk[g];
    const std::uint8_t qh = blk[32 + (g >> 1)];
    const float dl = d * static_cast<float>(
                             2 *
                                 ((scw[ib / 2] >> (6 * (ib % 2) + 3 * (l >> 1))) &
                                  7) +
                             1);
    const std::uint32_t idx =
        static_cast<std::uint32_t>(qs) |
        ((static_cast<std::uint32_t>(qh) << ((l & 1U) ? 4U : 8U)) & 0x700U);
    const float delta =
        ((qh >> ((l & 1U) ? 7U : 3U)) & 1U) != 0 ? -0.125F : 0.125F;
    const v16h part = GridHalves(kIq1sGridGpu[idx], dl, delta);
    const __half* pp = reinterpret_cast<const __half*>(&part);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      halves[8 * sel + j] = pp[j];
    }
  }
  return frag;
}

/// grid.x = m / kRowsPerBlock, grid.y = routed tile entries, one kernel
/// over the fused [gate | up] width run as separate launches for v1.
template <int kTileTokens>
__global__ void __launch_bounds__(kThreads) RoutedHalfIq1Kernel(
    const std::uint8_t* __restrict__ w, const __half* __restrict__ x,
    const std::int32_t* __restrict__ tiles,
    const std::int32_t* __restrict__ pad_bounds,
    const std::int32_t* __restrict__ rows_in, float* __restrict__ out,
    std::uint32_t m, std::uint32_t k) {
  constexpr int kTokTiles = kTileTokens / 16;
  constexpr int kChunksPerTok = kStageK / 8;
  constexpr int kActChunks = kTileTokens * kChunksPerTok;
  constexpr int kActPer = (kActChunks + kThreads - 1) / kThreads;
  __shared__ __attribute__((aligned(16))) std::uint8_t
      s_rows[kRowsPerBlock * kRowStride];
  __shared__ __attribute__((aligned(16))) __half
      s_act[kTileTokens * kActStride];

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * kTileTokens;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  if (t_local >= bucket_rows) {
    return;
  }
  const int live_tiles = min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int sub = lane & 15;
  const int half = lane >> 4;
  const std::size_t row_bytes = std::size_t{k} / 256 * kIq1MBytes;

  // Weight fetch: two threads per staged row move 7 words (28 B) each of
  // the stage's 56-byte super-block (word loads: blocks sit 56 B apart).
  const int f_row = tid >> 1;
  const int f_part = tid & 1;
  const std::uint32_t f_global = blockIdx.x * kRowsPerBlock + f_row;
  const bool f_live = f_global < m;
  const std::uint8_t* f_ptr =
      w + (std::size_t(expert) * m + (f_live ? f_global : m - 1)) * row_bytes;
  std::uint32_t f_word[7];
  const auto fetch_weights = [&](int stage) {
    const std::uint8_t* block = f_ptr + stage * kIq1MBytes + 28 * f_part;
#pragma unroll
    for (int i = 0; i < 7; ++i) {
      f_word[i] = U32At(block + 4 * i);
    }
  };
  const auto commit_weights = [&] {
    auto* dst = reinterpret_cast<std::uint32_t*>(s_rows + f_row * kRowStride);
#pragma unroll
    for (int i = 0; i < 7; ++i) {
      dst[7 * f_part + i] = f_word[i];
    }
  };

  // Activation fetch (flint chunking: chunk c = token c / 32, eight
  // values at K offset 8 * (c % 32)).
  std::int32_t a_src[kActPer];
#pragma unroll
  for (int i = 0; i < kActPer; ++i) {
    const int chunk = tid + i * kThreads;
    const int t = chunk / kChunksPerTok;
    a_src[i] = -1;
    if (chunk < kActChunks && t_local + t < bucket_rows) {
      const std::int32_t src = rows_in[bucket_begin + t_local + t];
      if (src >= 0) {
        a_src[i] =
            src * static_cast<std::int32_t>(k) + (chunk % kChunksPerTok) * 8;
      }
    }
  }
  uint4 a_data[kActPer];
  const auto fetch_act = [&](int stage) {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      a_data[i] = a_src[i] >= 0
                      ? *reinterpret_cast<const uint4*>(x + a_src[i] +
                                                        stage * kStageK)
                      : make_uint4(0U, 0U, 0U, 0U);
    }
  };
  const auto commit_act = [&] {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      const int chunk = tid + i * kThreads;
      if (chunk < kActChunks) {
        *reinterpret_cast<uint4*>(s_act + (chunk / kChunksPerTok) * kActStride +
                                  (chunk % kChunksPerTok) * 8) = a_data[i];
      }
    }
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  }
  const int stages = static_cast<int>(k / kStageK);
  fetch_weights(0);
  fetch_act(0);
  for (int stage = 0; stage < stages; ++stage) {
    commit_weights();
    commit_act();
    __syncthreads();
    if (stage + 1 < stages) {
      fetch_weights(stage + 1);
      fetch_act(stage + 1);
    }
    const std::uint8_t* row = s_rows + (wave * 16 + sub) * kRowStride;
#pragma unroll
    for (int s = 0; s < kStageK / 32; ++s) {
      const v16h a_lo = DecodeIq1FragmentHalf(row, 32 * s);
      const v16h a_hi = DecodeIq1FragmentHalf(row, 32 * s + 16);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if (j < live_tiles) {
          const __half* b = s_act + (j * 16 + sub) * kActStride + s * 32;
          v16h b_lo;
          v16h b_hi;
          __builtin_memcpy(&b_lo, b, 32);
          __builtin_memcpy(&b_hi, b + 16, 32);
          acc[j] = Wmma(a_lo, b_lo, acc[j]);
          acc[j] = Wmma(a_hi, b_hi, acc[j]);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    const int t = t_local + j * 16 + sub;
    if (t >= bucket_rows) {
      continue;
    }
    const std::int32_t src = rows_in[bucket_begin + t];
    if (src < 0) {
      continue;
    }
    const std::size_t base =
        static_cast<std::size_t>(bucket_begin + t) * m;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const std::uint32_t r =
          blockIdx.x * kRowsPerBlock + wave * 16 + 2 * i + half;
      if (r < m) {
        out[base + r] = acc[j][i];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Scalar reference (contract oracle): glm5 DecodeIq1M8 semantics verbatim.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float RefIq1ValueAt(const std::uint8_t* blk,
                                               int elem) {
  const int g = elem >> 3;
  const int j = elem & 7;
  const std::uint8_t qs = blk[g];
  const std::uint8_t qh = blk[32 + (g >> 1)];
  const int ib = g >> 2;
  const int l = g & 3;
  std::uint32_t scw[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    scw[i] = static_cast<std::uint32_t>(blk[48 + 2 * i]) |
             (static_cast<std::uint32_t>(blk[48 + 2 * i + 1]) << 8);
  }
  const std::uint16_t packed = static_cast<std::uint16_t>(
      (scw[0] >> 12) | ((scw[1] >> 8) & 0x00F0U) | ((scw[2] >> 4) & 0x0F00U) |
      (scw[3] & 0xF000U));
  __half_raw d_raw;
  d_raw.x = packed;
  const float d = __half2float(*reinterpret_cast<const __half*>(&d_raw));
  const float dl = d * static_cast<float>(
                           2 *
                               ((scw[ib / 2] >> (6 * (ib % 2) + 3 * (l >> 1))) &
                                7) +
                           1);
  const std::uint32_t idx =
      static_cast<std::uint32_t>(qs) |
      ((static_cast<std::uint32_t>(qh) << ((l & 1U) ? 4U : 8U)) & 0x700U);
  const float delta =
      ((qh >> ((l & 1U) ? 7U : 3U)) & 1U) != 0 ? -0.125F : 0.125F;
  const int nib = static_cast<int>(
      (kIq1sGridGpu[idx] >> (4 * (2 * (j & 3) + (j >> 2)))) & 0xFU);
  return dl * (static_cast<float>(nib - 1) + delta);
}

__global__ void RefIq1Gemm(const std::uint8_t* __restrict__ w,
                           const float* __restrict__ x,
                           const std::int32_t* __restrict__ rows_in,
                           const std::int32_t* __restrict__ expert_of_row,
                           float* __restrict__ out, std::uint32_t m,
                           std::uint32_t k, std::uint32_t rows_total) {
  const std::size_t idx = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
  if (idx >= std::size_t{rows_total} * m) {
    return;
  }
  const std::uint32_t t = static_cast<std::uint32_t>(idx / m);
  const std::uint32_t r = static_cast<std::uint32_t>(idx % m);
  const std::int32_t src = rows_in[t];
  if (src < 0) {
    return;
  }
  const std::uint32_t expert = static_cast<std::uint32_t>(expert_of_row[t]);
  const std::size_t row_bytes = std::size_t{k} / 256 * kIq1MBytes;
  const std::uint8_t* wrow = w + (std::size_t(expert) * m + r) * row_bytes;
  const float* xr = x + std::size_t(src) * k;
  float s = 0.0F;
  for (std::uint32_t blk = 0; blk < k; blk += 256) {
    const std::uint8_t* b = wrow + (blk >> 8) * kIq1MBytes;
    for (int e = 0; e < 256; ++e) {
      s += xr[blk + e] * RefIq1ValueAt(b, e);
    }
  }
  out[idx] = s;
}

struct Bench {
  std::vector<std::uint8_t> w;
  std::vector<float> x;
  std::vector<__half> xh;
  std::vector<std::int32_t> tiles;
  std::vector<std::int32_t> pad_bounds;
  std::vector<std::int32_t> rows_in;
  std::vector<std::int32_t> expert_of_row;
  std::uint32_t experts{288};
  std::uint32_t m{2048};
  std::uint32_t k{4096};
  std::uint32_t n_tokens{0};
  std::uint32_t topk{8};
  std::size_t rows_total{0};
  std::size_t weight_bytes{0};
};

void BuildBench(Bench& b, std::uint32_t tokens) {
  b.n_tokens = tokens;
  const std::uint64_t slots = std::uint64_t{tokens} * b.topk;
  std::vector<std::int32_t> ids(slots);
  for (std::uint64_t tok = 0; tok < tokens; ++tok) {
    for (std::uint32_t j = 0; j < b.topk; ++j) {
      ids[tok * b.topk + j] = static_cast<std::int32_t>(
          (tok * 13 + j * 89 + tok * tok * 7) % b.experts);
    }
  }
  std::vector<std::uint32_t> count(b.experts, 0);
  for (std::int32_t e : ids) {
    ++count[e];
  }
  b.pad_bounds.assign(b.experts + 1, 0);
  for (std::uint32_t e = 0; e < b.experts; ++e) {
    b.pad_bounds[e + 1] =
        b.pad_bounds[e] + static_cast<std::int32_t>((count[e] + 15) & ~15U);
  }
  b.rows_total = static_cast<std::size_t>(b.pad_bounds[b.experts]);
  b.rows_in.assign(b.rows_total, -1);
  b.expert_of_row.assign(b.rows_total, -1);
  std::vector<std::uint32_t> cursor(b.experts, 0);
  for (std::uint64_t s = 0; s < slots; ++s) {
    const std::int32_t e = ids[s];
    const std::uint32_t pos =
        static_cast<std::uint32_t>(b.pad_bounds[e]) + cursor[e]++;
    b.rows_in[pos] = static_cast<std::int32_t>(s / b.topk);
    b.expert_of_row[pos] = e;
  }
  b.tiles.clear();
  for (std::uint32_t e = 0; e < b.experts; ++e) {
    const std::int32_t rows = b.pad_bounds[e + 1] - b.pad_bounds[e];
    for (std::int32_t tt = 0; tt * 16 < rows; ++tt) {
      b.tiles.push_back(static_cast<std::int32_t>(e) | (tt << 16));
    }
  }
  b.weight_bytes = std::size_t(b.experts) * b.m * (b.k / 256) * kIq1MBytes;
  b.w.resize(b.weight_bytes);
  std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
  auto next = [&]() {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return static_cast<std::uint8_t>(rng >> 33);
  };
  for (std::size_t i = 0; i < b.w.size(); ++i) {
    b.w[i] = next();
  }
  // Random 3-bit scales/grid bytes, but pin the reconstructed d to
  // exponent 11, mantissa 0xC00 bits: d = 0.109375 (see DecodeIq1M8
  // top-nibble packing), so all products stay finite in f16/f32.
  for (std::size_t off = 0; off < b.w.size(); off += kIq1MBytes) {
    for (int i = 0; i < 4; ++i) {
      b.w[off + 48 + 2 * i] = next();
      b.w[off + 48 + 2 * i + 1] = next();
    }
    b.w[off + 49] &= 0x0FU;  // scw0[15:12] -> packed[3:0]
    b.w[off + 51] &= 0x0FU;  // scw1[15:12] -> packed[7:4]
    b.w[off + 53] =
        static_cast<std::uint8_t>((b.w[off + 53] & 0x0FU) | 0xC0U);
    b.w[off + 55] = static_cast<std::uint8_t>((b.w[off + 55] & 0x0FU) | 0x20U);
  }
  b.x.assign(std::size_t(tokens) * b.k, 0.0F);
  for (std::size_t i = 0; i < b.x.size(); ++i) {
    const std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U;
    b.x[i] = static_cast<float>(h >> 19) / 512.0F - 8.0F;
  }
  b.xh.resize(b.x.size());
  for (std::size_t i = 0; i < b.x.size(); ++i) {
    b.xh[i] = __float2half(b.x[i]);
  }
}

template <int kTile>
void LaunchRouted(const Bench& b, uint8_t* d_w, __half* d_x, float* d_out,
                  std::int32_t* d_tiles, std::int32_t* d_pad,
                  std::int32_t* d_rows) {
  dim3 grid(b.m / kRowsPerBlock, static_cast<unsigned>(b.tiles.size()));
  RoutedHalfIq1Kernel<kTile><<<grid, kThreads>>>(d_w, d_x, d_tiles, d_pad,
                                                 d_rows, d_out, b.m, b.k);
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t tokens = 2048;
  bool small = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--tokens" && i + 1 < argc) {
      tokens = static_cast<std::uint32_t>(std::atoi(argv[++i]));
    } else if (a == "--small") {
      small = true;
    }
  }
  Bench b;
  if (small) {
    b.experts = 8;
    b.topk = 2;
  }
  BuildBench(b, tokens);
  std::printf("experts=%u m=%u k=%u tokens=%u topk=%u rows_total=%zu tiles=%zu\n",
              b.experts, b.m, b.k, b.n_tokens, b.topk, b.rows_total,
              b.tiles.size());

  std::uint8_t* d_w = nullptr;
  __half* d_x = nullptr;
  float* d_out = nullptr;
  std::int32_t *d_tiles = nullptr, *d_pad = nullptr, *d_rows = nullptr,
                     *d_eor = nullptr;
  const std::size_t xh_bytes = b.xh.size() * sizeof(__half);
  const std::size_t out_bytes = b.rows_total * b.m * sizeof(float);
  hipMalloc(&d_w, b.weight_bytes);
  hipMalloc(&d_x, xh_bytes);
  hipMalloc(&d_out, out_bytes);
  hipMalloc(&d_tiles, b.tiles.size() * sizeof(std::int32_t));
  hipMalloc(&d_pad, b.pad_bounds.size() * sizeof(std::int32_t));
  hipMalloc(&d_rows, b.rows_in.size() * sizeof(std::int32_t));
  hipMalloc(&d_eor, b.expert_of_row.size() * sizeof(std::int32_t));
  hipMemcpy(d_w, b.w.data(), b.weight_bytes, hipMemcpyHostToDevice);
  hipMemcpy(d_x, b.xh.data(), xh_bytes, hipMemcpyHostToDevice);
  hipMemcpy(d_tiles, b.tiles.data(), b.tiles.size() * sizeof(std::int32_t),
            hipMemcpyHostToDevice);
  hipMemcpy(d_pad, b.pad_bounds.data(),
            b.pad_bounds.size() * sizeof(std::int32_t), hipMemcpyHostToDevice);
  hipMemcpy(d_rows, b.rows_in.data(), b.rows_in.size() * sizeof(std::int32_t),
            hipMemcpyHostToDevice);
  hipMemcpy(d_eor, b.expert_of_row.data(),
            b.expert_of_row.size() * sizeof(std::int32_t),
            hipMemcpyHostToDevice);
  hipMemset(d_out, 0, out_bytes);

  if (small) {
    float* d_xf = nullptr;
    float* d_ref = nullptr;
    hipMalloc(&d_xf, b.x.size() * sizeof(float));
    hipMalloc(&d_ref, out_bytes);
    hipMemcpy(d_xf, b.x.data(), b.x.size() * sizeof(float),
              hipMemcpyHostToDevice);
    hipMemset(d_ref, 0, out_bytes);
    const std::size_t n_out = b.rows_total * b.m;
    RefIq1Gemm<<<static_cast<unsigned>((n_out + 255) / 256), 256>>>(
        d_w, d_xf, d_rows, d_eor, d_ref, b.m, b.k,
        static_cast<std::uint32_t>(b.rows_total));
    LaunchRouted<32>(b, d_w, d_x, d_out, d_tiles, d_pad, d_rows);
    hipDeviceSynchronize();
    std::vector<float> ref(n_out), got(n_out);
    hipMemcpy(ref.data(), d_ref, out_bytes, hipMemcpyDeviceToHost);
    hipMemcpy(got.data(), d_out, out_bytes, hipMemcpyDeviceToHost);
    double max_abs = 0.0, mse = 0.0, ref_ms = 0.0;
    std::size_t n = 0, got_nan = 0;
    for (std::size_t i = 0; i < n_out; ++i) {
      const float a = ref[i];
      if (!(got[i] == got[i])) {
        ++got_nan;
      }
      if (a == 0.0F && got[i] == 0.0F) {
        continue;
      }
      const double d = std::fabs(static_cast<double>(a) - got[i]);
      max_abs = max_abs < d ? d : max_abs;
      mse += d * d;
      ref_ms += static_cast<double>(a) * a;
      ++n;
    }
    const double nmse = n > 0 ? mse / n / (ref_ms / n + 1e-12) : 0.0;
    std::printf(
        "contract: rows_checked=%zu max_abs=%.6g nmse=%.3e scale_rms=%.4g "
        "got_nan=%zu\n",
        n, max_abs, nmse, n > 0 ? std::sqrt(ref_ms / n) : 0.0, got_nan);
    if (got_nan != 0 || nmse > 1e-4) {
      std::printf("contract FAILED\n");
      return 1;
    }
    hipFree(d_ref);
    hipFree(d_xf);
  }

  hipEvent_t t0, t1;
  hipEventCreate(&t0);
  hipEventCreate(&t1);
  const int reps = 5;
  const double eff_flops =
      2.0 * static_cast<double>(b.n_tokens) * b.topk * (2.0 * b.m * b.k);
  const double pad_flops =
      2.0 * static_cast<double>(b.rows_total) * (2.0 * b.m * b.k);
  const double read_bytes = static_cast<double>(b.tiles.size()) *
                            static_cast<double>(b.m / kRowsPerBlock) *
                            kRowsPerBlock * (b.k / 256) * kIq1MBytes;
  const auto run = [&](auto tile_c, const char* name) {
    constexpr int kTile = decltype(tile_c)::value;
    LaunchRouted<kTile>(b, d_w, d_x, d_out, d_tiles, d_pad, d_rows);
    hipDeviceSynchronize();
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
      std::printf("%-8s launch error: %s\n", name, hipGetErrorString(le));
      return;
    }
    hipEventRecord(t0);
    for (int r = 0; r < reps; ++r) {
      LaunchRouted<kTile>(b, d_w, d_x, d_out, d_tiles, d_pad, d_rows);
    }
    hipEventRecord(t1);
    hipDeviceSynchronize();
    float ms = 0.0F;
    hipEventElapsedTime(&ms, t0, t1);
    ms /= reps;
    std::printf(
        "%-8s %8.3f ms  eff %6.2f TF/s  padded %6.2f TF/s  w-read %6.1f GB/s\n",
        name, ms, eff_flops / (ms * 1e9), pad_flops / (ms * 1e9),
        read_bytes / (ms * 1e6));
  };
  run(std::integral_constant<int, 16>{}, "tile16");
  run(std::integral_constant<int, 32>{}, "tile32");
  run(std::integral_constant<int, 48>{}, "tile48");
  run(std::integral_constant<int, 64>{}, "tile64");
  std::printf("done err=%s\n", hipGetErrorString(hipGetLastError()));
  return 0;
}
#else
int main() {
  return 0;
}
#endif
