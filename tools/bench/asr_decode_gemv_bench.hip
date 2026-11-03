// Batch-one BF16 GEMV ablation for the Qwen3-ASR text decoder.
//
// The retained decode route (Wave32GEMVKernel_1Row in
// src/families/qwen/kernels/gemv.hip) reaches roughly half of the gfx1151
// DRAM read ceiling on the ASR decoder shapes, and it is 79% of request GPU
// time.
//
// The harness allocates the decoder's whole per-token weight footprint -- 28
// layers of q/k/v/o/gate/up/down plus the language-model head, 3.44 GB -- and
// times a complete token pass. A per-shape loop is not a valid substitute: one
// projection is 4-25 MB and stays resident in the 32 MB MALL across repeats,
// which reports 400-860 GB/s for a kernel that sustains 124 GB/s against cold
// memory. Streaming the full arena reproduces the real cache behavior.
//
// Two mechanisms are separated:
//
//   * per-wave memory level parallelism -- unroll depth, i.e. how many row
//     loads a wave issues before consuming the first;
//   * grid occupancy -- one wave per row starves the machine when M is 1024
//     (6.4 waves/SIMD against a 16-wave limit), so q/k/v and gate/up are also
//     measured as single fused dispatches over their concatenated rows.
//
// `read ceiling` is the same dispatch geometry doing nothing but streaming the
// weights, which bounds what any GEMV can reach at that size.
//
// Every candidate reports whether it is bit-identical to the retained kernel.
// Greedy ASR decode reproduces the official token IDs only when the
// accumulation order is preserved; fusing dispatches and deepening the unroll
// both leave each row's summation order untouched.
//
//   nix develop -c tools/bench/build.sh asr_decode_gemv_bench
//   /tmp/asr_decode_gemv_bench [token_passes]

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(expr)                                          \
  do {                                                       \
    const hipError_t status = (expr);                        \
    if (status != hipSuccess) {                              \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(status));               \
      std::exit(1);                                          \
    }                                                        \
  } while (0)

namespace {

constexpr double kDramReadPeakGBs = 240.24;

// __builtin_nontemporal_load rejects HIP's uint4 (a class template), so the
// candidate kernels load through a native 16-byte vector instead.
using Vec16 = __attribute__((__vector_size__(16))) unsigned int;

constexpr std::uint32_t kMaxFused = 3;

// One dispatch covering up to three weight tensors that share an activation
// vector: q/k/v, or gate/up. Rows are concatenated so a single grid keeps every
// SIMD busy even when the individual projections are narrow.
struct FusedDesc {
  const hip_bfloat16* a[kMaxFused];
  float* y[kMaxFused];
  std::uint32_t rows[kMaxFused];
  std::uint32_t prefix[kMaxFused];  // first global row of each tensor
  std::uint32_t count;
  std::uint32_t total_rows;
};

// ---------------------------------------------------------------------------
// v0: retained route. One wave per row, float32 activations, no unroll.
// ---------------------------------------------------------------------------
template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Baseline1Row(const hip_bfloat16* __restrict__ A,
                      const float* __restrict__ x, float* __restrict__ y,
                      std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }
  const auto* row = A + (m * K);
  const auto* row_vec = reinterpret_cast<const uint4*>(row);
  const std::size_t K_vec = K / 8;

  float dot = 0.0F;
  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const uint4 v = row_vec[vi];
    const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
    dot += (static_cast<float>(vals[0]) * x0.x) +
           (static_cast<float>(vals[1]) * x0.y) +
           (static_cast<float>(vals[2]) * x0.z) +
           (static_cast<float>(vals[3]) * x0.w) +
           (static_cast<float>(vals[4]) * x1.x) +
           (static_cast<float>(vals[5]) * x1.y) +
           (static_cast<float>(vals[6]) * x1.z) +
           (static_cast<float>(vals[7]) * x1.w);
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    dot += static_cast<float>(row[k]) * x[k];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }
  if (lane_id == 0) {
    y[m] = dot;
  }
}

__device__ __forceinline__ float Bf16Bits(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

// Eight products accumulated left to right, matching the retained kernel.
__device__ __forceinline__ float Dot8(const Vec16 w, const Vec16 xv) {
  const auto* wb = reinterpret_cast<const std::uint16_t*>(&w);
  const auto* xb = reinterpret_cast<const std::uint16_t*>(&xv);
  return (Bf16Bits(wb[0]) * Bf16Bits(xb[0])) +
         (Bf16Bits(wb[1]) * Bf16Bits(xb[1])) +
         (Bf16Bits(wb[2]) * Bf16Bits(xb[2])) +
         (Bf16Bits(wb[3]) * Bf16Bits(xb[3])) +
         (Bf16Bits(wb[4]) * Bf16Bits(xb[4])) +
         (Bf16Bits(wb[5]) * Bf16Bits(xb[5])) +
         (Bf16Bits(wb[6]) * Bf16Bits(xb[6])) +
         (Bf16Bits(wb[7]) * Bf16Bits(xb[7]));
}

// Accumulates one row against a bf16 activation vector with an `Unroll`-deep
// load prefetch. The accumulator is single and the adds stay in ascending k
// order, so the value is whatever the retained kernel produces.
template<std::uint32_t Unroll>
__device__ __forceinline__ float RowDot(const Vec16* __restrict__ row_vec,
                                        const Vec16* __restrict__ x_vec,
                                        std::size_t K_vec,
                                        std::size_t lane_id) {
  constexpr std::size_t kStep = 32 * Unroll;
  float dot = 0.0F;
  std::size_t vi = lane_id;
  for (; vi + (kStep - 32) < K_vec; vi += kStep) {
    Vec16 w[Unroll];
    Vec16 a[Unroll];
#pragma unroll
    for (std::uint32_t u = 0; u < Unroll; ++u) {
      w[u] = row_vec[vi + (u * 32)];
      a[u] = x_vec[vi + (u * 32)];
    }
#pragma unroll
    for (std::uint32_t u = 0; u < Unroll; ++u) {
      dot += Dot8(w[u], a[u]);
    }
  }
  for (; vi < K_vec; vi += 32) {
    dot += Dot8(row_vec[vi], x_vec[vi]);
  }
  return dot;
}

// ---------------------------------------------------------------------------
// One wave per row, bf16 activations, templated unroll depth.
// ---------------------------------------------------------------------------
template<std::uint32_t WavesPerBlock, std::uint32_t Unroll>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Bf16Gemv(const hip_bfloat16* __restrict__ A,
                  const hip_bfloat16* __restrict__ x, float* __restrict__ y,
                  std::size_t M, std::size_t K) {
  const std::size_t lane_id = threadIdx.x & 31u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + (threadIdx.x >> 5u);
  if (m >= M) {
    return;
  }
  float dot = RowDot<Unroll>(reinterpret_cast<const Vec16*>(A + (m * K)),
                             reinterpret_cast<const Vec16*>(x), K / 8, lane_id);
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }
  if (lane_id == 0) {
    y[m] = dot;
  }
}

// ---------------------------------------------------------------------------
// One dispatch over concatenated rows of several tensors sharing x.
// ---------------------------------------------------------------------------
template<std::uint32_t WavesPerBlock, std::uint32_t Unroll>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Bf16GemvFused(FusedDesc d, const hip_bfloat16* __restrict__ x,
                       std::size_t K) {
  const std::size_t lane_id = threadIdx.x & 31u;
  const std::uint32_t row = static_cast<std::uint32_t>(
      (blockIdx.x * WavesPerBlock) + (threadIdx.x >> 5u));
  if (row >= d.total_rows) {
    return;
  }
  std::uint32_t tensor = 0;
#pragma unroll
  for (std::uint32_t t = 1; t < kMaxFused; ++t) {
    if (t < d.count && row >= d.prefix[t]) {
      tensor = t;
    }
  }
  const std::uint32_t local = row - d.prefix[tensor];
  float dot =
      RowDot<Unroll>(reinterpret_cast<const Vec16*>(d.a[tensor] + (local * K)),
                     reinterpret_cast<const Vec16*>(x), K / 8, lane_id);
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }
  if (lane_id == 0) {
    d.y[tensor][local] = dot;
  }
}

// Fused dispatch, `Rows` rows per wave. The activation vector is loaded once
// per row group instead of once per row, which is the only remaining redundant
// read once the grid is large enough. Per-row order is unchanged.
template<std::uint32_t WavesPerBlock, std::uint32_t Rows, std::uint32_t Unroll>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Bf16GemvFusedRows(FusedDesc d, const hip_bfloat16* __restrict__ x,
                           std::size_t K) {
  const std::size_t lane_id = threadIdx.x & 31u;
  const std::uint32_t group = static_cast<std::uint32_t>(
      (blockIdx.x * WavesPerBlock) + (threadIdx.x >> 5u));
  const std::uint32_t row0 = group * Rows;
  if (row0 >= d.total_rows) {
    return;
  }
  const auto* x_vec = reinterpret_cast<const Vec16*>(x);
  const std::size_t K_vec = K / 8;
  constexpr std::size_t kStep = 32 * Unroll;

  // Rows in one group can straddle two tensors, so each row resolves its own
  // base pointer and destination.
  const hip_bfloat16* bases[Rows];
  float* outs[Rows];
  std::uint32_t active = 0;
#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
    const std::uint32_t row = row0 + r;
    if (row < d.total_rows) {
      std::uint32_t tensor = 0;
#pragma unroll
      for (std::uint32_t t = 1; t < kMaxFused; ++t) {
        if (t < d.count && row >= d.prefix[t]) {
          tensor = t;
        }
      }
      const std::uint32_t local = row - d.prefix[tensor];
      bases[r] = d.a[tensor] + (local * K);
      outs[r] = d.y[tensor] + local;
      ++active;
    } else {
      bases[r] = nullptr;
      outs[r] = nullptr;
    }
  }

  float dot[Rows];
#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
    dot[r] = 0.0F;
  }

  if (active == Rows) {
    const Vec16* rows_vec[Rows];
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) {
      rows_vec[r] = reinterpret_cast<const Vec16*>(bases[r]);
    }
    std::size_t vi = lane_id;
    for (; vi + (kStep - 32) < K_vec; vi += kStep) {
      Vec16 a[Unroll];
      Vec16 wv[Rows][Unroll];
#pragma unroll
      for (std::uint32_t u = 0; u < Unroll; ++u) {
        a[u] = x_vec[vi + (u * 32)];
#pragma unroll
        for (std::uint32_t r = 0; r < Rows; ++r) {
          wv[r][u] = rows_vec[r][vi + (u * 32)];
        }
      }
#pragma unroll
      for (std::uint32_t u = 0; u < Unroll; ++u) {
#pragma unroll
        for (std::uint32_t r = 0; r < Rows; ++r) {
          dot[r] += Dot8(wv[r][u], a[u]);
        }
      }
    }
    for (; vi < K_vec; vi += 32) {
      const Vec16 a = x_vec[vi];
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r) {
        dot[r] += Dot8(rows_vec[r][vi], a);
      }
    }
  } else {
    for (std::uint32_t r = 0; r < active; ++r) {
      dot[r] = RowDot<Unroll>(reinterpret_cast<const Vec16*>(bases[r]), x_vec,
                              K_vec, lane_id);
    }
  }

#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot[r] += __shfl_xor(dot[r], offset);
    }
  }
  if (lane_id == 0) {
    for (std::uint32_t r = 0; r < active; ++r) {
      *outs[r] = dot[r];
    }
  }
}

// Fused dispatch with the activation vector staged in LDS once per block, so
// the vector cache serves only the weight stream.
template<std::uint32_t WavesPerBlock, std::uint32_t Unroll>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Bf16GemvFusedLds(FusedDesc d, const hip_bfloat16* __restrict__ x,
                          std::size_t K) {
  extern __shared__ Vec16 shared_x[];
  const std::size_t K_vec = K / 8;
  for (std::size_t i = threadIdx.x; i < K_vec; i += blockDim.x) {
    shared_x[i] = reinterpret_cast<const Vec16*>(x)[i];
  }
  __syncthreads();

  const std::size_t lane_id = threadIdx.x & 31u;
  const std::uint32_t row = static_cast<std::uint32_t>(
      (blockIdx.x * WavesPerBlock) + (threadIdx.x >> 5u));
  if (row >= d.total_rows) {
    return;
  }
  std::uint32_t tensor = 0;
#pragma unroll
  for (std::uint32_t t = 1; t < kMaxFused; ++t) {
    if (t < d.count && row >= d.prefix[t]) {
      tensor = t;
    }
  }
  const std::uint32_t local = row - d.prefix[tensor];
  const auto* row_vec =
      reinterpret_cast<const Vec16*>(d.a[tensor] + (local * K));
  constexpr std::size_t kStep = 32 * Unroll;

  float dot = 0.0F;
  std::size_t vi = lane_id;
  for (; vi + (kStep - 32) < K_vec; vi += kStep) {
    Vec16 w[Unroll];
#pragma unroll
    for (std::uint32_t u = 0; u < Unroll; ++u) {
      w[u] = row_vec[vi + (u * 32)];
    }
#pragma unroll
    for (std::uint32_t u = 0; u < Unroll; ++u) {
      dot += Dot8(w[u], shared_x[vi + (u * 32)]);
    }
  }
  for (; vi < K_vec; vi += 32) {
    dot += Dot8(row_vec[vi], shared_x[vi]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }
  if (lane_id == 0) {
    d.y[tensor][local] = dot;
  }
}

// ---------------------------------------------------------------------------
// Streaming-read ceiling: identical dispatch geometry and load pattern, no
// arithmetic beyond an accumulate that keeps the loads live.
// ---------------------------------------------------------------------------
template<std::uint32_t WavesPerBlock, std::uint32_t Unroll>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void ReadCeiling(const hip_bfloat16* __restrict__ A, float* __restrict__ y,
                     std::size_t M, std::size_t K) {
  const std::size_t lane_id = threadIdx.x & 31u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + (threadIdx.x >> 5u);
  if (m >= M) {
    return;
  }
  const auto* row_vec = reinterpret_cast<const Vec16*>(A + (m * K));
  const std::size_t K_vec = K / 8;
  constexpr std::size_t kStep = 32 * Unroll;
  Vec16 acc = {0u, 0u, 0u, 0u};
  std::size_t vi = lane_id;
  for (; vi + (kStep - 32) < K_vec; vi += kStep) {
    Vec16 w[Unroll];
#pragma unroll
    for (std::uint32_t u = 0; u < Unroll; ++u) {
      w[u] = row_vec[vi + (u * 32)];
    }
#pragma unroll
    for (std::uint32_t u = 0; u < Unroll; ++u) {
      acc += w[u];
    }
  }
  for (; vi < K_vec; vi += 32) {
    acc += row_vec[vi];
  }
  float dot = static_cast<float>(acc[0] + acc[1] + acc[2] + acc[3]);
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }
  if (lane_id == 0) {
    y[m] = dot;
  }
}

template<std::uint32_t WavesPerBlock, std::uint32_t Unroll>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void ReadCeilingWithX(const hip_bfloat16* __restrict__ A,
                          const hip_bfloat16* __restrict__ x,
                          float* __restrict__ y, std::size_t M, std::size_t K) {
  const std::size_t lane_id = threadIdx.x & 31u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + (threadIdx.x >> 5u);
  if (m >= M) {
    return;
  }
  const auto* row_vec = reinterpret_cast<const Vec16*>(A + (m * K));
  const auto* x_vec = reinterpret_cast<const Vec16*>(x);
  const std::size_t K_vec = K / 8;
  constexpr std::size_t kStep = 32 * Unroll;
  Vec16 acc = {0u, 0u, 0u, 0u};
  std::size_t vi = lane_id;
  for (; vi + (kStep - 32) < K_vec; vi += kStep) {
    Vec16 w[Unroll];
    Vec16 a[Unroll];
#pragma unroll
    for (std::uint32_t u = 0; u < Unroll; ++u) {
      w[u] = row_vec[vi + (u * 32)];
      a[u] = x_vec[vi + (u * 32)];
    }
#pragma unroll
    for (std::uint32_t u = 0; u < Unroll; ++u) {
      acc += w[u] ^ a[u];
    }
  }
  for (; vi < K_vec; vi += 32) {
    acc += row_vec[vi] ^ x_vec[vi];
  }
  float dot = static_cast<float>(acc[0] + acc[1] + acc[2] + acc[3]);
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }
  if (lane_id == 0) {
    y[m] = dot;
  }
}

__global__ void FillBf16(std::uint16_t* out, std::size_t count,
                         std::uint32_t seed) {
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  std::uint32_t state = seed + (blockIdx.x * blockDim.x) + threadIdx.x;
  for (std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < count;
       i += stride) {
    state = (state * 1664525u) + 1013904223u;
    // bf16 near +-2^-4 with a random sign and mantissa. Magnitudes must stay
    // well clear of the denormal range or the exactness check compares noise.
    const std::uint32_t mantissa = (state >> 9) & 0x7fu;
    const std::uint32_t sign = (state >> 31) << 15;
    out[i] = static_cast<std::uint16_t>(sign | (0x7b << 7) | mantissa);
  }
}

__global__ void Bf16ToF32(const std::uint16_t* in, float* out,
                          std::size_t count) {
  const std::size_t i = (std::size_t(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i < count) {
    out[i] = __uint_as_float(static_cast<std::uint32_t>(in[i]) << 16);
  }
}

// Qwen3-ASR-1.7B text decoder geometry.
constexpr std::size_t kLayers = 28;
constexpr std::size_t kHidden = 2048;
constexpr std::size_t kFfn = 6144;
constexpr std::size_t kKvDim = 1024;  // 8 KV heads x 128
constexpr std::size_t kVocab = 151936;

enum class Kind : std::uint8_t { kQ, kK, kV, kO, kGate, kUp, kDown, kHead };

const char* KindName(Kind kind) {
  switch (kind) {
    case Kind::kQ:
      return "q_proj";
    case Kind::kK:
      return "k_proj";
    case Kind::kV:
      return "v_proj";
    case Kind::kO:
      return "o_proj";
    case Kind::kGate:
      return "ffn_gate";
    case Kind::kUp:
      return "ffn_up";
    case Kind::kDown:
      return "ffn_down";
    case Kind::kHead:
      return "lm_head";
  }
  return "?";
}

struct Projection {
  Kind kind;
  std::size_t m;
  std::size_t k;
  std::size_t offset;  // element offset into the arena
};

// A dispatch group: one or more projections launched together.
struct Group {
  std::vector<std::size_t> members;  // indices into the projection list
};

}  // namespace

int main(int argc, char** argv) {
  const int passes = (argc > 1) ? std::atoi(argv[1]) : 8;
  hipDeviceProp_t props{};
  CHECK(hipGetDeviceProperties(&props, 0));

  std::vector<Projection> projections;
  std::size_t arena = 0;
  auto push = [&](Kind kind, std::size_t m, std::size_t k) {
    projections.push_back({kind, m, k, arena});
    arena += m * k;
  };
  for (std::size_t layer = 0; layer < kLayers; ++layer) {
    push(Kind::kQ, kHidden, kHidden);
    push(Kind::kK, kKvDim, kHidden);
    push(Kind::kV, kKvDim, kHidden);
    push(Kind::kO, kHidden, kHidden);
    push(Kind::kGate, kFfn, kHidden);
    push(Kind::kUp, kFfn, kHidden);
    push(Kind::kDown, kHidden, kFfn);
  }
  push(Kind::kHead, kVocab, kHidden);

  // Unfused schedule: one dispatch per projection, as the runtime does today.
  std::vector<Group> unfused;
  for (std::size_t i = 0; i < projections.size(); ++i) {
    unfused.push_back({{i}});
  }
  // Fused schedule: q/k/v in one dispatch, gate/up in another.
  std::vector<Group> fused_schedule;
  for (std::size_t layer = 0; layer < kLayers; ++layer) {
    const std::size_t base = layer * 7;
    fused_schedule.push_back({{base + 0, base + 1, base + 2}});
    fused_schedule.push_back({{base + 3}});
    fused_schedule.push_back({{base + 4, base + 5}});
    fused_schedule.push_back({{base + 6}});
  }
  fused_schedule.push_back({{projections.size() - 1}});

  const double arena_bytes = static_cast<double>(arena) * 2.0;
  std::printf("device: %s arch=%s CUs=%d\n", props.name, props.gcnArchName,
              props.multiProcessorCount);
  std::printf("decoder arena: %.3f GB across %zu projections\n",
              arena_bytes / 1e9, projections.size());
  std::printf("DRAM read peak: %.2f GB/s   token passes: %d\n\n",
              kDramReadPeakGBs, passes);

  std::uint16_t* d_weights = nullptr;
  CHECK(hipMalloc(&d_weights, arena * sizeof(std::uint16_t)));
  FillBf16<<<4096, 256>>>(d_weights, arena, 7u);
  CHECK(hipDeviceSynchronize());

  std::uint16_t* d_x = nullptr;
  float* d_x_f32 = nullptr;
  float* d_y = nullptr;
  CHECK(hipMalloc(&d_x, kFfn * sizeof(std::uint16_t)));
  CHECK(hipMalloc(&d_x_f32, kFfn * sizeof(float)));
  CHECK(hipMalloc(&d_y, 3 * kVocab * sizeof(float)));
  FillBf16<<<64, 256>>>(d_x, kFfn, 99u);
  Bf16ToF32<<<(kFfn + 255) / 256, 256>>>(d_x, d_x_f32, kFfn);
  CHECK(hipDeviceSynchronize());

  const auto* w = reinterpret_cast<const hip_bfloat16*>(d_weights);
  const auto* xb = reinterpret_cast<const hip_bfloat16*>(d_x);

  // A variant launches one dispatch group. Outputs are written at successive
  // offsets in d_y so a fused group can be validated per member.
  using LaunchGroup =
      void (*)(const hip_bfloat16* arena_base, const hip_bfloat16* x_bf16,
               const float* x_f32, float* y, const Projection* projs,
               const std::size_t* members, std::size_t count);

  struct Variant {
    const char* name;
    const std::vector<Group>* schedule;
    LaunchGroup launch;
  };

  const std::vector<Variant> variants = {
      {"v0 retained 1row f32x", &unfused,
       [](const hip_bfloat16* base, const hip_bfloat16*, const float* xf,
          float* y, const Projection* projs, const std::size_t* members,
          std::size_t) {
         constexpr std::uint32_t kW = 4;
         const auto& p = projs[members[0]];
         Baseline1Row<kW><<<(p.m + kW - 1) / kW, kW * 32>>>(base + p.offset, xf,
                                                            y, p.m, p.k);
       }},
      {"v1 unfused bf16x u8", &unfused,
       [](const hip_bfloat16* base, const hip_bfloat16* x, const float*,
          float* y, const Projection* projs, const std::size_t* members,
          std::size_t) {
         constexpr std::uint32_t kW = 4;
         const auto& p = projs[members[0]];
         Bf16Gemv<kW, 8><<<(p.m + kW - 1) / kW, kW * 32>>>(base + p.offset, x,
                                                           y, p.m, p.k);
       }},
      {"v2 fused u4", &fused_schedule,
       [](const hip_bfloat16* base, const hip_bfloat16* x, const float*,
          float* y, const Projection* projs, const std::size_t* members,
          std::size_t count) {
         constexpr std::uint32_t kW = 4;
         FusedDesc d{};
         d.count = static_cast<std::uint32_t>(count);
         std::uint32_t total = 0;
         for (std::size_t i = 0; i < count; ++i) {
           const auto& p = projs[members[i]];
           d.a[i] = base + p.offset;
           d.y[i] = y + (i * kVocab);
           d.rows[i] = static_cast<std::uint32_t>(p.m);
           d.prefix[i] = total;
           total += static_cast<std::uint32_t>(p.m);
         }
         d.total_rows = total;
         const std::size_t K = projs[members[0]].k;
         Bf16GemvFused<kW, 4><<<(total + kW - 1) / kW, kW * 32>>>(d, x, K);
       }},
      {"v3 fused u8", &fused_schedule,
       [](const hip_bfloat16* base, const hip_bfloat16* x, const float*,
          float* y, const Projection* projs, const std::size_t* members,
          std::size_t count) {
         constexpr std::uint32_t kW = 4;
         FusedDesc d{};
         d.count = static_cast<std::uint32_t>(count);
         std::uint32_t total = 0;
         for (std::size_t i = 0; i < count; ++i) {
           const auto& p = projs[members[i]];
           d.a[i] = base + p.offset;
           d.y[i] = y + (i * kVocab);
           d.rows[i] = static_cast<std::uint32_t>(p.m);
           d.prefix[i] = total;
           total += static_cast<std::uint32_t>(p.m);
         }
         d.total_rows = total;
         const std::size_t K = projs[members[0]].k;
         Bf16GemvFused<kW, 8><<<(total + kW - 1) / kW, kW * 32>>>(d, x, K);
       }},
      {"v4 fused 2row u4", &fused_schedule,
       [](const hip_bfloat16* base, const hip_bfloat16* x, const float*,
          float* y, const Projection* projs, const std::size_t* members,
          std::size_t count) {
         constexpr std::uint32_t kW = 4;
         FusedDesc d{};
         d.count = static_cast<std::uint32_t>(count);
         std::uint32_t total = 0;
         for (std::size_t i = 0; i < count; ++i) {
           const auto& p = projs[members[i]];
           d.a[i] = base + p.offset;
           d.y[i] = y + (i * kVocab);
           d.rows[i] = static_cast<std::uint32_t>(p.m);
           d.prefix[i] = total;
           total += static_cast<std::uint32_t>(p.m);
         }
         d.total_rows = total;
         const std::size_t K = projs[members[0]].k;
         const std::size_t groups = (total + 1) / 2;
         Bf16GemvFusedRows<kW, 2, 4>
             <<<(groups + kW - 1) / kW, kW * 32>>>(d, x, K);
       }},
      {"v5 fused 2row u8", &fused_schedule,
       [](const hip_bfloat16* base, const hip_bfloat16* x, const float*,
          float* y, const Projection* projs, const std::size_t* members,
          std::size_t count) {
         constexpr std::uint32_t kW = 4;
         FusedDesc d{};
         d.count = static_cast<std::uint32_t>(count);
         std::uint32_t total = 0;
         for (std::size_t i = 0; i < count; ++i) {
           const auto& p = projs[members[i]];
           d.a[i] = base + p.offset;
           d.y[i] = y + (i * kVocab);
           d.rows[i] = static_cast<std::uint32_t>(p.m);
           d.prefix[i] = total;
           total += static_cast<std::uint32_t>(p.m);
         }
         d.total_rows = total;
         const std::size_t K = projs[members[0]].k;
         const std::size_t groups = (total + 1) / 2;
         Bf16GemvFusedRows<kW, 2, 8>
             <<<(groups + kW - 1) / kW, kW * 32>>>(d, x, K);
       }},
      {"v6 fused 4row u4", &fused_schedule,
       [](const hip_bfloat16* base, const hip_bfloat16* x, const float*,
          float* y, const Projection* projs, const std::size_t* members,
          std::size_t count) {
         constexpr std::uint32_t kW = 4;
         FusedDesc d{};
         d.count = static_cast<std::uint32_t>(count);
         std::uint32_t total = 0;
         for (std::size_t i = 0; i < count; ++i) {
           const auto& p = projs[members[i]];
           d.a[i] = base + p.offset;
           d.y[i] = y + (i * kVocab);
           d.rows[i] = static_cast<std::uint32_t>(p.m);
           d.prefix[i] = total;
           total += static_cast<std::uint32_t>(p.m);
         }
         d.total_rows = total;
         const std::size_t K = projs[members[0]].k;
         const std::size_t groups = (total + 3) / 4;
         Bf16GemvFusedRows<kW, 4, 4>
             <<<(groups + kW - 1) / kW, kW * 32>>>(d, x, K);
       }},
      {"v7 fused lds u8", &fused_schedule,
       [](const hip_bfloat16* base, const hip_bfloat16* x, const float*,
          float* y, const Projection* projs, const std::size_t* members,
          std::size_t count) {
         constexpr std::uint32_t kW = 4;
         FusedDesc d{};
         d.count = static_cast<std::uint32_t>(count);
         std::uint32_t total = 0;
         for (std::size_t i = 0; i < count; ++i) {
           const auto& p = projs[members[i]];
           d.a[i] = base + p.offset;
           d.y[i] = y + (i * kVocab);
           d.rows[i] = static_cast<std::uint32_t>(p.m);
           d.prefix[i] = total;
           total += static_cast<std::uint32_t>(p.m);
         }
         d.total_rows = total;
         const std::size_t K = projs[members[0]].k;
         const std::size_t lds = (K / 8) * 16;
         Bf16GemvFusedLds<kW, 8>
             <<<(total + kW - 1) / kW, kW * 32, lds>>>(d, x, K);
       }},
      {"v8 fused u8 w8", &fused_schedule,
       [](const hip_bfloat16* base, const hip_bfloat16* x, const float*,
          float* y, const Projection* projs, const std::size_t* members,
          std::size_t count) {
         constexpr std::uint32_t kW = 8;
         FusedDesc d{};
         d.count = static_cast<std::uint32_t>(count);
         std::uint32_t total = 0;
         for (std::size_t i = 0; i < count; ++i) {
           const auto& p = projs[members[i]];
           d.a[i] = base + p.offset;
           d.y[i] = y + (i * kVocab);
           d.rows[i] = static_cast<std::uint32_t>(p.m);
           d.prefix[i] = total;
           total += static_cast<std::uint32_t>(p.m);
         }
         d.total_rows = total;
         Bf16GemvFused<kW, 8>
             <<<(total + kW - 1) / kW, kW * 32>>>(d, x, projs[members[0]].k);
       }},
      {"ceiling weights u8", &unfused,
       [](const hip_bfloat16* base, const hip_bfloat16*, const float*, float* y,
          const Projection* projs, const std::size_t* members, std::size_t) {
         constexpr std::uint32_t kW = 4;
         const auto& p = projs[members[0]];
         ReadCeiling<kW, 8>
             <<<(p.m + kW - 1) / kW, kW * 32>>>(base + p.offset, y, p.m, p.k);
       }},
      {"ceiling weights+x u8", &unfused,
       [](const hip_bfloat16* base, const hip_bfloat16* x, const float*,
          float* y, const Projection* projs, const std::size_t* members,
          std::size_t) {
         constexpr std::uint32_t kW = 4;
         const auto& p = projs[members[0]];
         ReadCeilingWithX<kW, 8><<<(p.m + kW - 1) / kW, kW * 32>>>(
             base + p.offset, x, y, p.m, p.k);
       }},
  };

  // Reference outputs from the retained kernel, one entry per projection.
  std::vector<std::vector<float>> reference(projections.size());
  for (std::size_t p = 0; p < projections.size(); ++p) {
    const auto& proj = projections[p];
    const std::size_t member = p;
    variants[0].launch(w, xb, d_x_f32, d_y, projections.data(), &member, 1);
    CHECK(hipDeviceSynchronize());
    reference[p].resize(proj.m);
    CHECK(hipMemcpy(reference[p].data(), d_y, proj.m * sizeof(float),
                    hipMemcpyDeviceToHost));
  }

  std::vector<float> candidate(kVocab);
  std::printf("%-24s %11s %10s %8s %9s  %s\n", "variant", "ms/token", "GB/s",
              "of peak", "dispatch", "numerics");
  double baseline_ms = 0.0;
  for (const auto& variant : variants) {
    const bool is_ceiling = std::string(variant.name).find("ceiling") == 0;
    std::size_t mismatched = 0;
    double max_abs = 0.0;
    if (!is_ceiling) {
      for (const auto& group : *variant.schedule) {
        variant.launch(w, xb, d_x_f32, d_y, projections.data(),
                       group.members.data(), group.members.size());
        CHECK(hipDeviceSynchronize());
        for (std::size_t i = 0; i < group.members.size(); ++i) {
          const std::size_t p = group.members[i];
          const std::size_t out = (group.members.size() > 1) ? i * kVocab : 0;
          CHECK(hipMemcpy(candidate.data(), d_y + out,
                          projections[p].m * sizeof(float),
                          hipMemcpyDeviceToHost));
          for (std::size_t r = 0; r < projections[p].m; ++r) {
            if (candidate[r] != reference[p][r]) {
              ++mismatched;
              max_abs = std::max(max_abs, static_cast<double>(std::abs(
                                              candidate[r] - reference[p][r])));
            }
          }
        }
      }
    }

    std::size_t dispatches = variant.schedule->size();
    auto token_pass = [&] {
      for (const auto& group : *variant.schedule) {
        variant.launch(w, xb, d_x_f32, d_y, projections.data(),
                       group.members.data(), group.members.size());
      }
    };
    token_pass();
    CHECK(hipDeviceSynchronize());
    hipEvent_t start;
    hipEvent_t stop;
    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));
    CHECK(hipEventRecord(start));
    for (int i = 0; i < passes; ++i) {
      token_pass();
    }
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    float ms = 0.0F;
    CHECK(hipEventElapsedTime(&ms, start, stop));
    CHECK(hipEventDestroy(start));
    CHECK(hipEventDestroy(stop));

    const double per_token = static_cast<double>(ms) / passes;
    const double gbs = arena_bytes / (per_token * 1e-3) / 1e9;
    if (baseline_ms == 0.0) {
      baseline_ms = per_token;
    }
    std::printf("%-24s %8.3f ms %7.1f GB/s %7.1f%% %9zu  ", variant.name,
                per_token, gbs, 100.0 * gbs / kDramReadPeakGBs, dispatches);
    if (is_ceiling) {
      std::printf("n/a");
    } else if (mismatched == 0) {
      std::printf("bit-exact");
    } else {
      std::printf("DIFFERS rows=%zu max_abs=%.3e", mismatched, max_abs);
    }
    std::printf("   x%.3f\n", baseline_ms / per_token);
  }

  CHECK(hipFree(d_weights));
  CHECK(hipFree(d_x));
  CHECK(hipFree(d_x_f32));
  CHECK(hipFree(d_y));
  return 0;
}
