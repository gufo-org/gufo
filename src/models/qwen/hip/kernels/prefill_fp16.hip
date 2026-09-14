#if defined(ENGINE_ENABLE_HIP)
#include <stdexcept>

#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
namespace gufo::hip {
namespace {
__device__ __forceinline__ __half
RoundActivation(float value) {  // Keep the FP32 rounding boundary before the
                                // FP16 conversion. Without this
  // register dependency, combining the producer with conversion changes rare
  // halfway values relative to the standalone FP32 producer. No instruction
  // or scratch memory is emitted by this compiler barrier.
  asm volatile("" : "+v"(value));
  return __float2half_rn(value);
}
using v16bf = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f WmmaHalf(v16bf a, v16bf b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// Permute 16-byte halves and the LDS row to avoid repeated bank conflicts.
__device__ __forceinline__ v16bf LoadSwizzled(const short* p) {
  union {
    v16bf f;
    uint4 q[2];
  } bits;
  const unsigned row = threadIdx.x & 15, shift = (row ^ (row >> 2)) & 1;
  bits.q[0] = reinterpret_cast<const uint4*>(p)[shift];
  bits.q[1] = reinterpret_cast<const uint4*>(p)[shift ^ 1];
  return bits.f;
}
__device__ __forceinline__ void StoreSwizzled(short* p, uint4 lo, uint4 hi,
                                              unsigned row) {
  const unsigned shift = (row ^ (row >> 2)) & 1;
  reinterpret_cast<uint4*>(p)[shift] = lo;
  reinterpret_cast<uint4*>(p)[shift ^ 1] = hi;
}

template<gufo::core::GgmlType Type>
struct RawWeights {
  static_assert(Type == gufo::core::GgmlType::kQ4_K ||
                Type == gufo::core::GgmlType::kQ5_K ||
                Type == gufo::core::GgmlType::kQ6_K ||
                Type == gufo::core::GgmlType::kQ8_0);
  std::uint32_t low[4], high[Type == gufo::core::GgmlType::kQ5_K ||
                                     Type == gufo::core::GgmlType::kQ6_K
                                 ? 4
                                 : 1];
  __half d, dmin;
  unsigned sc0, sc1, sc2, sub;
  int signed_scale;
};
template<gufo::core::GgmlType Type>
__device__ __forceinline__ RawWeights<Type> LoadRaw(const void* row,
                                                    unsigned sub) {
  using namespace gufo::hip;
  using gufo::core::GgmlType;
  RawWeights<Type> r{};
  r.sub = sub;
  const unsigned sb = (sub / 2) % 8, lane = (sub % 2) * 16;
  if constexpr (Type == GgmlType::kQ8_0) {
    const auto& b = static_cast<const Q8_0Block*>(row)[sub / 2];
    r.d = b.d;
    __builtin_memcpy(r.low, b.qs + lane, 16);
  } else if constexpr (Type == GgmlType::kQ4_K || Type == GgmlType::kQ5_K) {
    using Block =
        std::conditional_t<Type == GgmlType::kQ5_K, Q5KBlock, Q4KBlock>;
    const auto& b = static_cast<const Block*>(row)[sub / 16];
    r.d = b.d;
    r.dmin = b.dmin;
    r.sc0 = b.scales[sb & 3];
    r.sc1 = b.scales[(sb & 3) + 4];
    r.sc2 = b.scales[(sb & 3) + 8];
    LoadQuantWords16(b.qs + (sb / 2) * 32 + lane, r.low);
    if constexpr (Type == GgmlType::kQ5_K)
      LoadQuantWords16(b.qh + lane, r.high);
  } else if constexpr (Type == GgmlType::kQ6_K) {
    const auto& b = static_cast<const Q6KBlock*>(row)[sub / 16];
    unsigned half = sb / 4, segment = sb % 4;
    r.d = b.d;
    r.signed_scale = b.scales[half * 8 + segment * 2 + (sub % 2)];
    LoadQuantWords16(b.ql + half * 64 + (segment % 2) * 32 + lane, r.low);
    LoadQuantWords16(b.qh + half * 32 + lane, r.high);
  }
  return r;
}
template<gufo::core::GgmlType Type>
__device__ __forceinline__ gufo::hip::QuantSub16 DecodeRaw(
    const RawWeights<Type>& r) {
  using namespace gufo::hip;
  using gufo::core::GgmlType;
  QuantSub16 q;
  q.offset = 0;
  unsigned sb = (r.sub / 2) % 8;
  if constexpr (Type == GgmlType::kQ8_0) {
    __builtin_memcpy(q.w, r.low, 16);
    q.scale = __half2float(r.d);
  } else if constexpr (Type == GgmlType::kQ4_K || Type == GgmlType::kQ5_K) {
    unsigned mask = 0u - (sb >> 2), lo = r.sc0 & 63,
             up = (r.sc2 & 15) | ((r.sc0 >> 6) << 4), lm = r.sc1 & 63,
             um = (r.sc2 >> 4) | ((r.sc1 >> 6) << 4);
    q.scale = __half2float(r.d) * static_cast<float>(lo ^ ((lo ^ up) & mask));
    q.offset =
        __half2float(r.dmin) * static_cast<float>(lm ^ ((lm ^ um) & mask));
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
      q.w[i] = (r.low[i] >> (4 * (sb & 1))) & 0x0f0f0f0f;
      if constexpr (Type == GgmlType::kQ5_K)
        q.w[i] |= ((r.high[i] >> sb) & 0x01010101) << 4;
    }
  } else if constexpr (Type == GgmlType::kQ6_K) {
    q.scale = __half2float(r.d) * static_cast<float>(r.signed_scale);
    unsigned seg = sb % 4;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
      q.w[i] = SubBytes(((r.low[i] >> (seg >= 2 ? 4 : 0)) & 0x0f0f0f0f) |
                            (((r.high[i] >> (2 * seg)) & 0x03030303) << 4),
                        0x20202020);
  }
  return q;
}

enum class HalfEpilogue { kStore, kSwiGLU, kGateUp, kResidual };

// Fully scaled FP16 weights and activations; FP32 accumulation in K16 order.
template<int BM, int BN, int WM, int WN, gufo::core::GgmlType Type,
         HalfEpilogue Epilogue, bool Complete = false>
__launch_bounds__(WM * WN * 32) __global__
    void HalfPrefillGemmKernel(const void* __restrict__ w,
                               const short* __restrict__ x,
                               float* __restrict__ y, int batch, int m, int k,
                               const float* __restrict__ gate,
                               const void* __restrict__ up = nullptr) {
  constexpr bool Paired = Epilogue == HalfEpilogue::kGateUp;
  constexpr int BK = 4, WaveSize = 32, Group = 8;
  constexpr int DecodeStage = (Type == gufo::core::GgmlType::kQ4_K ||
                               Type == gufo::core::GgmlType::kQ5_K ||
                               Type == gufo::core::GgmlType::kQ6_K ||
                               Type == gufo::core::GgmlType::kQ8_0)
                                  ? 2
                                  : 0;
  constexpr int kThreads = WM * WN * WaveSize;
  constexpr int RS = BM / 16;
  constexpr int TS = BN / 16;
  constexpr int WRS = RS / WM;
  constexpr int WTS = TS / WN;
  constexpr int KS = 16;
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);

  constexpr int PFA = (BM * BK + kThreads - 1) / kThreads;
  constexpr int PFB = (BN * BK + kThreads - 1) / kThreads;

  __shared__ short s_a[BK][RS][16][KS];
  __shared__ short s_b[BK][TS][16][KS];

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / WaveSize;
  const int lane = tid % WaveSize;
  const int sl = lane & 15;
  const int wr = wave / WN;
  const int wt = wave % WN;

  unsigned row_tile = blockIdx.y, token_tile = blockIdx.x;
  {
    unsigned first = (blockIdx.y / Group) * Group,
             within = (blockIdx.y % Group) * gridDim.x + blockIdx.x,
             rows = min(unsigned(Group), gridDim.y - first);
    row_tile = first + within % rows;
    token_tile = within / rows;
  }
  const int r_block = row_tile * (Paired ? BM / 2 : BM),
            t_block = token_tile * BN;
  const int ksteps = k / 16;

  v8f acc[WRS][WTS] = {};

  // Hoisted 64-bit row bases, clamped so an out-of-range row reads row m-1 and
  // is masked out at the store instead of branching in the K loop.
  const std::uint8_t* w_row[PFA];
  const std::size_t row_bytes = gufo::hip::QuantRowBytes(Type, k);
#pragma unroll
  for (int p = 0; p < PFA; ++p) {
    const int physical_row = ((p * kThreads) + tid) / BK;
    const int r =
        r_block +
        (Paired ? (physical_row / 32) * 16 + physical_row % 16 : physical_row);
    const void* source = Paired && physical_row % 32 >= 16 ? up : w;
    w_row[p] = static_cast<const std::uint8_t*>(source) +
               (static_cast<std::size_t>((Complete || r < m) ? r : (m - 1)) *
                row_bytes);
  }
  const short* x_row[PFB];
#pragma unroll
  for (int p = 0; p < PFB; ++p) {
    const int t = t_block + (((p * kThreads) + tid) / BK);
    x_row[p] =
        x +
        (static_cast<std::size_t>((Complete || t < batch) ? t : (batch - 1)) *
         k);
  }

  uint4 ra[PFA][2], rb[PFB][2];
  struct NoRaw {};
  std::conditional_t<DecodeStage == 2, RawWeights<Type>, NoRaw> raw[PFA];
  const auto fetch = [&](int ks0) {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      if (idx < BM * BK && (Complete || kk < ksteps)) {
        if constexpr (DecodeStage == 2)
          raw[p] = LoadRaw<Type>(w_row[p], kk);
        else {
          gufo::hip::QuantSub16 q;
          gufo::hip::DecodeQuantSub16<true>(Type, w_row[p], kk, q);
          union {
            v16bf f;
            uint4 u[2];
          } packed;
#pragma unroll
          for (int j = 0; j < 16; ++j)
            packed.f[j] = static_cast<_Float16>(__fsub_rn(
                __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
          ra[p][0] = packed.u[0];
          ra[p][1] = packed.u[1];
        }
      } else {
        ra[p][0] = make_uint4(0, 0, 0, 0);
        ra[p][1] = make_uint4(0, 0, 0, 0);
        if constexpr (DecodeStage == 2)
          raw[p] = {};
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      if (idx < BN * BK && (Complete || kk < ksteps)) {
        const short* src = x_row[p] + static_cast<std::size_t>(kk) * 16;
        rb[p][0] = *reinterpret_cast<const uint4*>(src);
        rb[p][1] = *reinterpret_cast<const uint4*>(src + 8);
      } else {
        rb[p][0] = make_uint4(0, 0, 0, 0);
        rb[p][1] = make_uint4(0, 0, 0, 0);
      }
    }
  };
  const auto commit = [&]() {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < BM * BK) {
        const int placed = (row ^ (ks & 3));
        short* dst = &s_a[ks][placed / 16][placed % 16][0];
        if constexpr (DecodeStage != 0) {
          gufo::hip::QuantSub16 q = DecodeRaw<Type>(raw[p]);
          union {
            v16bf f;
            uint4 u[2];
          } packed;
#pragma unroll
          for (int j = 0; j < 16; ++j)
            packed.f[j] = static_cast<_Float16>(__fsub_rn(
                __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
          StoreSwizzled(dst, packed.u[0], packed.u[1], row);
        } else
          StoreSwizzled(dst, ra[p][0], ra[p][1], row);
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < BN * BK) {
        const int placed = (row ^ (ks & 3));
        short* dst = &s_b[ks][placed / 16][placed % 16][0];
        StoreSwizzled(dst, rb[p][0], rb[p][1], row);
      }
    }
  };
  fetch(0);
  for (int ks0 = 0; ks0 < ksteps; ks0 += BK) {
    commit();
    __syncthreads();
    if (ks0 + BK < ksteps)
      fetch(ks0 + BK);
#pragma unroll
    for (int ks = 0; ks < BK; ++ks) {
      v16bf a[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        a[i] = LoadSwizzled(&s_a[ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
      }
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const v16bf b =
            LoadSwizzled(&s_b[ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
          if constexpr (Type == gufo::core::GgmlType::kIQ3_S)
            acc[i][j] = WmmaHalf(a[i], b, acc[i][j]);
          else
            acc[i][j] = WmmaHalf(b, a[i], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // Orient WMMA output as [token][row] for coalesced direct stores. IQ3_S
  // keeps the original orientation and LDS transpose, which is faster for
  // its decoder's register schedule.
  const auto store = [&](std::size_t index, float value) {
    if constexpr (Epilogue == HalfEpilogue::kSwiGLU) {
      const float g = gate[index];
      const float sig = 1.0F / (1.0F + expf(-g));
      reinterpret_cast<__half*>(y)[index] =
          RoundActivation(__fmul_rn(g * sig, value));
    } else if constexpr (Epilogue == HalfEpilogue::kResidual) {
      // Add only after the dot product, preserving the FP32 residual link.
      y[index] = __fadd_rn(y[index], value);
    } else
      y[index] = value;
  };
  const int hi = lane / 16;
  if constexpr (Paired) {
    static_assert(WRS % 2 == 0 && Type != gufo::core::GgmlType::kIQ3_S);
    // Adjacent 16-row fragments hold gate and up for the same logical rows.
    // One wave owns both accumulators, so no intermediate store or exchange
    // between waves is needed.
#pragma unroll
    for (int i = 0; i < WRS; i += 2) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const int row = r_block + (wr * (WRS / 2) + i / 2) * 16 + sl;
        const int t0 = t_block + (wt * WTS + j) * 16;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const int token = t0 + 2 * l + hi;
          if (Complete || (row < m && token < batch)) {
            const float g = acc[i][j][l], u = acc[i + 1][j][l];
            const float sig = 1.0F / (1.0F + expf(-g));
            reinterpret_cast<__half*>(y)[std::size_t(token) * m + row] =
                RoundActivation(__fmul_rn(g * sig, u));
          }
        }
      }
    }
  } else if constexpr (Type != gufo::core::GgmlType::kIQ3_S) {
#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const int row = r_block + ((wr * WRS + i) * 16) + sl;
        const int t0 = t_block + ((wt * WTS + j) * 16);
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const int token = t0 + 2 * l + hi;
          if (Complete || (row < m && token < batch))
            store(std::size_t(token) * m + row, acc[i][j][l]);
        }
      }
    }
  } else {
    __syncthreads();
    constexpr int Slots =
        (sizeof(s_a) >= sizeof(s_b) ? sizeof(s_a) : sizeof(s_b)) / (256 * 4);
    static_assert(Slots >= WM * WN);
    float* scratch = (sizeof(s_a) >= sizeof(s_b)
                          ? reinterpret_cast<float*>(&s_a[0][0][0][0])
                          : reinterpret_cast<float*>(&s_b[0][0][0][0])) +
                     (wave * 256);

#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
#pragma unroll
        for (int l = 0; l < 256 / WaveSize; ++l) {
          scratch[(sl * 16) + ((WaveSize / 16) * l) + hi] = acc[i][j][l];
        }
        __builtin_amdgcn_wave_barrier();
        const int r0 = r_block + (((wr * WRS) + i) * 16);
        const int t0 = t_block + (((wt * WTS) + j) * 16);
        if (t0 + 16 <= batch && r0 + 16 <= m) {
          const std::size_t base =
              (static_cast<std::size_t>(t0) * m) + static_cast<std::size_t>(r0);
#pragma unroll
          for (int s = 0; s < 256 / WaveSize; ++s) {
            const int flat = (s * WaveSize) + lane;
            store(
                base + (static_cast<std::size_t>(flat >> 4) * m) + (flat & 15),
                scratch[flat]);
          }
        } else {
#pragma unroll
          for (int s = 0; s < 256 / WaveSize; ++s) {
            const int flat = (s * WaveSize) + lane;
            const int tok = t0 + (flat >> 4);
            const int r = r0 + (flat & 15);
            if (tok < batch && r < m) {
              store((static_cast<std::size_t>(tok) * m) + r, scratch[flat]);
            }
          }
        }
        __builtin_amdgcn_wave_barrier();
      }
    }
  }
}

__global__ void HalfNorm(const float* x, const float* residual,
                         const float* weight, float* sum_out, __half* out,
                         std::size_t dim, float eps) {
  const std::size_t row = blockIdx.x, tid = threadIdx.x, base = row * dim;
  float sum_sq = 0;
  for (std::size_t i = tid; i < dim; i += 256) {
    const float v =
        residual ? __fadd_rn(x[base + i], residual[base + i]) : x[base + i];
    sum_sq += v * v;
  }
  __shared__ float sums[256];
  sums[tid] = sum_sq;
  __syncthreads();
  for (unsigned stride = 128; stride; stride /= 2) {
    if (tid < stride)
      sums[tid] += sums[tid + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(sums[0] / static_cast<float>(dim) + eps);
  for (std::size_t i = tid; i < dim; i += 256) {
    const float v =
        residual ? __fadd_rn(x[base + i], residual[base + i]) : x[base + i];
    if (sum_out)
      sum_out[base + i] = v;
    out[base + i] = RoundActivation(__fmul_rn(v * inv, weight[i]));
  }
}
__global__ void HalfCast(const float* x, __half* y, std::size_t n) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = __float2half_rn(x[i]);
}
}  // namespace
void LaunchBatchedRMSNormFp16(const float* x, const float* residual,
                              const float* weight, float* sum_out, void* out,
                              std::size_t batch, std::size_t dim, float eps,
                              hipStream_t stream) {
  HalfNorm<<<batch, 256, 0, stream>>>(x, residual, weight, sum_out,
                                      static_cast<__half*>(out), dim, eps);
}
void LaunchFloatToFp16(const float* x, void* y, std::size_t n,
                       hipStream_t stream) {
  HalfCast<<<(n + 255) / 256, 256, 0, stream>>>(x, static_cast<__half*>(y), n);
}
template<HalfEpilogue Epilogue>
void DirectGemm(gufo::core::GgmlType type, const void* w, const void* x,
                float* y, std::size_t batch, std::size_t m, std::size_t k,
                const float* gate, hipStream_t stream) {
  const auto launch = [&]<gufo::core::GgmlType Type>() {
    constexpr bool LargeKvTile = Type == gufo::core::GgmlType::kQ4_K ||
                                 Type == gufo::core::GgmlType::kQ5_K ||
                                 Type == gufo::core::GgmlType::kQ6_K ||
                                 Type == gufo::core::GgmlType::kQ8_0;
    const auto large = [&]<bool Full>() {
      HalfPrefillGemmKernel<256, 256, 8, 4, Type, Epilogue, Full>
          <<<dim3((batch + 255) / 256, (m + 255) / 256), 1024, 0, stream>>>(
              w, static_cast<const short*>(x), y, batch, m, k, gate);
    };
    if (Epilogue == HalfEpilogue::kSwiGLU || m >= 4096 ||
        (LargeKvTile && m >= 1024 && batch >= 1024)) {
      // Only Q4/Q5 benefit from removing the tail checks on complete tiles.
      // Other formats retain the general kernel's better register schedule.
      if constexpr (Type == gufo::core::GgmlType::kQ4_K ||
                    Type == gufo::core::GgmlType::kQ5_K) {
        if (m % 256 == 0 && batch % 256 == 0) {
          large.template operator()<true>();
          return;
        }
      }
      large.template operator()<false>();
    } else if constexpr (Epilogue != HalfEpilogue::kSwiGLU) {
      if (m >= 1024) {
        HalfPrefillGemmKernel<64, 128, 2, 4, Type, Epilogue>
            <<<dim3((batch + 127) / 128, (m + 63) / 64), 256, 0, stream>>>(
                w, static_cast<const short*>(x), y, batch, m, k, nullptr);
      } else {
        HalfPrefillGemmKernel<32, 64, 2, 2, Type, Epilogue>
            <<<dim3((batch + 63) / 64, (m + 31) / 32), 128, 0, stream>>>(
                w, static_cast<const short*>(x), y, batch, m, k, nullptr);
      }
    }
  };
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      launch.template operator()<GgmlType::kQ4_K>();
      break;
    case GgmlType::kQ5_K:
      launch.template operator()<GgmlType::kQ5_K>();
      break;
    case GgmlType::kQ6_K:
      launch.template operator()<GgmlType::kQ6_K>();
      break;
    case GgmlType::kIQ4_XS:
      launch.template operator()<GgmlType::kIQ4_XS>();
      break;
    case GgmlType::kIQ4_NL:
      launch.template operator()<GgmlType::kIQ4_NL>();
      break;
    case GgmlType::kQ8_0:
      launch.template operator()<GgmlType::kQ8_0>();
      break;
    case GgmlType::kQ3_K:
      launch.template operator()<GgmlType::kQ3_K>();
      break;
    case GgmlType::kIQ3_S:
      launch.template operator()<GgmlType::kIQ3_S>();
      break;
    default:
      throw std::runtime_error("unexpected half prefill weight format");
  }
}
void LaunchBatchedQuantGEMMFp16(gufo::core::GgmlType type, const void* w,
                                const void* x, float* y, std::size_t batch,
                                std::size_t m, std::size_t k,
                                hipStream_t stream) {
  DirectGemm<HalfEpilogue::kStore>(type, w, x, y, batch, m, k, nullptr, stream);
}
void LaunchBatchedQuantGEMMSwiGLUFp16(gufo::core::GgmlType type, const void* w,
                                      const void* x, const float* gate, void* y,
                                      std::size_t batch, std::size_t m,
                                      std::size_t k, hipStream_t stream) {
  DirectGemm<HalfEpilogue::kSwiGLU>(type, w, x, static_cast<float*>(y), batch,
                                    m, k, gate, stream);
}

void LaunchBatchedQuantGEMMResidualFp16(gufo::core::GgmlType type,
                                        const void* w, const void* x,
                                        float* residual, std::size_t batch,
                                        std::size_t m, std::size_t k,
                                        hipStream_t stream) {
  DirectGemm<HalfEpilogue::kResidual>(type, w, x, residual, batch, m, k,
                                      nullptr, stream);
}

bool TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
    gufo::core::GgmlType type, const void* gate_weights, const void* up_weights,
    const void* x, void* y, std::size_t batch, std::size_t m, std::size_t k,
    hipStream_t stream) {
  const auto launch = [&]<gufo::core::GgmlType Type>() {
    const auto tile = [&]<bool Full>() {
      HalfPrefillGemmKernel<256, 256, 8, 4, Type, HalfEpilogue::kGateUp, Full>
          <<<dim3((batch + 255) / 256, (m + 127) / 128), 1024, 0, stream>>>(
              gate_weights, static_cast<const short*>(x),
              static_cast<float*>(y), batch, m, k, nullptr, up_weights);
    };
    if constexpr (Type == gufo::core::GgmlType::kQ4_K ||
                  Type == gufo::core::GgmlType::kQ5_K) {
      if (m % 128 == 0 && batch % 256 == 0) {
        tile.template operator()<true>();
        return;
      }
    }
    tile.template operator()<false>();
  };
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      launch.template operator()<GgmlType::kQ4_K>();
      return true;
    case GgmlType::kQ5_K:
      launch.template operator()<GgmlType::kQ5_K>();
      return true;
    case GgmlType::kIQ4_XS:
      launch.template operator()<GgmlType::kIQ4_XS>();
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::hip
#endif
