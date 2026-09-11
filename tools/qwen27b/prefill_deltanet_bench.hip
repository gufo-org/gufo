// Standalone microbenchmark + ablation harness for the Qwen3.8-27B batched
// DeltaNet (gated delta rule) prefill recurrence.
//
// Self-contained (no repository headers) so it compiles directly with hipcc on
// the supported gfx1151 host for fast kernel-design iteration. Winning variants
// are ported into src/models/qwen/hip/kernels/ssm_recurrence.hip and validated
// through `nix build` plus the CTest oracles.
//
// Build inside Nix with tools/bench/build.sh and this source path.
// Run: /tmp/prefill_deltanet_bench -b 2048 -i 3
//
// Shapes are the production Qwen3.8-27B linear-attention layer: 48 value heads
// over 16 key heads, key_dim = val_dim = 128, so the carried state is
// 48 * 128 * 128 floats = 3 MiB per layer.
//
// The recurrence per token t and head h, with S the [val_dim][key_dim] state:
//
//   S' = alpha_t * S
//   u  = S' k_t,  p = S' q_t          (k, q L2-normalized; q also 1/sqrt(dk))
//   d  = (v_t - u) * beta_t
//   o  = p + d * (k_t . q_t)          equals S_t q_t
//   S  = S' + d k_t^T
//
// Roofline: the state is touched 4 times per element per token (scale, two dot
// accumulations, update), so 48 * 16384 * 4 = 3.15 M lane-ops per token. At
// 2560 FP32 lanes/cycle that is a 1229-cycle/token floor, or 1.19 TFLOP per
// 2048-token layer pass. There is no matrix-core path worth taking: the
// chunkwise-parallel form needs 1.2x (chunk 16) to 1.9x (chunk 64) more MACs,
// which cancels the 2x BF16 WMMA rate.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    hipError_t err_ = (expr);                                           \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

constexpr int kNumHeads = 48;
constexpr int kNumKeyHeads = 16;
constexpr int kKeyDim = 128;
constexpr int kValDim = 128;
constexpr int kQkvSize = (2 * kNumKeyHeads * kKeyDim) + (kNumHeads * kValDim);
constexpr int kInnerSize = kNumHeads * kValDim;
constexpr int kStatePerHead = kKeyDim * kValDim;

// ---------------------------------------------------------------------------
// Baseline: verbatim copy of the production kernel
// (src/models/qwen/hip/kernels/ssm_recurrence.hip). Two lanes own one state
// row; every token runs four block-wide barriers and two single-thread
// cross-warp reductions.
// ---------------------------------------------------------------------------

__launch_bounds__(256) __global__ void BaselineDeltaNet(
    const float* __restrict__ conv_out, float* __restrict__ deltanet_state,
    const float* __restrict__ alpha_buf, const float* __restrict__ beta_buf,
    const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
    float* __restrict__ raw_out, std::size_t batch_size, std::size_t qkv_size,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::uint32_t kh_idx = h % num_key_heads;

  float* s_matrix = deltanet_state + (static_cast<std::size_t>(h) * key_dim *
                                      static_cast<std::size_t>(val_dim));

  const float dt = ssm_dt[h];
  const float a_val = ssm_a[h];

  __shared__ float s_k[128];
  __shared__ float s_q[128];
  __shared__ float s_warp_k[8];
  __shared__ float s_warp_q[8];
  __shared__ float s_warp_kq[8];
  __shared__ float s_scales[3];

  constexpr std::size_t row_vectors_per_thread = 16;
  constexpr std::size_t row_values_per_thread =
      row_vectors_per_thread * sizeof(float4) / sizeof(float);
  const std::size_t j = tid >> 1U;
  const std::size_t row_segment = tid & 1U;
  const std::size_t row_offset = row_segment * row_values_per_thread;

  float4 s_reg[row_vectors_per_thread];
  if (j < val_dim) {
    const float* s_row = s_matrix + (j * key_dim) + row_offset;
    const auto* s_row_vec = reinterpret_cast<const float4*>(s_row);
#pragma unroll
    for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
      s_reg[vi] = s_row_vec[vi];
    }
  }

  for (std::size_t t = 0; t < batch_size; ++t) {
    const float* conv_t = conv_out + (t * qkv_size);
    const float* q_h = conv_t + (kh_idx * key_dim);
    const float* k_h = conv_t + (num_key_heads * key_dim) + (kh_idx * key_dim);
    const float* v_h = conv_t + (2 * num_key_heads * key_dim) + (h * val_dim);
    const std::size_t inner_size =
        static_cast<std::size_t>(num_heads) * val_dim;
    float* o_h = raw_out + (t * inner_size) + (h * val_dim);

    const float alpha_in = (alpha_buf + (t * num_heads))[h];
    const float beta_in = (beta_buf + (t * num_heads))[h];

    const float alpha_biased = alpha_in + dt;
    const float alpha_softplus =
        (alpha_biased > 20.0F) ? alpha_biased : log1pf(expf(alpha_biased));
    const float alpha_h = expf(alpha_softplus * a_val);
    const float beta_h = 1.0F / (1.0F + expf(-beta_in));

    const float raw_k = (tid < key_dim) ? k_h[tid] : 0.0F;
    const float raw_q = (tid < key_dim) ? q_h[tid] : 0.0F;
    float k_sq = raw_k * raw_k;
    float q_sq = raw_q * raw_q;

    for (int offset = 16; offset > 0; offset >>= 1) {
      k_sq += __shfl_xor(k_sq, offset);
      q_sq += __shfl_xor(q_sq, offset);
    }
    if (lane_id == 0) {
      s_warp_k[warp_id] = k_sq;
      s_warp_q[warp_id] = q_sq;
    }
    __syncthreads();

    if (tid == 0) {
      float total_k_sq = 0.0F;
      float total_q_sq = 0.0F;
#pragma unroll
      for (std::size_t warp = 0; warp < 8; ++warp) {
        total_k_sq += s_warp_k[warp];
        total_q_sq += s_warp_q[warp];
      }
      s_scales[0] = 1.0F / sqrtf(total_k_sq + 1e-6F);
      s_scales[1] = (1.0F / sqrtf(static_cast<float>(key_dim))) *
                    (1.0F / sqrtf(total_q_sq + 1e-6F));
    }
    __syncthreads();

    const float inv_k_norm = s_scales[0];
    const float q_scale = s_scales[1];
    const float norm_k = raw_k * inv_k_norm;
    const float norm_q = raw_q * q_scale;

    if (tid < 128) {
      s_k[tid] = norm_k;
      s_q[tid] = norm_q;
    }

    float local_kq = norm_k * norm_q;
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_kq += __shfl_xor(local_kq, offset);
    }
    if (lane_id == 0) {
      s_warp_kq[warp_id] = local_kq;
    }
    __syncthreads();

    if (tid == 0) {
      float total_kq = 0.0F;
#pragma unroll
      for (std::size_t warp = 0; warp < 8; ++warp) {
        total_kq += s_warp_kq[warp];
      }
      s_scales[2] = total_kq;
    }
    __syncthreads();
    const float kq_dot = s_scales[2];

    if (j < val_dim) {
      const auto* k_vec = reinterpret_cast<const float4*>(s_k) +
                          (row_segment * row_vectors_per_thread);
      const auto* q_vec = reinterpret_cast<const float4*>(s_q) +
                          (row_segment * row_vectors_per_thread);

      float u_j = 0.0F;
      float p_j = 0.0F;

#pragma unroll
      for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
        float4 s4 = s_reg[vi];
        s4.x *= alpha_h;
        s4.y *= alpha_h;
        s4.z *= alpha_h;
        s4.w *= alpha_h;
        s_reg[vi] = s4;

        const float4 kv = k_vec[vi];
        const float4 qv = q_vec[vi];

        u_j += (s4.x * kv.x) + (s4.y * kv.y) + (s4.z * kv.z) + (s4.w * kv.w);
        p_j += (s4.x * qv.x) + (s4.y * qv.y) + (s4.z * qv.z) + (s4.w * qv.w);
      }

      u_j += __shfl_xor(u_j, 1);
      p_j += __shfl_xor(p_j, 1);
      const float d_j = (v_h[j] - u_j) * beta_h;
      if (row_segment == 0) {
        o_h[j] = p_j + (d_j * kq_dot);
      }

#pragma unroll
      for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
        float4 s4 = s_reg[vi];
        const float4 kv = k_vec[vi];
        s4.x += (d_j * kv.x);
        s4.y += (d_j * kv.y);
        s4.z += (d_j * kv.z);
        s4.w += (d_j * kv.w);
        s_reg[vi] = s4;
      }
    }
  }

  if (j < val_dim) {
    float* s_row = s_matrix + (j * key_dim) + row_offset;
    auto* s_row_vec = reinterpret_cast<float4*>(s_row);
#pragma unroll
    for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
      s_row_vec[vi] = s_reg[vi];
    }
  }
}

// ---------------------------------------------------------------------------
// Shared prologue for the row-split variants: everything in the token loop that
// is uniform across state rows, hoisted out of the recurrence.
//
//   per (t, key head): inv_k_norm, q_scale, raw_kq  (unnormalized k . q)
//   per (t, value head): alpha_h, beta_h
//
// The recurrence then works on the raw k and q and applies the scalars to the
// reduced dot products instead of to the 128-wide vectors, which is why
// raw_kq rather than the normalized dot is enough:
//
//   u = inv_k_norm * (S' . k_raw),  p = q_scale * (S' . q_raw)
//   k . q = inv_k_norm * q_scale * (k_raw . q_raw)
//   S += (d * inv_k_norm) * k_raw
//
// In production this folds into the conv1d kernel, which already streams the
// same values; here it is separate and included in the reported time.
// ---------------------------------------------------------------------------

// One wave per (token, key head): 128 keys are exactly four values per lane, so
// the whole reduction is a single 5-step butterfly with no LDS and no barrier.
__launch_bounds__(32) __global__
    void PrepKqScalesKernel(const float* __restrict__ conv_out,
                            float* __restrict__ kq_scales,
                            std::size_t batch_size, std::size_t qkv_size,
                            std::uint32_t num_key_heads,
                            std::uint32_t key_dim) {
  const std::uint32_t kh = blockIdx.x;
  const std::size_t t = blockIdx.y;
  const int lane = static_cast<int>(threadIdx.x);

  const float* conv_t = conv_out + (t * qkv_size);
  const auto* qp = reinterpret_cast<const float4*>(
      conv_t + (static_cast<std::size_t>(kh) * key_dim));
  const auto* kp = reinterpret_cast<const float4*>(
      conv_t + (static_cast<std::size_t>(num_key_heads) * key_dim) +
      (static_cast<std::size_t>(kh) * key_dim));
  const float4 q4 = qp[lane];
  const float4 k4 = kp[lane];

  float k_sq = (k4.x * k4.x) + (k4.y * k4.y) + (k4.z * k4.z) + (k4.w * k4.w);
  float q_sq = (q4.x * q4.x) + (q4.y * q4.y) + (q4.z * q4.z) + (q4.w * q4.w);
  float kq = (k4.x * q4.x) + (k4.y * q4.y) + (k4.z * q4.z) + (k4.w * q4.w);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    k_sq += __shfl_xor(k_sq, offset);
    q_sq += __shfl_xor(q_sq, offset);
    kq += __shfl_xor(kq, offset);
  }
  if (lane == 0) {
    float* out = kq_scales + (((t * num_key_heads) + kh) * 3);
    out[0] = 1.0F / sqrtf(k_sq + 1e-6F);
    out[1] = (1.0F / sqrtf(static_cast<float>(key_dim))) *
             (1.0F / sqrtf(q_sq + 1e-6F));
    out[2] = kq;
  }
}

__launch_bounds__(256) __global__ void PrepAlphaBetaKernel(
    const float* __restrict__ alpha_buf, const float* __restrict__ beta_buf,
    const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
    float* __restrict__ ab, std::size_t count, std::uint32_t num_heads) {
  const std::size_t i =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::uint32_t h = static_cast<std::uint32_t>(i % num_heads);
  const float alpha_biased = alpha_buf[i] + ssm_dt[h];
  const float alpha_softplus =
      (alpha_biased > 20.0F) ? alpha_biased : log1pf(expf(alpha_biased));
  ab[i * 2] = expf(alpha_softplus * ssm_a[h]);
  ab[(i * 2) + 1] = 1.0F / (1.0F + expf(-beta_buf[i]));
}

// ---------------------------------------------------------------------------
// Row-split recurrence.
//
// State rows are independent given (k, q, k.q, alpha, beta, v), so the 128 rows
// of a head can be spread over several blocks instead of being pinned to one.
// That is the whole point: the baseline runs 48 blocks of 8 waves = 384 waves
// on 80 SIMDs, and every one of them stalls on four block-wide barriers per
// token. Here `kKeysPerLane` lanes cooperate on one row, no lane needs a value
// another lane owns, and the kernel has no barrier at all.
//
//   kKeysPerLane  keys of the 128-wide row held per lane
//   kRowsPerLane  independent rows held per lane (amortizes the k/q loads)
//
// State registers per lane = kKeysPerLane * kRowsPerLane, and total resident
// waves = 48 * 16384 / (32 * that), so shrinking the per-lane tile is what buys
// latency hiding. The k and q loads for the next token are issued before the
// current token's arithmetic so the L2 round trip overlaps compute.
// ---------------------------------------------------------------------------

template<int kKeysPerLane, int kRowsPerLane, bool kPrefetch>
__launch_bounds__(256) __global__
    void RowSplitDeltaNet(const float* __restrict__ conv_out,
                          const float* __restrict__ kq_scales,
                          const float* __restrict__ ab,
                          float* __restrict__ deltanet_state,
                          float* __restrict__ raw_out, std::size_t batch_size,
                          std::size_t qkv_size, std::uint32_t num_key_heads,
                          std::uint32_t num_heads) {
  constexpr int kLanesPerRow = kKeyDim / kKeysPerLane;
  constexpr int kRowGroups = 32 / kLanesPerRow;
  constexpr int kRowsPerWave = kRowGroups * kRowsPerLane;
  constexpr int kRowsPerBlock = kRowsPerWave * 8;
  constexpr int kVec = kKeysPerLane / 4;
  static_assert(kKeysPerLane % 4 == 0, "float4 loads");
  static_assert(kLanesPerRow <= 32, "a row group must fit in one wave");

  const std::uint32_t h = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int seg = lane % kLanesPerRow;
  const int grp = lane / kLanesPerRow;

  const int row0 = (static_cast<int>(blockIdx.x) * kRowsPerBlock) +
                   (wave * kRowsPerWave) + grp;
  const std::uint32_t kh = h % num_key_heads;
  const int key0 = seg * kKeysPerLane;

  // State tile: rows row0 + r * kRowGroups, keys [key0, key0 + kKeysPerLane).
  float4 s_reg[kRowsPerLane][kVec];
  float* s_head =
      deltanet_state + (static_cast<std::size_t>(h) * kStatePerHead);
#pragma unroll
  for (int r = 0; r < kRowsPerLane; ++r) {
    const auto* src = reinterpret_cast<const float4*>(
        s_head + (static_cast<std::size_t>(row0 + (r * kRowGroups)) * kKeyDim) +
        key0);
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      s_reg[r][vi] = src[vi];
    }
  }

  const std::size_t q_off = (static_cast<std::size_t>(kh) * kKeyDim) + key0;
  const std::size_t k_off =
      (static_cast<std::size_t>(num_key_heads) * kKeyDim) + q_off;
  const std::size_t v_off =
      (static_cast<std::size_t>(2 * num_key_heads) * kKeyDim) +
      (static_cast<std::size_t>(h) * kValDim);

  float4 k_next[kVec];
  float4 q_next[kVec];
  const auto load_kq = [&](std::size_t t, float4* kd, float4* qd) {
    const float* conv_t = conv_out + (t * qkv_size);
    const auto* kp = reinterpret_cast<const float4*>(conv_t + k_off);
    const auto* qp = reinterpret_cast<const float4*>(conv_t + q_off);
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      kd[vi] = kp[vi];
      qd[vi] = qp[vi];
    }
  };
  if (kPrefetch) {
    load_kq(0, k_next, q_next);
  }

  for (std::size_t t = 0; t < batch_size; ++t) {
    float4 kv[kVec];
    float4 qv[kVec];
    if (kPrefetch) {
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        kv[vi] = k_next[vi];
        qv[vi] = q_next[vi];
      }
      if (t + 1 < batch_size) {
        load_kq(t + 1, k_next, q_next);
      }
    } else {
      load_kq(t, kv, qv);
    }

    const float* scales = kq_scales + ((((t * num_key_heads) + kh)) * 3);
    const float inv_k_norm = scales[0];
    const float q_scale = scales[1];
    const float kq_dot = inv_k_norm * q_scale * scales[2];
    const float* abp = ab + ((((t * num_heads) + h)) * 2);
    const float alpha_h = abp[0];
    const float beta_h = abp[1];

    const float* v_h = conv_out + (t * qkv_size) + v_off;
    float* o_h =
        raw_out + (t * kInnerSize) + (static_cast<std::size_t>(h) * kValDim);

#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
      float u = 0.0F;
      float p = 0.0F;
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        float4 s4 = s_reg[r][vi];
        s4.x *= alpha_h;
        s4.y *= alpha_h;
        s4.z *= alpha_h;
        s4.w *= alpha_h;
        s_reg[r][vi] = s4;
        const float4 kk = kv[vi];
        const float4 qq = qv[vi];
        u += (s4.x * kk.x) + (s4.y * kk.y) + (s4.z * kk.z) + (s4.w * kk.w);
        p += (s4.x * qq.x) + (s4.y * qq.y) + (s4.z * qq.z) + (s4.w * qq.w);
      }
#pragma unroll
      for (int offset = 1; offset < kLanesPerRow; offset <<= 1) {
        u += __shfl_xor(u, offset);
        p += __shfl_xor(p, offset);
      }
      const int row = row0 + (r * kRowGroups);
      const float d = ((v_h[row] - (u * inv_k_norm)) * beta_h);
      if (seg == 0) {
        o_h[row] = (p * q_scale) + (d * kq_dot);
      }
      const float dk = d * inv_k_norm;
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        float4 s4 = s_reg[r][vi];
        const float4 kk = kv[vi];
        s4.x += dk * kk.x;
        s4.y += dk * kk.y;
        s4.z += dk * kk.z;
        s4.w += dk * kk.w;
        s_reg[r][vi] = s4;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerLane; ++r) {
    auto* dst = reinterpret_cast<float4*>(
        s_head + (static_cast<std::size_t>(row0 + (r * kRowGroups)) * kKeyDim) +
        key0);
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      dst[vi] = s_reg[r][vi];
    }
  }
}

// ---------------------------------------------------------------------------
// Row-split v2: same arithmetic, addressed so the token loop carries no 64-bit
// pointer math.
//
// v1 recomputes `conv_out + t * qkv_size + <lane-dependent offset>` every
// token, which the compiler lowers to 32 s_add_u32/s_addc_u32 plus 30
// v_add_co_u32 pairs -- about 60 of roughly 200 instructions per token spent on
// addresses. Here every base pointer is block-uniform (so it lives in SGPRs and
// advances with one scalar add) and every lane-dependent part is a
// loop-invariant 32-bit element index, which is exactly the SGPR-base +
// VGPR-offset form of global_load_b128.
//
// kLdsKq additionally stages k and q through LDS. All eight waves of a block
// share one head, so the block reads 1 KiB per token instead of each lane
// pulling its own slice through L1; the cost is one barrier per token, against
// the four the baseline pays.
// ---------------------------------------------------------------------------

// __shfl_xor lowers to ds_bpermute_b32, which takes the LDS path. For xor masks
// inside a 16-lane DPP row the same exchange is a pure ALU-path DPP modifier.
template<int kMask>
__device__ __forceinline__ float XorAddDpp(float x) {
  const int y = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x),
                                            0x160 | kMask, 0xF, 0xF, false);
  return x + __builtin_bit_cast(float, y);
}

template<int kLanesPerRow, bool kDpp>
__device__ __forceinline__ void RowReduce2(float& u, float& p) {
  if (kDpp) {
    if (kLanesPerRow > 1) {
      u = XorAddDpp<1>(u);
      p = XorAddDpp<1>(p);
    }
    if (kLanesPerRow > 2) {
      u = XorAddDpp<2>(u);
      p = XorAddDpp<2>(p);
    }
    if (kLanesPerRow > 4) {
      u = XorAddDpp<4>(u);
      p = XorAddDpp<4>(p);
    }
    if (kLanesPerRow > 8) {
      u = XorAddDpp<8>(u);
      p = XorAddDpp<8>(p);
    }
  } else {
#pragma unroll
    for (int offset = 1; offset < kLanesPerRow; offset <<= 1) {
      u += __shfl_xor(u, offset);
      p += __shfl_xor(p, offset);
    }
  }
}

template<int kKeysPerLane, int kRowsPerLane, bool kLdsKq, bool kPrefetch,
         bool kDpp>
__launch_bounds__(256) __global__
    void RowSplitDeltaNetV2(const float* __restrict__ conv_out,
                            const float* __restrict__ kq_scales,
                            const float* __restrict__ ab,
                            float* __restrict__ deltanet_state,
                            float* __restrict__ raw_out, int batch_size,
                            int num_key_heads, int num_heads) {
  constexpr int kLanesPerRow = kKeyDim / kKeysPerLane;
  constexpr int kRowGroups = 32 / kLanesPerRow;
  constexpr int kRowsPerWave = kRowGroups * kRowsPerLane;
  constexpr int kRowsPerBlock = kRowsPerWave * 8;
  constexpr int kVec = kKeysPerLane / 4;

  const int h = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int seg = lane % kLanesPerRow;
  const int grp = lane / kLanesPerRow;
  const int row0 = (static_cast<int>(blockIdx.x) * kRowsPerBlock) +
                   (wave * kRowsPerWave) + grp;
  const int kh = h % num_key_heads;

  // Only the LDS variant needs the staging buffer; sizing it away keeps the
  // direct variant's occupancy free of a phantom 2 KiB allocation.
  constexpr int kLdsFloats = kLdsKq ? (2 * 2 * kKeyDim) : 4;
  __shared__ __align__(16) float s_kq_store[kLdsFloats];

  float4 s_reg[kRowsPerLane][kVec];
  {
    float* s_head =
        deltanet_state + (static_cast<std::size_t>(h) * kStatePerHead);
#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
      const auto* src = reinterpret_cast<const float4*>(
          s_head +
          (static_cast<std::size_t>(row0 + (r * kRowGroups)) * kKeyDim));
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        s_reg[r][vi] = src[(seg * kVec) + vi];
      }
    }
  }

  // Block-uniform bases. Only these advance in the loop, by one scalar add.
  const float* q_base = conv_out + (kh * kKeyDim);
  const float* k_base = conv_out + ((num_key_heads + kh) * kKeyDim);
  const float* v_base =
      conv_out + ((2 * num_key_heads * kKeyDim) + (h * kValDim));
  const float* kq_base = kq_scales + (kh * 3);
  const float* ab_base = ab + (h * 2);
  float* o_base = raw_out + (h * kValDim);

  // Loop-invariant, lane-dependent element indices.
  const int kq_idx = seg * kVec;
  const int lds_idx = seg * kKeysPerLane;

  const auto stage = [&](const float* q_row, const float* k_row, int buf) {
    // 256 threads cover 2 * 128 floats as 64 float4 loads.
    if (tid < 64) {
      const auto* src =
          reinterpret_cast<const float4*>((tid < 32) ? k_row : q_row);
      reinterpret_cast<float4*>(s_kq_store + (buf * 2 * kKeyDim))[tid] =
          src[tid & 31];
    }
  };

  float4 k_next[kVec];
  float4 q_next[kVec];
  if (kLdsKq) {
    stage(q_base, k_base, 0);
    __syncthreads();
  } else if (kPrefetch) {
    const auto* kp = reinterpret_cast<const float4*>(k_base);
    const auto* qp = reinterpret_cast<const float4*>(q_base);
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      k_next[vi] = kp[kq_idx + vi];
      q_next[vi] = qp[kq_idx + vi];
    }
  }

  for (int t = 0; t < batch_size; ++t) {
    float4 kv[kVec];
    float4 qv[kVec];
    if (kLdsKq) {
      const float* ks = s_kq_store + ((t & 1) * 2 * kKeyDim);
      const float* qs = ks + kKeyDim;
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        kv[vi] = reinterpret_cast<const float4*>(ks + lds_idx)[vi];
        qv[vi] = reinterpret_cast<const float4*>(qs + lds_idx)[vi];
      }
      if (t + 1 < batch_size) {
        stage(q_base + kQkvSize, k_base + kQkvSize, (t + 1) & 1);
      }
    } else if (kPrefetch) {
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        kv[vi] = k_next[vi];
        qv[vi] = q_next[vi];
      }
      if (t + 1 < batch_size) {
        const auto* kp = reinterpret_cast<const float4*>(k_base + kQkvSize);
        const auto* qp = reinterpret_cast<const float4*>(q_base + kQkvSize);
#pragma unroll
        for (int vi = 0; vi < kVec; ++vi) {
          k_next[vi] = kp[kq_idx + vi];
          q_next[vi] = qp[kq_idx + vi];
        }
      }
    } else {
      const auto* kp = reinterpret_cast<const float4*>(k_base);
      const auto* qp = reinterpret_cast<const float4*>(q_base);
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        kv[vi] = kp[kq_idx + vi];
        qv[vi] = qp[kq_idx + vi];
      }
    }

    const float inv_k_norm = kq_base[0];
    const float q_scale = kq_base[1];
    const float kq_dot = inv_k_norm * q_scale * kq_base[2];
    const float alpha_h = ab_base[0];
    const float beta_h = ab_base[1];

#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
      float u = 0.0F;
      float p = 0.0F;
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        float4 s4 = s_reg[r][vi];
        s4.x *= alpha_h;
        s4.y *= alpha_h;
        s4.z *= alpha_h;
        s4.w *= alpha_h;
        s_reg[r][vi] = s4;
        const float4 kk = kv[vi];
        const float4 qq = qv[vi];
        u += (s4.x * kk.x) + (s4.y * kk.y) + (s4.z * kk.z) + (s4.w * kk.w);
        p += (s4.x * qq.x) + (s4.y * qq.y) + (s4.z * qq.z) + (s4.w * qq.w);
      }
      RowReduce2<kLanesPerRow, kDpp>(u, p);
      const int row = row0 + (r * kRowGroups);
      const float d = (v_base[row] - (u * inv_k_norm)) * beta_h;
      if (seg == 0) {
        o_base[row] = (p * q_scale) + (d * kq_dot);
      }
      const float dk = d * inv_k_norm;
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        float4 s4 = s_reg[r][vi];
        const float4 kk = kv[vi];
        s4.x += dk * kk.x;
        s4.y += dk * kk.y;
        s4.z += dk * kk.z;
        s4.w += dk * kk.w;
        s_reg[r][vi] = s4;
      }
    }

    q_base += kQkvSize;
    k_base += kQkvSize;
    v_base += kQkvSize;
    kq_base += num_key_heads * 3;
    ab_base += num_heads * 2;
    o_base += kInnerSize;
    if (kLdsKq) {
      __syncthreads();
    }
  }

  {
    float* s_head =
        deltanet_state + (static_cast<std::size_t>(h) * kStatePerHead);
#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
      auto* dst = reinterpret_cast<float4*>(
          s_head +
          (static_cast<std::size_t>(row0 + (r * kRowGroups)) * kKeyDim));
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        dst[(seg * kVec) + vi] = s_reg[r][vi];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

template<typename Fn>
float TimeMs(int iters, Fn&& fn) {
  fn();
  HIP_OK(hipDeviceSynchronize());
  hipEvent_t a = nullptr;
  hipEvent_t b = nullptr;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  HIP_OK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  HIP_OK(hipEventRecord(b));
  HIP_OK(hipEventSynchronize(b));
  float ms = 0.0F;
  HIP_OK(hipEventElapsedTime(&ms, a, b));
  HIP_OK(hipEventDestroy(a));
  HIP_OK(hipEventDestroy(b));
  return ms / static_cast<float>(iters);
}

int main(int argc, char** argv) {
  std::size_t batch = 2048;
  int iters = 3;
  std::string only;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-b" && i + 1 < argc) {
      batch = std::strtoull(argv[++i], nullptr, 10);
    } else if (arg == "-i" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (arg == "-v" && i + 1 < argc) {
      only = argv[++i];
    }
  }

  std::vector<float> h_conv(batch * kQkvSize);
  for (std::size_t i = 0; i < h_conv.size(); ++i) {
    h_conv[i] = 0.6F * std::sin(0.017F * static_cast<float>(i % 997)) +
                0.2F * std::cos(0.003F * static_cast<float>(i % 31));
  }
  std::vector<float> h_alpha(batch * kNumHeads);
  std::vector<float> h_beta(batch * kNumHeads);
  for (std::size_t i = 0; i < h_alpha.size(); ++i) {
    h_alpha[i] = 1.5F + 0.5F * std::sin(0.011F * static_cast<float>(i % 401));
    h_beta[i] = 0.3F * std::cos(0.007F * static_cast<float>(i % 211));
  }
  std::vector<float> h_a(kNumHeads);
  std::vector<float> h_dt(kNumHeads);
  for (int i = 0; i < kNumHeads; ++i) {
    h_a[i] = -0.04F - 0.01F * static_cast<float>(i % 5);
    h_dt[i] = 0.1F * static_cast<float>(i % 3);
  }
  std::vector<float> h_state0(static_cast<std::size_t>(kNumHeads) *
                              kStatePerHead);
  for (std::size_t i = 0; i < h_state0.size(); ++i) {
    h_state0[i] = 0.01F * std::sin(0.013F * static_cast<float>(i % 577));
  }

  float* d_conv = nullptr;
  float* d_alpha = nullptr;
  float* d_beta = nullptr;
  float* d_a = nullptr;
  float* d_dt = nullptr;
  float* d_state = nullptr;
  float* d_out = nullptr;
  float* d_ref = nullptr;
  float* d_ref_state = nullptr;
  float* d_kq = nullptr;
  float* d_ab = nullptr;
  const std::size_t state_elems =
      static_cast<std::size_t>(kNumHeads) * kStatePerHead;
  HIP_OK(hipMalloc(&d_conv, h_conv.size() * sizeof(float)));
  HIP_OK(hipMalloc(&d_alpha, h_alpha.size() * sizeof(float)));
  HIP_OK(hipMalloc(&d_beta, h_beta.size() * sizeof(float)));
  HIP_OK(hipMalloc(&d_a, h_a.size() * sizeof(float)));
  HIP_OK(hipMalloc(&d_dt, h_dt.size() * sizeof(float)));
  HIP_OK(hipMalloc(&d_state, state_elems * sizeof(float)));
  HIP_OK(hipMalloc(&d_ref_state, state_elems * sizeof(float)));
  HIP_OK(hipMalloc(&d_out, batch * kInnerSize * sizeof(float)));
  HIP_OK(hipMalloc(&d_ref, batch * kInnerSize * sizeof(float)));
  HIP_OK(hipMalloc(&d_kq, batch * kNumKeyHeads * 3 * sizeof(float)));
  HIP_OK(hipMalloc(&d_ab, batch * kNumHeads * 2 * sizeof(float)));
  HIP_OK(hipMemcpy(d_conv, h_conv.data(), h_conv.size() * sizeof(float),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_alpha, h_alpha.data(), h_alpha.size() * sizeof(float),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(float),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_dt, h_dt.data(), h_dt.size() * sizeof(float),
                   hipMemcpyHostToDevice));

  float* d_state_init = nullptr;
  HIP_OK(hipMalloc(&d_state_init, state_elems * sizeof(float)));
  HIP_OK(hipMemcpy(d_state_init, h_state0.data(), state_elems * sizeof(float),
                   hipMemcpyHostToDevice));
  // Device-to-device so the reset never appears inside a timed region.
  const auto reset_state = [&](float* dst) {
    HIP_OK(hipMemcpy(dst, d_state_init, state_elems * sizeof(float),
                     hipMemcpyDeviceToDevice));
  };

  // Reference run.
  reset_state(d_ref_state);
  BaselineDeltaNet<<<kNumHeads, 256>>>(
      d_conv, d_ref_state, d_alpha, d_beta, d_a, d_dt, d_ref, batch, kQkvSize,
      kNumKeyHeads, kNumHeads, kKeyDim, kValDim);
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> ref(batch * kInnerSize);
  std::vector<float> ref_state(state_elems);
  HIP_OK(hipMemcpy(ref.data(), d_ref, ref.size() * sizeof(float),
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(ref_state.data(), d_ref_state,
                   ref_state.size() * sizeof(float), hipMemcpyDeviceToHost));

  double ref_mag = 0.0;
  for (float x : ref) {
    ref_mag = std::fmax(ref_mag, std::fabs(static_cast<double>(x)));
  }

  const double lane_ops = static_cast<double>(kNumHeads) * kStatePerHead * 4.0 *
                          static_cast<double>(batch);
  std::printf("batch=%zu iters=%d  heads=%d key_heads=%d dk=%d dv=%d\n", batch,
              iters, kNumHeads, kNumKeyHeads, kKeyDim, kValDim);
  std::printf(
      "state=%.1f MiB  core lane-ops=%.2f G  (fp32 VALU floor at 2560 "
      "lanes/cyc, 2.64 GHz = %.2f ms)\n",
      static_cast<double>(state_elems) * 4.0 / 1048576.0, lane_ops / 1e9,
      lane_ops / 2560.0 / 2.64e6);
  std::printf("%-30s %10s %9s %10s %9s\n", "variant", "ms", "cyc/tok", "maxrel",
              "state_rel");

  std::vector<float> got(batch * kInnerSize);
  std::vector<float> got_state(state_elems);
  const auto check = [&](const char* name, float ms) {
    HIP_OK(hipMemcpy(got.data(), d_out, got.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(got_state.data(), d_state,
                     got_state.size() * sizeof(float), hipMemcpyDeviceToHost));
    double maxabs = 0.0;
    bool bad = false;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      if (!std::isfinite(got[i])) {
        bad = true;
        break;
      }
      maxabs = std::fmax(maxabs, std::fabs(static_cast<double>(got[i]) -
                                           static_cast<double>(ref[i])));
    }
    double smaxabs = 0.0;
    double smag = 0.0;
    for (std::size_t i = 0; i < ref_state.size(); ++i) {
      smaxabs =
          std::fmax(smaxabs, std::fabs(static_cast<double>(got_state[i]) -
                                       static_cast<double>(ref_state[i])));
      smag = std::fmax(smag, std::fabs(static_cast<double>(ref_state[i])));
    }
    const double cyc =
        static_cast<double>(ms) * 1e-3 * 2.64e9 / static_cast<double>(batch);
    if (bad) {
      std::printf("%-30s %10.3f %9.0f %10s %9s\n", name, ms, cyc, "NONFINITE",
                  "-");
      return;
    }
    std::printf("%-30s %10.3f %9.0f %10.2e %9.2e\n", name, ms, cyc,
                maxabs / ref_mag, smaxabs / smag);
  };

  if (only.empty() || only == "baseline") {
    const auto launch = [&] {
      BaselineDeltaNet<<<kNumHeads, 256>>>(
          d_conv, d_state, d_alpha, d_beta, d_a, d_dt, d_out, batch, kQkvSize,
          kNumKeyHeads, kNumHeads, kKeyDim, kValDim);
    };
    // Correctness from a clean state, then time with the reset outside the loop
    // (the state drifts across iterations, which does not change the work).
    reset_state(d_state);
    launch();
    HIP_OK(hipGetLastError());
    HIP_OK(hipDeviceSynchronize());
    std::vector<float> keep_out(batch * kInnerSize);
    std::vector<float> keep_state(state_elems);
    HIP_OK(hipMemcpy(keep_out.data(), d_out, keep_out.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(keep_state.data(), d_state,
                     keep_state.size() * sizeof(float), hipMemcpyDeviceToHost));
    reset_state(d_state);
    const float ms = TimeMs(iters, launch);
    HIP_OK(hipMemcpy(d_out, keep_out.data(), keep_out.size() * sizeof(float),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_state, keep_state.data(),
                     keep_state.size() * sizeof(float), hipMemcpyHostToDevice));
    check("baseline (production)", ms);
  }

  const auto run_prep = [&] {
    const dim3 g(kNumKeyHeads, static_cast<unsigned int>(batch));
    PrepKqScalesKernel<<<g, 32>>>(d_conv, d_kq, batch, kQkvSize, kNumKeyHeads,
                                  kKeyDim);
    const std::size_t n = batch * kNumHeads;
    PrepAlphaBetaKernel<<<static_cast<unsigned int>((n + 255) / 256), 256>>>(
        d_alpha, d_beta, d_a, d_dt, d_ab, n, kNumHeads);
  };

  if (only.empty() || only == "prep") {
    const float ms = TimeMs(iters, run_prep);
    std::printf("%-30s %10.3f %9.0f %10s %9s\n",
                "  prep only (kq + alpha/beta)", ms,
                static_cast<double>(ms) * 1e-3 * 2.64e9 / batch, "-", "-");
  }

  const auto bench_rowsplit = [&](const char* name, auto launch) {
    if (!only.empty() && only != name) {
      return;
    }
    const auto step = [&] {
      run_prep();
      launch();
    };
    reset_state(d_state);
    HIP_OK(hipMemset(d_out, 0, batch * kInnerSize * sizeof(float)));
    step();
    HIP_OK(hipGetLastError());
    HIP_OK(hipDeviceSynchronize());
    std::vector<float> keep_out(batch * kInnerSize);
    std::vector<float> keep_state(state_elems);
    HIP_OK(hipMemcpy(keep_out.data(), d_out, keep_out.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(keep_state.data(), d_state,
                     keep_state.size() * sizeof(float), hipMemcpyDeviceToHost));
    reset_state(d_state);
    const float ms = TimeMs(iters, step);
    HIP_OK(hipMemcpy(d_out, keep_out.data(), keep_out.size() * sizeof(float),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_state, keep_state.data(),
                     keep_state.size() * sizeof(float), hipMemcpyHostToDevice));
    check(name, ms);
  };

#define ROWSPLIT(KPL, RPL, PF, NAME)                                           \
  bench_rowsplit(NAME, [&] {                                                   \
    constexpr int kRowsPerBlock = ((32 / (kKeyDim / (KPL))) * (RPL)) * 8;      \
    static_assert(kValDim % kRowsPerBlock == 0, "row blocks must tile 128");   \
    const dim3 grid(kValDim / kRowsPerBlock, kNumHeads);                       \
    RowSplitDeltaNet<KPL, RPL, PF><<<grid, 256>>>(d_conv, d_kq, d_ab, d_state, \
                                                  d_out, batch, kQkvSize,      \
                                                  kNumKeyHeads, kNumHeads);    \
  })

  ROWSPLIT(16, 1, false, "rowsplit k16 r1");
  ROWSPLIT(16, 1, true, "rowsplit k16 r1 prefetch");
  ROWSPLIT(16, 2, true, "rowsplit k16 r2 prefetch");
  ROWSPLIT(16, 4, true, "rowsplit k16 r4 prefetch");
  ROWSPLIT(32, 1, true, "rowsplit k32 r1 prefetch");
  ROWSPLIT(32, 2, true, "rowsplit k32 r2 prefetch");
  ROWSPLIT(64, 1, true, "rowsplit k64 r1 prefetch");
  ROWSPLIT(8, 2, true, "rowsplit k8 r2 prefetch");
  ROWSPLIT(8, 4, true, "rowsplit k8 r4 prefetch");
#undef ROWSPLIT

#define ROWSPLIT2(KPL, RPL, LDS, PF, DPP, NAME)                              \
  bench_rowsplit(NAME, [&] {                                                 \
    constexpr int kRowsPerBlock = ((32 / (kKeyDim / (KPL))) * (RPL)) * 8;    \
    static_assert(kValDim % kRowsPerBlock == 0, "row blocks must tile 128"); \
    const dim3 grid(kValDim / kRowsPerBlock, kNumHeads);                     \
    RowSplitDeltaNetV2<KPL, RPL, LDS, PF, DPP>                               \
        <<<grid, 256>>>(d_conv, d_kq, d_ab, d_state, d_out,                  \
                        static_cast<int>(batch), kNumKeyHeads, kNumHeads);   \
  })

  ROWSPLIT2(16, 2, false, false, false, "v2 k16 r2");
  ROWSPLIT2(32, 1, false, false, false, "v2 k32 r1");
  ROWSPLIT2(16, 1, false, true, false, "v2 k16 r1 pf");
  ROWSPLIT2(16, 2, false, true, false, "v2 k16 r2 pf");
  ROWSPLIT2(16, 4, false, true, false, "v2 k16 r4 pf");
  ROWSPLIT2(32, 1, false, true, false, "v2 k32 r1 pf");
  ROWSPLIT2(32, 2, false, true, false, "v2 k32 r2 pf");
  ROWSPLIT2(8, 4, false, true, false, "v2 k8 r4 pf");
  ROWSPLIT2(16, 2, false, false, true, "v2 k16 r2 dpp");
  ROWSPLIT2(32, 1, false, false, true, "v2 k32 r1 dpp");
  ROWSPLIT2(16, 2, false, true, true, "v2 k16 r2 pf dpp");
  ROWSPLIT2(16, 4, false, true, true, "v2 k16 r4 pf dpp");
  ROWSPLIT2(32, 1, false, true, true, "v2 k32 r1 pf dpp");
  ROWSPLIT2(32, 2, false, true, true, "v2 k32 r2 pf dpp");
  ROWSPLIT2(8, 4, false, true, true, "v2 k8 r4 pf dpp");
#undef ROWSPLIT2

  return 0;
}
