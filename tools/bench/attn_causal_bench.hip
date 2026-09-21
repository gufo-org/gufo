// Standalone microbenchmark + ablation harness for the Qwen3.8-27B masked
// (causal) prefill attention.
//
// Self-contained (no repository headers) so it compiles directly with hipcc on
// the supported gfx1151 host for fast kernel-design iteration.
//
//   nix develop -c tools/bench/build.sh attn_causal_bench
//   /tmp/attn_causal_bench -b 2048 -i 3
//
// Shapes are the production Qwen3.8-27B full-attention layer: 24 query heads
// over 4 KV heads (GQA 6), head_dim 256, an FP16 K/V cache laid out
// [position][kv_head][dim], FP32 queries and output laid out
// [token][head][dim].
//
// Compare native WMMA tiling against the scalar tiled control. This harness
// measures matrix throughput, memory traffic and independent FP64 error.
#include <hip/hip_fp16.h>
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

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kQueryHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
constexpr std::uint32_t kAttnWidth = kQueryHeads * kHeadDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;

// ---------------------------------------------------------------------------
// Baseline: the production tiled kernel (src/models/qwen/hip/kernels/
// attention_tile.hip), copied verbatim apart from the constants it pulls from
// the policy header.
// ---------------------------------------------------------------------------

constexpr std::uint32_t kTileQueryRows = 16;
constexpr std::uint32_t kTileHeadsPerBlock = 2;
constexpr std::uint32_t kTileColumns = kTileQueryRows * kTileHeadsPerBlock;
constexpr std::uint32_t kTileKeyRows = 64;
constexpr std::uint32_t kTileHalf2Dim = kHeadDim / 2;
constexpr std::uint32_t kTileKeyHalf2Dim = kTileHalf2Dim / 2;
constexpr std::uint32_t kTileValueRows = 32;
constexpr std::uint32_t kTileKeyRowStride = kTileKeyHalf2Dim + 1;

__device__ __forceinline__ float TileWaveReduceMax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ float TileWaveReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down(value, offset);
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ void TileHalf2Dot(float& accumulator, __half2 lhs,
                                             __half2 rhs) {
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0"
               : "+v"(accumulator)
               : "v"(lhs), "v"(rhs));
}

__launch_bounds__(256, 3) __global__
    void TiledAttention(const float* __restrict__ q,
                        const float* __restrict__ gate,
                        const __half* __restrict__ k_cache_f16,
                        const __half* __restrict__ v_cache_f16,
                        float* __restrict__ out_context,
                        std::uint32_t start_pos, std::size_t batch_size,
                        float* __restrict__ lse_out, std::uint32_t key_begin) {
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t wave = threadIdx.y;
  const std::uint32_t query_start = blockIdx.x * kTileQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kGqa / 2);
  const std::uint32_t pair_in_group = head_pair % (kGqa / 2);
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kTileHeadsPerBlock);
  constexpr std::size_t kv_half2_stride = kKvWidth / 2;
  constexpr float attention_scale = 1.0F / 16.0F;

  __shared__ __half2 q_tile[kTileColumns][kTileHalf2Dim];
  __shared__ __half2 kv_tile[kTileKeyRows * kTileKeyRowStride];
  __shared__ __half weights[kTileColumns][kTileKeyRows];

  constexpr std::uint32_t columns_per_wave = kTileColumns / 8;
  float running_max[columns_per_wave];
  float running_sum[columns_per_wave];
  float2 output[columns_per_wave][kTileHalf2Dim / 32];
#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    running_max[column] = -INFINITY;
    running_sum[column] = 0.0F;
#pragma unroll
    for (std::uint32_t slot = 0; slot < kTileHalf2Dim / 32; ++slot) {
      output[column][slot] = make_float2(0.0F, 0.0F);
    }
  }

#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
    const std::uint32_t query_offset = tiled_column / kTileHeadsPerBlock;
    const std::uint32_t query_head =
        first_query_head + (tiled_column % kTileHeadsPerBlock);
    const std::uint32_t local_query_position = query_start + query_offset;
#pragma unroll
    for (std::uint32_t half2_index = lane; half2_index < kTileHalf2Dim;
         half2_index += 32) {
      __half2 packed =
          __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
      if (local_query_position < batch_size) {
        const std::size_t source =
            (static_cast<std::size_t>(local_query_position) * kAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (2 * half2_index);
        packed =
            __halves2half2(__float2half_rn(q[source] * attention_scale),
                           __float2half_rn(q[source + 1] * attention_scale));
      }
      q_tile[tiled_column][half2_index] = packed;
    }
  }
  __syncthreads();

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kTileQueryRows);
  for (std::uint32_t key_start = key_begin; key_start < max_visible;
       key_start += kTileKeyRows) {
    float scores[columns_per_wave][2] = {};
#pragma unroll
    for (std::uint32_t key_dimension = 0; key_dimension < 2; ++key_dimension) {
      for (std::uint32_t key_row = wave; key_row < kTileKeyRows; key_row += 8) {
        const std::uint32_t key_position = key_start + key_row;
#pragma unroll
        for (std::uint32_t half2_index = lane; half2_index < kTileKeyHalf2Dim;
             half2_index += 32) {
          __half2 packed =
              __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
          if (key_position < context_end) {
            const auto* layer_key =
                reinterpret_cast<const __half2*>(k_cache_f16);
            packed =
                layer_key[(static_cast<std::size_t>(key_position) *
                           kv_half2_stride) +
                          (static_cast<std::size_t>(kv_head) * kTileHalf2Dim) +
                          (key_dimension * kTileKeyHalf2Dim) + half2_index];
          }
          kv_tile[(key_row * kTileKeyRowStride) + half2_index] = packed;
        }
      }
      __syncthreads();

#pragma unroll 4
      for (std::uint32_t half2_index = 0; half2_index < kTileKeyHalf2Dim;
           ++half2_index) {
        const __half2 key_values[2] = {
            kv_tile[(lane * kTileKeyRowStride) + half2_index],
            kv_tile[((lane + 32) * kTileKeyRowStride) + half2_index]};
#pragma unroll
        for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
          const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
          const __half2 query_value =
              q_tile[tiled_column]
                    [(key_dimension * kTileKeyHalf2Dim) + half2_index];
          TileHalf2Dot(scores[column][0], key_values[0], query_value);
          TileHalf2Dot(scores[column][1], key_values[1], query_value);
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
      const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
      const std::uint32_t query_offset = tiled_column / kTileHeadsPerBlock;
      const std::uint32_t local_query_position = query_start + query_offset;
      const std::uint32_t absolute_query_position =
          start_pos + local_query_position;
      const std::uint32_t key_positions[2] = {key_start + lane,
                                              key_start + lane + 32};
      const bool valid[2] = {local_query_position < batch_size &&
                                 key_positions[0] <= absolute_query_position &&
                                 key_positions[0] < context_end,
                             local_query_position < batch_size &&
                                 key_positions[1] <= absolute_query_position &&
                                 key_positions[1] < context_end};
      const float score_values[2] = {valid[0] ? scores[column][0] : -INFINITY,
                                     valid[1] ? scores[column][1] : -INFINITY};
      const float tile_max =
          TileWaveReduceMax(fmaxf(score_values[0], score_values[1]));
      const float next_max = fmaxf(running_max[column], tile_max);
      const float prior_scale = isfinite(running_max[column])
                                    ? expf(running_max[column] - next_max)
                                    : 0.0F;
      const float weight_values[2] = {
          valid[0] ? expf(score_values[0] - next_max) : 0.0F,
          valid[1] ? expf(score_values[1] - next_max) : 0.0F};
      running_max[column] = next_max;
      running_sum[column] = (running_sum[column] * prior_scale) +
                            weight_values[0] + weight_values[1];
#pragma unroll
      for (std::uint32_t slot = 0; slot < kTileHalf2Dim / 32; ++slot) {
        output[column][slot].x *= prior_scale;
        output[column][slot].y *= prior_scale;
      }
      weights[tiled_column][lane] = __float2half_rn(weight_values[0]);
      weights[tiled_column][lane + 32] = __float2half_rn(weight_values[1]);
    }
    __syncthreads();

    for (std::uint32_t value_group = 0; value_group < 2; ++value_group) {
      for (std::uint32_t value_row = wave; value_row < kTileValueRows;
           value_row += 8) {
        const std::uint32_t key_position =
            key_start + (value_group * kTileValueRows) + value_row;
#pragma unroll
        for (std::uint32_t half2_index = lane; half2_index < kTileHalf2Dim;
             half2_index += 32) {
          __half2 packed =
              __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
          if (key_position < context_end) {
            const auto* layer_value =
                reinterpret_cast<const __half2*>(v_cache_f16);
            packed = layer_value
                [(static_cast<std::size_t>(key_position) * kv_half2_stride) +
                 (static_cast<std::size_t>(kv_head) * kTileHalf2Dim) +
                 half2_index];
          }
          kv_tile[(value_row * kTileHalf2Dim) + half2_index] = packed;
        }
      }
      __syncthreads();

#pragma unroll
      for (std::uint32_t value_row = 0; value_row < kTileValueRows;
           value_row += 2) {
        __half2 weight_values[columns_per_wave];
#pragma unroll
        for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
          const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
          weight_values[column] = __halves2half2(
              weights[tiled_column][(value_group * kTileValueRows) + value_row],
              weights[tiled_column]
                     [(value_group * kTileValueRows) + value_row + 1]);
        }
#pragma unroll
        for (std::uint32_t slot = 0; slot < kTileHalf2Dim / 32; ++slot) {
          const std::uint32_t half2_index = lane + (slot * 32);
          const __half2 value0 =
              kv_tile[(value_row * kTileHalf2Dim) + half2_index];
          const __half2 value1 =
              kv_tile[((value_row + 1) * kTileHalf2Dim) + half2_index];
          const __half2 values_x =
              __halves2half2(__low2half(value0), __low2half(value1));
          const __half2 values_y =
              __halves2half2(__high2half(value0), __high2half(value1));
#pragma unroll
          for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
            TileHalf2Dot(output[column][slot].x, values_x,
                         weight_values[column]);
            TileHalf2Dot(output[column][slot].y, values_y,
                         weight_values[column]);
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
    const std::uint32_t query_offset = tiled_column / kTileHeadsPerBlock;
    const std::uint32_t query_head =
        first_query_head + (tiled_column % kTileHeadsPerBlock);
    const std::uint32_t local_query_position = query_start + query_offset;
    const float denominator = TileWaveReduceSum(running_sum[column]);
    if (local_query_position >= batch_size) {
      continue;
    }
    if (lse_out != nullptr && lane == 0) {
      lse_out[(static_cast<std::size_t>(query_head) * batch_size) +
              local_query_position] =
          (denominator > 0.0F) ? (running_max[column] + logf(denominator))
                               : -INFINITY;
    }
    if (denominator == 0.0F) {
      if (lse_out != nullptr) {
#pragma unroll
        for (std::uint32_t slot = 0; slot < kTileHalf2Dim / 32; ++slot) {
          const std::uint32_t dimension = 2 * (lane + (slot * 32));
          const std::size_t output_offset =
              (static_cast<std::size_t>(local_query_position) * kAttnWidth) +
              (static_cast<std::size_t>(query_head) * kHeadDim) + dimension;
          out_context[output_offset] = 0.0F;
          out_context[output_offset + 1] = 0.0F;
        }
      }
      continue;
    }
#pragma unroll
    for (std::uint32_t slot = 0; slot < kTileHalf2Dim / 32; ++slot) {
      const std::uint32_t half2_index = lane + (slot * 32);
      const std::uint32_t dimension = 2 * half2_index;
      const std::size_t output_offset =
          (static_cast<std::size_t>(local_query_position) * kAttnWidth) +
          (static_cast<std::size_t>(query_head) * kHeadDim) + dimension;
      const float2 result = output[column][slot];
      const float gate0 = (gate == nullptr || lse_out != nullptr)
                              ? 1.0F
                              : 1.0F / (1.0F + expf(-gate[output_offset]));
      const float gate1 = (gate == nullptr || lse_out != nullptr)
                              ? 1.0F
                              : 1.0F / (1.0F + expf(-gate[output_offset + 1]));
      out_context[output_offset] = (result.x / denominator) * gate0;
      out_context[output_offset + 1] = (result.y / denominator) * gate1;
    }
  }
}

// ---------------------------------------------------------------------------
// Candidate: masked flash attention on the WMMA matrix cores.
//
// Block geometry mirrors the tiled kernel so the K/V re-read traffic is
// unchanged (16 queries x 2 query heads sharing one KV head), but every dot
// product becomes `wmma_f32_16x16x16_f16`. Wave32 fragment layout, verified
// against a CPU reference: A holds row L%16 and 16 contiguous k, B holds column
// L%16 and 16 contiguous k (so B is fed a transposed tile), and C element i is
// row 2i + L/16, column L%16.
//
// The 32 query rows form two 16-row blocks, one per query head, and a key tile
// is 32 keys = two 16-key blocks, so S is four 16x16 tiles. Eight waves cover
// them as four tiles x two halves of the 16-step head dimension, and the two
// halves are summed through LDS. O is two row blocks x sixteen 16-wide dim
// tiles = 32 tiles, four per wave, which is 32 accumulator floats per lane --
// the same register budget the tiled kernel already lives within.
//
// Q never reaches LDS: each wave keeps the eight A fragments for its row block
// and its half of the head dimension in registers for the whole key loop, which
// is 64 VGPRs and removes 16 KiB of LDS pressure. K and the transposed V share
// one staging buffer, since S is complete before PV starts.
// ---------------------------------------------------------------------------

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A fragment is 16 contiguous halves = 32 bytes. Reading it element by element
// costs 16 ds_read_u16 plus 16 __half -> _Float16 conversions; the two formats
// are bit-identical, so two 16-byte loads and a bitcast do the same work in two
// instructions. Every fragment base in this kernel is 16-byte aligned by
// construction (all row strides and offsets are multiples of 8 halves).
#define ABLATE_SYNC()   \
  do {                  \
    if (kAblate != 5) { \
      __syncthreads();  \
    }                   \
  } while (0)

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kWmmaHeads =
    2;  // query heads per block, must divide GQA
constexpr std::uint32_t kWmmaKSteps = kHeadDim / 16;

// Row padding keeps a 16-byte-per-lane fragment read off a single bank group.
constexpr std::uint32_t kWmmaKStride = kHeadDim + 8;

// kQueryRows queries x kWmmaHeads heads form (kQueryRows / 16) * kWmmaHeads
// 16-row blocks, and kKeys keys form kKeys / 16 key blocks. Their product is
// pinned to half the wave count, so the waves split as one S tile each over two
// halves of the head dimension, which is what keeps Q in registers at 64 VGPRs.
// Growing kQueryRows and shrinking kKeys in step therefore trades nothing in
// the tiling but halves the K/V bytes each block pulls per query row -- the
// dominant cost. Raising kWaves alongside kQueryRows keeps the O accumulator
// per lane fixed while doing the same.
template<std::uint32_t kQueryRows, std::uint32_t kKeys, std::uint32_t kWaves,
         int kAblate>
__launch_bounds__(kWaves * 32, 1) __global__
    void WmmaCausalAttention(const float* __restrict__ q,
                             const float* __restrict__ gate,
                             const __half* __restrict__ k_cache_f16,
                             const __half* __restrict__ v_cache_f16,
                             float* __restrict__ out_context,
                             std::uint32_t start_pos, std::size_t batch_size,
                             float* __restrict__ lse_out,
                             std::uint32_t key_begin) {
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kThreads = kWaves * 32;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (kWaves / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kDimTilesPerWave = kWmmaKSteps / kWaves;
  constexpr std::uint32_t kSoftmaxLanes = kThreads / kRows;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles * 2 == kWaves,
                "the waves cover the S tiles in two head-dimension halves");
  static_assert(kKeys % 16 == 0 && kQueryRows % 16 == 0, "16-row WMMA tiles");
  static_assert(kKStepsPerWave * (kWaves / kSTiles) == kWmmaKSteps, "k split");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t query_start = blockIdx.x * kQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kGqa / kWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (kGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  // Row block -> (query head, 16-query sub-block of this block's rows).
  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  constexpr std::uint32_t kKvLdsHalves =
      (kKeys * kWmmaKStride > kHeadDim * kVtStride) ? (kKeys * kWmmaKStride)
                                                    : (kHeadDim * kVtStride);
  __shared__ __half kv_lds[kKvLdsHalves];
  __shared__ float s_lds[2][kSTiles][16][16];
  __shared__ __half p_lds[kRows][kKeys];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < batch_size;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * kAttnWidth) +
        (static_cast<std::size_t>(row_block_head(s_rb)) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  // This wave owns dim tiles `wave` and `wave + 8` for every row block, so both
  // halves of the head dimension stay evenly loaded.
  v8f o_acc[kRowBlocks][kDimTilesPerWave] = {};
  for (std::uint32_t r = tid; r < kRows; r += kThreads) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0F;
  }

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  for (std::uint32_t key_start = key_begin; key_start < max_visible;
       key_start += kKeys) {
    // V's global load is issued before the K staging and the whole S phase, and
    // parked in registers until the transpose. Issued at its point of use it
    // sits behind two barriers with nothing able to cover its latency.
    constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (kThreads * 8);
    uint4 v_raw[kVRegs];
    {
      const std::uint32_t v_key = lane % kKeys;
      const std::uint32_t v_slice = ((tid / kKeys) * (kVRegs * 8));
      const std::uint32_t key_position = key_start + v_key;
      const auto* src =
          v_cache_f16 + (static_cast<std::size_t>(key_position) * kKvWidth) +
          (static_cast<std::size_t>(kv_head) * kHeadDim) + v_slice;
      const bool live = key_position < context_end;
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        if (kAblate == 6) {
          v_raw[j] = make_uint4(key_start + j, lane, wave, j);
        } else {
          v_raw[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                          : make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }

    // --- stage K (coalesced: a wave reads one key row's 512 contiguous bytes)
    ABLATE_SYNC();
    for (std::uint32_t idx = tid; idx < kKeys * (kHeadDim / 8);
         idx += kThreads) {
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      const std::uint32_t key_position = key_start + key_row;
      __half packed[8] = {};
      if (kAblate == 6) {
        *reinterpret_cast<uint4*>(packed) = make_uint4(idx, d8, key_row, 1u);
      } else if (key_position < context_end) {
        const auto* src = k_cache_f16 +
                          (static_cast<std::size_t>(key_position) * kKvWidth) +
                          (static_cast<std::size_t>(kv_head) * kHeadDim) + d8;
        *reinterpret_cast<uint4*>(packed) =
            *reinterpret_cast<const uint4*>(src);
      }
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          *reinterpret_cast<const uint4*>(packed);
    }
    ABLATE_SYNC();

    // --- S = Q K^T ---
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        if (kAblate != 4) {
          s_acc = Wmma(q_frag[ks], k_frag, s_acc);
        }
      }
      // Each half writes its own slot, so the reader sums them instead of a
      // read-modify-write that would need a barrier between the halves.
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    ABLATE_SYNC();

    // --- online softmax: kSoftmaxLanes threads per query row ---
    if (kAblate != 2) {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + row;
      const std::uint32_t absolute_query = start_pos + local_query;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = key_start + col;
        const bool valid = local_query < batch_size &&
                           key_position <= absolute_query &&
                           key_position < context_end;
        const std::uint32_t tile = ((col / 16) * kRowBlocks) + rb;
        vals[m] = valid ? (s_lds[0][tile][row][col % 16] +
                           s_lds[1][tile][row][col % 16])
                        : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float prev_max = row_max[rg];
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      if (seg == 0) {
        row_max[rg] = next_max;
        row_sum[rg] = (row_sum[rg] * prior_scale) + part_sum;
        row_scale[rg] = prior_scale;
      }
    }
    ABLATE_SYNC();

    // --- rescale the running O by the new maximum. A lane touches only rows
    // 2i + half_id, so the factors are read once and reused across its dim
    // tiles.
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }

    // --- stage V transposed. No barrier is needed here: the barrier after the
    // softmax already separates every read of K in the S phase from these
    // writes to the same buffer.
    //
    // Lane L owns key L % kKeys and each thread a dim slice, so consecutive
    // lanes write consecutive *columns* of V^T: two 16-bit writes share a dword
    // and the wave covers 16 banks. The mapping the coalesced global read would
    // prefer -- each lane taking 8 contiguous dims of one key -- puts all 32
    // lanes on rows kVtStride halves apart, which is one bank, and costs a
    // 32-way conflict on every write.
    if (kAblate != 1) {
      const std::uint32_t v_key = lane % kKeys;
      const std::uint32_t v_slice = ((tid / kKeys) * (kVRegs * 8));
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        const auto* packed = reinterpret_cast<const __half*>(&v_raw[j]);
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          kv_lds[((v_slice + (j * 8) + i) * kVtStride) + v_key] = packed[i];
        }
      }
    }
    ABLATE_SYNC();

    // --- O += P V ---
#pragma unroll
    for (std::uint32_t t = 0; (kAblate != 3) && t < kDimTilesPerWave; ++t) {
      const std::uint32_t dim_tile = wave + (t * kWaves);
      v16h v_frag[kKeyBlocks];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
        v_frag[kb] = LoadFrag(
            &kv_lds[(((dim_tile * 16) + sub) * kVtStride) + (kb * 16)]);
      }
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          o_acc[rb][t] = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
        }
      }
    }
  }
  __syncthreads();

  // --- epilogue ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
      const std::uint32_t dim_tile = wave + (t * kWaves);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = query_start + row_offset + row;
        if (local_query >= batch_size) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * kAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr && lse_out == nullptr) {
          value *= 1.0F / (1.0F + __expf(-gate[offset]));
        }
        out_context[offset] = value;
      }
    }
  }
  if (lse_out != nullptr && wave == 0) {
    for (std::uint32_t rg = lane; rg < kRows; rg += 32) {
      const std::uint32_t rb = rg / 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + (rg % 16);
      if (local_query >= batch_size) {
        continue;
      }
      const float denominator = row_sum[rg];
      lse_out[(static_cast<std::size_t>(row_block_head(rb)) * batch_size) +
              local_query] =
          (denominator > 0.0F) ? (row_max[rg] + logf(denominator)) : -INFINITY;
    }
  }
}

// ---------------------------------------------------------------------------
// Pipelined variant
// ---------------------------------------------------------------------------
//
// The ablations above price the two dominant costs of the shipped kernel at
// batch 2048, depth 8192: the K/V global loads at 33% (27.25 -> 18.32 ms with
// them removed) and V's LDS transpose at 23% (-> 20.94). Neither is a bandwidth
// or a bank-cycle limit. Per SIMD the kernel issues one WMMA every 223 cycles
// against a ~64-cycle issue cost, so the SIMD is idle 71% of the time: with
// 23296 bytes of LDS only two blocks fit per CU, giving four waves per SIMD to
// cover a per-key-tile dependency chain that runs
//
//   global load -> barrier -> stage K -> barrier -> S -> barrier -> softmax
//   -> barrier -> transpose V -> barrier -> PV
//
// and all eight waves of a block walk it in lockstep. Three changes here, each
// aimed at that chain rather than at instruction count:
//
//   * Both K and V for tile i+1 are loaded into registers immediately after
//     tile i's staging barrier, so their latency is covered by tile i's S,
//     softmax and PV phases instead of standing at the head of the chain. Costs
//     16 VGPRs of second buffer and 16 v_mov per tile.
//   * V's global read is remapped so four lanes cover 32 contiguous dims of one
//     key. Each instruction's eight key pairs then land on eight fully used
//     64-byte lines; the shipped mapping (one key per lane) issues 16 half-used
//     line requests per instruction for the same bytes.
//   * Each thread holds the same 8 dims of *two* adjacent keys, which are
//     adjacent columns of V^T, so the transpose writes 8 packed dwords instead
//     of 16 halves. A 16-half pad every 8 dims (`kVtSwizzle`) then spreads
//     those writes over all 32 banks: without it the four lane groups differ
//     only in dim, and 8 dims x 12 dwords is 0 mod 32, so they collide 4 ways.
template<std::uint32_t kVtStride, std::uint32_t kVtSwizzle>
__device__ __forceinline__ constexpr std::uint32_t VtRow(std::uint32_t d) {
  return (d * kVtStride) + ((d / 8) * kVtSwizzle);
}

template<std::uint32_t kQueryRows, std::uint32_t kKeys, std::uint32_t kWaves,
         std::uint32_t kVtSwizzle, bool kPipeline, bool kPackedV = true,
         std::uint32_t kVtPad = 8, int kAblate = 0,
         std::uint32_t kDepth = 1>
__launch_bounds__(kWaves * 32, 1) __global__ void WmmaCausalAttentionPipe(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache_f16,
    const __half* __restrict__ v_cache_f16, float* __restrict__ out_context,
    std::uint32_t start_pos, std::size_t batch_size,
    float* __restrict__ lse_out, std::uint32_t key_begin) {
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kThreads = kWaves * 32;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (kWaves / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kDimTilesPerWave = kWmmaKSteps / kWaves;
  constexpr std::uint32_t kSoftmaxLanes = kThreads / kRows;
  constexpr std::uint32_t kVtStride = kKeys + kVtPad;
  constexpr std::uint32_t kKRegs = (kKeys * kHeadDim) / (kThreads * 8);
  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (kThreads * 8);
  static_assert(kSTiles * 2 == kWaves, "waves cover S tiles in two halves");
  static_assert(!kPackedV || (kKeys == 16 && kWaves == 8 && kHeadDim == 256),
                "the coalesced V mapping assumes 4 lanes x 32 dims per key");
  static_assert(kVtStride % 8 == 0 && kVtSwizzle % 8 == 0,
                "fragment rows stay 16-byte aligned");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t query_start = blockIdx.x * kQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kGqa / kWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (kGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  constexpr std::uint32_t kVtHalves =
      VtRow<kVtStride, kVtSwizzle>(kHeadDim - 1) + kKeys;
  constexpr std::uint32_t kKvLdsHalves =
      (kKeys * kWmmaKStride > kVtHalves) ? (kKeys * kWmmaKStride) : kVtHalves;
  __shared__ __half kv_lds[kKvLdsHalves];
  __shared__ float s_lds[2][kSTiles][16][16];
  __shared__ __half p_lds[kRows][kKeys];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < batch_size;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * kAttnWidth) +
        (static_cast<std::size_t>(row_block_head(s_rb)) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  v8f o_acc[kRowBlocks][kDimTilesPerWave] = {};
  for (std::uint32_t r = tid; r < kRows; r += kThreads) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0F;
  }

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  // V: lane L holds dims [wave*32 + (L%4)*8, +8) of keys 2*(L/4) and 2*(L/4)+1.
  const std::uint32_t v_pair = lane >> 2u;
  const std::uint32_t v_dim = (wave * 32u) + ((lane & 3u) * 8u);
  const auto* v_base =
      v_cache_f16 + (static_cast<std::size_t>(kv_head) * kHeadDim) + v_dim;
  const auto* k_base =
      k_cache_f16 + (static_cast<std::size_t>(kv_head) * kHeadDim);

  // The shipped mapping, kept so the coalesced read can be priced on its own:
  // one key per lane, each thread a 16-dim slice.
  const std::uint32_t s_key = lane % kKeys;
  const std::uint32_t s_dim = (tid / kKeys) * (kVRegs * 8);

  const auto load_v = [&](std::uint32_t key_start, uint4* dst) {
    if constexpr (kPackedV) {
#pragma unroll
      for (std::uint32_t j = 0; j < 2; ++j) {
        const std::uint32_t key_position = key_start + (2 * v_pair) + j;
        dst[j] = (key_position < context_end)
                     ? *reinterpret_cast<const uint4*>(
                           v_base +
                           (static_cast<std::size_t>(key_position) * kKvWidth))
                     : make_uint4(0u, 0u, 0u, 0u);
      }
    } else {
      const std::uint32_t key_position = key_start + s_key;
      const auto* src = v_cache_f16 +
                        (static_cast<std::size_t>(key_position) * kKvWidth) +
                        (static_cast<std::size_t>(kv_head) * kHeadDim) + s_dim;
      const bool live = key_position < context_end;
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        dst[j] = (kAblate == 5)
                     ? make_uint4(key_start + j, lane, wave, j)
                     : (live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                             : make_uint4(0u, 0u, 0u, 0u));
      }
    }
  };
  const auto load_k = [&](std::uint32_t key_start, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * kThreads);
      const std::uint32_t key_position = key_start + (idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      dst[n] =
          (kAblate == 5)
              ? make_uint4(idx, d8, key_position, 1u)
              : ((key_position < context_end)
                     ? *reinterpret_cast<const uint4*>(
                           k_base +
                           (static_cast<std::size_t>(key_position) * kKvWidth) +
                           d8)
                     : make_uint4(0u, 0u, 0u, 0u));
    }
  };

  // kDepth is how many key tiles ahead the K/V loads run. At depth the reads
  // miss the MALL, so one tile of compute may not cover the latency; kDepth 2
  // gives two, for one more pair of register buffers.
  constexpr std::uint32_t kVBuf = kPackedV ? 2 : kVRegs;
  uint4 k_buf[kDepth + 1][kKRegs];
  uint4 v_buf[kDepth + 1][kVBuf];
#pragma unroll
  for (std::uint32_t d = 0; d <= kDepth; ++d) {
    const std::uint32_t start = key_begin + (d * kKeys);
    if (d == 0 || start < max_visible) {
      load_k(start, k_buf[d]);
      load_v(start, v_buf[d]);
    }
  }

  for (std::uint32_t key_start = key_begin; key_start < max_visible;
       key_start += kKeys) {
    // --- stage K from the registers the previous iteration prefetched
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * kThreads);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          k_buf[0][n];
    }
    __syncthreads();

    // --- prefetch kDepth tiles ahead. Everything below covers the latency.
    const std::uint32_t next_start = key_start + (kDepth * kKeys);
    if (kPipeline && next_start < max_visible) {
      load_k(next_start, k_buf[kDepth]);
      load_v(next_start, v_buf[kDepth]);
    }

    // --- S = Q K^T ---
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        if (kAblate != 3) {
          s_acc = Wmma(q_frag[ks], k_frag, s_acc);
        }
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    // --- online softmax ---
    if (kAblate != 2) {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + row;
      const std::uint32_t absolute_query = start_pos + local_query;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = key_start + col;
        const bool valid = local_query < batch_size &&
                           key_position <= absolute_query &&
                           key_position < context_end;
        const std::uint32_t tile = ((col / 16) * kRowBlocks) + rb;
        vals[m] = valid ? (s_lds[0][tile][row][col % 16] +
                           s_lds[1][tile][row][col % 16])
                        : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float prev_max = row_max[rg];
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      if (seg == 0) {
        row_max[rg] = next_max;
        row_sum[rg] = (row_sum[rg] * prior_scale) + part_sum;
        row_scale[rg] = prior_scale;
      }
    }
    __syncthreads();

    // --- rescale the running O ---
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }

    // --- transpose V: 8 packed dwords, two adjacent V^T columns each
    if (kAblate == 1) {
      // priced out
    } else if constexpr (kPackedV) {
      const auto* lo = reinterpret_cast<const __half*>(&v_buf[0][0]);
      const auto* hi = reinterpret_cast<const __half*>(&v_buf[0][1]);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t packed =
            static_cast<std::uint32_t>(__half_as_ushort(lo[i])) |
            (static_cast<std::uint32_t>(__half_as_ushort(hi[i])) << 16);
        *reinterpret_cast<std::uint32_t*>(
            &kv_lds[VtRow<kVtStride, kVtSwizzle>(v_dim + i) + (2 * v_pair)]) =
            packed;
      }
    } else {
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        const auto* packed = reinterpret_cast<const __half*>(&v_buf[0][j]);
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          kv_lds[VtRow<kVtStride, kVtSwizzle>(s_dim + (j * 8) + i) + s_key] =
              packed[i];
        }
      }
    }
    __syncthreads();

    // --- O += P V ---
#pragma unroll
    for (std::uint32_t t = 0; (kAblate != 4) && t < kDimTilesPerWave; ++t) {
      const std::uint32_t dim_tile = wave + (t * kWaves);
      v16h v_frag[kKeyBlocks];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
        v_frag[kb] = LoadFrag(
            &kv_lds[VtRow<kVtStride, kVtSwizzle>((dim_tile * 16) + sub) +
                    (kb * 16)]);
      }
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          o_acc[rb][t] = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
        }
      }
    }

    if (kPipeline) {
#pragma unroll
      for (std::uint32_t d = 0; d < kDepth; ++d) {
#pragma unroll
        for (std::uint32_t n = 0; n < kKRegs; ++n) {
          k_buf[d][n] = k_buf[d + 1][n];
        }
#pragma unroll
        for (std::uint32_t n = 0; n < kVBuf; ++n) {
          v_buf[d][n] = v_buf[d + 1][n];
        }
      }
    } else {
      load_k(key_start + kKeys, k_buf[0]);
      load_v(key_start + kKeys, v_buf[0]);
    }
  }
  __syncthreads();

  // --- epilogue ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
      const std::uint32_t dim_tile = wave + (t * kWaves);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = query_start + row_offset + row;
        if (local_query >= batch_size) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * kAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr && lse_out == nullptr) {
          value *= 1.0F / (1.0F + __expf(-gate[offset]));
        }
        out_context[offset] = value;
      }
    }
  }
  if (lse_out != nullptr && wave == 0) {
    for (std::uint32_t rg = lane; rg < kRows; rg += 32) {
      const std::uint32_t rb = rg / 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + (rg % 16);
      if (local_query >= batch_size) {
        continue;
      }
      const float denominator = row_sum[rg];
      lse_out[(static_cast<std::size_t>(row_block_head(rb)) * batch_size) +
              local_query] =
          (denominator > 0.0F) ? (row_max[rg] + logf(denominator)) : -INFINITY;
    }
  }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

template<typename Fn>
float TimeMs(int iters, Fn&& fn) {
  fn();
  HIP_OK(hipGetLastError());
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

float HalfToFloat(__half h) {
  return __half2float(h);
}

// CPU reference in double precision, same masking rule as the kernels.
void ReferenceAttention(const std::vector<float>& q,
                        const std::vector<__half>& k,
                        const std::vector<__half>& v, std::uint32_t start_pos,
                        std::uint32_t batch, std::uint32_t key_begin,
                        std::vector<float>& out) {
  const std::uint32_t context_end = start_pos + batch;
  const double scale = 1.0 / 16.0;
  for (std::uint32_t t = 0; t < batch; ++t) {
    const std::uint32_t absolute = start_pos + t;
    for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
      const std::uint32_t kvh = h / kGqa;
      std::vector<double> scores;
      std::vector<std::uint32_t> keys;
      for (std::uint32_t kp = key_begin; kp < context_end; ++kp) {
        if (kp > absolute) {
          break;
        }
        double dot = 0.0;
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
          const double qv =
              static_cast<double>(q[(t * kAttnWidth) + (h * kHeadDim) + d]) *
              scale;
          const double kvv = static_cast<double>(
              HalfToFloat(k[(kp * kKvWidth) + (kvh * kHeadDim) + d]));
          dot += qv * kvv;
        }
        scores.push_back(dot);
        keys.push_back(kp);
      }
      double m = -INFINITY;
      for (double s : scores) {
        m = std::fmax(m, s);
      }
      double denom = 0.0;
      for (double s : scores) {
        denom += std::exp(s - m);
      }
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        double acc = 0.0;
        for (std::size_t i = 0; i < scores.size(); ++i) {
          acc += std::exp(scores[i] - m) *
                 static_cast<double>(HalfToFloat(
                     v[(keys[i] * kKvWidth) + (kvh * kHeadDim) + d]));
        }
        out[(t * kAttnWidth) + (h * kHeadDim) + d] =
            (denom > 0.0) ? static_cast<float>(acc / denom) : 0.0F;
      }
    }
  }
}

int main(int argc, char** argv) {
  std::uint32_t batch = 2048;
  std::uint32_t start_pos = 0;
  int iters = 3;
  bool check = false;
  bool whole_range = false;
  // At depth 16384 a full sweep is long enough to thermally throttle the APU,
  // which penalizes whichever variant runs last by up to 45%. `-v` runs one
  // variant so a caller can interleave candidates with cooldowns between them.
  std::string only;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-b" && i + 1 < argc) {
      batch = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "-s" && i + 1 < argc) {
      start_pos =
          static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "-i" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (arg == "-v" && i + 1 < argc) {
      only = argv[++i];
    } else if (arg == "-c") {
      check = true;
    } else if (arg == "-k") {
      // Cover the prefix and the diagonal in one kernel instead of masking only
      // the N x N diagonal, which is what replacing the split path would mean.
      whole_range = true;
    }
  }

  const std::uint32_t context_end = start_pos + batch;
  std::vector<float> h_q(static_cast<std::size_t>(batch) * kAttnWidth);
  std::vector<__half> h_k(static_cast<std::size_t>(context_end) * kKvWidth);
  std::vector<__half> h_v(h_k.size());
  for (std::size_t i = 0; i < h_q.size(); ++i) {
    h_q[i] = 0.9F * std::sin(0.0131F * static_cast<float>(i % 811)) +
             0.2F * std::cos(0.0027F * static_cast<float>(i % 53));
  }
  for (std::size_t i = 0; i < h_k.size(); ++i) {
    h_k[i] =
        __float2half(0.8F * std::sin(0.0173F * static_cast<float>(i % 907)));
    h_v[i] =
        __float2half(0.7F * std::cos(0.0119F * static_cast<float>(i % 643)));
  }

  float* d_q = nullptr;
  __half* d_k = nullptr;
  __half* d_v = nullptr;
  float* d_out_ref = nullptr;
  float* d_out_new = nullptr;
  float* d_lse_ref = nullptr;
  float* d_lse_new = nullptr;
  const std::size_t out_elems = static_cast<std::size_t>(batch) * kAttnWidth;
  const std::size_t lse_elems = static_cast<std::size_t>(batch) * kQueryHeads;
  HIP_OK(hipMalloc(&d_q, h_q.size() * sizeof(float)));
  HIP_OK(hipMalloc(&d_k, h_k.size() * sizeof(__half)));
  HIP_OK(hipMalloc(&d_v, h_v.size() * sizeof(__half)));
  HIP_OK(hipMalloc(&d_out_ref, out_elems * sizeof(float)));
  HIP_OK(hipMalloc(&d_out_new, out_elems * sizeof(float)));
  HIP_OK(hipMalloc(&d_lse_ref, lse_elems * sizeof(float)));
  HIP_OK(hipMalloc(&d_lse_new, lse_elems * sizeof(float)));
  HIP_OK(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__half),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_v, h_v.data(), h_v.size() * sizeof(__half),
                   hipMemcpyHostToDevice));

  const std::uint32_t key_begin = whole_range ? 0u : start_pos;
  const dim3 tiled_grid((batch + kTileQueryRows - 1) / kTileQueryRows,
                        kKvHeads * (kGqa / kTileHeadsPerBlock));
  const dim3 tiled_block(32, 8);
  const dim3 wmma_grid16((batch + 15) / 16, kKvHeads * (kGqa / kWmmaHeads));
  const dim3 wmma_grid32((batch + 31) / 32, kKvHeads * (kGqa / kWmmaHeads));
  const dim3 wmma_grid64((batch + 63) / 64, kKvHeads * (kGqa / kWmmaHeads));

  const auto run_tiled = [&] {
    TiledAttention<<<tiled_grid, tiled_block>>>(d_q, nullptr, d_k, d_v,
                                                d_out_ref, start_pos, batch,
                                                d_lse_ref, key_begin);
  };
  const auto run_wmma = [&] {
    WmmaCausalAttention<16, 32, 8, 0>
        <<<wmma_grid16, 256>>>(d_q, nullptr, d_k, d_v, d_out_new, start_pos,
                               batch, d_lse_new, key_begin);
  };
  const auto run_wmma32 = [&] {
    WmmaCausalAttention<32, 16, 8, 0>
        <<<wmma_grid32, 256>>>(d_q, nullptr, d_k, d_v, d_out_new, start_pos,
                               batch, d_lse_new, key_begin);
  };

  // Causal work actually computed: sum over query rows of the visible keys.
  double pairs = 0.0;
  for (std::uint32_t t = 0; t < batch; ++t) {
    const std::uint32_t absolute = start_pos + t;
    const std::uint32_t last =
        (absolute < context_end - 1) ? absolute : context_end - 1;
    pairs += (last >= key_begin) ? (last - key_begin + 1) : 0;
  }
  const double gflop = pairs * kQueryHeads * kHeadDim * 4.0 / 1e9;

  std::printf("batch=%u start_pos=%u iters=%d  heads=%u/%u dk=%u\n", batch,
              start_pos, iters, kQueryHeads, kKvHeads, kHeadDim);
  std::printf(
      "useful work %.2f GFLOP  (WMMA f16 peak 55.05 TFLOPS, v_dot2 "
      "peak 29.7)\n",
      gflop);
  std::printf("%-26s %10s %9s %7s %11s\n", "variant", "ms", "TFLOPS", "%peak",
              "maxrel");

  std::vector<float> ref(out_elems);
  std::vector<float> got(out_elems);
  const auto report = [&](const char* name, float ms, float* d_out,
                          const std::vector<float>* against) {
    double maxrel = -1.0;
    if (against != nullptr) {
      HIP_OK(hipMemcpy(got.data(), d_out, out_elems * sizeof(float),
                       hipMemcpyDeviceToHost));
      double max_abs = 0.0;
      double magnitude = 0.0;
      std::size_t non_finite = 0;
      for (std::size_t i = 0; i < out_elems; ++i) {
        if (!std::isfinite(got[i])) {
          ++non_finite;
          continue;
        }
        max_abs =
            std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) -
                                         static_cast<double>((*against)[i])));
        magnitude = std::fmax(magnitude, std::fabs((*against)[i]));
      }
      maxrel = (non_finite != 0)
                   ? 9.99
                   : ((magnitude > 0.0) ? (max_abs / magnitude) : max_abs);
    }
    const double tflops = gflop / 1e3 / (static_cast<double>(ms) / 1e3);
    if (maxrel < 0.0) {
      std::printf("%-26s %10.3f %9.2f %6.1f%% %11s\n", name, ms, tflops,
                  100.0 * tflops / 55.05, "-");
    } else {
      std::printf("%-26s %10.3f %9.2f %6.1f%% %11.2e\n", name, ms, tflops,
                  100.0 * tflops / 55.05, maxrel);
    }
  };

  run_tiled();
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipMemcpy(ref.data(), d_out_ref, out_elems * sizeof(float),
                   hipMemcpyDeviceToHost));

  if (check) {
    std::vector<float> cpu(out_elems);
    ReferenceAttention(h_q, h_k, h_v, start_pos, batch, key_begin, cpu);
    double worst = 0.0;
    double mag = 0.0;
    for (std::size_t i = 0; i < out_elems; ++i) {
      worst = std::fmax(worst, std::fabs(static_cast<double>(ref[i]) - cpu[i]));
      mag = std::fmax(mag, std::fabs(cpu[i]));
    }
    std::printf("tiled vs CPU double reference: maxrel = %.2e\n", worst / mag);
    ref = cpu;  // hold the kernels to the CPU reference when checking
  }

  const auto bench = [&](const char* name, auto&& fn, float* d_out,
                         const std::vector<float>* against) {
    if (!only.empty() && std::string(name).find(only) == std::string::npos) {
      return;
    }
    report(name, TimeMs(iters, fn), d_out, against);
  };

  bench("tiled (production)", run_tiled, d_out_ref, check ? &ref : nullptr);
  bench("wmma 32 rows/block", run_wmma, d_out_new, &ref);
  bench("wmma 64 rows/block", run_wmma32, d_out_new, &ref);
  bench(
      "wmma 128 rows/block",
      [&] {
        WmmaCausalAttention<64, 16, 16, 0>
            <<<wmma_grid64, 512>>>(d_q, nullptr, d_k, d_v, d_out_new, start_pos,
                                   batch, d_lse_new, key_begin);
      },
      d_out_new, &ref);

  // Pipelined 64-row variants: coalesced V read plus packed transpose writes,
  // with and without the prefetch and the bank swizzle, so the three changes
  // can be priced apart.
#define PIPE(SWIZZLE, PIPELINE, PACKED, PAD, NAME)                          \
  bench(                                                                    \
      NAME,                                                                 \
      [&] {                                                                 \
        WmmaCausalAttentionPipe<32, 16, 8, SWIZZLE, PIPELINE, PACKED, PAD>  \
            <<<wmma_grid32, 256>>>(d_q, nullptr, d_k, d_v, d_out_new,       \
                                   start_pos, batch, d_lse_new, key_begin); \
      },                                                                    \
      d_out_new, &ref)
  PIPE(0, false, true, 8, "pipe: packed V only");
  PIPE(16, false, true, 8, "pipe: packed+swizzle");
  PIPE(0, true, false, 8, "pipe: prefetch only");
  PIPE(0, true, true, 8, "pipe: packed+prefetch");
  PIPE(16, true, true, 8, "pipe: all three");
  // kVtPad 0 drops the V^T row padding, so LDS falls under a third of the 64 KB
  // and three blocks fit per CU instead of two. The swizzle keeps the packed
  // writes conflict-free without it.
  PIPE(16, true, true, 0, "pipe: all three, 3 blk/CU");
  PIPE(16, true, false, 0, "pipe: prefetch, 3 blk/CU");
  PIPE(0, true, false, 0, "pipe: prefetch, pad0");
#undef PIPE

  // Prefetch distance. At depth the K/V reads miss the MALL, so one key tile of
  // compute may not be enough to cover the latency.
#define DEPTH(D, NAME)                                                   \
  bench(                                                                 \
      NAME,                                                              \
      [&] {                                                              \
        WmmaCausalAttentionPipe<32, 16, 8, 0, true, false, 8, 0, D>       \
            <<<wmma_grid32, 256>>>(d_q, nullptr, d_k, d_v, d_out_new,     \
                                   start_pos, batch, d_lse_new,          \
                                   key_begin);                           \
      },                                                                 \
      d_out_new, &ref)
  DEPTH(2, "pipe: prefetch depth 2");
  DEPTH(3, "pipe: prefetch depth 3");
#undef DEPTH

  // A 32-key tile halves the number of dependency-chain traversals per unit of
  // work. The eight-wave split cannot hold it, so this is 16 waves / 512
  // threads at the same 64 query rows per block, hence the same K/V traffic.
  bench(
      "pipe: 32 keys, 16 waves",
      [&] {
        WmmaCausalAttentionPipe<32, 32, 16, 0, true, false, 8>
            <<<wmma_grid32, 512>>>(d_q, nullptr, d_k, d_v, d_out_new, start_pos,
                                   batch, d_lse_new, key_begin);
      },
      d_out_new, &ref);

  // Phase pricing again, this time on the pipelined kernel, to show where the
  // critical path went after the prefetch.
#define PABLATE(N, NAME)                                                    \
  bench(                                                                    \
      NAME,                                                                 \
      [&] {                                                                 \
        WmmaCausalAttentionPipe<32, 16, 8, 0, true, false, 8, N>            \
            <<<wmma_grid32, 256>>>(d_q, nullptr, d_k, d_v, d_out_new,       \
                                   start_pos, batch, d_lse_new, key_begin); \
      },                                                                    \
      d_out_new, nullptr)
  PABLATE(1, "  pipe ablate: no V transpose");
  PABLATE(2, "  pipe ablate: no softmax");
  PABLATE(3, "  pipe ablate: no S wmma");
  PABLATE(4, "  pipe ablate: no PV wmma");
  PABLATE(5, "  pipe ablate: no K/V global");
#undef PABLATE

  // Phase pricing. These are deliberately incorrect; only the time is
  // meaningful.
#define ABLATE(N, NAME)                                                     \
  bench(                                                                    \
      NAME,                                                                 \
      [&] {                                                                 \
        WmmaCausalAttention<32, 16, 8, N>                                   \
            <<<wmma_grid32, 256>>>(d_q, nullptr, d_k, d_v, d_out_new,       \
                                   start_pos, batch, d_lse_new, key_begin); \
      },                                                                    \
      d_out_new, nullptr)
  ABLATE(1, "  ablate: no V transpose");
  ABLATE(2, "  ablate: no softmax");
  ABLATE(3, "  ablate: no PV wmma");
  ABLATE(4, "  ablate: no S wmma");
  ABLATE(5, "  ablate: no barriers");
  ABLATE(6, "  ablate: no K/V global loads");
#undef ABLATE

  return 0;
}
