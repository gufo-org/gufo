#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "mmvq.hpp"
#include "unary.hpp"
#include "vecdotq.hpp"

// Each wave handles up to eight dense inputs for one weight row on gfx1151.
// row_waves packs several output rows into one block. A wave still owns one row
// and keeps that row's exact kbx sequence and reduction tree, so grouping only
// changes which rows a block covers. It pays off on tall, narrow matrices whose
// K loop is too short to fill a wave: the per-row warp reduction then costs more
// than the dot products, and one block per row leaves the SM with one load in
// flight per wave.
// `scatter` reduces the inputs' sums together and stores them from one lane
// each (same bits as reducing each sum on every lane and storing from lane 0).
template<int ncols_dst, bool has_gate, int token_waves = 1, bool ragged = false,
         int row_waves = 1, bool scatter = false>
__launch_bounds__(32 * token_waves * row_waves, 1) static __global__
    void mul_mat_vec_q8(const void* __restrict__ weights,
                        const void* __restrict__ gate,
                        const block_q8_1* __restrict__ input,
                        float* __restrict__ output, const uint32_t ncols_x,
                        const uint32_t nrows_x, const uint32_t stride_col_y,
                        const uint32_t valid_tokens = ncols_dst) {
  constexpr int qi = QI8_0;
  constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  constexpr bool split = token_waves * row_waves > 1;
  const int wave = split ? static_cast<int>(threadIdx.x) / 32 : 0;
  const int lane =
      split ? static_cast<int>(threadIdx.x) % 32 : static_cast<int>(threadIdx.x);
  const int first_token =
      token_waves > 1 ? (wave % token_waves) * ncols_dst : 0;
  const int row = blockIdx.x * row_waves + (row_waves > 1 ? wave / token_waves : 0);
  if constexpr (row_waves > 1) {
    if (row >= static_cast<int>(nrows_x)) {
      return;
    }
  }
  const int blocks_per_row = ncols_x / QK8_0;
  const int row_offset = row * blocks_per_row;
  const int kqs = vdr * (lane % (qi / vdr));
  float sum[ncols_dst] = {};
  float gate_sum[ncols_dst] = {};
  for (int kbx = lane / (qi / vdr); kbx < blocks_per_row;
       kbx += blocks_per_iter) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      // The last wave may contain fewer than eight inputs. Its inactive
      // columns reuse a valid row; their results are never stored.
      const int token =
          ragged ? min(first_token + j, int(valid_tokens) - 1) : first_token + j;
      sum[j] += vec_dot_q8_0_q8_1(
          weights, input + token * stride_col_y + kbx,
          row_offset + kbx, kqs);
      if constexpr (has_gate) {
        gate_sum[j] += vec_dot_q8_0_q8_1(gate, input + j * stride_col_y + kbx,
                                         row_offset + kbx, kqs);
      }
    }
  }
  if constexpr (scatter) {
    static_assert(token_waves == 1 && !ragged);
    constexpr int shift = 5 - ScatterLevels(ncols_dst);
    // Both reductions run on the full wave; only the stores are per lane.
    float value = scatter_reduce_sum(sum, lane);
    float gate_value = 0.0f;
    if constexpr (has_gate)
      gate_value = scatter_reduce_sum(gate_sum, lane);
    const int token = lane >> shift;
    if ((lane & ((1 << shift) - 1)) == 0 && token < ncols_dst) {
      if constexpr (has_gate)
        value *= ggml_hip_op_silu_single(gate_value);
      output[token * nrows_x + row] = value;
    }
    return;
  }
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    sum[j] = warp_reduce_sum<32>(sum[j]);
    if constexpr (has_gate)
      gate_sum[j] = warp_reduce_sum<32>(gate_sum[j]);
    if (lane == 0 && (token_waves == 1 || first_token + j < valid_tokens)) {
      float value = sum[j];
      if constexpr (has_gate)
        value *= ggml_hip_op_silu_single(gate_sum[j]);
      output[(first_token + j) * nrows_x + row] = value;
    }
  }
}

__device__ __forceinline__ float Q8Materialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

// mul_mat_vec_q8 for one input over a few long rows (the draft block's
// 4-row HC inject, K 10240): a wave is a row's whole latency chain, so each
// lane loads `steps` of its blocks before the first product. The lane's
// blocks, products and order are the one-row kernel's, spelled out as it
// compiles them: d0 * sumi, then one fma with d1 into the sum.
template<int steps>
__launch_bounds__(32, 1) static __global__
    void mul_mat_vec_q8_ahead(const void* __restrict__ weights,
                              const block_q8_1* __restrict__ input,
                              float* __restrict__ output,
                              const uint32_t ncols_x) {
  constexpr int qi = QI8_0;
  constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int blocks_per_row = ncols_x / QK8_0;
  const block_q8_0* bx =
      static_cast<const block_q8_0*>(weights) + row * blocks_per_row;
  const int kqs = vdr * (lane % (qi / vdr));
  float sum = 0.0f;
  for (int kb = lane / (qi / vdr); kb < blocks_per_row;
       kb += steps * blocks_per_iter) {
    int v[steps][vdr];
    int u[steps][vdr];
    half d0[steps];
    half d1[steps];
#pragma unroll
    for (int s = 0; s < steps; ++s) {
      const block_q8_0* w = bx + kb + s * blocks_per_iter;
      const block_q8_1* y = input + kb + s * blocks_per_iter;
#pragma unroll
      for (int i = 0; i < vdr; ++i) {
        v[s][i] = get_int_b2(w->qs, kqs + i);
        u[s][i] = get_int_b4(y->qs, kqs + i);
      }
      d0[s] = w->d;
      d1[s] = __low2half(y->ds);
    }
#pragma unroll
    for (int s = 0; s < steps; ++s) {
      int sumi = 0;
#pragma unroll
      for (int i = 0; i < vdr; ++i)
        sumi = ggml_hip_dp4a(v[s][i], u[s][i], sumi);
      const float p =
          Q8Materialize(__half2float(d0[s]) * static_cast<float>(sumi));
      sum = Q8Materialize(__builtin_fmaf(p, __half2float(d1[s]), sum));
    }
  }
  sum = warp_reduce_sum<32>(sum);
  if (lane == 0)
    output[row] = sum;
}

// Integer matrix products reuse each Q8 weight across up to 48 inputs.
// Keep four separate K8 sums per wave: merging them into a K32 integer sum
// would change the scalar kernel's rounded products, FMA chain and reduction.
template<int token_tiles, bool ragged, bool grouped = false>
__launch_bounds__(256) static __global__ void mul_mat_q8_decode_batch(
    const block_q8_0* __restrict__ weights,
    const block_q8_1* __restrict__ input, float* __restrict__ output,
    int k, int rows, int input_stride, int tokens) {
  if constexpr (grouped) {
    input += std::size_t{blockIdx.y} * token_tiles * 16 * input_stride;
    output += std::size_t{blockIdx.y} * token_tiles * 16 * rows;
  }
  using Int4 = __attribute__((ext_vector_type(4))) int;
  using Int8 = __attribute__((ext_vector_type(8))) int;
  __shared__ float partial[16 * 16 * 32];
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int sub = lane % 16;
  const int half = lane / 16;
  const int first_row = blockIdx.x * 16;
  const int blocks = k / QK8_0;
  float acc[token_tiles][4][8]{};
  for (int kb = wave; kb < blocks; kb += 8) {
    const auto* w = weights + min(first_row + sub, rows - 1) * blocks + kb;
    const float scale = __half2float(w->d);
    float scales[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      scales[i] = __shfl(scale, 2 * i + half, 32);
    Int4 codes[4];
#pragma unroll
    for (int part = 0; part < 4; ++part)
      codes[part] = {get_int_b2(w->qs, 2 * part),
                     get_int_b2(w->qs, 2 * part + 1), 0, 0};
#pragma unroll
    for (int tile = 0; tile < token_tiles; ++tile) {
      const int token = ragged && tile == token_tiles - 1
                            ? min(tile * 16 + sub, tokens - 1)
                            : tile * 16 + sub;
      const auto* x = input + token * input_stride + kb;
      const float x_scale = __low2float(x->ds);
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const Int4 values = {get_int_b4(x->qs, 2 * part),
                              get_int_b4(x->qs, 2 * part + 1), 0, 0};
        Int8 dots{};
        dots = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, codes[part], true, values, dots, false);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          acc[tile][part][i] =
              __fmaf_rn(x_scale, __fmul_rn(scales[i], float(dots[i])),
                        acc[tile][part][i]);
      }
    }
  }
  // Reuse the 32 KiB transpose for each token tile. One thread reduces each
  // output in the original lane-zero tree, avoiding duplicate wave sums.
#pragma unroll
  for (int tile = 0; tile < token_tiles; ++tile) {
#pragma unroll
    for (int part = 0; part < 4; ++part) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int out = sub * 16 + 2 * i + half;
        const int swizzle = (out ^ (out >> 4)) & 31;
        partial[out * 32 + ((wave * 4 + part) ^ swizzle)] = acc[tile][part][i];
      }
    }
    __syncthreads();
    const int out = threadIdx.x;
    const int swizzle = (out ^ (out >> 4)) & 31;
    float sums[32];
#pragma unroll
    for (int i = 0; i < 32; ++i)
      sums[i] = partial[out * 32 + (i ^ swizzle)];
#pragma unroll
    for (int delta = 16; delta; delta /= 2) {
#pragma unroll
      for (int i = 0; i < delta; ++i)
        sums[i] = __fadd_rn(sums[i], sums[i + delta]);
    }
    const int row = first_row + out % 16;
    const int token = tile * 16 + out / 16;
    if (row < rows && (!ragged || token < tokens))
      output[token * rows + row] = sums[0];
    __syncthreads();
  }
}

typedef float (*vec_dot_q_hip_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_hip_t get_vec_dot_q_hip(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

template<ggml_type type, int c_rows_per_block, bool gated = false>
__launch_bounds__(mmvq_moe_max_batch(type) * (gated ? 64 : 32),
                  1) static __global__
    void mul_mat_vec_q_moe(const void* __restrict__ weights,
                           const void* __restrict__ vy,
                           const int32_t* __restrict__ ids,
                           float* __restrict__ dst, const uint32_t ncols_x,
                           const uint32_t nrows_x, const uint32_t stride_row_x,
                           const uint32_t stride_col_y,
                           const uint32_t stride_col_dst,
                           const uint32_t stride_channel_x,
                           const uint32_t stride_channel_dst,
                           const uint32_t ncols_dst, const uint32_t ids_stride,
                           const void* __restrict__ up_weights = nullptr,
                           const int32_t* __restrict__ dst_slots = nullptr) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = get_vdr_mmvq(type);
  constexpr int warp_size = 32;

  constexpr vec_dot_q_hip_t vec_dot_q_hip = get_vec_dot_q_hip(type);

  const bool is_up = gated && threadIdx.y != 0;
  const uint32_t token_idx =
      gated ? 0 : blockIdx.z * blockDim.y + threadIdx.y;
  const void* vx = is_up ? up_weights : weights;
  const int row0 = c_rows_per_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * warp_size / qi;

  const uint32_t channel_dst = blockIdx.y;

  if (token_idx >= ncols_dst) {
    return;
  }

  // Inactive experts still write zero to every output row. The expert ID
  // and its validity are uniform within a wave.
  const int32_t id_raw = ids[channel_dst + token_idx * ids_stride];
  const bool invalid_id = id_raw < 0;
  // Expert-ordered slot rows store to their source slot; the load is issued
  // with the expert's, not after the dot product.
  const uint32_t dst_token = dst_slots ? dst_slots[token_idx] : token_idx;
  const uint32_t channel_x = invalid_id ? 0u : (uint32_t)id_raw;

  const block_q8_1* y = ((const block_q8_1*)vy) + token_idx * stride_col_y;
  uint32_t row_offsets[c_rows_per_block];
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        // A ragged tile reads the last valid row again for its unused lane.
        // Keep the bounds check out of the quantized dot-product loop.
        const uint32_t row = min(uint32_t(row0 + i), nrows_x - 1);
        row_offsets[i] = channel_x*stride_channel_x + row*stride_row_x;
    }

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_hip(vx, &y[kby], row_offsets[i] + kbx, kqs);
        }
    }

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if constexpr (gated) {
      // Separate waves preserve each projection's original dot-product
      // grouping. Combining two accumulators in one wave lets fast-math
      // reassociate their shared input scales.
      __shared__ float values[2][c_rows_per_block];
      if (threadIdx.x < c_rows_per_block) {
        const float value = tmp[threadIdx.x];
        values[is_up][threadIdx.x] = isfinite(value) ? value : 0.0f;
      }
      __syncthreads();
      if (!is_up && threadIdx.x < c_rows_per_block &&
          uint32_t(row0 + threadIdx.x) < nrows_x) {
        const float g = values[0][threadIdx.x];
        const float u = values[1][threadIdx.x];
        dst[channel_dst * stride_channel_dst + row0 + threadIdx.x] =
            (g * (1.0f / (1.0f + __expf(-g)))) * u;
      }
      return;
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        const float value = tmp[threadIdx.x];
        dst[channel_dst*stride_channel_dst + dst_token*stride_col_dst + row0 + threadIdx.x] =
            isfinite(value) ? value : 0.0f;
    }
}

// Dense projection of up to eight tokens: each wave owns one weight row and
// reads it once for every token. Per row and token, the lane's kbx order,
// warp reduction and non-finite guard match mul_mat_vec_q_moe, so the result
// equals one expert-0 slot per token.
template<ggml_type type, int ncols_dst, int row_waves>
__launch_bounds__(32 * row_waves, 1) static __global__
    void mul_mat_vec_q_dense(const void* __restrict__ weights,
                             const block_q8_1* __restrict__ input,
                             float* __restrict__ output, const uint32_t ncols_x,
                             const uint32_t nrows_x,
                             const uint32_t stride_col_y) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = get_vdr_mmvq(type);
  constexpr int warp_size = 32;
  constexpr vec_dot_q_hip_t vec_dot_q_hip = get_vec_dot_q_hip(type);
  constexpr int blocks_per_iter = vdr * warp_size / qi;

  const int lane = static_cast<int>(threadIdx.x) % warp_size;
  const uint32_t row =
      blockIdx.x * row_waves + static_cast<uint32_t>(threadIdx.x) / warp_size;
  if (row >= nrows_x) {
    return;
  }
  const int blocks_per_row_x = ncols_x / qk;
  const uint32_t row_offset = row * blocks_per_row_x;
  const int kqs = vdr * (lane % (qi / vdr));

  float tmp[ncols_dst] = {};
  for (int kbx = lane / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      tmp[j] += vec_dot_q_hip(weights, &input[j * stride_col_y + kby],
                              row_offset + kbx, kqs);
    }
  }
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    const float value = warp_reduce_sum<warp_size>(tmp[j]);
    if (lane == 0) {
      output[j * nrows_x + row] = isfinite(value) ? value : 0.0f;
    }
  }
}

template<ggml_type type, int ncols_dst>
static void launch_dense(const void* weights, const block_q8_1* input,
                         float* output, int k, int rows, int input_stride,
                         hipStream_t stream) {
  constexpr int row_waves = 4;
  mul_mat_vec_q_dense<type, ncols_dst, row_waves>
      <<<(rows + row_waves - 1) / row_waves, 32 * row_waves, 0, stream>>>(
          weights, input, output, k, rows, input_stride);
}

template<ggml_type type>
static void dispatch_dense(const void* weights, const block_q8_1* input,
                           float* output, int k, int rows, int tokens,
                           int input_stride, hipStream_t stream) {
  switch (tokens) {
#define DENSE_LAUNCH(N)                                                       \
  case N:                                                                     \
    launch_dense<type, N>(weights, input, output, k, rows, input_stride,      \
                          stream);                                            \
    break;
    DENSE_LAUNCH(1)
    DENSE_LAUNCH(2)
    DENSE_LAUNCH(3)
    DENSE_LAUNCH(4)
    DENSE_LAUNCH(5)
    DENSE_LAUNCH(6)
    DENSE_LAUNCH(7)
    DENSE_LAUNCH(8)
#undef DENSE_LAUNCH
    default:
      GGML_ABORT("invalid dense vector batch width");
  }
}

void mul_mat_vec_dense_dispatch(const void* weights, ggml_type type,
                                const block_q8_1* input, float* output, int k,
                                int rows, int tokens, int input_stride,
                                hipStream_t stream) {
  GGML_ASSERT(k % ggml_blck_size(type) == 0 && rows > 0);
  switch (type) {
    case GGML_TYPE_Q4_K:
      dispatch_dense<GGML_TYPE_Q4_K>(weights, input, output, k, rows, tokens,
                                     input_stride, stream);
      break;
    case GGML_TYPE_Q5_K:
      dispatch_dense<GGML_TYPE_Q5_K>(weights, input, output, k, rows, tokens,
                                     input_stride, stream);
      break;
    case GGML_TYPE_Q5_1:
      dispatch_dense<GGML_TYPE_Q5_1>(weights, input, output, k, rows, tokens,
                                     input_stride, stream);
      break;
    case GGML_TYPE_Q6_K:
      dispatch_dense<GGML_TYPE_Q6_K>(weights, input, output, k, rows, tokens,
                                     input_stride, stream);
      break;
    default:
      GGML_ABORT("unsupported dense vector weight format");
  }
}

// Keep one anchor per expert and a slot mask for each token. Duplicate
// experts share their weights; inactive slots still get explicit zeroes.
// Two anchor lists follow the groups, each a count and then its anchors in
// ascending order: list 0 holds the inactive slots and the groups the first
// launch computes, list 1 (split launches only) the groups with one active
// token. One block of at most kMoeGroupAnchors threads covers every anchor.
constexpr int kMoeGroupAnchors = 256;

static __global__ void group_moe_slots(const int32_t* ids, int32_t* groups,
                                       int tokens, int experts_used,
                                       bool split) {
  const int anchors = tokens * experts_used;
  const int anchor = threadIdx.x;
  // The scans below read the routing from LDS, not one dependent global
  // load per step.
  __shared__ int32_t routed[kMoeGroupAnchors];
  if (anchor < anchors)
    routed[anchor] = ids[anchor];
  __syncthreads();
  int list = -1;
  if (anchor < anchors) {
    const int expert = routed[anchor];
    int32_t* group = groups + anchor * (tokens + 1);
    group[0] = -1;
    if (expert < 0) {
      group[0] = -2;
      list = 0;
    } else {
      bool leader = true;
#pragma unroll 8
      for (int i = 0; i < anchor; ++i)
        leader &= routed[i] != expert;
      if (leader) {
        group[0] = expert;
        int active = 0;
        for (int t = 0; t < tokens; ++t) {
          uint32_t slots = 0;
          for (int j = 0; j < experts_used; ++j)
            if (routed[t * experts_used + j] == expert)
              slots |= uint32_t{1} << j;
          group[t + 1] = static_cast<int32_t>(slots);
          active += slots != 0;
        }
        list = split && active == 1 ? 1 : 0;
      }
    }
  }
  // Each list keeps its anchors in ascending order: a rank is the members
  // in lower lanes of this wave plus the members of lower waves.
  __shared__ int wave_counts[2][kMoeGroupAnchors / 32];
  const int lane = anchor % 32;
  const int wave = anchor / 32;
  const uint32_t member[2] = {static_cast<uint32_t>(__ballot(list == 0)),
                              static_cast<uint32_t>(__ballot(list == 1))};
  if (lane == 0) {
    wave_counts[0][wave] = __popc(member[0]);
    wave_counts[1][wave] = __popc(member[1]);
  }
  __syncthreads();
  int32_t* lists = groups + anchors * (tokens + 1);
  const int waves = blockDim.x / 32;
  if (anchor == 0) {
    int counts[2] = {};
    for (int w = 0; w < waves; ++w) {
      counts[0] += wave_counts[0][w];
      counts[1] += wave_counts[1][w];
    }
    lists[0] = counts[0];
    lists[anchors + 1] = counts[1];
  }
  if (list < 0)
    return;
  int rank = __popc(member[list] & ((uint32_t{1} << lane) - 1));
  for (int w = 0; w < wave; ++w)
    rank += wave_counts[list][w];
  lists[list * (anchors + 1) + 1 + rank] = anchor;
}


template<ggml_type type, int tokens, bool single_request = false>
__launch_bounds__(64) static __global__
    void mul_mat_vec_moe_grouped(const void* __restrict__ gate,
                                 const void* __restrict__ up,
                                 const block_q8_1* __restrict__ input,
                                 const int32_t* __restrict__ groups,
                                 const int32_t* __restrict__ list,
                                 float* __restrict__ output, int k, int rows,
                                 int experts_used, int input_stride) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = get_vdr_mmvq(type);
  constexpr int blocks_per_iter = vdr * 32 / qi;
  constexpr auto dot = get_vec_dot_q_hip(type);
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const bool is_up = tid >= 32;
  const int row0 = blockIdx.x * 2;
  const int listed = list != nullptr ? list[0] : static_cast<int>(gridDim.y);
  for (int entry = blockIdx.y; entry < listed; entry += gridDim.y) {
  const int anchor = list != nullptr ? list[1 + entry] : entry;
  const int32_t* group = groups + anchor * (tokens + 1);
  const int expert = group[0];
  if (expert < 0) {
    if (!single_request && expert == -2 && tid < 2 && row0 + tid < rows)
      output[anchor * rows + row0 + tid] = 0.0f;
    continue;
  }

  constexpr bool split = type == GGML_TYPE_Q4_K && tokens >= 3;
  constexpr int columns = single_request ? 1 : tokens;
  int token = 0;
  if constexpr (split) {
    int active = 0;
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
      if (group[t + 1] != 0) {
        ++active;
        token = t;
      }
    }
    if (single_request ? active != 1 : active <= 1)
      continue;
  }
  const void* weights = is_up ? up : gate;
  const int blocks_per_row = k / qk;
  uint32_t slots[columns];
#pragma unroll
  for (int t = 0; t < columns; ++t)
    slots[t] = static_cast<uint32_t>(group[(single_request ? token : t) + 1]);
  float sum[columns][2] = {};
  for (int kb = lane / (qi / vdr); kb < blocks_per_row; kb += blocks_per_iter) {
    const int kqs = vdr * (lane % (qi / vdr));
    if constexpr (split && !single_request) {
      const Q4MoeFragment w0(
          weights, (expert * rows + row0) * blocks_per_row + kb, kqs);
      const Q4MoeFragment w1(
          weights, (expert * rows + min(row0 + 1, rows - 1)) * blocks_per_row + kb,
          kqs);
#pragma unroll
      for (int t = 0; t < columns; ++t) {
        if (slots[t] == 0)
          continue;
        const auto* x = input + t * input_stride + kb * (qk / QK8_1);
        sum[t][0] += w0.Dot(x, kqs);
        sum[t][1] += w1.Dot(x, kqs);
      }
    } else {
#pragma unroll
      for (int t = 0; t < columns; ++t) {
        if (slots[t] == 0)
          continue;
#pragma unroll
        for (int r = 0; r < 2; ++r)
          sum[t][r] += dot(
              weights,
              input + (single_request ? token : t) * input_stride +
                  kb * (qk / QK8_1),
              (expert * rows + min(row0 + r, rows - 1)) * blocks_per_row + kb,
              kqs);
      }
    }
  }

  // Gate and up stay in separate waves to retain the scalar projection's
  // sum order, including the treatment of nonfinite quantization scales.
  __shared__ float values[columns][2][2];
#pragma unroll
  for (int t = 0; t < columns; ++t) {
    if (slots[t] == 0)
      continue;
#pragma unroll
    for (int r = 0; r < 2; ++r)
      sum[t][r] = warp_reduce_sum<32>(sum[t][r]);
    if (lane < 2)
      values[t][is_up][lane] = isfinite(sum[t][lane]) ? sum[t][lane] : 0.0f;
  }
  __syncthreads();
  if (!is_up && lane < 2 && row0 + lane < rows) {
#pragma unroll
    for (int t = 0; t < columns; ++t) {
      if (slots[t] == 0)
        continue;
      const float g = values[t][0][lane];
      const float u = values[t][1][lane];
      uint32_t bits = slots[t];
      while (bits) {
        const int slot = __ffs(static_cast<int>(bits)) - 1;
        output[((single_request ? token : t) * experts_used + slot) * rows +
               row0 + lane] =
            (g * (1.0f / (1.0f + __expf(-g)))) * u;
        bits &= bits - 1;
      }
    }
  }
  // The next listed group rewrites values.
  __syncthreads();
  }
}

// Group equal experts across the complete request batch. Each distinct token
// appears once; its mask retains duplicate output slots. The append order
// affects scheduling only: no two groups write the same output element.
static __global__ void group_moe_batch(const int32_t* ids, int32_t* storage,
                                       int tokens, int experts_used) {
  const int anchor = blockIdx.x * blockDim.x + threadIdx.x;
  if (anchor >= tokens * experts_used)
    return;
  const int expert = ids[anchor];
  const int first = anchor / experts_used;
  auto* groups = reinterpret_cast<MoeBatchGroup*>(storage + 1);
  if (expert < 0) {
    auto& group = groups[atomicAdd(storage, 1)];
    group.expert = -2;
    group.anchor = anchor;
    return;
  }
  for (int j = 0; j < anchor % experts_used; ++j)
    if (ids[first * experts_used + j] == expert)
      return;
  int before = 0;
  for (int t = 0; t < first; ++t) {
    bool found = false;
    for (int j = 0; j < experts_used; ++j)
      found |= ids[t * experts_used + j] == expert;
    before += found;
  }
  if (before % MMVQ_MAX_BATCH_SIZE != 0)
    return;
  auto& group = groups[atomicAdd(storage, 1)];
  group.expert = expert;
  group.anchor = anchor;
  int count = 0;
  for (int t = first; t < tokens && count < MMVQ_MAX_BATCH_SIZE; ++t) {
    uint32_t mask = 0;
    for (int j = 0; j < experts_used; ++j)
      if (ids[t * experts_used + j] == expert)
        mask |= uint32_t{1} << j;
    if (mask) {
      group.token[count] = t;
      group.slots[count++] = mask;
    }
  }
  for (; count < MMVQ_MAX_BATCH_SIZE; ++count) {
    group.token[count] = 0;
    group.slots[count] = 0;
  }
}


template<ggml_type type, int tokens = 2>
static void launch_moe_grouped(const void* gate, const void* up,
                               const block_q8_1* input, const int32_t* groups,
                               float* output, int k, int rows, int n_tokens,
                               int experts_used, int input_stride,
                               hipStream_t stream) {
  if (n_tokens == tokens) {
    // Q4_K splits groups across two launches from three tokens on, so most
    // of a block row per slot would only read a group header and exit. Two
    // block rows walk each launch's anchor list instead. Two tokens keep one
    // block row per slot.
    constexpr bool split = type == GGML_TYPE_Q4_K && tokens >= 3;
    constexpr int kListRows = 2;
    const int anchors = tokens * experts_used;
    const int32_t* lists = split ? groups + anchors * (tokens + 1) : nullptr;
    const dim3 grid((rows + 1) / 2,
                    split ? std::min(kListRows, anchors) : anchors);
    mul_mat_vec_moe_grouped<type, tokens><<<grid, 64, 0, stream>>>(
        gate, up, input, groups, lists, output, k, rows, experts_used,
        input_stride);
    if constexpr (split) {
      mul_mat_vec_moe_grouped<type, tokens, true><<<grid, 64, 0, stream>>>(
          gate, up, input, groups, lists + anchors + 1, output, k, rows,
          experts_used, input_stride);
    }
  } else if constexpr (tokens < MMVQ_MAX_BATCH_SIZE) {
    launch_moe_grouped<type, tokens + 1>(gate, up, input, groups, output, k,
                                         rows, n_tokens, experts_used,
                                         input_stride, stream);
  } else {
    GGML_ABORT("invalid gated vector batch width");
  }
}

// Tall narrow matrices get several rows per block. A wave is 32 lanes, so the
// grouped launch below is 32*row_waves threads covering row_waves rows.
static inline int q8_row_waves(int k, int rows) {
    // blocks_per_row below 2*blocks_per_iter (8) leaves a mostly idle second
    // trip, which is the signature of a launch-bound rather than a
    // bandwidth-bound projection. Only that shape is grouped: extending the
    // rule to the 248320-row Q8_0 head was measured and regressed both tg and
    // pp, and sixteen waves per block measured the same as eight, so tall and
    // wide shapes keep one block per row.
    return (rows >= 2048 && k <= 512) ? 8 : 1;
}

template <int tokens, int row_waves = 1>
static void launch_q8(const void* weights, const void* gate, const block_q8_1* input,
                      float* output, int k, int rows, int input_stride, hipStream_t stream) {
    const dim3 grid((rows + row_waves - 1) / row_waves);
    const dim3 block(32 * row_waves);
    if (gate && tokens > 1) {
        mul_mat_vec_q8<tokens, true, 1, false, 1, true><<<rows, 32, 0, stream>>>(
            weights, gate, input, output, k, rows, input_stride);
    } else if (gate) {
        mul_mat_vec_q8<tokens, true, 1, false, row_waves><<<grid, block, 0, stream>>>(
            weights, gate, input, output, k, rows, input_stride);
    } else if (tokens == 1 && rows <= 64 && k % (32 * 8 * 20) == 0) {
        // Few rows cannot hide a lane's load latency behind other waves.
        mul_mat_vec_q8_ahead<20><<<rows, 32, 0, stream>>>(weights, input,
                                                          output, k);
    } else if (tokens > 1) {
        mul_mat_vec_q8<tokens, false, 1, false, 1, true><<<rows, 32, 0, stream>>>(
            weights, nullptr, input, output, k, rows, input_stride);
    } else {
        mul_mat_vec_q8<tokens, false, 1, false, row_waves><<<grid, block, 0, stream>>>(
            weights, nullptr, input, output, k, rows, input_stride);
    }
}

template<int waves>
static void launch_q8_batch(const void* weights, const block_q8_1* input,
                            float* output, int k, int rows, int tokens,
                            int input_stride, hipStream_t stream) {
  if (tokens % 8 != 0) {
    mul_mat_vec_q8<8, false, waves, true><<<rows, 32 * waves, 0, stream>>>(
        weights, nullptr, input, output, k, rows, input_stride, tokens);
  } else {
    mul_mat_vec_q8<8, false, waves><<<rows, 32 * waves, 0, stream>>>(
        weights, nullptr, input, output, k, rows, input_stride, tokens);
  }
}

template<int token_tiles>
static void launch_q8_matrix(const void* weights, const block_q8_1* input,
                             float* output, int k, int rows, int tokens,
                             int input_stride, hipStream_t stream) {
  if (tokens == 16 * token_tiles)
    mul_mat_q8_decode_batch<token_tiles, false>
        <<<(rows + 15) / 16, 256, 0, stream>>>(
            static_cast<const block_q8_0*>(weights), input, output, k, rows,
            input_stride, tokens);
  else
    mul_mat_q8_decode_batch<token_tiles, true>
        <<<(rows + 15) / 16, 256, 0, stream>>>(
            static_cast<const block_q8_0*>(weights), input, output, k, rows,
            input_stride, tokens);
}

void mul_mat_vec_q8_dispatch(const void* weights, const void* gate,
                             const block_q8_1* input, float* output, int k,
                             int rows, int tokens, int input_stride,
                             hipStream_t stream) {
  const bool matrix_shape =
      (k == 2560 && rows >= 1024) || (k == 320 && rows == 10240);
  if (!gate && matrix_shape && tokens > 48 && tokens % 32 == 0) {
    // Independent 32-row tiles retain the cache-invariant arithmetic.
    // Put the tile loop in the grid instead of launching once per tile.
    mul_mat_q8_decode_batch<2, false, true>
        <<<dim3((rows + 15) / 16, tokens / 32), 256, 0, stream>>>(
            static_cast<const block_q8_0*>(weights), input, output, k, rows,
            input_stride, 32);
    return;
  }
  if (!gate && matrix_shape && tokens >= 9 && tokens <= 48) {
    if (tokens <= 16)
      launch_q8_matrix<1>(weights, input, output, k, rows, tokens, input_stride,
                           stream);
    else if (tokens <= 32)
      launch_q8_matrix<2>(weights, input, output, k, rows, tokens, input_stride,
                           stream);
    else
      launch_q8_matrix<3>(weights, input, output, k, rows, tokens, input_stride,
                           stream);
    return;
  }
  GGML_ASSERT(k % QK8_0 == 0 && rows > 0);
  if (tokens > 8) {
    GGML_ASSERT(tokens <= 32 && !gate);
    // Each wave retains the eight-row arithmetic. Adjacent waves work
    // on the same weight row, reusing cache lines across requests.
    if (tokens <= 16) {
      launch_q8_batch<2>(weights, input, output, k, rows, tokens, input_stride,
                          stream);
    } else if (tokens <= 24) {
      launch_q8_batch<3>(weights, input, output, k, rows, tokens, input_stride,
                          stream);
    } else {
      launch_q8_batch<4>(weights, input, output, k, rows, tokens, input_stride,
                          stream);
    }
    return;
  }
  // Tall narrow rows pack several rows per block; the wave count per row is
  // unchanged, so each row keeps its own kbx order and warp reduction.
  const int row_waves = gate ? 1 : q8_row_waves(k, rows);
#define Q8_LAUNCH(N)                                                            \
  case N:                                                                       \
    if (row_waves == 8) {                                                       \
      launch_q8<N, 8>(weights, gate, input, output, k, rows, input_stride,      \
                      stream);                                                   \
    } else {                                                                    \
      launch_q8<N, 1>(weights, gate, input, output, k, rows, input_stride,      \
                      stream);                                                   \
    }                                                                           \
    break;
  switch (tokens) {
    Q8_LAUNCH(1)
    Q8_LAUNCH(2)
    Q8_LAUNCH(3)
    Q8_LAUNCH(4)
    Q8_LAUNCH(5)
    Q8_LAUNCH(6)
    Q8_LAUNCH(7)
    Q8_LAUNCH(8)
#undef Q8_LAUNCH
    default:
      GGML_ABORT("invalid vector batch width");
  }
}

template<ggml_type type, int rows_per_wave = 2>
static void launch_moe(const void* weights, const block_q8_1* input,
                       const int32_t* ids, float* output, int k, int rows,
                       int tokens, int experts_used, int input_stride,
                       hipStream_t stream,
                       const int32_t* dst_slots = nullptr) {
  GGML_ASSERT(k % ggml_blck_size(type) == 0 && rows > 0);
  GGML_ASSERT(tokens > 0);
  const int block_tokens = std::min(tokens, mmvq_moe_max_batch(type));
  const int row_stride = k / ggml_blck_size(type);
  mul_mat_vec_q_moe<type, rows_per_wave>
      <<<dim3((rows + rows_per_wave - 1) / rows_per_wave, experts_used,
              (tokens + block_tokens - 1) / block_tokens),
         dim3(32, block_tokens), 0, stream>>>(
          weights, input, ids, output, k, rows, row_stride, input_stride,
          rows * experts_used, rows * row_stride, rows, tokens, experts_used,
          nullptr, dst_slots);
}

void mul_mat_vec_moe_dispatch(const void* weights, ggml_type type,
                             const block_q8_1* input, const int32_t* ids, float* output,
                             int k, int rows, int tokens, int experts_used,
                             int input_stride, hipStream_t stream,
                             const int32_t* dst_slots) {
    // dst_slots (expert-ordered slot rows) is only used by the down view.
    GGML_ASSERT(!dst_slots || (k == 640 && experts_used == 1 && tokens > 1 &&
                               (type == GGML_TYPE_Q5_1 ||
                                type == GGML_TYPE_Q8_0)));
    switch (type) {
        case GGML_TYPE_Q5_1:
          // Down projection slots have independent inputs. Wider row tiles
          // reuse each short input without changing its dot-product sum.
          if (k == 640 && experts_used == 1 && tokens > 1) {
            if (tokens > 20) {
              launch_moe<GGML_TYPE_Q5_1, 8>(
                  weights, input, ids, output, k, rows, tokens, experts_used,
                  input_stride, stream, dst_slots);
            } else {
              launch_moe<GGML_TYPE_Q5_1, 4>(
                  weights, input, ids, output, k, rows, tokens, experts_used,
                  input_stride, stream, dst_slots);
            }
            break;
          }
            launch_moe<GGML_TYPE_Q5_1>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
          // Down projection slots have independent inputs. Four rows reuse
          // each short input across more weights without changing its sum.
          if (k == 640 && experts_used == 1 && tokens > 1) {
            launch_moe<GGML_TYPE_Q8_0, 4>(weights, input, ids, output, k, rows,
                                          tokens, experts_used, input_stride,
                                          stream, dst_slots);
            break;
          }
            launch_moe<GGML_TYPE_Q8_0>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_moe<GGML_TYPE_Q4_K>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_moe<GGML_TYPE_Q5_K>(weights, input, ids, output, k, rows, tokens,
                                      experts_used, input_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_moe<GGML_TYPE_Q6_K>(weights, input, ids, output, k, rows, tokens,
                                       experts_used, input_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            launch_moe<GGML_TYPE_IQ3_S>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            launch_moe<GGML_TYPE_IQ4_XS>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            launch_moe<GGML_TYPE_IQ4_NL>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        default: GGML_ABORT("unsupported vector weight format");
    }
}

// IQ3_S gated experts for 1-8 tokens on F16 activations. One block per
// (row tile, first slot of an expert); each wave owns rows, each lane the
// 32-weight sub-blocks lane, lane + 32, ... of a row. The grid codebook and
// the routed tokens' activations sit in LDS. A sub-block decodes to exact F16
// codes (+-grid) and each routed token accumulates d (1 + 2 scale) times its
// v_dot2 sum, so a token's arithmetic never depends on the batch width or on
// which other tokens share the expert.
// One 32-weight IQ3_S sub-block (block_iq3_s is 110 bytes, so its fields are
// 2-byte aligned: they are read as 16-bit pairs) as exact F16 codes +-grid,
// with d (1 + 2 scale) returned separately.
static __device__ __forceinline__ void iq3s_subblock_half2(
    const uint8_t* __restrict__ b, int ib32, const uint32_t* grid, half2 w[16],
    float& scale) {
  const auto* h = reinterpret_cast<const uint16_t*>(b);
  const uint32_t q0 =
      h[1 + ib32 * 4] | (static_cast<uint32_t>(h[2 + ib32 * 4]) << 16);
  const uint32_t q1 =
      h[3 + ib32 * 4] | (static_cast<uint32_t>(h[4 + ib32 * 4]) << 16);
  const uint32_t sg =
      h[37 + ib32 * 2] | (static_cast<uint32_t>(h[38 + ib32 * 2]) << 16);
  const uint32_t qh = (h[33 + ib32 / 2] >> (8 * (ib32 & 1))) & 0xFFU;
  const uint32_t sc =
      (h[53 + ib32 / 4] >> (8 * ((ib32 / 2) & 1) + 4 * (ib32 & 1))) & 0x0FU;
  scale = __half2float(__ushort_as_half(h[0])) * static_cast<float>(1 + 2 * sc);
  const half2 magic = __floats2half2_rn(-1152.0f, -1152.0f);
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const uint32_t qsb = ((g < 4 ? q0 : q1) >> (8 * (g % 4))) & 0xFFU;
    const uint32_t index = qsb | ((qh << (8 - g)) & 256U);
    const uint32_t magnitudes = grid[index];
    const uint32_t bits = (sg >> (4 * g)) & 0xFU;
    // 0x01 in each negated byte; magnitudes are at least 1, so the per-byte
    // two's complement never carries.
    const uint32_t neg = ((bits * 0x00204081U) & 0x01010101U);
    const uint32_t codes = ((magnitudes ^ (neg * 0xFFU)) + neg) ^ 0x80808080U;
    const uint32_t p0 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x01050004U);
    const uint32_t p1 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x03070206U);
    w[2 * g] = __hadd2(__builtin_bit_cast(half2, p0), magic);
    w[2 * g + 1] = __hadd2(__builtin_bit_cast(half2, p1), magic);
  }
}

// IQ3_S gated experts for 1-8 tokens on F16 activations. One block per
// (row tile, first slot of an expert); each wave owns kRowsPerWave rows. A
// wave streams its gate and up rows with coalesced dword loads into its own
// LDS buffer, with the next rows' loads in flight while the current rows
// decode; each lane decodes the 32-weight sub-blocks lane, lane + 32, ...
// The grid codebook is in LDS; activations are read as 128-bit F16 chunks.
// Each routed token accumulates d (1 + 2 scale) times its v_dot2 sum, so a
// token's arithmetic never depends on the batch width or on which other
// tokens share the expert.
constexpr int kIq3sRowDwords = 2560 / QK_K * 110 / 4;  // 275 for K = 2560
constexpr int kIq3sRowFetch = (kIq3sRowDwords + 31) / 32;

template<int kRowsPerWave, int kMaxTokens>
__launch_bounds__(256) static __global__
    void iq3s_moe_gated_f16(const uint32_t* __restrict__ gate,
                            const uint32_t* __restrict__ up,
                            const half* __restrict__ x,
                            const int32_t* __restrict__ groups,
                            float* __restrict__ output, int k, int rows,
                            int tokens, int experts_used) {
  __shared__ uint32_t grid[512];
  __shared__ uint32_t buf[8][2][kIq3sRowDwords + 1];
  const int anchor = blockIdx.y;
  const int32_t* group = groups + anchor * (tokens + 1);
  const int expert = group[0];
  const int tid = threadIdx.x;
  const int row0 = blockIdx.x * 8 * kRowsPerWave;
  if (expert == -1)
    return;
  if (expert < 0) {
    // An inactive slot still gets explicit zeroes.
    for (int r = tid; r < 8 * kRowsPerWave; r += 256)
      if (row0 + r < rows)
        output[static_cast<size_t>(anchor) * rows + row0 + r] = 0.0f;
    return;
  }
  int active[kMaxTokens];
  uint32_t masks[kMaxTokens];
  int count = 0;
  for (int t = 0; t < tokens && count < kMaxTokens; ++t) {
    const uint32_t m = static_cast<uint32_t>(group[t + 1]);
    if (m != 0) {
      active[count] = t;
      masks[count] = m;
      ++count;
    }
  }
  for (int i = tid; i < 512; i += 256)
    grid[i] = iq3s_grid[i];
  __syncthreads();

  const int lane = tid & 31;
  const int wave = tid >> 5;
  constexpr int kK = 2560;  // hidden size; the launcher requires it
  constexpr int sub_blocks = kK / 32;
  const size_t expert_rows = static_cast<size_t>(expert) * rows;
  uint32_t pg[kIq3sRowFetch];
  uint32_t pu[kIq3sRowFetch];
  const auto fetch = [&](int row) {
    const size_t off = (expert_rows + row) * kIq3sRowDwords;
#pragma unroll
    for (int i = 0; i < kIq3sRowFetch; ++i) {
      const int idx = lane + 32 * i;
      pg[i] = idx < kIq3sRowDwords ? gate[off + idx] : 0U;
      pu[i] = idx < kIq3sRowDwords ? up[off + idx] : 0U;
    }
  };
  const int first = row0 + wave * kRowsPerWave;
  if (first < rows)
    fetch(first);
#pragma unroll 1
  for (int rr = 0; rr < kRowsPerWave; ++rr) {
    const int row = first + rr;
    if (row >= rows)
      break;
#pragma unroll
    for (int i = 0; i < kIq3sRowFetch; ++i) {
      const int idx = lane + 32 * i;
      if (idx < kIq3sRowDwords) {
        buf[wave][0][idx] = pg[i];
        buf[wave][1][idx] = pu[i];
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    if (rr + 1 < kRowsPerWave && row + 1 < rows)
      fetch(row + 1);
    const auto* gb = reinterpret_cast<const uint8_t*>(buf[wave][0]);
    const auto* ub = reinterpret_cast<const uint8_t*>(buf[wave][1]);
    float acc_g[kMaxTokens] = {};
    float acc_u[kMaxTokens] = {};
    for (int sb = lane; sb < sub_blocks; sb += 32) {
      // One matrix at a time keeps a single sub-block's weights live.
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        half2 w[16];
        float scale = 0.0f;
        iq3s_subblock_half2((m == 0 ? gb : ub) + (sb / 8) * 110, sb % 8, grid,
                            w, scale);
        float* acc = m == 0 ? acc_g : acc_u;
#pragma unroll
        for (int c = 0; c < kMaxTokens; ++c) {
          if (c >= count)
            break;
          uint4 xv[4];
          const auto* xsrc = reinterpret_cast<const uint4*>(
              x + static_cast<size_t>(active[c]) * kK + sb * 32);
#pragma unroll
          for (int q = 0; q < 4; ++q)
            xv[q] = xsrc[q];
          const half2* xs = reinterpret_cast<const half2*>(xv);
          float sum = 0.0f;
#pragma unroll
          for (int i = 0; i < 16; ++i)
            ggml_hip_mad(sum, w[i], xs[i]);
          acc[c] = __fmaf_rn(scale, sum, acc[c]);
        }
      }
    }
#pragma unroll
    for (int c = 0; c < kMaxTokens; ++c) {
      if (c >= count)
        break;
      float g = warp_reduce_sum<32>(acc_g[c]);
      float u = warp_reduce_sum<32>(acc_u[c]);
      if (lane == 0) {
        g = isfinite(g) ? g : 0.0f;
        u = isfinite(u) ? u : 0.0f;
        const float value = (g * (1.0f / (1.0f + __expf(-g)))) * u;
        for (int slot = 0; slot < experts_used; ++slot)
          if ((masks[c] >> slot) & 1U)
            output[(static_cast<size_t>(active[c]) * experts_used + slot) *
                       rows +
                   row] = value;
      }
    }
    // Every lane has read this row before the next overwrites the buffer.
    __builtin_amdgcn_wave_barrier();
  }
}

static __global__ void narrow_rows_f16(const float* __restrict__ x,
                                       half* __restrict__ out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = __float2half_rn(x[i]);
}

void mul_mat_vec_iq3s_gated_f16(const void* gate, const void* up,
                                const float* x, const int32_t* ids,
                                half* x_half, int32_t* groups, float* output,
                                int k, int rows, int tokens, int experts_used,
                                hipStream_t stream) {
  GGML_ASSERT(tokens > 0 && tokens <= MMVQ_MAX_BATCH_SIZE &&
              experts_used <= 32 && k == 2560);
  const int count = tokens * k;
  narrow_rows_f16<<<(count + 255) / 256, 256, 0, stream>>>(x, x_half, count);
  const int anchors = tokens * experts_used;
  GGML_ASSERT(anchors <= kMoeGroupAnchors);
  group_moe_slots<<<1, (anchors + 31) / 32 * 32, 0, stream>>>(
      ids, groups, tokens, experts_used, false);
  constexpr int kRowsPerWave = 8;
  const dim3 grid((rows + 8 * kRowsPerWave - 1) / (8 * kRowsPerWave),
                  tokens * experts_used);
  // Decode instantiates one token's registers; the per-token instruction
  // sequence is the same in both instantiations, so verify matches decode.
  if (tokens == 1) {
    iq3s_moe_gated_f16<kRowsPerWave, 1><<<grid, 256, 0, stream>>>(
        static_cast<const uint32_t*>(gate), static_cast<const uint32_t*>(up),
        x_half, groups, output, k, rows, tokens, experts_used);
  } else {
    iq3s_moe_gated_f16<kRowsPerWave, MMVQ_MAX_BATCH_SIZE>
        <<<grid, 256, 0, stream>>>(static_cast<const uint32_t*>(gate),
                                   static_cast<const uint32_t*>(up), x_half,
                                   groups, output, k, rows, tokens,
                                   experts_used);
  }
}

void mul_mat_vec_moe_gated(const void* gate, const void* up, ggml_type type,
                           const block_q8_1* input, const int32_t* ids,
                           int32_t* groups, float* output, int k, int rows,
                           int tokens, int experts_used, int input_stride,
                           hipStream_t stream) {
  GGML_ASSERT(tokens > 0 && tokens <= MMVQ_MAX_ROUTED_BATCH);
  if (tokens > MMVQ_MAX_BATCH_SIZE) {
    GGML_ASSERT(type == GGML_TYPE_Q4_K && groups && experts_used <= 32);
    group_moe_batch<<<(tokens * experts_used + 127) / 128, 128, 0, stream>>>(
        ids, groups, tokens, experts_used);
    mul_mat_vec_moe_batch_wave64(gate, up, input, groups, output, k, rows,
                                 experts_used, input_stride, stream);
    return;
  }
  if (tokens > 1) {
    GGML_ASSERT(groups && experts_used <= 32);
    const int anchors = tokens * experts_used;
    GGML_ASSERT(anchors <= kMoeGroupAnchors);
    group_moe_slots<<<1, (anchors + 31) / 32 * 32, 0, stream>>>(
        ids, groups, tokens, experts_used,
        type == GGML_TYPE_Q4_K && tokens >= 3);
    if (type == GGML_TYPE_Q4_K) {
      launch_moe_grouped<GGML_TYPE_Q4_K>(gate, up, input, groups, output, k,
                                         rows, tokens, experts_used,
                                         input_stride, stream);
    } else if (type == GGML_TYPE_Q8_0) {
      launch_moe_grouped<GGML_TYPE_Q8_0>(gate, up, input, groups, output, k,
                                         rows, tokens, experts_used,
                                         input_stride, stream);
    } else if (type == GGML_TYPE_Q5_K) {
      launch_moe_grouped<GGML_TYPE_Q5_K>(gate, up, input, groups, output, k,
                                         rows, tokens, experts_used,
                                         input_stride, stream);
    } else if (type == GGML_TYPE_IQ3_S) {
      launch_moe_grouped<GGML_TYPE_IQ3_S>(gate, up, input, groups, output, k,
                                         rows, tokens, experts_used,
                                         input_stride, stream);
    } else if (type == GGML_TYPE_IQ4_NL) {
      launch_moe_grouped<GGML_TYPE_IQ4_NL>(gate, up, input, groups, output, k,
                                          rows, tokens, experts_used,
                                          input_stride, stream);
    } else if (type == GGML_TYPE_IQ4_XS) {
      launch_moe_grouped<GGML_TYPE_IQ4_XS>(gate, up, input, groups, output, k,
                                          rows, tokens, experts_used,
                                          input_stride, stream);
    } else {
      GGML_ABORT("unsupported gated vector weight format");
    }
    return;
  }
  const int row_stride = k / ggml_blck_size(type);
  const dim3 grid((rows + 1) / 2, experts_used);
  const dim3 block(32, 2);
  if (type == GGML_TYPE_Q4_K) {
    mul_mat_vec_q_moe<GGML_TYPE_Q4_K, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  } else if (type == GGML_TYPE_Q8_0) {
    mul_mat_vec_q_moe<GGML_TYPE_Q8_0, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  } else if (type == GGML_TYPE_IQ4_XS) {
    mul_mat_vec_q_moe<GGML_TYPE_IQ4_XS, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  } else if (type == GGML_TYPE_IQ4_NL) {
    mul_mat_vec_q_moe<GGML_TYPE_IQ4_NL, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  } else if (type == GGML_TYPE_IQ3_S) {
    mul_mat_vec_q_moe<GGML_TYPE_IQ3_S, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  } else if (type == GGML_TYPE_Q5_K) {
    mul_mat_vec_q_moe<GGML_TYPE_Q5_K, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  } else {
    GGML_ABORT("unsupported gated vector weight format");
  }
}

}  // namespace qfn_mmq
