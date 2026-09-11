#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <hipcub/hipcub.hpp>

__global__ static void indexer_topk_kernel(uint32_t* selected,
                                           const float* scores, uint32_t n_comp,
                                           uint32_t n_tokens, uint32_t top_k) {
  uint32_t t = blockIdx.x;
  if (t >= n_tokens || threadIdx.x != 0)
    return;
  const float* row = scores + (uint64_t)t * n_comp;
  uint32_t* sel = selected + (uint64_t)t * top_k;
  for (uint32_t k = 0; k < top_k; k++)
    sel[k] = 0;
  for (uint32_t c = 0; c < n_comp; c++) {
    float v = row[c];
    for (uint32_t k = 0; k < top_k; k++) {
      if ((k >= c) || v > row[sel[k]]) {
        for (uint32_t j = top_k - 1; j > k; j--)
          sel[j] = sel[j - 1];
        sel[k] = c;
        break;
      }
    }
  }
}

__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai,
                                                         float bv,
                                                         uint32_t bi) {
  return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ static uint32_t topk_float_ordered_key(float v) {
  uint32_t u = __float_as_uint(v);
  // Match the numeric comparator: +0 and -0 tie, then lower index wins.
  if ((u & 0x7fffffffu) == 0u)
    u = 0u;
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t topk_pack_key(float v,
                                                         uint32_t idx) {
  return ((uint64_t)topk_float_ordered_key(v) << 32u) |
         (uint64_t)(0xffffffffu - idx);
}

__global__ static void indexer_topk_8192_cub_kernel(uint32_t* selected,
                                                    const float* scores,
                                                    uint32_t n_comp,
                                                    uint32_t n_tokens,
                                                    uint32_t top_k) {
  constexpr uint32_t BLOCK_THREADS = 512u;
  constexpr uint32_t ITEMS_PER_THREAD = 16u;
  using BlockSort =
      hipcub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
  extern __shared__ __align__(16) unsigned char sort_smem[];
  typename BlockSort::TempStorage& sort_storage =
      *reinterpret_cast<typename BlockSort::TempStorage*>(sort_smem);

  const uint32_t t = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  if (t >= n_tokens || tid >= BLOCK_THREADS)
    return;

  const float* row = scores + (uint64_t)t * n_comp;
  uint64_t keys[ITEMS_PER_THREAD];
#pragma unroll
  for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
    const uint32_t i = tid * ITEMS_PER_THREAD + item;
    if (i < n_comp) {
      keys[item] = topk_pack_key(row[i], i);
    } else {
      keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
    }
  }

  BlockSort(sort_storage).SortDescending(keys);

#pragma unroll
  for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
    const uint32_t i = tid * ITEMS_PER_THREAD + item;
    if (i < top_k) {
      selected[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
    }
  }
}

__global__ static void indexer_topk_1024_kernel(uint32_t* selected,
                                                const float* scores,
                                                uint32_t n_comp,
                                                uint32_t n_tokens,
                                                uint32_t top_k) {
  uint32_t t = blockIdx.x;
  uint32_t tid = threadIdx.x;
  if (t >= n_tokens || tid >= 1024u)
    return;
  __shared__ float vals[1024];
  __shared__ uint32_t idxs[1024];

  const float* row = scores + (uint64_t)t * n_comp;
  if (tid < n_comp) {
    vals[tid] = row[tid];
    idxs[tid] = tid;
  } else {
    vals[tid] = -INFINITY;
    idxs[tid] = UINT32_MAX;
  }
  __syncthreads();

  for (uint32_t k = 2u; k <= 1024u; k <<= 1u) {
    for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
      uint32_t other = tid ^ j;
      if (other > tid && other < 1024u) {
        const float av = vals[tid];
        const float bv = vals[other];
        const uint32_t ai = idxs[tid];
        const uint32_t bi = idxs[other];
        const bool desc_half = (tid & k) == 0u;
        const bool swap = desc_half ? topk_score_better(bv, bi, av, ai)
                                    : topk_score_better(av, ai, bv, bi);
        if (swap) {
          vals[tid] = bv;
          idxs[tid] = bi;
          vals[other] = av;
          idxs[other] = ai;
        }
      }
      __syncthreads();
    }
  }

  if (tid < top_k)
    selected[(uint64_t)t * top_k + tid] = idxs[tid];
}

template<uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(uint32_t* selected,
                                                const float* scores,
                                                uint32_t n_comp,
                                                uint32_t n_tokens,
                                                uint32_t top_k) {
  uint32_t t = blockIdx.x;
  uint32_t tid = threadIdx.x;
  if (t >= n_tokens)
    return;
  __shared__ float vals[SORT_N];
  __shared__ uint32_t idxs[SORT_N];

  const float* row = scores + (uint64_t)t * n_comp;
  for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
    if (i < n_comp) {
      vals[i] = row[i];
      idxs[i] = i;
    } else {
      vals[i] = -INFINITY;
      idxs[i] = UINT32_MAX;
    }
  }
  __syncthreads();

  for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
    for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
      for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t other = i ^ j;
        if (other > i && other < SORT_N) {
          const float av = vals[i];
          const float bv = vals[other];
          const uint32_t ai = idxs[i];
          const uint32_t bi = idxs[other];
          const bool desc_half = (i & k) == 0u;
          const bool swap = desc_half ? topk_score_better(bv, bi, av, ai)
                                      : topk_score_better(av, ai, bv, bi);
          if (swap) {
            vals[i] = bv;
            idxs[i] = bi;
            vals[other] = av;
            idxs[other] = ai;
          }
        }
      }
      __syncthreads();
    }
  }

  for (uint32_t i = tid; i < top_k; i += blockDim.x) {
    selected[(uint64_t)t * top_k + i] = idxs[i];
  }
}

template<uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(uint32_t* selected,
                                                    const float* scores,
                                                    uint32_t n_comp,
                                                    uint32_t n_tokens,
                                                    uint32_t top_k) {
  uint32_t t = blockIdx.x;
  uint32_t tid = threadIdx.x;
  if (t >= n_tokens)
    return;
  __shared__ float vals[SORT_N];
  __shared__ uint16_t idxs[SORT_N];

  const float* row = scores + (uint64_t)t * n_comp;
  for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
    if (i < n_comp) {
      vals[i] = row[i];
      idxs[i] = (uint16_t)i;
    } else {
      vals[i] = -INFINITY;
      idxs[i] = UINT16_MAX;
    }
  }
  __syncthreads();

  for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
    for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
      for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t other = i ^ j;
        if (other > i && other < SORT_N) {
          const float av = vals[i];
          const float bv = vals[other];
          const uint32_t ai = idxs[i];
          const uint32_t bi = idxs[other];
          const bool desc_half = (i & k) == 0u;
          const bool swap = desc_half ? topk_score_better(bv, bi, av, ai)
                                      : topk_score_better(av, ai, bv, bi);
          if (swap) {
            vals[i] = bv;
            idxs[i] = (uint16_t)bi;
            vals[other] = av;
            idxs[other] = (uint16_t)ai;
          }
        }
      }
      __syncthreads();
    }
  }

  for (uint32_t i = tid; i < top_k; i += blockDim.x) {
    selected[(uint64_t)t * top_k + i] = idxs[i];
  }
}

template<uint32_t SORT_N>
__global__ static void indexer_topk_chunk_pow2_kernel(
    uint32_t* candidates, const float* scores, uint32_t n_comp,
    uint32_t n_tokens, uint32_t top_k, uint32_t candidate_stride) {
  uint32_t t = blockIdx.x;
  uint32_t chunk = blockIdx.y;
  uint32_t tid = threadIdx.x;
  if (t >= n_tokens)
    return;

  const uint32_t chunk_start = chunk * SORT_N;
  if (chunk_start >= n_comp)
    return;
  const uint32_t chunk_n =
      n_comp - chunk_start < SORT_N ? n_comp - chunk_start : SORT_N;
  __shared__ float vals[SORT_N];
  __shared__ uint32_t idxs[SORT_N];

  const float* row = scores + (uint64_t)t * n_comp;
  for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
    if (i < chunk_n) {
      vals[i] = row[chunk_start + i];
      idxs[i] = chunk_start + i;
    } else {
      vals[i] = -INFINITY;
      idxs[i] = UINT32_MAX;
    }
  }
  __syncthreads();

  for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
    for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
      for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t other = i ^ j;
        if (other > i && other < SORT_N) {
          const float av = vals[i];
          const float bv = vals[other];
          const uint32_t ai = idxs[i];
          const uint32_t bi = idxs[other];
          const bool desc_half = (i & k) == 0u;
          const bool swap = desc_half ? topk_score_better(bv, bi, av, ai)
                                      : topk_score_better(av, ai, bv, bi);
          if (swap) {
            vals[i] = bv;
            idxs[i] = bi;
            vals[other] = av;
            idxs[other] = ai;
          }
        }
      }
      __syncthreads();
    }
  }

  uint32_t* out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
  for (uint32_t i = tid; i < top_k; i += blockDim.x) {
    out[i] = idxs[i];
  }
}

template<uint32_t SORT_N>
__global__ static void indexer_topk_merge_pow2_kernel(
    uint32_t* selected, const uint32_t* candidates, const float* scores,
    uint32_t n_comp, uint32_t n_tokens, uint32_t top_k,
    uint32_t candidate_count, uint32_t candidate_stride) {
  uint32_t t = blockIdx.x;
  uint32_t tid = threadIdx.x;
  if (t >= n_tokens)
    return;
  __shared__ float vals[SORT_N];
  __shared__ uint32_t idxs[SORT_N];

  const float* row = scores + (uint64_t)t * n_comp;
  const uint32_t* cand = candidates + (uint64_t)t * candidate_stride;
  for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
    uint32_t idx = UINT32_MAX;
    float v = -INFINITY;
    if (i < candidate_count) {
      idx = cand[i];
      if (idx < n_comp)
        v = row[idx];
    }
    vals[i] = v;
    idxs[i] = idx;
  }
  __syncthreads();

  for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
    for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
      for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t other = i ^ j;
        if (other > i && other < SORT_N) {
          const float av = vals[i];
          const float bv = vals[other];
          const uint32_t ai = idxs[i];
          const uint32_t bi = idxs[other];
          const bool desc_half = (i & k) == 0u;
          const bool swap = desc_half ? topk_score_better(bv, bi, av, ai)
                                      : topk_score_better(av, ai, bv, bi);
          if (swap) {
            vals[i] = bv;
            idxs[i] = bi;
            vals[other] = av;
            idxs[other] = ai;
          }
        }
      }
      __syncthreads();
    }
  }

  for (uint32_t i = tid; i < top_k; i += blockDim.x) {
    selected[(uint64_t)t * top_k + i] = idxs[i];
  }
}

template<uint32_t SORT_N>
__global__ static void indexer_topk_tree_merge_pow2_kernel(
    uint32_t* out, const uint32_t* candidates, const float* scores,
    uint32_t n_comp, uint32_t n_tokens, uint32_t top_k, uint32_t n_sets,
    uint32_t merge_group, uint32_t candidate_stride, uint32_t out_stride) {
  uint32_t t = blockIdx.x;
  uint32_t group = blockIdx.y;
  uint32_t tid = threadIdx.x;
  if (t >= n_tokens)
    return;

  const uint32_t set0 = group * merge_group;
  if (set0 >= n_sets)
    return;
  uint32_t set_count = n_sets - set0;
  if (set_count > merge_group)
    set_count = merge_group;
  const uint32_t candidate_count = set_count * top_k;

  __shared__ float vals[SORT_N];
  __shared__ uint32_t idxs[SORT_N];

  const float* row = scores + (uint64_t)t * n_comp;
  const uint32_t* cand =
      candidates + (uint64_t)t * candidate_stride + set0 * top_k;
  for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
    uint32_t idx = UINT32_MAX;
    float v = -INFINITY;
    if (i < candidate_count) {
      idx = cand[i];
      if (idx < n_comp)
        v = row[idx];
    }
    vals[i] = v;
    idxs[i] = idx;
  }
  __syncthreads();

  for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
    for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
      for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t other = i ^ j;
        if (other > i && other < SORT_N) {
          const float av = vals[i];
          const float bv = vals[other];
          const uint32_t ai = idxs[i];
          const uint32_t bi = idxs[other];
          const bool desc_half = (i & k) == 0u;
          const bool swap = desc_half ? topk_score_better(bv, bi, av, ai)
                                      : topk_score_better(av, ai, bv, bi);
          if (swap) {
            vals[i] = bv;
            idxs[i] = bi;
            vals[other] = av;
            idxs[other] = ai;
          }
        }
      }
      __syncthreads();
    }
  }

  uint32_t* dst = out + (uint64_t)t * out_stride + group * top_k;
  for (uint32_t i = tid; i < top_k; i += blockDim.x) {
    dst[i] = idxs[i];
  }
}

// Exact MSD radix selection: narrow the boundary bucket until at most 1024
// candidates remain, then sort their complete score/index keys. Keeping the
// input keys in registers avoids rescanning score memory. Prefix compression
// skips identical digits, including large ties in the floating-point score.
template<uint32_t MAX_N>
__global__ static void indexer_partial_topk_kernel(uint32_t* selected,
                                                   const float* scores,
                                                   uint32_t n,
                                                   uint32_t tokens) {
  constexpr uint32_t threads = 256, waves = threads / 32, cap = 1024;
  using Sort = hipcub::BlockRadixSort<uint64_t, threads, cap / threads>;
  __shared__ typename Sort::TempStorage sort_storage;
  __shared__ uint32_t histogram[waves][16];
  __shared__ uint64_t key_and[waves], key_or[waves];
  __shared__ int digit_shift;
  __shared__ uint64_t candidates[cap];
  __shared__ uint64_t prefix, prefix_mask;
  __shared__ uint32_t remaining, above_total, done, output_count;
  const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5u;
  const uint32_t token = blockIdx.x;
  if (token >= tokens)
    return;
  scores += uint64_t(token) * n;
  selected += uint64_t(token) * 512;
  constexpr uint32_t items = MAX_N / threads;
  uint64_t input_keys[items];
#pragma unroll
  for (uint32_t item = 0; item < items; ++item) {
    const uint32_t i = item * threads + tid;
    input_keys[item] = i < n ? topk_pack_key(scores[i], i) : 0;
  }
  if (tid == 0) {
    prefix = 0;
    prefix_mask = 0;
    remaining = 512;
    above_total = 0;
    done = 0;
    output_count = 0;
  }
  __syncthreads();
  for (int shift = 60; shift >= 0; shift -= 4) {
    // Jump over leading digits shared by every remaining candidate.
    uint64_t local_and = ~uint64_t(0), local_or = 0;
#pragma unroll
    for (uint32_t item = 0; item < items; ++item) {
      const uint32_t i = item * threads + tid;
      const uint64_t key = input_keys[item];
      if (i < n && (key & prefix_mask) == prefix) {
        local_and &= key;
        local_or |= key;
      }
    }
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      local_and &= __shfl_down(local_and, offset, 32);
      local_or |= __shfl_down(local_or, offset, 32);
    }
    if (lane == 0) {
      key_and[wave] = local_and;
      key_or[wave] = local_or;
    }
    __syncthreads();
    if (tid == 0) {
      uint64_t all = ~uint64_t(0), any = 0;
#pragma unroll
      for (uint32_t w = 0; w < waves; ++w) {
        all &= key_and[w];
        any |= key_or[w];
      }
      const uint64_t different = all ^ any;
      digit_shift = different ? int((63u - __clzll(different)) & ~3u) : 0;
      prefix_mask = digit_shift == 60 ? 0 : ~uint64_t(0) << (digit_shift + 4);
      prefix = any & prefix_mask;
    }
    __syncthreads();
    shift = digit_shift;
    uint32_t count = 0;
#pragma unroll
    for (uint32_t item = 0; item < items; ++item) {
      const uint32_t base = item * threads;
      if (base >= n)
        break;
      const uint32_t i = base + tid;
      const uint64_t key = input_keys[item];
      const uint32_t digit = uint32_t(key >> shift) & 15u;
      uint32_t mask = __ballot(i < n && (key & prefix_mask) == prefix);
      const uint32_t b0 = __ballot(digit & 1u);
      const uint32_t b1 = __ballot(digit & 2u);
      const uint32_t b2 = __ballot(digit & 4u);
      const uint32_t b3 = __ballot(digit & 8u);
      if (lane < 16) {
        mask &= (lane & 1u) ? b0 : ~b0;
        mask &= (lane & 2u) ? b1 : ~b1;
        mask &= (lane & 4u) ? b2 : ~b2;
        mask &= (lane & 8u) ? b3 : ~b3;
        count += __popc(mask);
      }
    }
    if (lane < 16)
      histogram[wave][lane] = count;
    __syncthreads();
    uint32_t bin_count = 0;
    if (tid < 16) {
#pragma unroll
      for (uint32_t w = 0; w < waves; ++w)
        bin_count += histogram[w][tid];
    }
    uint32_t suffix = bin_count;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1)
      suffix += __shfl_down(suffix, offset, 32);
    const uint32_t above = suffix - bin_count;
    if (tid < 16 && above < remaining && suffix >= remaining) {
      prefix |= uint64_t(tid) << shift;
      prefix_mask |= uint64_t(15) << shift;
      remaining -= above;
      above_total += above;
      done = above_total + bin_count <= cap;
    }
    __syncthreads();
    if (done)
      break;
  }
  for (uint32_t i = tid; i < cap; i += threads)
    candidates[i] = 0;
  __syncthreads();
#pragma unroll
  for (uint32_t item = 0; item < items; ++item) {
    const uint32_t base = item * threads;
    if (base >= n)
      break;
    const uint32_t i = base + tid;
    const uint64_t key = input_keys[item];
    const bool keep = i < n && key >= prefix;
    const uint32_t mask = __ballot(keep);
    uint32_t offset = 0;
    if (lane == 0 && mask)
      offset = atomicAdd(&output_count, __popc(mask));
    offset = __shfl(offset, 0, 32);
    if (keep) {
      const uint32_t rank = __popc(mask & ((1u << lane) - 1u));
      candidates[offset + rank] = key;
    }
  }
  __syncthreads();
  uint64_t keys[cap / threads];
#pragma unroll
  for (uint32_t j = 0; j < cap / threads; ++j)
    keys[j] = candidates[tid * (cap / threads) + j];
  Sort(sort_storage).SortDescending(keys);
#pragma unroll
  for (uint32_t j = 0; j < cap / threads; ++j) {
    const uint32_t index = tid * (cap / threads) + j;
    if (index < 512)
      selected[index] = 0xffffffffu - uint32_t(keys[j]);
  }
}

// The parallel tree wins beyond this range; wide prefill retains its full sort.
static bool indexer_partial_topk_launch(uint32_t* selected, const float* scores,
                                        uint32_t n_comp, uint32_t n_tokens,
                                        uint32_t top_k) {
  if (top_k != 512u || n_comp <= 1024u || n_comp > 16384u || n_tokens == 0u ||
      n_tokens > 6u)
    return false;
  if (n_comp <= 8192u)
    indexer_partial_topk_kernel<8192>
        <<<n_tokens, 256>>>(selected, scores, n_comp, n_tokens);
  else
    indexer_partial_topk_kernel<16384>
        <<<n_tokens, 256>>>(selected, scores, n_comp, n_tokens);
  return true;
}
