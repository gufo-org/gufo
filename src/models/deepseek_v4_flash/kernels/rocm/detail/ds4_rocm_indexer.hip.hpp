#include "ds4_dspark_sampling.hip.hpp"
#include "ds4_rocm_indexer_score.hip.hpp"
#include "ds4_rocm_indexer_topk.hip.hpp"

template <typename Kernel>
static hipError_t ds4_hip_set_dynamic_shared_memory(Kernel kernel,
                                                     int bytes) {
    return hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                               hipFuncAttributeMaxDynamicSharedMemorySize,
                               bytes);
}

/* Hadamard transform and FP4 (E2M1) round trip of one 128-value indexer row.
 * Every thread of the block calls it (barriers); threads 0..127 own a value.
 * `vals` and `absbuf` are 128 shared floats each. */
__device__ static void indexer_hadamard_fp4_row(float *xr, uint32_t tid, float *vals, float *absbuf) {
    const bool own = tid < 128u;
    if (own) vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if (own && (tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = own ? vals[tid] * 0.08838834764831845f : 0.0f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    if (own) absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (own && lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    if (!own) return;
    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
}

/* kRope first rotates each row's RoPE tail at `pos`, one pair per thread as
 * rope_tail_kernel does, then reads the rotated row back. */
template <bool kRope>
__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim,
                                                   uint32_t pos, ds4_rope_tail_args rope) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    if (kRope) {
        if (tid < rope.n_rot / 2u) {
            rope_tail_rotate_pair(x + (uint64_t)row * head_dim + head_dim - rope.n_rot, tid, pos, rope);
        }
        __syncthreads();
    }
    __shared__ float vals[128];
    __shared__ float absbuf[128];
    indexer_hadamard_fp4_row(x + (uint64_t)row * head_dim, tid, vals, absbuf);
}

__global__ static void indexer_scores_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
    uint32_t c = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;
    if (causal) {
        uint32_t n_visible = (pos0 + t + 1u) / ratio;
        if (c >= n_visible) {
            if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
        const float *kh = index_comp + (uint64_t)c * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) dot += qh[d] * kh[d];
        __shared__ float partial[256];
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        total += fmaxf(partial[0], 0.0f) * weights[(uint64_t)t * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = total * scale;
}

/* DSpark Markov correction: select argmax(logits + W2 * W1[prev]) without
 * moving the vocabulary row back to the host. W1 and W2 are Q8_0. */
template<bool DEVICE_PREVIOUS = false>
__global__ static void dspark_markov_argmax_kernel(
    unsigned long long* out_key, const float* logits,
    const unsigned char* w1_row, const unsigned char* w2, uint32_t vocab,
    uint32_t rank_blocks, const int32_t* previous = nullptr) {
  if constexpr (DEVICE_PREVIOUS) {
    const int32_t token = *previous;
    if (token < 0 || static_cast<uint32_t>(token) >= vocab)
      return;
    w1_row += static_cast<uint64_t>(token) * rank_blocks * 34u;
  }
  __shared__ float state[256];
  const uint32_t tid = threadIdx.x;
  if (tid < rank_blocks * 32u) {
    const uint32_t block = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const unsigned char* qblock = w1_row + (uint64_t)block * 34u;
    const float scale = __half2float(*(const __half*)qblock);
    state[tid] = scale * (float)((const int8_t*)(qblock + 2u))[lane];
  }
  __syncthreads();

  float best_value = -INFINITY;
  uint32_t best_index = 0;
  for (uint32_t i = blockIdx.x * blockDim.x + tid; i < vocab;
       i += gridDim.x * blockDim.x) {
    const unsigned char* row = w2 + (uint64_t)i * rank_blocks * 34u;
    float acc = 0.0f;
    for (uint32_t block = 0; block < rank_blocks; block++) {
      const unsigned char* qblock = row + (uint64_t)block * 34u;
      const float scale = __half2float(*(const __half*)qblock);
      const int8_t* quants = (const int8_t*)(qblock + 2u);
      float sum = 0.0f;
#pragma unroll
            for (uint32_t lane = 0; lane < 32u; lane++) {
                sum += (float)quants[lane] * state[block * 32u + lane];
            }
            acc += scale * sum;
        }
        const float value = logits[i] + acc;
        if (topk_score_better(value, i, best_value, best_index)) {
            best_value = value;
            best_index = i;
        }
    }

    __shared__ float values[256];
    __shared__ uint32_t indices[256];
    values[tid] = best_value;
    indices[tid] = best_index;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride &&
            topk_score_better(values[tid + stride], indices[tid + stride],
                              values[tid], indices[tid])) {
            values[tid] = values[tid + stride];
            indices[tid] = indices[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        const unsigned int bits = __float_as_uint(values[0]);
        const unsigned int value_key =
            (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        const unsigned long long key =
            ((unsigned long long)value_key << 32) |
            (unsigned int)(~indices[0]);
        atomicMax(out_key, key);
    }
}

/*
 * Resolve one DSpark position for several requests while streaming Markov W2
 * once. Each request keeps the scalar kernel's block/lane accumulation and
 * reduction order; only the independent request dimension is fused.
 */
template <uint32_t N_ROWS>
__global__ static void dspark_markov_argmax_batch_kernel(
        unsigned long long *out_keys,
        const float *logits,
        uint64_t logits_row_stride,
        const int32_t *previous_tokens,
        const unsigned char *w1,
        const unsigned char *w2,
        uint32_t vocab,
        uint32_t rank_blocks) {
    __shared__ float states[N_ROWS][256];
    __shared__ float values[N_ROWS][256];
    __shared__ uint32_t indices[N_ROWS][256];
    const uint32_t tid = threadIdx.x;
    const uint64_t row_bytes = (uint64_t)rank_blocks * 34u;

#pragma unroll
    for (uint32_t request = 0; request < N_ROWS; ++request) {
        if (tid < rank_blocks * 32u) {
            const uint32_t block = tid >> 5u;
            const uint32_t lane = tid & 31u;
            const unsigned char *qblock =
                w1 + (uint64_t)(uint32_t)previous_tokens[request] * row_bytes +
                (uint64_t)block * 34u;
            const float scale = __half2float(*(const __half *)qblock);
            states[request][tid] =
                scale * (float)((const int8_t *)(qblock + 2u))[lane];
        }
    }
    __syncthreads();

    float best_values[N_ROWS];
    uint32_t best_indices[N_ROWS];
#pragma unroll
    for (uint32_t request = 0; request < N_ROWS; ++request) {
        best_values[request] = -INFINITY;
        best_indices[request] = 0u;
    }

    for (uint32_t i = blockIdx.x * blockDim.x + tid; i < vocab;
         i += gridDim.x * blockDim.x) {
        const unsigned char *row = w2 + (uint64_t)i * row_bytes;
        float acc[N_ROWS] = {};
        for (uint32_t block = 0; block < rank_blocks; ++block) {
            const unsigned char *qblock = row + (uint64_t)block * 34u;
            const float scale = __half2float(*(const __half *)qblock);
            const int8_t *quants = (const int8_t *)(qblock + 2u);
            float sums[N_ROWS] = {};
#pragma unroll
            for (uint32_t lane = 0; lane < 32u; ++lane) {
                const float quant = (float)quants[lane];
#pragma unroll
                for (uint32_t request = 0; request < N_ROWS; ++request) {
                    sums[request] +=
                        quant * states[request][block * 32u + lane];
                }
            }
#pragma unroll
            for (uint32_t request = 0; request < N_ROWS; ++request) {
                acc[request] += scale * sums[request];
            }
        }
#pragma unroll
        for (uint32_t request = 0; request < N_ROWS; ++request) {
            const float value =
                logits[(uint64_t)request * logits_row_stride + i] +
                acc[request];
            if (topk_score_better(value, i, best_values[request],
                                  best_indices[request])) {
                best_values[request] = value;
                best_indices[request] = i;
            }
        }
    }

#pragma unroll
    for (uint32_t request = 0; request < N_ROWS; ++request) {
        values[request][tid] = best_values[request];
        indices[request][tid] = best_indices[request];
    }
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
#pragma unroll
            for (uint32_t request = 0; request < N_ROWS; ++request) {
                if (topk_score_better(
                        values[request][tid + stride],
                        indices[request][tid + stride],
                        values[request][tid],
                        indices[request][tid])) {
                    values[request][tid] =
                        values[request][tid + stride];
                    indices[request][tid] =
                        indices[request][tid + stride];
                }
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
#pragma unroll
        for (uint32_t request = 0; request < N_ROWS; ++request) {
            const unsigned int bits =
                __float_as_uint(values[request][0]);
            const unsigned int value_key =
                (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
            const unsigned long long key =
                ((unsigned long long)value_key << 32) |
                (unsigned int)(~indices[request][0]);
            atomicMax(out_keys + request, key);
        }
    }
}

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

static int indexer_scores_launch(
    ds4_gpu_tensor* scores, const ds4_gpu_tensor* q,
    const ds4_gpu_tensor* weights, const ds4_gpu_tensor* index_comp,
    uint32_t n_comp, uint32_t n_tokens, uint32_t pos0, uint32_t n_head,
    uint32_t head_dim, uint32_t ratio, float scale, uint32_t causal,
    ds4_gpu_tensor* score_scratch) {
  if (!scores || !q || !weights || !index_comp || n_comp == 0 ||
      n_tokens == 0 || n_head == 0 || head_dim == 0 ||
      q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
      weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
      index_comp->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
      scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
    return 0;
  }
  if (causal && ratio == 0)
    return 0;
  if ((n_tokens == 1u || ds4_rocm_verifier_batch_mode()) && head_dim == 128u &&
      n_head == 64u) {
    const uint32_t tiles =
        (n_comp + indexer_score_direct_keys - 1u) / indexer_score_direct_keys;
    indexer_score_one_direct_kernel<<<dim3(tiles, n_tokens), 128>>>(
        (float*)scores->ptr, (const float*)q->ptr, (const float*)weights->ptr,
        (const float*)index_comp->ptr, n_comp, pos0, ratio, scale,
        causal ? 1 : 0);
    return hip_ok(hipGetLastError(), "indexer score one direct launch");
  }
  if (head_dim == 128u && n_head == 64u) {
    dim3 grid((n_comp + 127u) / 128u, (n_tokens + 15u) / 16u, 1);
    const uint64_t padded = ((uint64_t(n_comp) + 127u) / 128u) * 128u * 128u;
    const uint64_t query_elements =
        ((uint64_t(n_tokens) + 15u) / 16u) * 16u * 64u * 128u;
    if (score_scratch &&
        score_scratch->bytes >= (padded + query_elements) * sizeof(__half)) {
      auto* packed = static_cast<__half*>(score_scratch->ptr);
      auto* packed_queries = packed + padded;
      indexer_pack_keys_kernel<<<(padded + 255u) / 256u, 256>>>(
          static_cast<const float*>(index_comp->ptr), packed, n_comp, padded);
      indexer_pack_queries_kernel<<<dim3(64, (n_tokens + 15u) / 16u), 256>>>(
          static_cast<const float*>(q->ptr), packed_queries, n_tokens);
      indexer_scores_wmma128_kernel<true><<<grid, 256>>>(
          static_cast<float*>(scores->ptr), packed_queries,
          static_cast<const float*>(weights->ptr), packed, n_comp, n_tokens,
          pos0, n_head, head_dim, ratio, scale, causal ? 1 : 0);
    } else {
      indexer_scores_wmma128_kernel<false><<<grid, 256>>>(
          static_cast<float*>(scores->ptr), static_cast<const float*>(q->ptr),
          static_cast<const float*>(weights->ptr), index_comp->ptr, n_comp,
          n_tokens, pos0, n_head, head_dim, ratio, scale, causal ? 1 : 0);
    }
    return hip_ok(hipGetLastError(), "indexer scores wmma128 launch");
  }
  dim3 grid(n_comp, n_tokens, 1);
  indexer_scores_kernel<<<grid, 256>>>(
      (float*)scores->ptr, (const float*)q->ptr, (const float*)weights->ptr,
      (const float*)index_comp->ptr, n_comp, n_tokens, pos0, n_head, head_dim,
      ratio, scale, causal ? 1 : 0);
  return hip_ok(hipGetLastError(), "indexer scores launch");
}

extern "C" int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
  return indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                               n_head, head_dim, 1, scale, 0, nullptr);
}

extern "C" int ds4_gpu_indexer_scores_prefill_tensor(
    ds4_gpu_tensor* scores, const ds4_gpu_tensor* q,
    const ds4_gpu_tensor* weights, const ds4_gpu_tensor* index_comp,
    uint32_t n_comp, uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
    uint32_t ratio, float scale, ds4_gpu_tensor* score_scratch) {
  return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens,
                               0, n_head, head_dim, ratio, scale, 1,
                               score_scratch);
}

extern "C" int ds4_gpu_indexer_scores_decode_batch_tensor(
    ds4_gpu_tensor* scores, const ds4_gpu_tensor* q,
    const ds4_gpu_tensor* weights, const ds4_gpu_tensor* index_comp,
    uint32_t n_comp, uint32_t n_tokens, uint32_t pos0, uint32_t n_head,
    uint32_t head_dim, uint32_t ratio, float scale,
    ds4_gpu_tensor* score_scratch) {
  return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens,
                               pos0, n_head, head_dim, ratio, scale, 1,
                               score_scratch);
}

extern "C" int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    if (indexer_partial_topk_launch((uint32_t*)selected->ptr,
                                    (const float*)scores->ptr, n_comp, n_tokens,
                                    top_k)) {
      return hip_ok(hipGetLastError(), "indexer partial topk launch");
    }
    if (top_k == 512u && n_comp <= 1024u) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return hip_ok(hipGetLastError(), "indexer topk 1024 launch");
    }
    if (top_k == 1024u && n_comp <= 1024u) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return hip_ok(hipGetLastError(), "indexer topk 1024x1024 launch");
    }
    if (top_k == 1024u && n_comp <= 2048u) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return hip_ok(hipGetLastError(), "indexer topk 2048x1024 launch");
    }
    if (top_k == 1024u && n_comp <= 4096u) {
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return hip_ok(hipGetLastError(), "indexer topk 4096x1024 launch");
    }
    if (top_k == 1024u && n_comp <= 8192u) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            hipError_t attr_err = hipGetDevice(&dev);
            if (attr_err == hipSuccess) {
                attr_err = hipDeviceGetAttribute(&max_optin_smem,
                                                  hipDeviceAttributeSharedMemPerBlockOptin,
                                                  dev);
            }
            if (attr_err == hipSuccess && max_optin_smem >= smem) {
                attr_err = ds4_hip_set_dynamic_shared_memory(indexer_topk_8192_cub_kernel, smem);
                if (attr_err == hipSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return hip_ok(hipGetLastError(), "indexer topk 8192x1024 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return hip_ok(hipGetLastError(), "indexer topk 8192x1024 launch");
    }
    if (top_k == 2048u && n_comp <= 4096u) {
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return hip_ok(hipGetLastError(), "indexer topk 4096x2048 launch");
    }
    if (top_k == 2048u && n_comp <= 8192u) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            hipError_t attr_err = hipGetDevice(&dev);
            if (attr_err == hipSuccess) {
                attr_err = hipDeviceGetAttribute(&max_optin_smem,
                                                  hipDeviceAttributeSharedMemPerBlockOptin,
                                                  dev);
            }
            if (attr_err == hipSuccess && max_optin_smem >= smem) {
                attr_err = ds4_hip_set_dynamic_shared_memory(indexer_topk_8192_cub_kernel, smem);
                if (attr_err == hipSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return hip_ok(hipGetLastError(), "indexer topk 8192x2048 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return hip_ok(hipGetLastError(), "indexer topk 8192x2048 launch");
    }
    if (top_k == 512u || top_k == 1024u || top_k == 2048u) {
        const uint32_t chunk_n = 4096u;
        const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
        const uint32_t merge_group = chunk_n / top_k;
        const uint64_t candidate_stride64 = (uint64_t)n_chunks * top_k;
        if (candidate_stride64 > UINT32_MAX) return 0;
        const uint32_t candidate_stride = (uint32_t)candidate_stride64;
        uint32_t n_sets = n_chunks;
        uint64_t scratch_u32_per_token = candidate_stride;
        while (n_sets > merge_group) {
            n_sets = (n_sets + merge_group - 1u) / merge_group;
            scratch_u32_per_token += (uint64_t)n_sets * top_k;
        }
        if (scratch_u32_per_token > UINT64_MAX / n_tokens / sizeof(uint32_t)) return 0;
        const uint64_t tmp_bytes = (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
        uint32_t *scratch = (uint32_t *)hip_tmp_alloc(tmp_bytes, "indexer topk tree");
        if (!scratch) return 0;

        uint32_t *cur = scratch;
        n_sets = n_chunks;
        uint32_t cur_stride = candidate_stride;
        dim3 grid_chunks(n_tokens, n_chunks, 1);
        indexer_topk_chunk_pow2_kernel<4096><<<grid_chunks, 1024>>>(cur,
                                                                    (const float *)scores->ptr,
                                                                    n_comp,
                                                                    n_tokens,
                                                                    top_k,
                                                                    candidate_stride);
        if (!hip_ok(hipGetLastError(), "indexer topk chunk launch")) return 0;

        while (n_sets > merge_group) {
            const uint32_t next_sets = (n_sets + merge_group - 1u) / merge_group;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 grid_merge(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<grid_merge, 1024>>>(
                    next,
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets,
                    merge_group,
                    cur_stride,
                    next_stride);
            if (!hip_ok(hipGetLastError(), "indexer topk tree merge launch")) return 0;
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }

        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                                 cur,
                                                                 (const float *)scores->ptr,
                                                                 n_comp,
                                                                 n_tokens,
                                                                 top_k,
                                                                 n_sets * top_k,
                                                                 cur_stride);
        return hip_ok(hipGetLastError(), "indexer topk tree final launch");
    }
    indexer_topk_kernel<<<n_tokens, 1>>>((uint32_t *)selected->ptr,
                                         (const float *)scores->ptr,
                                         n_comp, n_tokens, top_k);
    return hip_ok(hipGetLastError(), "indexer topk launch");
}

extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<false><<<n_rows, 128>>>((float *)x->ptr, n_rows, head_dim, 0u,
                                                         ds4_rope_tail_args{});
    return hip_ok(hipGetLastError(), "indexer_hadamard_fp4 launch");
}

extern "C" int ds4_gpu_dsv4_indexer_rope_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim, uint32_t n_rot, uint32_t pos, uint32_t n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!x || n_rows == 0 || head_dim != 128u || n_rot > head_dim || (n_rot & 1u) ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<true><<<n_rows, 128>>>(
            (float *)x->ptr, n_rows, head_dim, pos,
            ds4_rope_tail_args{n_rot, n_ctx_orig, 0, freq_base, freq_scale, ext_factor,
                               attn_factor, beta_fast, beta_slow});
    return hip_ok(hipGetLastError(), "indexer rope hadamard_fp4 launch");
}

/* Decode the packed (value, index) key produced by the Markov argmax reduce. */
__global__ static void dspark_markov_key_decode_kernel(
        int32_t *out_index,
        const unsigned long long *key) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    out_index[0] = (int32_t)(~(unsigned int)(key[0] & 0xffffffffull));
}

__global__ static void dspark_markov_key_decode_batch_kernel(
        int32_t *out_indices,
        const unsigned long long *keys,
        uint32_t n_rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    out_indices[row] =
        (int32_t)(~(unsigned int)(keys[row] & 0xffffffffull));
}

/*
 * DSpark path selection for one drafted position.
 *
 * Selecting each position independently by argmax lets the block drift, because
 * position t's best token given the block's own context is not the best token
 * given what position t-1 actually emitted. The Markov head restores that
 * coupling with a rank-256 bilinear term, and it stays on the device: the
 * vocabulary row never moves to the host.
 */
extern "C" int ds4_gpu_dspark_candidates_tensor(
    ds4_gpu_tensor* out, ds4_gpu_tensor* scratch, const ds4_gpu_tensor* logits,
    uint64_t logits_stride, const ds4_gpu_tensor* hidden,
    uint64_t hidden_stride, const ds4_gpu_tensor* previous,
    const void* model_map, uint64_t model_size, uint64_t w1_offset,
    uint64_t w2_offset, uint64_t confidence_offset, uint32_t vocab,
    uint32_t rank, uint32_t hidden_width, uint32_t requests) {
  if (!out || !scratch || !logits || !hidden || !previous || !model_map ||
      !vocab || !rank || rank > 256 || rank % 32 || !hidden_width ||
      hidden_width % 32 || !requests || requests > 8 || logits_stride < vocab ||
      hidden_stride < hidden_width)
    return 0;
  const uint32_t blocks = (vocab + 255u) / 256u;
  const uint64_t table_bytes = (uint64_t)vocab * (rank / 32) * 34;
  const uint64_t confidence_bytes = (uint64_t)(hidden_width + rank) / 32 * 34;
  if (!hip_tensor_has_elems(out, requests, sizeof(ds4_dspark_candidates)) ||
      !hip_tensor_has_elems(scratch,
                            (uint64_t)requests * blocks * DS4_DSPARK_CANDIDATES,
                            sizeof(uint64_t)) ||
      !hip_tensor_has_elems(logits, (requests - 1) * logits_stride + vocab,
                            sizeof(float)) ||
      !hip_tensor_has_elems(hidden,
                            (requests - 1) * hidden_stride + hidden_width,
                            sizeof(float)) ||
      !hip_tensor_has_elems(previous, requests, sizeof(int32_t)) ||
      !hip_model_range_fits(model_size, w1_offset, table_bytes) ||
      !hip_model_range_fits(model_size, w2_offset, table_bytes) ||
      !hip_model_range_fits(model_size, confidence_offset, confidence_bytes))
    return 0;
  const auto* w1 = reinterpret_cast<const unsigned char*>(hip_model_range_ptr(
      model_map, w1_offset, table_bytes, "dspark_markov_w1"));
  const auto* w2 = reinterpret_cast<const unsigned char*>(hip_model_range_ptr(
      model_map, w2_offset, table_bytes, "dspark_markov_w2"));
  const auto* confidence =
      reinterpret_cast<const unsigned char*>(hip_model_range_ptr(
          model_map, confidence_offset, confidence_bytes, "dspark_confidence"));
  if (!w1 || !w2 || !confidence)
    return 0;
  dspark_markov_candidates_kernel<<<dim3(blocks, requests), 256>>>(
      (uint64_t*)scratch->ptr, (const float*)logits->ptr, logits_stride,
      (const int32_t*)previous->ptr, w1, w2, vocab, rank);
  if (!hip_ok(hipGetLastError(), "dspark candidates launch"))
    return 0;
  dspark_merge_candidates_kernel<<<requests, 256>>>(
      (ds4_dspark_candidates*)out->ptr, (const uint64_t*)scratch->ptr, blocks,
      (const float*)hidden->ptr, hidden_stride, hidden_width,
      (const int32_t*)previous->ptr, w1, confidence, rank);
  return hip_ok(hipGetLastError(), "dspark candidates merge launch");
}

extern "C" int ds4_gpu_dspark_markov_chain_tensor(
    ds4_gpu_tensor* out_index, ds4_gpu_tensor* scratch_key,
    const ds4_gpu_tensor* logits_rows, const void* model_map,
    uint64_t model_size, uint64_t w1_offset, uint64_t w2_offset, uint32_t vocab,
    uint32_t markov_rank, uint32_t previous_token, uint32_t rows) {
  if (!out_index || !scratch_key || !logits_rows || !model_map || vocab == 0u ||
      markov_rank == 0u || markov_rank % 32u != 0u || markov_rank > 256u ||
      previous_token >= vocab || rows == 0u || rows > 8u ||
      !hip_tensor_has_elems(out_index, rows, sizeof(int32_t)) ||
      !hip_tensor_has_elems(scratch_key, 1, sizeof(unsigned long long)) ||
      !hip_tensor_has_elems(logits_rows, (uint64_t)rows * vocab, sizeof(float)))
    return 0;
  const uint32_t rank_blocks = markov_rank / 32u;
  const uint64_t row_bytes = (uint64_t)rank_blocks * 34u;
  uint64_t table_bytes = 0;
  if (!hip_u64_mul_checked(vocab, row_bytes, &table_bytes) ||
      !hip_model_range_fits(model_size, w1_offset, table_bytes) ||
      !hip_model_range_fits(model_size, w2_offset, table_bytes))
    return 0;
  const auto* w1 = (const unsigned char*)hip_model_range_ptr(
      model_map, w1_offset, table_bytes, "dspark_markov_w1");
  const auto* w2 = (const unsigned char*)hip_model_range_ptr(
      model_map, w2_offset, table_bytes, "dspark_markov_w2");
  if (!w1 || !w2)
    return 0;
  auto* indices = (int32_t*)out_index->ptr;
  auto* key = (unsigned long long*)scratch_key->ptr;
  const uint32_t blocks = (vocab + 255u) / 256u;
  for (uint32_t row = 0; row < rows; ++row) {
    if (!hip_ok(hipMemsetAsync(key, 0, sizeof(*key), nullptr),
                "dspark markov key reset"))
      return 0;
    const auto* logits = (const float*)logits_rows->ptr + (uint64_t)row * vocab;
    // Preserve the sequential Markov dependency on the default stream.
    // Only the first previous token comes from the host; subsequent
    // positions consume the preceding kernel's index directly.
    if (row == 0u)
      dspark_markov_argmax_kernel<false><<<blocks, 256>>>(
          key, logits, w1 + (uint64_t)previous_token * row_bytes, w2, vocab,
          rank_blocks);
    else
      dspark_markov_argmax_kernel<true><<<blocks, 256>>>(
          key, logits, w1, w2, vocab, rank_blocks, indices + row - 1u);
    if (!hip_ok(hipGetLastError(), "dspark markov argmax launch"))
      return 0;
    dspark_markov_key_decode_kernel<<<1, 1>>>(indices + row, key);
    if (!hip_ok(hipGetLastError(), "dspark markov decode launch"))
      return 0;
  }
  return 1;
}

extern "C" int ds4_gpu_dspark_markov_argmax_batch_tensor(
        ds4_gpu_tensor       *out_index,
        ds4_gpu_tensor       *scratch_key,
        const ds4_gpu_tensor *logits_rows,
        const ds4_gpu_tensor *previous_tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                w1_offset,
        uint64_t                w2_offset,
        uint32_t                vocab,
        uint32_t                markov_rank,
        uint32_t                n_rows,
        uint64_t                logits_row_stride) {
    if (!out_index || !scratch_key || !logits_rows || !previous_tokens ||
        !model_map || vocab == 0u || markov_rank == 0u ||
        markov_rank % 32u != 0u ||
        (n_rows != 2u && n_rows != 4u && n_rows != 6u && n_rows != 8u) ||
        logits_row_stride < vocab ||
        !hip_tensor_has_elems(out_index, n_rows, sizeof(int32_t)) ||
        !hip_tensor_has_elems(scratch_key, n_rows,
                              sizeof(unsigned long long)) ||
        !hip_tensor_has_elems(previous_tokens, n_rows, sizeof(int32_t)) ||
        !hip_tensor_has_elems(
            logits_rows,
            (uint64_t)(n_rows - 1u) * logits_row_stride + vocab,
            sizeof(float))) {
        return 0;
    }
    const uint32_t rank_blocks = markov_rank / 32u;
    if (rank_blocks * 32u > 256u) return 0;
    const uint64_t row_bytes = (uint64_t)rank_blocks * 34u;
    uint64_t table_bytes = 0u;
    if (!hip_u64_mul_checked(vocab, row_bytes, &table_bytes) ||
        !hip_model_range_fits(model_size, w1_offset, table_bytes) ||
        !hip_model_range_fits(model_size, w2_offset, table_bytes)) {
        return 0;
    }
    const unsigned char *w1 =
        (const unsigned char *)hip_model_range_ptr(
            model_map, w1_offset, table_bytes, "dspark_markov_w1_batch");
    const unsigned char *w2 =
        (const unsigned char *)hip_model_range_ptr(
            model_map, w2_offset, table_bytes, "dspark_markov_w2_batch");
    if (!w1 || !w2) return 0;
    if (!hip_ok(hipMemset(scratch_key->ptr, 0,
                          (size_t)n_rows * sizeof(unsigned long long)),
                "dspark markov batch key reset")) {
        return 0;
    }
    const uint32_t blocks = (vocab + 255u) / 256u;
    switch (n_rows) {
        case 2u:
            dspark_markov_argmax_batch_kernel<2u><<<blocks, 256>>>(
                (unsigned long long *)scratch_key->ptr,
                (const float *)logits_rows->ptr, logits_row_stride,
                (const int32_t *)previous_tokens->ptr, w1, w2, vocab,
                rank_blocks);
            break;
        case 4u:
            dspark_markov_argmax_batch_kernel<4u><<<blocks, 256>>>(
                (unsigned long long *)scratch_key->ptr,
                (const float *)logits_rows->ptr, logits_row_stride,
                (const int32_t *)previous_tokens->ptr, w1, w2, vocab,
                rank_blocks);
            break;
        case 6u:
            dspark_markov_argmax_batch_kernel<6u><<<blocks, 256>>>(
                (unsigned long long *)scratch_key->ptr,
                (const float *)logits_rows->ptr, logits_row_stride,
                (const int32_t *)previous_tokens->ptr, w1, w2, vocab,
                rank_blocks);
            break;
        case 8u:
            dspark_markov_argmax_batch_kernel<8u><<<blocks, 256>>>(
                (unsigned long long *)scratch_key->ptr,
                (const float *)logits_rows->ptr, logits_row_stride,
                (const int32_t *)previous_tokens->ptr, w1, w2, vocab,
                rank_blocks);
            break;
        default:
            return 0;
    }
    if (!hip_ok(hipGetLastError(), "dspark markov batch argmax launch")) {
        return 0;
    }
    dspark_markov_key_decode_batch_kernel<<<1, 32>>>(
        (int32_t *)out_index->ptr,
        (const unsigned long long *)scratch_key->ptr, n_rows);
    return hip_ok(hipGetLastError(), "dspark markov batch decode launch");
}
