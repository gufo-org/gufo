// DS4 ROCm attention kernels (prefill/decode, raw/mixed KV).

/* Non-causal attention used by the small DSpark draft block. Each query row
 * sees the complete raw-KV ring rather than a causal prefix. */
#include "ds4_rocm_decode_attention.hip.hpp"
//
// Included from ds4_rocm.hip.cpp in the same translation unit to keep launch/API
// glue unchanged while kernel implementations are split into modules.

#define DS4_ROCM_ATTENTION_PREFILL_MIXED_SCORE_CAP 2048u
/* The wave32 rocWMMA producer is only dispatched at top_k == 512 (see
 * attention_indexed_mixed_batch_heads_tensor's eligibility check), so its row
 * table needs half the generic cap. The 2 KiB that frees is what lets the score
 * pass split K across four wave groups. */
#define DS4_ROCM_ATTENTION_WMMA_TOPK_CAP 512u
#define DS4_ROCM_ATTENTION_INDEXED_SCORE_CAP \
    (256u + DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
template <bool INDEXED, bool COMP_F16>
__global__ __launch_bounds__(1024, 1) static void attention_mixed_heads32_wmma_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const void *comp_kv,
        const int32_t *topk,
        float *score_cache,
        uint32_t score_stride,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t HEADS = 32u;
    constexpr uint32_t ROWS = 16u;
    constexpr uint32_t DIM = 512u;
    constexpr uint32_t LDS_DIM = DIM + 4u;

    const uint32_t t = (uint32_t)blockIdx.x;
    const uint32_t head0 = (uint32_t)blockIdx.y * HEADS;
    if (t >= n_tokens || head_dim != DIM || head0 + HEADS > n_head) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    /* The score cache existed so the second pass could skip recomputing QK.
     * The single-pass form below visits each row block once, so nothing reads
     * it; the parameters stay for ABI compatibility with the launchers. */
    (void)score_cache;
    (void)score_stride;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[
        INDEXED ? DS4_ROCM_ATTENTION_WMMA_TOPK_CAP : 1u];
    __shared__ uint32_t comp_count_s;
    /* Q is staged transposed, as [k][head], so the score pass reads both of its
     * operands row-major.
     *
     * The score and value passes issue the same number of matrix ops per row
     * block, yet an ablation harness (tools/bench/dsv4_attn_mixed_bench.hip)
     * priced the score pass at about eight times the value pass. Neither
     * spreading it over more waves nor halving its LDS reads moved it, which
     * left the operand layout: K was a col_major matrix_b, and rocwmma has to
     * gather that. With Q transposed the score product becomes
     * `scoresT = KV . Q^T`, both operands row-major, worth 31% of the kernel on
     * the indexed shape and 21% on the mixed window one.
     *
     * The score tile is then [kv row][head] rather than [head][kv row]; every
     * reader below indexes accordingly. */
    constexpr uint32_t QT_PITCH = HEADS + 2u;
    __shared__ half qt_half[DIM * QT_PITCH];
    __shared__ half kv_half[ROWS * LDS_DIM];
    __shared__ float scores[HEADS * ROWS];
    /* Second half of the K split; group 0 writes straight into `scores`. */
    /* Two groups, not four. Four measured -4.4% on this kernel in isolation but
     * nothing end-to-end (433.1 against 433.2 tok/s), and its extra
     * reassociation of the score sum moved the prefill envelope from rmse 0.41
     * to 0.48 -- no longer clearly better than the 16d5e30 baseline's 0.478146.
     * Not worth the margin. */
    constexpr uint32_t QK_WG = 2u;
    __shared__ float qk_part[2u * (QK_WG - 1u) * BM * ROWS];
    __shared__ half probs[HEADS * ROWS];
    __shared__ float softmax_max[HEADS];
    __shared__ float softmax_den[HEADS];
    __shared__ float softmax_rescale[HEADS];

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0u) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    /* Row-table setup, off the single-thread path.
     *
     * This block used to run entirely on lane 0: a `top_k` loop of 512 *dependent*
     * global reads of `topk`, plus up to 256 serial `raw_rows` writes, once per
     * block -- and there are `n_tokens * n_head / 32` blocks. The window bounds
     * are pure scalar arithmetic, so every thread can derive them without help;
     * `raw_rows` then fills in parallel; and the top-k reads become one coalesced
     * pass into scratch, leaving lane 0 only an LDS-local compaction with no
     * memory latency in the chain. The compaction order and the filter are
     * unchanged, so the row tables are identical. */
    uint32_t raw_count = 0u;
    uint32_t raw_first_idx = 0u;
    if (n_raw != 0u) {
        const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
        if (qpos >= first_raw_pos) {
            uint32_t lo = first_raw_pos;
            if (window != 0u && qpos + 1u > window) {
                const uint32_t wlo = qpos + 1u - window;
                if (wlo > lo) lo = wlo;
            }
            const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
            if (hi >= lo) {
                raw_first_idx = lo - first_raw_pos;
                raw_count = hi - lo + 1u;
                if (raw_count > 256u) raw_count = 256u;
            }
        }
    }
    for (uint32_t r = tid; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if constexpr (INDEXED) {
        /* `scores` is untouched until the first row block, so it doubles as the
         * candidate buffer: HEADS * ROWS is exactly the top-k cap. */
        uint32_t *cand = reinterpret_cast<uint32_t *>(scores);
        const uint32_t nk = top_k < DS4_ROCM_ATTENTION_WMMA_TOPK_CAP
                                ? top_k : DS4_ROCM_ATTENTION_WMMA_TOPK_CAP;
        for (uint32_t i = tid; i < nk; i += blockDim.x) {
            const int32_t ci = topk[(uint64_t)t * top_k + i];
            cand[i] = (ci >= 0 && (uint32_t)ci < n_comp &&
                       (uint32_t)ci < visible_comp)
                          ? (uint32_t)ci : UINT32_MAX;
        }
        __syncthreads();
        if (tid == 0u) {
            uint32_t comp_count = 0u;
            for (uint32_t i = 0u; i < nk; i++) {
                if (cand[i] != UINT32_MAX) comp_rows[comp_count++] = cand[i];
            }
            comp_count_s = comp_count;
        }
    } else if (tid == 0u) {
        comp_count_s = visible_comp;
    }
    __syncthreads();

    for (uint32_t j = tid; j < HEADS * DIM; j += blockDim.x) {
        const uint32_t h = j / DIM;
        const uint32_t d = j - h * DIM;
        qt_half[d * QT_PITCH + h] = __float2half(
            q[((uint64_t)t * n_head + head0 + h) * DIM + d]);
    }
    if (tid < HEADS) {
        softmax_max[tid] = sinks[head0 + tid];
        softmax_den[tid] = 1.0f;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<
        rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b_row = rocwmma::fragment<
        rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<
        rocwmma::accumulator, BM, BN, BK, float>;

    frag_c out0;
    frag_c out1;
    rocwmma::fill_fragment(out0, 0.0f);
    rocwmma::fill_fragment(out1, 0.0f);
    const uint32_t n_score = raw_count + comp_count_s;
    const float score_scale = rsqrtf((float)DIM);
    const uint32_t lane = tid & 31u;
    /* Accumulator element (row, col) for a wave32 16x16 F32 fragment on
     * gfx1151 is (2 * e + lane / 16, lane % 16), recovered with
     * `rocm/tools/wmma_acc_layout.cpp` rather than assumed. `out0` rows are
     * heads 0..15 of the group and `out1` rows heads 16..31, so a per-head
     * factor reaches the accumulator with register arithmetic alone. */
    const uint32_t acc_row_base = lane >> 4u;

    /* Single traversal of the KV rows.
     *
     * The two-pass form walked every row block twice -- once for the scores and
     * the online softmax statistics, once for the probabilities and PV -- and
     * staged, barriered and LDS-wrote the KV tile on both. The score cache
     * spared the second pass its QK matrix multiply but not the staging. Here
     * the running maximum is folded into the accumulator instead: each block
     * rescales the partial output by `exp(m_old - m_new)` before adding its own
     * contribution, and the denominator divides only at the end. Same algebra,
     * one traversal, and the score cache is no longer needed. */
    for (uint32_t row0 = 0u; row0 < n_score; row0 += ROWS) {
        const uint32_t nr = n_score - row0 < ROWS ? n_score - row0 : ROWS;
        /* Stage four values per lane, not one.
         *
         * One half per lane is a 2-byte LDS store: 32 lanes move 64 of the 128
         * bytes the LDS can retire per cycle, and adjacent lanes share banks. The
         * row pitch is a multiple of four halves and `d` is too, so a uint2 store
         * is always 8-byte aligned, and the F32 source is 16-byte aligned for a
         * float4 read. Same values, same locations, a quarter of the accesses. */
        constexpr uint32_t DIM4 = DIM / 4u;
        static_assert(LDS_DIM % 4u == 0u, "row pitch must allow 8-byte stores");
        for (uint32_t j = tid; j < ROWS * DIM4; j += blockDim.x) {
            const uint32_t rr = j / DIM4;
            const uint32_t d = (j - rr * DIM4) * 4u;
            uint2 packed = make_uint2(0u, 0u);
            if (rr < nr) {
                const uint32_t sr = row0 + rr;
                const half *src_h = nullptr;
                const float *src_f = nullptr;
                if (sr < raw_count) {
                    src_f = raw_kv + (uint64_t)raw_rows[sr] * DIM + d;
                } else {
                    uint32_t comp_row = sr - raw_count;
                    if constexpr (INDEXED) {
                        comp_row = comp_rows[comp_row];
                    }
                    if constexpr (COMP_F16) {
                        src_h = ((const half *)comp_kv) + (uint64_t)comp_row * DIM + d;
                    } else {
                        src_f = ((const float *)comp_kv) + (uint64_t)comp_row * DIM + d;
                    }
                }
                if (src_h != nullptr) {
                    packed = *reinterpret_cast<const uint2 *>(src_h);
                } else {
                    const float4 v4 = *reinterpret_cast<const float4 *>(src_f);
                    const __half2 lo = __floats2half2_rn(v4.x, v4.y);
                    const __half2 hi = __floats2half2_rn(v4.z, v4.w);
                    packed.x = *reinterpret_cast<const uint32_t *>(&lo);
                    packed.y = *reinterpret_cast<const uint32_t *>(&hi);
                }
            }
            *reinterpret_cast<uint2 *>(&kv_half[rr * LDS_DIM + d]) = packed;
        }
        __syncthreads();

        constexpr uint32_t QK_WAVES = 2u * QK_WG;
        constexpr uint32_t K_PER_WG = DIM / QK_WG;
        const uint32_t qk_head_block = wave / QK_WG;
        const uint32_t qk_group = wave % QK_WG;
        frag_c score_acc;
        if (wave < QK_WAVES) {
            rocwmma::fill_fragment(score_acc, 0.0f);
            for (uint32_t k0 = qk_group * K_PER_WG;
                 k0 < (qk_group + 1u) * K_PER_WG; k0 += BK) {
                frag_a ka;
                frag_b_row qb;
                rocwmma::load_matrix_sync(ka, kv_half + k0, LDS_DIM);
                rocwmma::load_matrix_sync(
                    qb, qt_half + k0 * QT_PITCH + qk_head_block * BM, QT_PITCH);
                rocwmma::mma_sync(score_acc, ka, qb, score_acc);
            }
            if (qk_group == 0u) {
                rocwmma::store_matrix_sync(scores + qk_head_block * BM,
                                            score_acc, HEADS,
                                            rocwmma::mem_row_major);
            } else {
                rocwmma::store_matrix_sync(
                    qk_part +
                        (qk_head_block * (QK_WG - 1u) + qk_group - 1u) * BM * ROWS,
                    score_acc, ROWS, rocwmma::mem_row_major);
            }
        }
        __syncthreads();
        for (uint32_t j = tid; j < 2u * BM * ROWS; j += blockDim.x) {
            const uint32_t b = j / (BM * ROWS);
            const uint32_t o = j - b * (BM * ROWS);
            float sum = 0.0f;
#pragma unroll
            for (uint32_t gq = 1u; gq < QK_WG; gq++) {
                sum += qk_part[(b * (QK_WG - 1u) + gq - 1u) * BM * ROWS + o];
            }
            const uint32_t r = o / BM;
            const uint32_t hl = o - r * BM;
            scores[r * HEADS + b * BM + hl] += sum;
        }
        __syncthreads();

        /* Advance the online statistics and publish this block's rescale.
         *
         * One wave per head, one row per lane. The obvious form walks the 16
         * rows on 32 of the block's 1024 threads with a carried dependency on
         * both the maximum and the denominator, so the whole workgroup waits on
         * a 16-deep serial chain of `expf`. Reducing across lanes instead needs
         * eight shuffles and keeps every wave busy. Rows beyond `nr` contribute
         * -inf to the maximum and 0 to the sum, so the tail needs no branch. */
        if (wave < HEADS) {
            const float s = lane < nr
                    ? scores[lane * HEADS + wave] * score_scale
                    : -INFINITY;
            float m_new = s;
            for (int off = 16; off > 0; off >>= 1) {
                m_new = fmaxf(m_new, __shfl_xor(m_new, off, 32));
            }
            const float m_old = softmax_max[wave];
            m_new = fmaxf(m_old, m_new);
            float block_sum = lane < nr ? expf(s - m_new) : 0.0f;
            for (int off = 16; off > 0; off >>= 1) {
                block_sum += __shfl_xor(block_sum, off, 32);
            }
            if (lane == 0u) {
                const float factor = expf(m_old - m_new);
                softmax_den[wave] = softmax_den[wave] * factor + block_sum;
                softmax_max[wave] = m_new;
                softmax_rescale[wave] = factor;
            }
            /* This wave already holds exp(score - m_new) for its own lane while
             * forming the block sum, which is exactly what the separate
             * probability pass recomputed -- so write it here and drop that pass
             * along with the barrier in front of it. Same value, same rounding. */
            if (lane < ROWS) {
                probs[wave * ROWS + lane] =
                    __float2half(lane < nr ? expf(s - m_new) : 0.0f);
            }
        }
        __syncthreads();

        if (row0 != 0u) {
#pragma unroll
            for (uint32_t e = 0u; e < out0.num_elements; e++) {
                const uint32_t r = 2u * e + acc_row_base;
                out0.x[e] *= softmax_rescale[r];
                out1.x[e] *= softmax_rescale[BM + r];
            }
        }

        if (wave < 32u) {
            frag_a p0;
            frag_a p1;
            frag_b_row vb;
            rocwmma::load_matrix_sync(p0, probs, ROWS);
            rocwmma::load_matrix_sync(p1, probs + BM * ROWS, ROWS);
            rocwmma::load_matrix_sync(vb, kv_half + wave * BN, LDS_DIM);
            rocwmma::mma_sync(out0, p0, vb, out0);
            rocwmma::mma_sync(out1, p1, vb, out1);
        }
        __syncthreads();
    }

    if (wave < 32u) {
        /* The denominator was deferred so that no block had to be revisited. */
#pragma unroll
        for (uint32_t e = 0u; e < out0.num_elements; e++) {
            const uint32_t r = 2u * e + acc_row_base;
            const float d0 = softmax_den[r];
            const float d1 = softmax_den[BM + r];
            if (d0 != 0.0f) out0.x[e] /= d0;
            if (d1 != 0.0f) out1.x[e] /= d1;
        }
        float *out = heads + ((uint64_t)t * n_head + head0) * DIM;
        rocwmma::store_matrix_sync(
            out + wave * BN, out0, DIM, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(
            out + BM * DIM + wave * BN,
            out1,
            DIM,
            rocwmma::mem_row_major);
    }
}
#endif
__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t raw_count = (window != 0u && t + 1u > window) ? window : t + 1u;
    uint32_t raw_start = t + 1u - raw_count;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[128];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    if (threadIdx.x == 0) {
        float den = expf(sinks[h] - max_s);
        for (uint32_t r = 0; r < raw_count; r++) {
            scores[r] = expf(scores[r] - max_s);
            den += scores[r];
        }
        denom = den;
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    __shared__ float scores[DS4_ROCM_ATTENTION_PREFILL_MIXED_SCORE_CAP];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;
    if (n_score > DS4_ROCM_ATTENTION_PREFILL_MIXED_SCORE_CAP) return;

    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kvrow = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
        float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            const float *kvrow = comp_kv + (uint64_t)c * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        bool valid = k <= t && (window == 0 || t - k < window);
        float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || ratio == 0) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t && (window == 0 || t - k < window)) s = row[k];
        } else {
            uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_mixed_softmax_tile_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t raw_tokens,
        uint32_t tile_start,
        uint32_t tile_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= tile_tokens || ratio == 0) return;
    const uint32_t global_t = tile_start + t;
    float *row = scores + ((uint64_t)h * tile_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (global_t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < raw_tokens) {
            if (k <= global_t && (window == 0 || global_t - k < window)) s = row[k];
        } else {
            uint32_t c = k - raw_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)global_t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t r = gid / head_dim;
    dst[gid] = r < n_tokens ? raw_kv[(uint64_t)r * head_dim + d]
                             : comp_kv[(uint64_t)(r - n_tokens) * head_dim + d];
}

__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint64_t q = gid / head_dim;
    uint32_t h = q % n_head;
    uint32_t t = q / n_head;
    heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
}

__global__ static void attention_pack_group_heads_f16_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * group_dim;
    if (gid >= n) return;
    uint32_t d = gid % group_dim;
    uint64_t q = gid / group_dim;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    dst[gid] = __float2half(heads[((uint64_t)t * n_groups + g) * group_dim + d]);
}

/* Four elements per thread.
 *
 * The pack is a (token, group) to (group, token) block permutation; both sides
 * stay contiguous in `d`, so the only thing costing bandwidth in the scalar form
 * is that a wave stores 64 B at a time. A `float4` load and paired-half stores
 * make both sides full cache lines. Use hip_launch_attention_pack_group_heads_f16
 * rather than calling either kernel directly. */
__global__ static void attention_pack_group_heads_f16_vec4_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim) {
    const uint64_t g4 = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t groups4 =
        (uint64_t)n_groups * n_tokens * (group_dim >> 2u);
    if (g4 >= groups4) return;
    const uint32_t d4 = (uint32_t)(g4 % (group_dim >> 2u));
    const uint64_t q = g4 / (group_dim >> 2u);
    const uint32_t t = (uint32_t)(q % n_tokens);
    const uint32_t g = (uint32_t)(q / n_tokens);
    const uint32_t d = d4 << 2u;
    const float4 v = *(const float4 *)(
        heads + ((uint64_t)t * n_groups + g) * group_dim + d);
    __half *out = dst + q * group_dim + d;
    *(__half2 *)out = __floats2half2_rn(v.x, v.y);
    *(__half2 *)(out + 2u) = __floats2half2_rn(v.z, v.w);
}

/* Inverse rotated tail folded into the group pack.
 *
 * The heads tensor is written by attention, rewritten in place by the inverse
 * rope, then read again here and converted to F16 -- three passes over 512 MiB
 * per layer for one rotation. Nothing else reads the roped F32, so the rotation
 * can happen in registers between this kernel's load and its store, which
 * removes a whole read-modify-write pass. The rope pairs are adjacent
 * (`tail[i]`, `tail[i + 1]` for `i = 2 * pair`), so one float4 covers exactly
 * two consecutive pairs whenever head_dim - n_rot is a multiple of four.
 *
 * Same expression and the same F32 inputs as rope_tail_kernel; the F32
 * intermediate the separate pass stored is exactly representable, so the F16
 * result matches it. */
__global__ static void attention_pack_group_heads_rope_f16_vec4_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint64_t g4 = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t groups4 =
        (uint64_t)n_groups * n_tokens * (group_dim >> 2u);
    if (g4 >= groups4) return;
    const uint32_t d4 = (uint32_t)(g4 % (group_dim >> 2u));
    const uint64_t q = g4 / (group_dim >> 2u);
    const uint32_t t = (uint32_t)(q % n_tokens);
    const uint32_t g = (uint32_t)(q / n_tokens);
    const uint32_t d = d4 << 2u;
    float4 v = *(const float4 *)(
        heads + ((uint64_t)t * n_groups + g) * group_dim + d);

    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t dh = (g * group_dim + d) % head_dim;
    if (dh >= n_nope) {
        float corr0 = 0.0f, corr1 = 0.0f;
        if (ext_factor != 0.0f) {
            const float denom = 2.0f * logf(freq_base);
            corr0 = floorf((float)n_rot *
                           logf((float)n_ctx_orig /
                                (beta_fast * 2.0f * (float)M_PI)) / denom);
            corr1 = ceilf((float)n_rot *
                          logf((float)n_ctx_orig /
                               (beta_slow * 2.0f * (float)M_PI)) / denom);
            corr0 = fmaxf(0.0f, corr0);
            corr1 = fminf((float)(n_rot - 1), corr1);
        }
        const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
        float *lane = &v.x;
#pragma unroll
        for (uint32_t half_pair = 0u; half_pair < 2u; half_pair++) {
            const uint32_t pair = (dh - n_nope) / 2u + half_pair;
            const uint32_t i = pair * 2u;
            const float theta_extrap =
                (float)(pos0 + t) * powf(theta_scale, (float)pair);
            const float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp;
            float mscale = attn_factor;
            if (ext_factor != 0.0f) {
                const float ramp_mix =
                    rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
                theta = theta_interp * (1.0f - ramp_mix) +
                        theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
            float c = cosf(theta) * mscale;
            float sn = sinf(theta) * mscale;
            if (inverse) sn = -sn;
            const float x0 = lane[half_pair * 2u];
            const float x1 = lane[half_pair * 2u + 1u];
            lane[half_pair * 2u] = x0 * c - x1 * sn;
            lane[half_pair * 2u + 1u] = x0 * sn + x1 * c;
        }
    }

    __half *out = dst + q * group_dim + d;
    *(__half2 *)out = __floats2half2_rn(v.x, v.y);
    *(__half2 *)(out + 2u) = __floats2half2_rn(v.z, v.w);
}

struct ds4_attn_pack_rope {
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    int inverse;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
};

/* Whether the fused rope-plus-pack can serve this shape. The pack walks the
 * token's heads row as float4, so a rope pair must never straddle a vector and
 * the nope prefix must be vector aligned. */
static int ds4_attn_pack_rope_eligible(const ds4_attn_pack_rope *rope,
                                       uint32_t group_dim) {
    if (!rope) return 0;
    if (rope->head_dim == 0u || rope->n_rot == 0u) return 0;
    if (rope->n_rot > rope->head_dim) return 0;
    if ((rope->head_dim & 3u) != 0u) return 0;
    if ((rope->n_rot & 3u) != 0u) return 0;
    if (((rope->head_dim - rope->n_rot) & 3u) != 0u) return 0;
    if (group_dim == 0u || (group_dim % rope->head_dim) != 0u) return 0;
    return 1;
}

static void hip_launch_attention_pack_group_heads_f16(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim,
        uint64_t count) {
    if (count == 0u) return;
    if ((group_dim & 3u) == 0u &&
        ((uintptr_t)heads & 15u) == 0u &&
        ((uintptr_t)dst & 7u) == 0u) {
        const uint64_t groups4 = count >> 2u;
        attention_pack_group_heads_f16_vec4_kernel<<<
                (uint32_t)((groups4 + 255u) / 256u), 256>>>(
                dst, heads, n_tokens, n_groups, group_dim);
        return;
    }
    attention_pack_group_heads_f16_kernel<<<
            (uint32_t)((count + 255u) / 256u), 256>>>(
            dst, heads, n_tokens, n_groups, group_dim);
}

/* Returns 0 when the fused form is unavailable, so the caller keeps the
 * separate rope_tail launch plus the plain pack. */
static int hip_launch_attention_pack_group_heads_rope_f16(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim,
        uint64_t count,
        const ds4_attn_pack_rope *rope) {
    if (count == 0u) return 0;
    if (!ds4_attn_pack_rope_eligible(rope, group_dim)) return 0;
    if ((group_dim & 3u) != 0u ||
        ((uintptr_t)heads & 15u) != 0u || ((uintptr_t)dst & 7u) != 0u) {
        return 0;
    }
    const uint64_t groups4 = count >> 2u;
    attention_pack_group_heads_rope_f16_vec4_kernel<<<
            (uint32_t)((groups4 + 255u) / 256u), 256>>>(
            dst, heads, n_tokens, n_groups, group_dim, rope->head_dim,
            rope->n_rot, rope->pos0, rope->n_ctx_orig, rope->inverse,
            rope->freq_base, rope->freq_scale, rope->ext_factor,
            rope->attn_factor, rope->beta_fast, rope->beta_slow);
    return 1;
}

__global__ static void attention_unpack_group_low_kernel(
        float *low,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t rank) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * rank;
    if (gid >= n) return;
    uint32_t r = gid % rank;
    uint64_t q = gid / rank;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    uint32_t low_dim = n_groups * rank;
    low[(uint64_t)t * low_dim + (uint64_t)g * rank + r] = tmp[gid];
}

__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_ROCM_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (visible_comp == 0 || n_tokens == 1u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
        for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            local_max = fmaxf(local_max, s);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float add = 0.0f;
                const float *kvrow = NULL;
                if (row < raw_count) {
                    kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                } else {
                    uint32_t c = row - raw_count;
                    add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                    if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                }
                float s = -INFINITY;
                if (kvrow) {
                    float dot = 0.0f;
                    for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                    for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                        dot += __shfl_down_sync(static_cast<MASK_T>(mask), dot, off, 8);
                    }
                    s = dot * scale + add;
                }
                if (qlane == 0) scores[row] = s;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_ROCM_ATTENTION_INDEXED_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if (threadIdx.x == 0) {
        for (uint32_t i = 0;
             i < top_k && comp_count < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP;
             i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                const float *kvrow = row < raw_count
                    ? raw_kv + (uint64_t)raw_rows[row] * head_dim
                    : comp_kv + (uint64_t)comp_rows[row - raw_count] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(static_cast<MASK_T>(mask), dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)comp_rows[c] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t s = 0; s < comp_count; s++) acc += comp_kv[(uint64_t)comp_rows[s] * head_dim + d] * scores[raw_count + s];
            oh[d] = acc / denom;
        }
    }
}

template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP>
__global__ static void attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * HEADS_PER_GROUP + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count_s;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * 128];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count_s = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        for (uint32_t i = 0;
             i < top_k && comp_count_s < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP;
             i++) {
            const int32_t ci = topk[(uint64_t)t * top_k + i];
            if (ci < 0) continue;
            const uint32_t c = (uint32_t)ci;
            if (c < n_comp && c < visible_comp) comp_rows[comp_count_s++] = c;
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t comp_count = comp_count_s;
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    float sum_s = valid_head ? 1.0f : 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = n_score - row0 < ROWS_PER_STAGE ? n_score - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(FULL_WARP_MASK, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_static_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ float4 kv_shared[4 * 128];

    const uint32_t raw_count = window != 0u && t + 1u > window ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    uint32_t comp_count = 0;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);

    /* Keep the fast heads8 staging, but use the same two-pass score/softmax
     * shape as the old-HIP warprows path.  The previous online recurrence was
     * close, but crossed greedy near-ties on long prompts. */
    __shared__ float scores[8 * 768];
    if (n_score > 768u) return;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            const float *qh = q + ((uint64_t)t * n_head + head) * head_dim;
            const float *kvf = (const float *)kv_shared;
            for (uint32_t rr = 0; rr < nr; rr++) {
                float dot = 0.0f;
#pragma unroll 16
                for (uint32_t d = lane; d < 512u; d += 32u) dot += qh[d] * kvf[(uint64_t)rr * 512u + d];
                dot = warp_sum_f32(dot);
                if (lane == 0u) scores[warp * 768u + row0 + rr] = dot * scale;
            }
        }
        __syncthreads();
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    if (valid_head) {
        const float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) max_s = fmaxf(max_s, score_row[i]);
        max_s = warp_max_f32(max_s);
        max_s = __shfl_sync(FULL_WARP_MASK, max_s, 0);
    }
    float den = 0.0f;
    if (valid_head) {
        float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) {
            float p = expf(score_row[i] - max_s);
            score_row[i] = p;
            den += p;
        }
        den = warp_sum_f32(den);
        den += expf(sinks[head] - max_s);
        den = __shfl_sync(FULL_WARP_MASK, den, 0);
    }

    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            const float *score_row = scores + warp * 768u;
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = den == 0.0f ? 0.0f : score_row[row0 + rr] / den;
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                o0.x += k0.x * p; o0.y += k0.y * p; o0.z += k0.z * p; o0.w += k0.w * p;
                o1.x += k1.x * p; o1.y += k1.y * p; o1.z += k1.z * p; o1.w += k1.w * p;
                o2.x += k2.x * p; o2.y += k2.y * p; o2.z += k2.z * p; o2.w += k2.w * p;
                o3.x += k3.x * p; o3.y += k3.y * p; o3.z += k3.z * p; o3.w += k3.w * p;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ float4 kv_shared[4 * 128];

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t comp_count = 0;
    if (n_comp != 0u) {
        if (n_tokens == 1u && ratio == 0u) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(FULL_WARP_MASK, score, 0);

                max_s = fmaxf(max_s, score);
            }
        }
        __syncthreads();
    }

    float sum_s = valid_head ? expf(sinks[head] - max_s) : 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(FULL_WARP_MASK, score, 0);

                const float row_scale = expf(score - max_s);
                sum_s += row_scale;
                o0.x += k0.x * row_scale; o0.y += k0.y * row_scale; o0.z += k0.z * row_scale; o0.w += k0.w * row_scale;
                o1.x += k1.x * row_scale; o1.y += k1.y * row_scale; o1.z += k1.z * row_scale; o1.w += k1.w * row_scale;
                o2.x += k2.x * row_scale; o2.y += k2.y * row_scale; o2.z += k2.z * row_scale; o2.w += k2.w * row_scale;
                o3.x += k3.x * row_scale; o3.y += k3.y * row_scale; o3.z += k3.z * row_scale; o3.w += k3.w * row_scale;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}
