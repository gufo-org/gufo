// DS4 ROCm routed-MoE quantization/device helpers and kernels.
//
// Included from the ROCm backend translation unit before
// ds4_rocm_moe_launch.hip.hpp so host policy/glue can keep using these static
// kernels directly.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <rocwmma/rocwmma.hpp>
#endif

#include "ds4_rocm_iq2_gate.hip.hpp"

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dev_dot_q4_K_q8_K_block(const hip_block_q4_K *x, const hip_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

__device__ static void dev_dot_q4_K_q8_K_block4(
        const hip_block_q4_K *x,
        const hip_block_q8_K *y0,
        const hip_block_q8_K *y1,
        const hip_block_q8_K *y2,
        const hip_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const hip_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t p = 0; p < n; p++) {
            if (!ys[p]) continue;
            summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
            isum[p] += (int)sc * dev_dot_q4_32(x->qs + byte_off, ys[p]->qs + j * 32u, shift);
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (ys[p]) acc[p] += ys[p]->d * xd * (float)isum[p] - ys[p]->d * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q4_K_q8_K_block8(
        const hip_block_q4_K *x,
        const hip_block_q8_K *y0,
        const hip_block_q8_K *y1,
        const hip_block_q8_K *y2,
        const hip_block_q8_K *y3,
        const hip_block_q8_K *y4,
        const hip_block_q8_K *y5,
        const hip_block_q8_K *y6,
        const hip_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const hip_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t p = 0; p < n; p++) {
            if (!ys[p]) continue;
            summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
            isum[p] += (int)sc * dev_dot_q4_32(x->qs + byte_off, ys[p]->qs + j * 32u, shift);
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (ys[p]) acc[p] += ys[p]->d * xd * (float)isum[p] - ys[p]->d * xmin * (float)summs[p];
    }
}

__device__ static float dev_dot_q2_K_q8_K_block(const hip_block_q2_K *x, const hip_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < ROCM_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}


__device__ static void dev_dot_q2_K_q8_K_block4(
        const hip_block_q2_K *x,
        const hip_block_q8_K *y0,
        const hip_block_q8_K *y1,
        const hip_block_q8_K *y2,
        const hip_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const hip_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < ROCM_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q2_K_q8_K_block8(
        const hip_block_q2_K *x,
        const hip_block_q8_K *y0,
        const hip_block_q8_K *y1,
        const hip_block_q8_K *y2,
        const hip_block_q8_K *y3,
        const hip_block_q8_K *y4,
        const hip_block_q8_K *y5,
        const hip_block_q8_K *y6,
        const hip_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const hip_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < ROCM_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(static_cast<MASK_T>(mask), v, offset, 16);
    }
    (void)lane16;
    return v;
}

__global__ static void q8_K_quantize_kernel(hip_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / ROCM_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * ROCM_QK_K;
    hip_block_q8_K *yb = out + (uint64_t)row * (in_dim / ROCM_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < ROCM_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < ROCM_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < ROCM_QK_K) yb->qs[tid] = 0;
        if (tid < ROCM_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < ROCM_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < ROCM_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const hip_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const hip_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const hip_block_iq2_xxs *gr = (const hip_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const hip_block_iq2_xxs *ur = (const hip_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < 256u; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[256] = sum;
    }
}

/*
 * Build the complete narrow-batch expert index in one workgroup.
 *
 * Pair threads build shared counts, then compute their stable rank among
 * earlier pairs. The block prefixes the 256 counts and writes both
 * deterministic pair order and the expert-tile map. This replaces memset plus
 * five tiny dependent kernels on C2-C8 without making every expert rescan the
 * pair list.
 */
__global__ static void moe_build_narrow_sorted_tiles_kernel(
        uint32_t *counts,
        uint32_t *offsets,
        uint32_t *sorted_pairs,
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t tile_m) {
    __shared__ uint32_t shared_experts[48];
    __shared__ uint32_t shared_counts[256];

    const uint32_t tid = threadIdx.x;
    const uint32_t expert = tid;
    shared_counts[expert] = 0u;
    if (tid < pair_count) {
        int32_t expert_i = selected[tid];
        if (expert_i < 0) expert_i = 0;
        shared_experts[tid] = (uint32_t)expert_i;
    }
    __syncthreads();

    if (tid < pair_count) {
        atomicAdd(&shared_counts[shared_experts[tid]], 1u);
    }
    __syncthreads();

    const uint32_t count = shared_counts[expert];
    counts[expert] = count;
    if (expert == 0u) {
        uint32_t pair_off = 0u;
        uint32_t tile_off = 0u;
        for (uint32_t e = 0; e < 256u; e++) {
            offsets[e] = pair_off;
            tile_offsets[e] = tile_off;
            pair_off += shared_counts[e];
            tile_off += (shared_counts[e] + tile_m - 1u) / tile_m;
        }
        offsets[256] = pair_off;
        tile_offsets[256] = tile_off;
        *tile_total = tile_off;
    }
    __syncthreads();

    if (tid < pair_count) {
        const uint32_t pair_expert = shared_experts[tid];
        uint32_t rank = 0u;
        for (uint32_t pair = 0; pair < tid; pair++) {
            rank += shared_experts[pair] == pair_expert;
        }
        sorted_pairs[offsets[pair_expert] + rank] = tid;
    }
    const uint32_t first_tile = tile_offsets[expert];
    const uint32_t n_tiles = (count + tile_m - 1u) / tile_m;
    for (uint32_t t = 0; t < n_tiles; t++) {
        tile_experts[first_tile + t] = expert;
        tile_starts[first_tile + t] = t * tile_m;
    }
}

/* Keep pair order stable inside each expert bucket.  The MoE WMMA kernels are
 * row-position sensitive enough that atomic append order changes logits. */
/* One block per expert, scanning the assignment list for its own pairs.
 *
 * This ran on a single thread per block, so all 256 blocks walked the whole
 * `pair_count` list serially: 24,576 dependent loads each at a 4,096-token
 * chunk, 539 us per call. Scanning in chunks of blockDim.x with a block-wide
 * stable prefix sum keeps the output byte-identical -- pair indices still land in
 * ascending order within each expert -- while cutting the serial depth by the
 * block width. */
__global__ static void moe_scatter_sorted_pairs_deterministic_kernel(
        uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const int32_t *selected,
        uint32_t pair_count) {
    const uint32_t expert = (uint32_t)blockIdx.x;
    if (expert >= 256u) return;
    extern __shared__ uint32_t scatter_scan[];
    const uint32_t tid = threadIdx.x;
    /* Contiguous range per thread, counted then written: thread t's range
     * precedes t+1's and each is walked in order, so the output is byte-identical
     * to the single-threaded form -- ascending pair indices within each expert --
     * with one block-wide prefix sum instead of a per-chunk one. */
    const uint32_t per = (pair_count + blockDim.x - 1u) / blockDim.x;
    const uint32_t lo = tid * per;
    const uint32_t hi = lo + per < pair_count ? lo + per : pair_count;
    uint32_t count = 0u;
    for (uint32_t pair = lo; pair < hi; pair++) {
        int32_t expert_i = selected[pair];
        if (expert_i < 0) expert_i = 0;
        if ((uint32_t)expert_i == expert) count++;
    }
    scatter_scan[tid] = count;
    __syncthreads();
    for (uint32_t off = 1u; off < blockDim.x; off <<= 1u) {
        const uint32_t add = tid >= off ? scatter_scan[tid - off] : 0u;
        __syncthreads();
        scatter_scan[tid] += add;
        __syncthreads();
    }
    uint32_t pos = offsets[expert] + scatter_scan[tid] - count;
    for (uint32_t pair = lo; pair < hi; pair++) {
        int32_t expert_i = selected[pair];
        if (expert_i < 0) expert_i = 0;
        if ((uint32_t)expert_i == expert) sorted_pairs[pos++] = pair;
    }
}

__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t block_m) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < 256u; e++) {
            tile_offsets[e] = sum;
            sum += (counts[e] + block_m - 1u) / block_m;
        }
        tile_offsets[256] = sum;
        *tile_total = sum;
    }
}

__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t block_m) {
    uint32_t e = threadIdx.x;
    if (e >= 256u) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}

__global__ static void moe_gate_up_mid_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const hip_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const hip_block_iq2_xxs *gr = (const hip_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const hip_block_iq2_xxs *ur = (const hip_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const hip_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

/*
 * Run two stable request-local tiles in one workgroup. Each 256-thread half
 * reuses each expert weight for up to four request-local rows. The sorted
 * tile index places matching weights in the same workgroup. This route is
 * selected only for six routed experts and does not materialize gate/up.
 */

__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const hip_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ hip_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const hip_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = hip_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = hip_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const hip_block_iq2_xxs *gr = (const hip_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const hip_block_iq2_xxs *ur = (const hip_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                            s_iq2_grid, s_iq2_signs);
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                            s_iq2_grid, s_iq2_signs);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row2048_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const hip_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ hip_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const hip_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = hip_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = hip_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const hip_block_iq2_xxs *gr = (const hip_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const hip_block_iq2_xxs *ur = (const hip_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const hip_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ hip_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const hip_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = hip_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = hip_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const hip_block_iq2_xxs *gr = (const hip_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const hip_block_iq2_xxs *ur = (const hip_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

__global__ static void moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const hip_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t pair_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= expert_mid_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const hip_block_iq2_xxs *gr = (const hip_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const hip_block_iq2_xxs *ur = (const hip_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const hip_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q4K_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const hip_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const hip_block_q4_K *gr = (const hip_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const hip_block_q4_K *ur = (const hip_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const hip_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
        up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q4K_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const hip_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ hip_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const hip_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const hip_block_q4_K *gr = (const hip_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const hip_block_q4_K *ur = (const hip_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_q4_K_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_q4K_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const hip_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ hip_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const hip_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const hip_block_q4_K *gr = (const hip_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const hip_block_q4_K *ur = (const hip_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate);
        dev_dot_q4_K_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp)
                  up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
              gate_out[off] = gate[p];
              up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] *
                           weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_down_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const hip_block_q2_K *wr = (const hip_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const hip_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_gate_up_mid_decode_q4K_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const hip_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const hip_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const hip_block_q4_K *gr = (const hip_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const hip_block_q4_K *ur = (const hip_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_q2K_decode_q8_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const hip_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row_lane = threadIdx.x >> 4u;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const hip_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ hip_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < 16u; rr++) {
        const uint32_t row = blockIdx.x * 256u + row_lane + rr * 16u;
        if (row >= expert_mid_dim) continue;
        const hip_block_q2_K *gr = (const hip_block_q2_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const hip_block_q2_K *ur = (const hip_block_q2_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 16u) {
            gate += dev_dot_q2_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q2_K_q8_K_block(ur + b, xqb + b);
        }
        gate = half_warp_sum_f32(gate, lane);
        up = half_warp_sum_f32(up, lane);
        if (lane == 0u) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_down_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
  const uint32_t token = blockIdx.y;
  out += (uint64_t)token * out_dim;
  selected += (uint64_t)token * 6u;
  midq += (uint64_t)token * 6u * midq_blocks;
  uint32_t lane = threadIdx.x & 7u;
  uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
  if (row >= out_dim)
    return;
  float total = 0.0f;
#pragma unroll 2
  for (uint32_t slot = 0; slot < 6u; slot++) {
    int32_t expert_i = selected[slot];
    if (expert_i < 0)
      expert_i = 0;
    const hip_block_q2_K* wr =
        (const hip_block_q2_K*)(down_base +
                                (uint64_t)(uint32_t)expert_i *
                                    down_expert_bytes +
                                (uint64_t)row * down_row_bytes);
    const hip_block_q8_K* xq = midq + (uint64_t)slot * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u)
      acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0)
      total += acc;
  }
  if (lane == 0)
    out[row] = total;
}

__global__ static void moe_down_q4K_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
  const uint32_t token = blockIdx.y;
  out += (uint64_t)token * out_dim;
  selected += (uint64_t)token * 6u;
  midq += (uint64_t)token * 6u * midq_blocks;
  uint32_t lane = threadIdx.x & 7u;
  uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
  if (row >= out_dim)
    return;
  float total = 0.0f;
#pragma unroll
  for (uint32_t slot = 0; slot < 6u; slot++) {
    int32_t expert_i = selected[slot];
    if (expert_i < 0)
      expert_i = 0;
    const hip_block_q4_K* wr =
        (const hip_block_q4_K*)(down_base +
                                (uint64_t)(uint32_t)expert_i *
                                    down_expert_bytes +
                                (uint64_t)row * down_row_bytes);
    const hip_block_q8_K* xq = midq + (uint64_t)slot * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u)
      acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0)
      total += acc;
  }
  if (lane == 0)
    out[row] = total;
}

__global__ static void moe_down_q4K_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const hip_block_q4_K *wr = (const hip_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const hip_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_q4K_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const hip_block_q4_K *wr = (const hip_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const hip_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_q4K_expert_tile4_row32_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ hip_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const hip_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const hip_block_q4_K *wr = (const hip_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_q4K_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ hip_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const hip_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const hip_block_q4_K *wr = (const hip_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const hip_block_q2_K *wr = (const hip_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const hip_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_expert_tile4_row32_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ hip_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const hip_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const hip_block_q2_K *wr = (const hip_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ hip_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const hip_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const hip_block_q2_K *wr = (const hip_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row32_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    __shared__ hip_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const hip_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const hip_block_q2_K *wr = (const hip_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[16] = {0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
        if (np > 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                     xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                     xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                     xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
        }
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row2048_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ hip_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const hip_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const hip_block_q2_K *wr = (const hip_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ hip_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const hip_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const hip_block_q2_K *wr = (const hip_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

__global__ static void moe_down_sorted_p2_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const hip_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t pair_count) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= out_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const hip_block_q2_K *wr = (const hip_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const hip_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
    out[gid] = acc;
}

/* SwiGLU epilogue for the MMQ routed gate/up pair.
 *
 * MMQ writes raw gate and up projections, so this applies the clamp, the SiLU
 * gate, and the router weight in one pass and emits the F16 mirror the wide Q2
 * down kernel reads. It runs only when MMQ could not fuse the epilogue itself. */
__global__ static void moe_mmq_swiglu_weighted_clamp_kernel(
        float *mid_out,
        __half *mid_out_h,
        const float *gate_buf,
        const float *up_buf,
        const float *weights,
        uint32_t expert_mid_dim,
        uint32_t n_tokens,
        uint32_t n_expert,
        float clamp) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_expert * expert_mid_dim;
    if (gid >= n) return;
    const uint64_t pair = gid / expert_mid_dim;
    const uint32_t tok = (uint32_t)(pair / n_expert);
    const uint32_t slot = (uint32_t)(pair - (uint64_t)tok * n_expert);
    float g = gate_buf[gid];
    float u = up_buf[gid];
    if (!isfinite(g)) g = 0.0f;
    if (!isfinite(u)) u = 0.0f;
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    const float w = weights[(uint64_t)tok * n_expert + slot];
    const float value = (g / (1.0f + expf(-g))) * u * w;
    if (mid_out) mid_out[gid] = value;
    if (mid_out_h) mid_out_h[gid] = __float2half(value);
}

__global__ static void moe_sum_f16_kernel(float *out, const __half *down_h, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += __half2float(down_h[((uint64_t)tok * n_expert + e) * out_dim + row]);
    out[gid] = acc;
}

__global__ static void moe_sum_f16x2_kernel(float *out, const __half *down_h, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t out_dim2 = out_dim >> 1u;
    const uint64_t n2 = (uint64_t)n_tokens * out_dim2;
    if (gid >= n2) return;
    const uint32_t tok = gid / out_dim2;
    const uint32_t row = (uint32_t)(gid - (uint64_t)tok * out_dim2) << 1u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint64_t off = ((uint64_t)tok * n_expert + e) * out_dim + row;
        const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(down_h + off));
        acc0 += v.x;
        acc1 += v.y;
    }
    const uint64_t out_off = (uint64_t)tok * out_dim + row;
    out[out_off] = acc0;
    out[out_off + 1u] = acc1;
}

__device__ __forceinline__ static void q2_K_scale_broadcast_w32(const unsigned char *blk, float *d, float *dmin) {
    float vd = 0.0f;
    float vm = 0.0f;
    if ((threadIdx.x & 31u) == 0u) {
        const uint16_t d_bits = (uint16_t)blk[80] | ((uint16_t)blk[81] << 8);
        const uint16_t dmin_bits = (uint16_t)blk[82] | ((uint16_t)blk[83] << 8);
        vd = dev_f16_to_f32(d_bits);
        vm = dev_f16_to_f32(dmin_bits);
    }
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    *d = __shfl(vd, 0, 32);
    *dmin = __shfl(vm, 0, 32);
#else
    *d = __shfl_sync(FULL_WARP_MASK, vd, 0, 32);
    *dmin = __shfl_sync(FULL_WARP_MASK, vm, 0, 32);
#endif
}

__device__ __forceinline__ static float q2_K_dequant_256_scaled_w32(
        const unsigned char *blk,
        uint32_t lane,
        uint32_t kk,
        float d,
        float dmin) {
    const unsigned char *sc = blk;
    const unsigned char *qs = blk + 16u;
    const uint32_t g = (lane >> 4u) + (kk << 1u);
    const uint32_t within = g & 7u;
    const uint32_t qi = (g >> 3u) * 32u + (within & 1u) * 16u + (lane & 15u);
    const uint32_t shift = (within >> 1u) * 2u;
    const float q = (float)((qs[qi] >> shift) & 3u);
    const float scale = (float)(sc[g] & 0x0fu);
    const float mn = (float)(sc[g] >> 4u);
    return d * scale * q - dmin * mn;
}

template <int BN, int BK>
__device__ __forceinline__ static void q2_K_dequant_tile_half_rowwise(
        __half *shB,
        const unsigned char *base,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 2u;
    for (uint32_t j = tid; j < (uint32_t)(BN * (BK / KG)); j += blockDim.x) {
        const uint32_t nn = j / (uint32_t)(BK / KG);
        const uint32_t kk0 = (j - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + nn;
        if (row < out_dim) {
            const unsigned char *blk = base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const float d = dev_f16_to_f32((uint16_t)blk[80] | ((uint16_t)blk[81] << 8));
            const float dm = dev_f16_to_f32((uint16_t)blk[82] | ((uint16_t)blk[83] << 8));
            const float s = (float)(blk[g] & 0x0fu);
            const float m = (float)(blk[g] >> 4u);
#pragma unroll
            for (uint32_t u = 0; u < KG; u++) {
                const uint32_t kk = kk0 + u;
                const float q = (float)((blk[16u + qbase + kk] >> shift) & 3u);
                shB[kk * (uint32_t)BN + nn] = __float2half(d * s * q - dm * m);
            }
        } else {
#pragma unroll
            for (uint32_t u = 0; u < KG; u++) {
                const uint32_t kk = kk0 + u;
                shB[kk * (uint32_t)BN + nn] = __float2half(0.0f);
            }
        }
    }
}

/* NFRAG-wide dequantizer over Q2_K blocks already staged in shared memory.
 *
 * The global-memory twin below re-reads the same 84-byte block on every one of
 * the sixteen BK steps inside a 256-value K slab, and each read is six scalar
 * sub-word loads from a two-byte-aligned block. Staging the slab once as
 * uint32 words and dequantizing from there leaves the arithmetic and the
 * emitted values identical. */
template <int BN, int BK, int NFRAG>
__device__ __forceinline__ static void q2_K_dequant_wide_tile_half_rowwise_staged(
        __half *shB,
        const uint32_t *raw_rows,
        uint32_t k0,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 4u;
    constexpr uint32_t RAW_DWORDS = 84u / sizeof(uint32_t);
    constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
    for (uint32_t j = tid; j < (uint32_t)NFRAG * UNITS_PER_TILE; j += blockDim.x) {
        const uint32_t tile = j / UNITS_PER_TILE;
        const uint32_t rem = j - tile * UNITS_PER_TILE;
        const uint32_t nn = rem / (uint32_t)(BK / KG);
        const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row_local = tile * (uint32_t)BN + nn;
        const unsigned char *blk = reinterpret_cast<const unsigned char *>(
                raw_rows + row_local * RAW_DWORDS);
        const uint32_t dm_bits = *reinterpret_cast<const uint32_t *>(blk + 80u);
        const float d = dev_f16_to_f32((uint16_t)dm_bits);
        const float dm = dev_f16_to_f32((uint16_t)(dm_bits >> 16u));
        const float s = (float)(blk[g] & 0x0fu);
        const float m = (float)(blk[g] >> 4u);
        const uint32_t qbits =
                *reinterpret_cast<const uint32_t *>(blk + 16u + qbase + kk0);
        const uint32_t q0 = (qbits >> shift) & 3u;
        const uint32_t q1 = (qbits >> (8u + shift)) & 3u;
        const uint32_t q2 = (qbits >> (16u + shift)) & 3u;
        const uint32_t q3 = (qbits >> (24u + shift)) & 3u;
        const float ds = d * s;
        const float dmm = dm * m;
        /* One 8-byte LDS store rather than two adjacent 4-byte ones. Split, the
         * compiler pairs them into a two-address form whose halves land in the
         * same bank; the same change in the IQ2 tile loader took its conflict
         * rate from 19.7% to 11.7%. `kk0` is a multiple of KG = 4, so `dst` is
         * always 8-byte aligned. Same bytes in the same order. */
        __half *dst = shB + tile * (uint32_t)(BK * BN) + nn * (uint32_t)BK + kk0;
        uint2 packed;
        packed.x = dev_pack_half2_bits(ds * (float)q0 - dmm, ds * (float)q1 - dmm);
        packed.y = dev_pack_half2_bits(ds * (float)q2 - dmm, ds * (float)q3 - dmm);
        *reinterpret_cast<uint2 *>(dst) = packed;
    }
}

template <int BN, int BK>
__device__ __forceinline__ static void q2_K_dequant_pair_tile_half_rowwise(
        __half *shB0,
        __half *shB1,
        const unsigned char *base,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 4u;
    constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
    for (uint32_t j = tid; j < 2u * UNITS_PER_TILE; j += blockDim.x) {
        const uint32_t tile = j / UNITS_PER_TILE;
        const uint32_t rem = j - tile * UNITS_PER_TILE;
        const uint32_t nn = rem / (uint32_t)(BK / KG);
        const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + tile * (uint32_t)BN + nn;
        __half *shB = tile == 0u ? shB0 : shB1;
        uint32_t v0 = 0u;
        uint32_t v1 = 0u;
        if (row < out_dim) {
            const unsigned char *blk = base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const float d = dev_f16_to_f32((uint16_t)blk[80] | ((uint16_t)blk[81] << 8));
            const float dm = dev_f16_to_f32((uint16_t)blk[82] | ((uint16_t)blk[83] << 8));
            const float s = (float)(blk[g] & 0x0fu);
            const float m = (float)(blk[g] >> 4u);
            const uint32_t qbits = *reinterpret_cast<const uint32_t *>(blk + 16u + qbase + kk0);
            const uint32_t q0 = (qbits >> shift) & 3u;
            const uint32_t q1 = (qbits >> (8u + shift)) & 3u;
            const uint32_t q2 = (qbits >> (16u + shift)) & 3u;
            const uint32_t q3 = (qbits >> (24u + shift)) & 3u;
            const float ds = d * s;
            const float dmm = dm * m;
            v0 = dev_pack_half2_bits(ds * (float)q0 - dmm,
                                     ds * (float)q1 - dmm);
            v1 = dev_pack_half2_bits(ds * (float)q2 - dmm,
                                     ds * (float)q3 - dmm);
        }
        __half *dst = shB + nn * (uint32_t)BK + kk0;
        /* One 8-byte store; see the note in the wide dequantizer. */
        *reinterpret_cast<uint2 *>(dst) = make_uint2(v0, v1);
    }
}

template <int BN, int BK>
__device__ __forceinline__ static void q2_K_dequant_dual_pair_tile_half_rowwise(
        __half *shB0g,
        __half *shB0u,
        __half *shB1g,
        __half *shB1u,
        const unsigned char *gate_base,
        const unsigned char *up_base,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 4u;
    constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
    for (uint32_t j = tid; j < 2u * UNITS_PER_TILE; j += blockDim.x) {
        const uint32_t tile = j / UNITS_PER_TILE;
        const uint32_t rem = j - tile * UNITS_PER_TILE;
        const uint32_t nn = rem / (uint32_t)(BK / KG);
        const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + tile * (uint32_t)BN + nn;
        __half *shBg = tile == 0u ? shB0g : shB1g;
        __half *shBu = tile == 0u ? shB0u : shB1u;
        uint32_t gv0 = 0u;
        uint32_t gv1 = 0u;
        uint32_t uv0 = 0u;
        uint32_t uv1 = 0u;
        if (row < out_dim) {
            const unsigned char *gb = gate_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const unsigned char *ub = up_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const float gd = dev_f16_to_f32((uint16_t)gb[80] | ((uint16_t)gb[81] << 8));
            const float gm = dev_f16_to_f32((uint16_t)gb[82] | ((uint16_t)gb[83] << 8));
            const float ud = dev_f16_to_f32((uint16_t)ub[80] | ((uint16_t)ub[81] << 8));
            const float um = dev_f16_to_f32((uint16_t)ub[82] | ((uint16_t)ub[83] << 8));
            const float gs = (float)(gb[g] & 0x0fu);
            const float gmn = (float)(gb[g] >> 4u);
            const float us = (float)(ub[g] & 0x0fu);
            const float umn = (float)(ub[g] >> 4u);
            const uint32_t gqbits = *reinterpret_cast<const uint32_t *>(gb + 16u + qbase + kk0);
            const uint32_t uqbits = *reinterpret_cast<const uint32_t *>(ub + 16u + qbase + kk0);
            const uint32_t gq0 = (gqbits >> shift) & 3u;
            const uint32_t gq1 = (gqbits >> (8u + shift)) & 3u;
            const uint32_t gq2 = (gqbits >> (16u + shift)) & 3u;
            const uint32_t gq3 = (gqbits >> (24u + shift)) & 3u;
            const uint32_t uq0 = (uqbits >> shift) & 3u;
            const uint32_t uq1 = (uqbits >> (8u + shift)) & 3u;
            const uint32_t uq2 = (uqbits >> (16u + shift)) & 3u;
            const uint32_t uq3 = (uqbits >> (24u + shift)) & 3u;
            const float gds = gd * gs;
            const float gmm = gm * gmn;
            const float uds = ud * us;
            const float umm = um * umn;
            gv0 = dev_pack_half2_bits(gds * (float)gq0 - gmm,
                                      gds * (float)gq1 - gmm);
            gv1 = dev_pack_half2_bits(gds * (float)gq2 - gmm,
                                      gds * (float)gq3 - gmm);
            uv0 = dev_pack_half2_bits(uds * (float)uq0 - umm,
                                      uds * (float)uq1 - umm);
            uv1 = dev_pack_half2_bits(uds * (float)uq2 - umm,
                                      uds * (float)uq3 - umm);
        }
        const uint32_t sj = nn * (uint32_t)BK + kk0;
        /* One 8-byte store each; see the note in the wide dequantizer. */
        *reinterpret_cast<uint2 *>(shBg + sj) = make_uint2(gv0, gv1);
        *reinterpret_cast<uint2 *>(shBu + sj) = make_uint2(uv0, uv1);
    }
}

__device__ __forceinline__ static float moe_silu_oldhip(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

__global__ __launch_bounds__(32) static void moe_gate_up_mid_q2K_rows_rpb1_w32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp,
        int store_gate_up) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x;
    const uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim || lane >= 32u) return;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    const unsigned char *gr = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const unsigned char *ur = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;

    float gate = 0.0f;
    float up = 0.0f;
    const uint32_t nb = expert_in_dim >> 8u;
    for (uint32_t b = 0; b < nb; b++) {
        const unsigned char *gb = gr + (uint64_t)b * 84u;
        const unsigned char *ub = ur + (uint64_t)b * 84u;
        float gd, gdmin, ud, udmin;
        q2_K_scale_broadcast_w32(gb, &gd, &gdmin);
        q2_K_scale_broadcast_w32(ub, &ud, &udmin);
        const uint64_t xbase = (uint64_t)b * 256u;
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; kk++) {
            const uint32_t i = lane + (kk << 5u);
            const float xv = xr[xbase + i];
            gate += q2_K_dequant_256_scaled_w32(gb, lane, kk, gd, gdmin) * xv;
            up += q2_K_dequant_256_scaled_w32(ub, lane, kk, ud, udmin) * xv;
        }
    }

    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (store_gate_up) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = moe_silu_oldhip(gate) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q2K_rows_w32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp,
        int store_gate_up) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    const unsigned char *gr = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const unsigned char *ur = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;

    float gate = 0.0f;
    float up = 0.0f;
    const uint32_t nb = expert_in_dim >> 8u;
    for (uint32_t b = 0; b < nb; b++) {
        const unsigned char *gb = gr + (uint64_t)b * 84u;
        const unsigned char *ub = ur + (uint64_t)b * 84u;
        float gd, gdmin, ud, udmin;
        q2_K_scale_broadcast_w32(gb, &gd, &gdmin);
        q2_K_scale_broadcast_w32(ub, &ud, &udmin);
        const uint64_t xbase = (uint64_t)b * 256u;
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; kk++) {
            const uint32_t i = lane + (kk << 5u);
            const float xv = xr[xbase + i];
            gate += q2_K_dequant_256_scaled_w32(gb, lane, kk, gd, gdmin) * xv;
            up += q2_K_dequant_256_scaled_w32(ub, lane, kk, ud, udmin) * xv;
        }
    }

    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (store_gate_up) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = moe_silu_oldhip(gate) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_q2K_sum_rows_w32_kernel(
        float *out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint32_t n_tokens,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens) return;

    float acc = 0.0f;
    const uint32_t nb = expert_mid_dim >> 8u;
#pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[(uint64_t)tok * 6u + slot];
        if (expert_i < 0) expert_i = 0;
        const unsigned char *dr = (const unsigned char *)down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes;
        const float *mr = mid + ((uint64_t)tok * 6u + slot) * expert_mid_dim;
        for (uint32_t b = 0; b < nb; b++) {
            const unsigned char *db = dr + (uint64_t)b * 84u;
            float d, dmin;
            q2_K_scale_broadcast_w32(db, &d, &dmin);
            const uint64_t mbase = (uint64_t)b * 256u;
#pragma unroll
            for (uint32_t kk = 0; kk < 8u; kk++) {
                const uint32_t i = lane + (kk << 5u);
                acc += q2_K_dequant_256_scaled_w32(db, lane, kk, d, dmin) * mr[mbase + i];
            }
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = acc;
}

template <uint32_t PAIR_TILE, bool OUT_F16=false>
__global__ static void moe_gate_up_mid_q2K_expert_batch_sharedx_kernel(
        float *mid_out,
        __half *mid_out_h,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const float *weights,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        uint32_t min_count,
        uint32_t max_count,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        float clamp) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t expert = blockIdx.y;
    if (expert >= 256u) return;
    const bool row_valid = row < expert_mid_dim;
    const uint32_t count = counts[expert];
    if (count == 0u || count < min_count || (max_count != 0u && count >= max_count)) return;
    const uint32_t first = offsets[expert];
    const unsigned char *grow = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)(row_valid ? row : 0u) * gate_row_bytes;
    const unsigned char *urow = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)(row_valid ? row : 0u) * gate_row_bytes;
    const uint32_t nb = expert_in_dim >> 8u;
    for (uint32_t p0 = 0; p0 < count; p0 += PAIR_TILE) {
        uint32_t pair[PAIR_TILE];
        float g_acc[PAIR_TILE];
        float u_acc[PAIR_TILE];
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            pair[u] = (p0 + u < count) ? pairs[first + p0 + u] : UINT32_MAX;
            g_acc[u] = 0.0f;
            u_acc[u] = 0.0f;
        }
        for (uint32_t b = 0; b < nb; b++) {
            const uint64_t xbase = (uint64_t)b * 256u;
            for (uint32_t j = tid; j < PAIR_TILE * 256u; j += blockDim.x) {
                const uint32_t u = j >> 8u;
                const uint32_t k = j & 255u;
                if (pair[u] != UINT32_MAX) {
                    const uint32_t tok = pair[u] / 6u;
                    shx[j] = x[(uint64_t)tok * expert_in_dim + xbase + k];
                } else {
                    shx[j] = 0.0f;
                }
            }
            __syncthreads();
            if (row_valid) {
                const unsigned char *gb = grow + (uint64_t)b * 84u;
                const unsigned char *ub = urow + (uint64_t)b * 84u;
                float gd, gdmin, ud, udmin;
                q2_K_scale_broadcast_w32(gb, &gd, &gdmin);
                q2_K_scale_broadcast_w32(ub, &ud, &udmin);
#pragma unroll
                for (uint32_t kk = 0; kk < 8u; kk++) {
                    const uint32_t i = lane + (kk << 5u);
                    const float gwv = q2_K_dequant_256_scaled_w32(gb, lane, kk, gd, gdmin);
                    const float uwv = q2_K_dequant_256_scaled_w32(ub, lane, kk, ud, udmin);
#pragma unroll
                    for (uint32_t u = 0; u < PAIR_TILE; u++) {
                        const float xv = shx[(u << 8u) + i];
                        g_acc[u] += gwv * xv;
                        u_acc[u] += uwv * xv;
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            g_acc[u] = warp_sum_f32(g_acc[u]);
            u_acc[u] = warp_sum_f32(u_acc[u]);
        }
        if (lane == 0u && row_valid) {
#pragma unroll
            for (uint32_t u = 0; u < PAIR_TILE; u++) {
                if (pair[u] != UINT32_MAX) {
                    float g = g_acc[u];
                    float upv = u_acc[u];
                    if (clamp > 1.0e-6f) {
                        if (g > clamp) g = clamp;
                        if (upv > clamp) upv = clamp;
                        if (upv < -clamp) upv = -clamp;
                    }
                    const float v = moe_silu_oldhip(g) * upv * weights[pair[u]];
                    if (OUT_F16) mid_out_h[(uint64_t)pair[u] * expert_mid_dim + row] = __float2half(v);
                    else mid_out[(uint64_t)pair[u] * expert_mid_dim + row] = v;
                }
            }
        }
    }
}

template <uint32_t PAIR_TILE, bool MID_F16=false, bool OUT_F16=false, bool SLOT_MAJOR=false>
__global__ static void moe_down_q2K_expert_batch_sharedmid_kernel(
        float *down_out,
        __half *down_out_h,
        const char *down_base,
        const float *mid,
        const __half *mid_h,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        uint32_t min_count,
        uint32_t max_count,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_tokens = 0u,
        const uint32_t *expert_indices = NULL) {
    extern __shared__ float shmid[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t expert =
        expert_indices ? expert_indices[blockIdx.y] : blockIdx.y;
    if (expert >= 256u) return;
    const bool row_valid = row < out_dim;
    const uint32_t count = counts[expert];
    if (count == 0u || count < min_count || (max_count != 0u && count >= max_count)) return;
    const uint32_t first = offsets[expert];
    const unsigned char *drow = (const unsigned char *)down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)(row_valid ? row : 0u) * down_row_bytes;
    const uint32_t nb = expert_mid_dim >> 8u;
    for (uint32_t p0 = 0; p0 < count; p0 += PAIR_TILE) {
        uint32_t pair[PAIR_TILE];
        float acc[PAIR_TILE];
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            pair[u] = (p0 + u < count) ? pairs[first + p0 + u] : UINT32_MAX;
            acc[u] = 0.0f;
        }
        for (uint32_t b = 0; b < nb; b++) {
            const uint64_t mbase = (uint64_t)b * 256u;
            for (uint32_t j = tid; j < PAIR_TILE * 256u; j += blockDim.x) {
                const uint32_t u = j >> 8u;
                const uint32_t k = j & 255u;
                if (pair[u] != UINT32_MAX) {
                    const uint64_t moff = (uint64_t)pair[u] * expert_mid_dim + mbase + k;
                    shmid[j] = MID_F16 ? __half2float(mid_h[moff]) : mid[moff];
                } else {
                    shmid[j] = 0.0f;
                }
            }
            __syncthreads();
            if (row_valid) {
                const unsigned char *db = drow + (uint64_t)b * 84u;
                float d, dmin;
                q2_K_scale_broadcast_w32(db, &d, &dmin);
#pragma unroll
                for (uint32_t kk = 0; kk < 8u; kk++) {
                    const uint32_t i = lane + (kk << 5u);
                    const float wv = q2_K_dequant_256_scaled_w32(db, lane, kk, d, dmin);
#pragma unroll
                    for (uint32_t u = 0; u < PAIR_TILE; u++) acc[u] += wv * shmid[(u << 8u) + i];
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) acc[u] = warp_sum_f32(acc[u]);
        if (lane == 0u && row_valid) {
#pragma unroll
            for (uint32_t u = 0; u < PAIR_TILE; u++) {
                if (pair[u] != UINT32_MAX) {
                    if (OUT_F16) {
                        uint64_t dst = (uint64_t)pair[u] * out_dim + row;
                        if (SLOT_MAJOR) {
                            const uint32_t tok = pair[u] / 6u;
                            const uint32_t slot = pair[u] - tok * 6u;
                            dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row;
                        }
                        down_out_h[dst] = __float2half(acc[u]);
                    } else {
                        uint64_t dst = (uint64_t)pair[u] * out_dim + row;
                        if (SLOT_MAJOR) {
                            const uint32_t tok = pair[u] / 6u;
                            const uint32_t slot = pair[u] - tok * 6u;
                            dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row;
                        }
                        down_out[dst] = acc[u];
                    }
                }
            }
        }
    }
}

/*
 * Compact exact verifier down projection.
 *
 * This keeps the retained path's arithmetic and storage boundary intact: one
 * wave accumulates one (token, routing-slot, output-row) dot in the same K
 * order as moe_down_q2K_expert_batch_sharedmid_kernel, rounds it through F16,
 * and leaves the existing moe_sum_f16x2_kernel to add slots in routing order.
 * Indexing the grid by the live pairs avoids launching against all 256 experts.
 */

/*
 * Narrow grouped float down projection.
 *
 * A tile contains up to four rows routed to the same expert. Singleton tiles
 * take the exact compact-pair path without LDS. Multi-pair tiles stage their
 * mid rows and dequantize each Q2_K weight once, while preserving each pair's
 * K order, warp reduction, and F16 output boundary.
 */

/*
 * One-launch form of the qualified request-local compact down projection.
 * Tiles retain their original request boundary, pair order, and 1/2/3/4-row
 * arithmetic; only the launch and final six-slot sum are shared.
 */

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
__device__ __forceinline__ static float iq2_xxs_dequant_256_direct(const unsigned char *blk, uint32_t i) {
    const uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    const uint16_t *q2 = reinterpret_cast<const uint16_t *>(blk + 2u);
    const uint32_t ib32 = i >> 5u;
    const uint32_t j = i & 31u;
    const uint32_t aux_g = (uint32_t)q2[ib32 * 4u + 0u] | ((uint32_t)q2[ib32 * 4u + 1u] << 16);
    const uint32_t aux_s = (uint32_t)q2[ib32 * 4u + 2u] | ((uint32_t)q2[ib32 * 4u + 3u] << 16);
    const uint32_t __half = j >> 4u;
    const uint32_t g = (j >> 3u) & 1u;
    const uint32_t ii = j & 7u;
    const uint32_t grid_idx = (aux_g >> (8u * ((__half << 1u) + g))) & 0xffu;
    const uint32_t sign_idx = (aux_s >> (14u * __half + 7u * g)) & 127u;
    const uint64_t grid = hip_iq2xxs_grid[grid_idx];
    const uint32_t signs = dev_unpack_iq2_signs(hip_ksigns_iq2xs[sign_idx]);
    float w = (float)((grid >> (8u * ii)) & 0xffu);
    if (signs & (1u << ii)) w = -w;
    return dev_f16_to_f32(d_bits) * (0.125f + 0.25f * (float)(aux_s >> 28u)) * w;
}

template <int BN, int BK>
__device__ __forceinline__ static void iq2_xxs_dequant_dual_pair_tile_half_rowwise(
        __half *shB0g,
        __half *shB0u,
        __half *shB1g,
        __half *shB1u,
        const unsigned char *gate_base,
        const unsigned char *up_base,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    constexpr uint32_t KG = 4u;
    constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
    for (uint32_t j = tid; j < 2u * UNITS_PER_TILE; j += blockDim.x) {
        const uint32_t tile = j / UNITS_PER_TILE;
        const uint32_t rem = j - tile * UNITS_PER_TILE;
        const uint32_t nn = rem / (uint32_t)(BK / KG);
        const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + tile * (uint32_t)BN + nn;
        __half *shBg = tile == 0u ? shB0g : shB1g;
        __half *shBu = tile == 0u ? shB0u : shB1u;
        uint32_t gv0 = 0u, gv1 = 0u, uv0 = 0u, uv1 = 0u;
        if (row < out_dim) {
            const unsigned char *gb = gate_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 66u;
            const unsigned char *ub = up_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 66u;
            const uint32_t kk_base = k0 & 255u;
            const float g0 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 0u);
            const float g1 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 1u);
            const float g2 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 2u);
            const float g3 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 3u);
            const float u0 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 0u);
            const float u1 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 1u);
            const float u2 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 2u);
            const float u3 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 3u);
            gv0 = dev_pack_half2_bits(g0, g1);
            gv1 = dev_pack_half2_bits(g2, g3);
            uv0 = dev_pack_half2_bits(u0, u1);
            uv1 = dev_pack_half2_bits(u2, u3);
        }
        const uint32_t sj = nn * (uint32_t)BK + kk0;
        /* One 8-byte store each; see the note in the wide dequantizer. */
        *reinterpret_cast<uint2 *>(shBg + sj) = make_uint2(gv0, gv1);
        *reinterpret_cast<uint2 *>(shBu + sj) = make_uint2(uv0, uv1);
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16, bool OUT_F16=false, bool X_F16=false>
__global__ static void moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel(
        float *mid_out,
        __half *mid_out_h,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const __half *x_h,
        const float *weights,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        float clamp) {
    extern __shared__ unsigned char raw_sh[];
    __half *shA = reinterpret_cast<__half *>(raw_sh);
    __half *shBg0 = shA + MTILES * BM * BK;
    __half *shBu0 = shBg0 + BK * BN;
    __half *shBg1 = shBu0 + BK * BN;
    __half *shBu1 = shBg1 + BK * BN;
    float *shCg0 = reinterpret_cast<float *>(shBu1 + BK * BN);
    float *shCu0 = shCg0 + MTILES * BM * BN;
    float *shCg1 = shCu0 + MTILES * BM * BN;
    float *shCu1 = shCg1 + MTILES * BM * BN;
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    if (expert >= 256u) return;
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * (2u * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];
    __shared__ uint32_t shPair[MTILES * BM];
    for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
        const uint32_t bucket_row = m_group0 + j;
        shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, __half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, __half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b bg0, bu0, bg1, bu1;
    frag_c accg0, accu0, accg1, accu1;
    if (wave < MTILES) {
        rocwmma::fill_fragment(accg0, 0.0f);
        rocwmma::fill_fragment(accu0, 0.0f);
        rocwmma::fill_fragment(accg1, 0.0f);
        rocwmma::fill_fragment(accu1, 0.0f);
    }

    const unsigned char *gew = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes;
    const unsigned char *uew = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes;
    for (uint32_t k0 = 0; k0 < expert_in_dim; k0 += BK) {
        if (X_F16) {
            for (uint32_t j = tid; j < MTILES * BM * (BK / 2); j += blockDim.x) {
                const uint32_t pair_row = j / (BK / 2);
                const uint32_t kk2 = j - pair_row * (BK / 2);
                const uint32_t pair = shPair[pair_row];
                uint32_t v = 0u;
                if (pair != UINT32_MAX) {
                    const uint32_t token = pair / 6u;
                    const uint64_t xoff = (uint64_t)token * expert_in_dim + k0 + kk2 * 2u;
                    v = *reinterpret_cast<const uint32_t *>(x_h + xoff);
                }
                *reinterpret_cast<uint32_t *>(shA + pair_row * BK + kk2 * 2u) = v;
            }
        } else {
            for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
                const uint32_t pair_row = j / BK;
                const uint32_t kk = j - pair_row * BK;
                const uint32_t pair = shPair[pair_row];
                shA[j] = pair != UINT32_MAX
                    ? __float2half(x[(uint64_t)(pair / 6u) * expert_in_dim + k0 + kk])
                    : __float2half(0.0f);
            }
        }
        iq2_xxs_dequant_dual_pair_tile_half_rowwise<BN, BK>(
                shBg0, shBu0, shBg1, shBu1, gew, uew, gate_row_bytes, n0, k0, expert_mid_dim, tid);
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(bg0, shBg0, BN);
            rocwmma::load_matrix_sync(bu0, shBu0, BN);
            rocwmma::load_matrix_sync(bg1, shBg1, BN);
            rocwmma::load_matrix_sync(bu1, shBu1, BN);
            rocwmma::mma_sync(accg0, a, bg0, accg0);
            rocwmma::mma_sync(accu0, a, bu0, accu0);
            rocwmma::mma_sync(accg1, a, bg1, accg1);
            rocwmma::mma_sync(accu1, a, bu1, accu1);
        }
        __syncthreads();
    }

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shCg0 + wave * BM * BN, accg0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu0 + wave * BM * BN, accu0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCg1 + wave * BM * BN, accg1, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu1 + wave * BM * BN, accu1, BN, rocwmma::mem_row_major);
    }
    __syncthreads();

    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row0 = n0 + nn;
            const uint32_t row1 = n0 + BN + nn;
            const float wt = weights[pair];
            if (row0 < expert_mid_dim) {
                float g = shCg0[j], u = shCu0[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row0] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row0] = v;
            }
            if (row1 < expert_mid_dim) {
                float g = shCg1[j], u = shCu1[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row1] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row1] = v;
            }
        }
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16, bool OUT_F16=false, bool X_F16=false>
__global__ static void moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel(
        float *mid_out,
        __half *mid_out_h,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const __half *x_h,
        const float *weights,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        float clamp) {
    extern __shared__ unsigned char raw_sh[];
    __half *shA = reinterpret_cast<__half *>(raw_sh);
    __half *shBg0 = shA + MTILES * BM * BK;
    __half *shBu0 = shBg0 + BK * BN;
    __half *shBg1 = shBu0 + BK * BN;
    __half *shBu1 = shBg1 + BK * BN;
    float *shCg0 = reinterpret_cast<float *>(shBu1 + BK * BN);
    float *shCu0 = shCg0 + MTILES * BM * BN;
    float *shCg1 = shCu0 + MTILES * BM * BN;
    float *shCu1 = shCg1 + MTILES * BM * BN;
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    if (expert >= 256u) return;
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * (2u * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];
    __shared__ uint32_t shPair[MTILES * BM];
    for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
        const uint32_t bucket_row = m_group0 + j;
        shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, __half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, __half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b bg0;
    frag_b bu0;
    frag_b bg1;
    frag_b bu1;
    frag_c accg0;
    frag_c accu0;
    frag_c accg1;
    frag_c accu1;
    if (wave < MTILES) {
        rocwmma::fill_fragment(accg0, 0.0f);
        rocwmma::fill_fragment(accu0, 0.0f);
        rocwmma::fill_fragment(accg1, 0.0f);
        rocwmma::fill_fragment(accu1, 0.0f);
    }

    const unsigned char *gew = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes;
    const unsigned char *uew = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes;
    for (uint32_t k0 = 0; k0 < expert_in_dim; k0 += BK) {
        if (X_F16) {
            for (uint32_t j = tid; j < MTILES * BM * (BK / 2); j += blockDim.x) {
                const uint32_t pair_row = j / (BK / 2);
                const uint32_t kk2 = j - pair_row * (BK / 2);
                const uint32_t pair = shPair[pair_row];
                uint32_t v = 0u;
                if (pair != UINT32_MAX) {
                    const uint32_t token = pair / 6u;
                    const uint64_t xoff = (uint64_t)token * expert_in_dim + k0 + kk2 * 2u;
                    v = *reinterpret_cast<const uint32_t *>(x_h + xoff);
                }
                *reinterpret_cast<uint32_t *>(shA + pair_row * BK + kk2 * 2u) = v;
            }
        } else {
            for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
                const uint32_t mt = j / (BM * BK);
                const uint32_t rem = j - mt * BM * BK;
                const uint32_t mm = rem / BK;
                const uint32_t kk = rem - mm * BK;
                const uint32_t pair = shPair[mt * BM + mm];
                if (pair != UINT32_MAX) {
                    const uint32_t token = pair / 6u;
                    shA[j] = __float2half(x[(uint64_t)token * expert_in_dim + k0 + kk]);
                } else {
                    shA[j] = __float2half(0.0f);
                }
            }
        }
        q2_K_dequant_dual_pair_tile_half_rowwise<BN, BK>(
                shBg0, shBu0, shBg1, shBu1, gew, uew, gate_row_bytes, n0, k0, expert_mid_dim, tid);
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(bg0, shBg0, BN);
            rocwmma::load_matrix_sync(bu0, shBu0, BN);
            rocwmma::load_matrix_sync(bg1, shBg1, BN);
            rocwmma::load_matrix_sync(bu1, shBu1, BN);
            rocwmma::mma_sync(accg0, a, bg0, accg0);
            rocwmma::mma_sync(accu0, a, bu0, accu0);
            rocwmma::mma_sync(accg1, a, bg1, accg1);
            rocwmma::mma_sync(accu1, a, bu1, accu1);
        }
        __syncthreads();
    }

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shCg0 + wave * BM * BN, accg0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu0 + wave * BM * BN, accu0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCg1 + wave * BM * BN, accg1, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu1 + wave * BM * BN, accu1, BN, rocwmma::mem_row_major);
    }
    __syncthreads();

    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row0 = n0 + nn;
            const uint32_t row1 = n0 + BN + nn;
            const float wt = weights[pair];
            if (row0 < expert_mid_dim) {
                float g = shCg0[j];
                float u = shCu0[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row0] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row0] = v;
            }
            if (row1 < expert_mid_dim) {
                float g = shCg1[j];
                float u = shCu1[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row1] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row1] = v;
            }
        }
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16>
__global__ static void moe_down_q2K_hotlist_wmma_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes) {
    extern __shared__ unsigned char raw_sh[];
    __half *shA = reinterpret_cast<__half *>(raw_sh);
    __half *shB = shA + MTILES * BM * BK;
    float *shC = reinterpret_cast<float *>(shB + BK * BN);
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    if (expert >= 256u) return;
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * BN;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, __half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, __half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b;
    frag_c acc;
    if (wave < MTILES) rocwmma::fill_fragment(acc, 0.0f);

    const unsigned char *dew = (const unsigned char *)down_base + (uint64_t)expert * down_expert_bytes;
    for (uint32_t k0 = 0; k0 < expert_mid_dim; k0 += BK) {
        for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
            const uint32_t mt = j / (BM * BK);
            const uint32_t rem = j - mt * BM * BK;
            const uint32_t mm = rem / BK;
            const uint32_t kk = rem - mm * BK;
            const uint32_t bucket_row = m_group0 + mt * BM + mm;
            if (bucket_row < count) {
                const uint32_t pair = pairs[first + bucket_row];
                shA[j] = __float2half(mid[(uint64_t)pair * expert_mid_dim + k0 + kk]);
            } else {
                shA[j] = __float2half(0.0f);
            }
        }
        q2_K_dequant_tile_half_rowwise<BN, BK>(
                shB, dew, down_row_bytes, n0, k0, out_dim, tid);
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(b, shB, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }

    if (wave < MTILES) rocwmma::store_matrix_sync(shC + wave * BM * BN, acc, BN, rocwmma::mem_row_major);
    __syncthreads();
    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t bucket_row = m_group0 + mt * BM + mm;
        const uint32_t row = n0 + nn;
        if (bucket_row < count && row < out_dim) {
            const uint32_t pair = pairs[first + bucket_row];
            down_out[(uint64_t)pair * out_dim + row] = shC[j];
        }
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16, bool MID_F16=false, bool OUT_F16=false, bool SLOT_MAJOR=false>
__global__ static void moe_down_q2K_hotlist_wmma_n2_kernel(
        float *down_out,
        __half *down_out_h,
        const char *down_base,
        const float *mid,
        const __half *mid_h,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_tokens = 0u) {
    extern __shared__ unsigned char raw_sh[];
    __half *shA = reinterpret_cast<__half *>(raw_sh);
    __half *shB0 = shA + MTILES * BM * BK;
    __half *shB1 = shB0 + BK * BN;
    /* C staging is dead until the K loop has retired every A/B read. */
    float *shC0 = reinterpret_cast<float *>(raw_sh);
    float *shC1 = shC0 + MTILES * BM * BN;
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    if (expert >= 256u) return;
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * (2u * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];
    __shared__ uint32_t shPair[MTILES * BM];
    for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
        const uint32_t bucket_row = m_group0 + j;
        shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, __half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, __half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b0;
    frag_b b1;
    frag_c acc0;
    frag_c acc1;
    if (wave < MTILES) {
        rocwmma::fill_fragment(acc0, 0.0f);
        rocwmma::fill_fragment(acc1, 0.0f);
    }

    const unsigned char *dew = (const unsigned char *)down_base + (uint64_t)expert * down_expert_bytes;
    for (uint32_t k0 = 0; k0 < expert_mid_dim; k0 += BK) {
        if (MID_F16) {
            for (uint32_t j = tid; j < MTILES * BM * (BK / 2); j += blockDim.x) {
                const uint32_t pair_row = j / (BK / 2);
                const uint32_t kk2 = j - pair_row * (BK / 2);
                const uint32_t pair = shPair[pair_row];
                uint32_t v = 0u;
                if (pair != UINT32_MAX) {
                    const uint64_t moff = (uint64_t)pair * expert_mid_dim + k0 + kk2 * 2u;
                    v = *reinterpret_cast<const uint32_t *>(mid_h + moff);
                }
                *reinterpret_cast<uint32_t *>(shA + pair_row * BK + kk2 * 2u) = v;
            }
        } else {
            for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
                const uint32_t mt = j / (BM * BK);
                const uint32_t rem = j - mt * BM * BK;
                const uint32_t mm = rem / BK;
                const uint32_t kk = rem - mm * BK;
                const uint32_t pair = shPair[mt * BM + mm];
                if (pair != UINT32_MAX) {
                    shA[j] = __float2half(mid[(uint64_t)pair * expert_mid_dim + k0 + kk]);
                } else {
                    shA[j] = __float2half(0.0f);
                }
            }
        }
        q2_K_dequant_pair_tile_half_rowwise<BN, BK>(
                shB0, shB1, dew, down_row_bytes, n0, k0, out_dim, tid);
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(b0, shB0, BN);
            rocwmma::load_matrix_sync(b1, shB1, BN);
            rocwmma::mma_sync(acc0, a, b0, acc0);
            rocwmma::mma_sync(acc1, a, b1, acc1);
        }
        __syncthreads();
    }

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shC0 + wave * BM * BN, acc0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shC1 + wave * BM * BN, acc1, BN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row0 = n0 + nn;
            const uint32_t row1 = n0 + BN + nn;
            const uint32_t tok = pair / 6u;
            const uint32_t slot = pair - tok * 6u;
            if (row0 < out_dim) {
                if (OUT_F16) {
                    uint64_t dst = (uint64_t)pair * out_dim + row0;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row0;
                    down_out_h[dst] = __float2half(shC0[j]);
                } else {
                    uint64_t dst = (uint64_t)pair * out_dim + row0;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row0;
                    down_out[dst] = shC0[j];
                }
            }
            if (row1 < out_dim) {
                if (OUT_F16) {
                    uint64_t dst = (uint64_t)pair * out_dim + row1;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row1;
                    down_out_h[dst] = __float2half(shC1[j]);
                } else {
                    uint64_t dst = (uint64_t)pair * out_dim + row1;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row1;
                    down_out[dst] = shC1[j];
                }
            }
        }
    }
}

/* Wide-N routed Q2_K down.
 *
 * The n2 kernel above is activation-traffic bound rather than compute or
 * dequantization bound: its 64-row mid tile is 92% of the bytes it touches, and
 * with 32 output columns per workgroup that tile is re-staged 128 times per
 * layer, once per column block.  Arithmetic intensity is about 30 FLOP/byte
 * where roughly 230 would be needed to saturate the matrix cores.
 *
 * This variant stages the same mid tile once and runs NFRAG column fragments
 * against it, halving (NFRAG=4) or quartering (NFRAG=8) the activation
 * re-reads.  The accumulators live in registers, and the epilogue writes them
 * back two fragments at a time through the same 8 KiB staging window the n2
 * kernel uses, so dynamic LDS and resident workgroups per CU are unchanged.
 * The K loop order, fragment shapes, and accumulation order per output element
 * are identical to the n2 kernel, so results stay bit-exact. */
template <int MTILES=4, int BM=16, int BN=16, int BK=16, int NFRAG=4,
          bool MID_F16=false, bool OUT_F16=false, bool SLOT_MAJOR=false>
__global__ static void moe_down_q2K_hotlist_wmma_wide_kernel(
        float *down_out,
        __half *down_out_h,
        const char *down_base,
        const float *mid,
        const __half *mid_h,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_tokens = 0u,
        /* When non-null, blockIdx.y indexes a compacted list of the
         * (hot expert, row group) pairs that actually have rows, packed as
         * (hot_index << 16) | row_group, and gridDim.z is 1. */
        const uint32_t *tile_map = nullptr) {
    extern __shared__ unsigned char raw_sh[];
    /* Two mid-tile buffers. Counters put this kernel at 9 to 18%
     * instruction-issue utilisation with 96 VGPRs and no scratch, so it is
     * waiting rather than working, and the mid tile is the only global read
     * left inside the K step once the weights are staged per slab. Fetching the
     * next step's tile into registers before this step's dequantize, barrier
     * and matrix ops lets that load retire underneath them. */
    constexpr uint32_t A_TILE = (uint32_t)(MTILES * BM * BK);
    __half *shA = reinterpret_cast<__half *>(raw_sh);
    __half *shB = shA + 2u * A_TILE;
    /* Raw Q2_K blocks for this workgroup's NFRAG * BN output rows, one 256-value
     * K slab at a time. shC still aliases the A and B staging area, which the K
     * loop has finished with by the time the epilogue runs, so the raw window
     * sits beyond both and is never overwritten. */
    constexpr uint32_t RAW_DWORDS = 84u / sizeof(uint32_t);
    constexpr uint32_t RAW_ROWS = (uint32_t)(NFRAG * BN);
    uint32_t *shW = reinterpret_cast<uint32_t *>(shB + (uint32_t)(NFRAG * BK * BN));
    /* MTILES*BM*(BK/2) uint32 over 32*MTILES threads is BM*(BK/2)/32 each,
     * independent of MTILES. */
    constexpr uint32_t MID_PRE = (uint32_t)(BM * (BK / 2) / 32);
    float *shC = reinterpret_cast<float *>(raw_sh);
    uint32_t hot_idx;
    uint32_t m_group0;
    if (tile_map) {
        const uint32_t packed = tile_map[blockIdx.y];
        hot_idx = packed >> 16u;
        m_group0 = (packed & 0xffffu) * MTILES * BM;
    } else {
        hot_idx = (uint32_t)blockIdx.z;
        m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    }
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    const uint32_t count = counts[expert];
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * (NFRAG * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];
    __shared__ uint32_t shPair[MTILES * BM];
    for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
        const uint32_t bucket_row = m_group0 + j;
        shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, __half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, __half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b[NFRAG];
    frag_c acc[NFRAG];
    if (wave < MTILES) {
#pragma unroll
        for (int f = 0; f < NFRAG; f++) rocwmma::fill_fragment(acc[f], 0.0f);
    }

    const unsigned char *dew = (const unsigned char *)down_base + (uint64_t)expert * down_expert_bytes;

    uint32_t pre[MID_PRE];
    const auto fetch_mid = [&](uint32_t k0, uint32_t *dst) {
#pragma unroll
        for (uint32_t u = 0; u < MID_PRE; u++) {
            const uint32_t j = tid + u * blockDim.x;
            const uint32_t pair_row = j / (BK / 2);
            const uint32_t kk2 = j - pair_row * (BK / 2);
            const uint32_t pair = shPair[pair_row];
            uint32_t v = 0u;
            if (pair != UINT32_MAX) {
                const uint64_t moff =
                    (uint64_t)pair * expert_mid_dim + k0 + kk2 * 2u;
                v = *reinterpret_cast<const uint32_t *>(mid_h + moff);
            }
            dst[u] = v;
        }
    };
    if (MID_F16) fetch_mid(0u, pre);
    uint32_t abuf = 0u;
    for (uint32_t kb = 0; kb < expert_mid_dim; kb += 256u) {
        for (uint32_t j = tid; j < RAW_ROWS * RAW_DWORDS; j += blockDim.x) {
            const uint32_t row_local = j / RAW_DWORDS;
            const uint32_t word = j - row_local * RAW_DWORDS;
            const uint32_t row = n0 + row_local;
            uint32_t v = 0u;
            if (row < out_dim) {
                const unsigned char *blk = dew + (uint64_t)row * down_row_bytes +
                                           (uint64_t)(kb >> 8u) * 84u;
                v = *reinterpret_cast<const uint32_t *>(
                        blk + word * sizeof(uint32_t));
            }
            shW[j] = v;
        }
        __syncthreads();

        for (uint32_t krel = 0; krel < 256u && kb + krel < expert_mid_dim;
             krel += BK) {
            const uint32_t k0 = kb + krel;
            uint32_t next[MID_PRE];
            if (MID_F16) {
#pragma unroll
                for (uint32_t u = 0; u < MID_PRE; u++) {
                    const uint32_t j = tid + u * blockDim.x;
                    const uint32_t pair_row = j / (BK / 2);
                    const uint32_t kk2 = j - pair_row * (BK / 2);
                    *reinterpret_cast<uint32_t *>(
                        shA + abuf * A_TILE + pair_row * BK + kk2 * 2u) = pre[u];
                }
                const uint32_t k_next = k0 + BK;
                if (k_next < expert_mid_dim) fetch_mid(k_next, next);
            } else {
                for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
                    const uint32_t mt = j / (BM * BK);
                    const uint32_t rem = j - mt * BM * BK;
                    const uint32_t mm = rem / BK;
                    const uint32_t kk = rem - mm * BK;
                    const uint32_t pair = shPair[mt * BM + mm];
                    if (pair != UINT32_MAX) {
                        shA[abuf * A_TILE + j] = __float2half(
                            mid[(uint64_t)pair * expert_mid_dim + k0 + kk]);
                    } else {
                        shA[abuf * A_TILE + j] = __float2half(0.0f);
                    }
                }
            }
            q2_K_dequant_wide_tile_half_rowwise_staged<BN, BK, NFRAG>(
                    shB, shW, krel, tid);
            __syncthreads();
            if (wave < MTILES) {
                rocwmma::load_matrix_sync(
                    a, shA + abuf * A_TILE + wave * BM * BK, BK);
#pragma unroll
                for (int f = 0; f < NFRAG; f++) {
                    rocwmma::load_matrix_sync(b[f], shB + f * (BK * BN), BN);
                    rocwmma::mma_sync(acc[f], a, b[f], acc[f]);
                }
            }
            if (MID_F16) {
#pragma unroll
                for (uint32_t u = 0; u < MID_PRE; u++) pre[u] = next[u];
            }
            abuf ^= 1u;
            __syncthreads();
        }
    }

    /* Page the accumulators out two fragments at a time so the C window stays
     * the same size the n2 kernel aliases. */
    /* Page the accumulators out one fragment at a time.
     *
     * Two at a time needed `2 * MTILES * BM * BN` floats, which made the C
     * window rather than the A/B staging the binding term and capped MTILES at
     * four for eight resident workgroups per CU. One fragment halves that
     * window, so MTILES can double without costing residency, and doubling the
     * row tile halves how often each weight column block is dequantized. The
     * epilogue writes the same values in the same order either way. */
#pragma unroll
    for (int f = 0; f < NFRAG; f++) {
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::store_matrix_sync(shC + wave * BM * BN, acc[f], BN, rocwmma::mem_row_major);
        }
        __syncthreads();
        for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
            const uint32_t mt = j / (BM * BN);
            const uint32_t rem = j - mt * BM * BN;
            const uint32_t mm = rem / BN;
            const uint32_t nn = rem - mm * BN;
            const uint32_t pair = shPair[mt * BM + mm];
            if (pair == UINT32_MAX) continue;
            const uint32_t tok = pair / 6u;
            const uint32_t slot = pair - tok * 6u;
            const uint32_t row = n0 + (uint32_t)f * BN + nn;
            if (row >= out_dim) continue;
            const float v = shC[j];
            uint64_t dst = (uint64_t)pair * out_dim + row;
            if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row;
            if (OUT_F16) down_out_h[dst] = __float2half(v);
            else down_out[dst] = v;
        }
    }
}
#endif

__device__ static float dev_iq2_xxs_dot_f32(const hip_block_iq2_xxs *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const hip_block_iq2_xxs *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const uint16_t *q2 = xb->qs;
        const float *xf = x + (uint64_t)b * ROCM_QK_K;
        for (uint32_t ib32 = 0; ib32 < ROCM_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            const uint8_t grids[4] = {
                (uint8_t)(aux_g & 0xffu),
                (uint8_t)((aux_g >> 8) & 0xffu),
                (uint8_t)((aux_g >> 16) & 0xffu),
                (uint8_t)((aux_g >> 24) & 0xffu),
            };
            for (uint32_t __half = 0; __half < 2; __half++) {
                for (uint32_t g = 0; g < 2; g++) {
                    const uint32_t gi = __half * 2 + g;
                    const uint64_t grid = hip_iq2xxs_grid[grids[gi]];
                    const uint8_t signs = hip_ksigns_iq2xs[(aux_s >> (14u * __half + 7u * g)) & 127u];
                    for (uint32_t i = 0; i < 8; i++) {
                        float w = (float)((grid >> (8u * i)) & 0xffu);
                        if (signs & (1u << i)) w = -w;
                        acc += dl * w * xf[ib32 * 32u + __half * 16u + g * 8u + i];
                    }
                }
            }
        }
    }
    return acc;
}

__device__ static float dev_q2_K_dot_f32(const hip_block_q2_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const hip_block_q2_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        for (uint32_t il = 0; il < 16; il++) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = xb->scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4);
            const uint8_t *q = xb->qs + 32u * chunk + 16u * pair;
            const float *xf = x + (uint64_t)b * ROCM_QK_K + chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16; i++) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * xf[i];
            }
        }
    }
    return acc;
}

__global__ static void moe_gate_up_mid_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / ROCM_QK_K;
    const hip_block_iq2_xxs *gr = (const hip_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const hip_block_iq2_xxs *ur = (const hip_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_iq2_xxs_dot_f32(gr + b, xr + (uint64_t)b * ROCM_QK_K, 1);
        up += dev_iq2_xxs_dot_f32(ur + b, xr + (uint64_t)b * ROCM_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q2K_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / ROCM_QK_K;
    const hip_block_q2_K *gr = (const hip_block_q2_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const hip_block_q2_K *ur = (const hip_block_q2_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_q2_K_dot_f32(gr + b, xr + (uint64_t)b * ROCM_QK_K, 1);
        up += dev_q2_K_dot_f32(ur + b, xr + (uint64_t)b * ROCM_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_f32_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t nb = expert_mid_dim / ROCM_QK_K;
    const hip_block_q2_K *wr = (const hip_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const float *xr = mid + (uint64_t)pair * expert_mid_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) acc += dev_q2_K_dot_f32(wr + b, xr + (uint64_t)b * ROCM_QK_K, 1);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}
