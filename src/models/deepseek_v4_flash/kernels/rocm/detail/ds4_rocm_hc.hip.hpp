// DS4 ROCm hierarchical-composition (HC) split/sum/expand kernels.
//
// Included from ds4_rocm.hip.cpp in the same translation unit to preserve current
// static helper visibility and launch behavior.

__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}

__global__ static void hc_split_sinkhorn_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t n_rows, uint32_t sinkhorn_iters, float epsv) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    hc4_split_one(out + (uint64_t)row * 24, mix + (uint64_t)row * 24, scale, base, sinkhorn_iters, epsv);
}

__global__ static void hc_weighted_sum_kernel(float *out, const float *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_tokens;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint32_t t = gid / n_embd;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += x[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d] *
               w[(uint64_t)t * weight_stride_f32 + h];
    }
    out[(uint64_t)t * n_embd + d] = acc;
}

__global__ static void hc_expand_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) block_v += block_add[(uint64_t)t * n_embd + d];
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}

__global__ static void hc_expand_half_kernel(
        float *out_hc,
        const __half *block_out,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    const float block_v = __half2float(block_out[(uint64_t)t * n_embd + d]);
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}

__global__ static void hc_expand_add_half_kernel(
        float *out_hc,
        const float *block_out,
        const __half *block_add,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    const float block_v = block_out[(uint64_t)t * n_embd + d] +
                          __half2float(block_add[(uint64_t)t * n_embd + d]);
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}

__global__ static void hc_expand4_kernel(
        float *out_hc,
        const float *block_out,
        const float *residual_hc,
        const float *split,
        uint32_t n_embd,
        uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    const uint32_t t = gid / n_embd;
    const uint64_t td = (uint64_t)t * n_embd + d;
    const uint64_t hc_base = (uint64_t)t * 4u * n_embd + d;
    const float bv = block_out[td];
    const float r0 = residual_hc[hc_base + 0u * (uint64_t)n_embd];
    const float r1 = residual_hc[hc_base + 1u * (uint64_t)n_embd];
    const float r2 = residual_hc[hc_base + 2u * (uint64_t)n_embd];
    const float r3 = residual_hc[hc_base + 3u * (uint64_t)n_embd];
    const float *sp = split + (uint64_t)t * 24u;
    const float *post = sp + 4u;
    const float *comb = sp + 8u;
#pragma unroll
    for (uint32_t dst = 0; dst < 4u; dst++) {
        float acc = bv * post[dst];
        acc += comb[0u * 4u + dst] * r0;
        acc += comb[1u * 4u + dst] * r1;
        acc += comb[2u * 4u + dst] * r2;
        acc += comb[3u * 4u + dst] * r3;
        out_hc[hc_base + (uint64_t)dst * n_embd] = acc;
    }
}

/* Four adjacent embedding lanes per thread.
 *
 * The scalar form keeps eight concurrent streams in flight (four residual
 * reads, four expanded writes) at one dword each, and measured 46% of the
 * 240 GB/s DRAM ceiling. A wave already covers a full cache line per stream at
 * one dword per lane, so the miss is memory-level parallelism rather than
 * coalescing: sixteen bytes per lane per stream quadruples the bytes each
 * thread has outstanding. Per-element arithmetic and its order are unchanged,
 * so the output is bit-identical. */
__global__ static void hc_expand4_vec4_kernel(
        float *out_hc,
        const float *block_out,
        const float *residual_hc,
        const float *split,
        uint32_t n_embd,
        uint32_t n_tokens) {
    const uint32_t n_vec = n_embd >> 2u;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (uint64_t)n_tokens * n_vec) return;
    const uint32_t dv = (uint32_t)(gid % n_vec);
    const uint32_t t = (uint32_t)(gid / n_vec);
    const uint64_t td = (uint64_t)t * n_embd + (uint64_t)dv * 4u;
    const uint64_t hc_base = (uint64_t)t * 4u * n_embd + (uint64_t)dv * 4u;
    const float4 bv = *reinterpret_cast<const float4 *>(block_out + td);
    const float4 r0 = *reinterpret_cast<const float4 *>(residual_hc + hc_base + 0u * (uint64_t)n_embd);
    const float4 r1 = *reinterpret_cast<const float4 *>(residual_hc + hc_base + 1u * (uint64_t)n_embd);
    const float4 r2 = *reinterpret_cast<const float4 *>(residual_hc + hc_base + 2u * (uint64_t)n_embd);
    const float4 r3 = *reinterpret_cast<const float4 *>(residual_hc + hc_base + 3u * (uint64_t)n_embd);
    const float *sp = split + (uint64_t)t * 24u;
    const float *post = sp + 4u;
    const float *comb = sp + 8u;
#pragma unroll
    for (uint32_t dst = 0; dst < 4u; dst++) {
        const float p = post[dst];
        const float c0 = comb[0u * 4u + dst];
        const float c1 = comb[1u * 4u + dst];
        const float c2 = comb[2u * 4u + dst];
        const float c3 = comb[3u * 4u + dst];
        float4 acc;
        acc.x = bv.x * p; acc.x += c0 * r0.x; acc.x += c1 * r1.x; acc.x += c2 * r2.x; acc.x += c3 * r3.x;
        acc.y = bv.y * p; acc.y += c0 * r0.y; acc.y += c1 * r1.y; acc.y += c2 * r2.y; acc.y += c3 * r3.y;
        acc.z = bv.z * p; acc.z += c0 * r0.z; acc.z += c1 * r1.z; acc.z += c2 * r2.z; acc.z += c3 * r3.z;
        acc.w = bv.w * p; acc.w += c0 * r0.w; acc.w += c1 * r1.w; acc.w += c2 * r2.w; acc.w += c3 * r3.w;
        *reinterpret_cast<float4 *>(out_hc + hc_base + (uint64_t)dst * n_embd) = acc;
    }
}

__global__ static void hc_expand4_add_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        uint32_t n_embd,
        uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    const uint32_t t = gid / n_embd;
    const uint64_t td = (uint64_t)t * n_embd + d;
    const uint64_t hc_base = (uint64_t)t * 4u * n_embd + d;
    const float bv = block_out[td] + block_add[td];
    const float r0 = residual_hc[hc_base + 0u * (uint64_t)n_embd];
    const float r1 = residual_hc[hc_base + 1u * (uint64_t)n_embd];
    const float r2 = residual_hc[hc_base + 2u * (uint64_t)n_embd];
    const float r3 = residual_hc[hc_base + 3u * (uint64_t)n_embd];
    const float *sp = split + (uint64_t)t * 24u;
    const float *post = sp + 4u;
    const float *comb = sp + 8u;
#pragma unroll
    for (uint32_t dst = 0; dst < 4u; dst++) {
        float acc = bv * post[dst];
        acc += comb[0u * 4u + dst] * r0;
        acc += comb[1u * 4u + dst] * r1;
        acc += comb[2u * 4u + dst] * r2;
        acc += comb[3u * 4u + dst] * r3;
        out_hc[hc_base + (uint64_t)dst * n_embd] = acc;
    }
}

/* hc_expand4_add with the routed expert sum folded in.
 *
 * The routed MoE wrote the 6-way sum to a float buffer that this kernel then
 * read back, one 128 MiB round trip per layer for a reduction that fits in a
 * register. Bit-identical: both forms accumulate the same F16 slots in ascending
 * expert order in F32, and the buffer the separate form stored is F32, so the
 * round trip was lossless. */
__global__ static void hc_expand4_add_moesum_kernel(
        float *out_hc,
        const __half *down_h,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        uint32_t n_embd,
        uint32_t n_expert,
        uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    const uint32_t t = gid / n_embd;
    const uint64_t td = (uint64_t)t * n_embd + d;
    const uint64_t hc_base = (uint64_t)t * 4u * n_embd + d;
    float routed = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) {
        routed += __half2float(
            down_h[((uint64_t)t * n_expert + e) * n_embd + d]);
    }
    const float bv = routed + block_add[td];
    const float r0 = residual_hc[hc_base + 0u * (uint64_t)n_embd];
    const float r1 = residual_hc[hc_base + 1u * (uint64_t)n_embd];
    const float r2 = residual_hc[hc_base + 2u * (uint64_t)n_embd];
    const float r3 = residual_hc[hc_base + 3u * (uint64_t)n_embd];
    const float *sp = split + (uint64_t)t * 24u;
    const float *post = sp + 4u;
    const float *comb = sp + 8u;
#pragma unroll
    for (uint32_t dst = 0; dst < 4u; dst++) {
        float acc = bv * post[dst];
        acc += comb[0u * 4u + dst] * r0;
        acc += comb[1u * 4u + dst] * r1;
        acc += comb[2u * 4u + dst] * r2;
        acc += comb[3u * 4u + dst] * r3;
        out_hc[hc_base + (uint64_t)dst * n_embd] = acc;
    }
}

/* Four adjacent embedding lanes per thread; see hc_expand4_vec4_kernel. The
 * routed sum keeps its ascending-expert F32 accumulation per element, so this
 * stays bit-identical to both the scalar fused form and the separate one. */
__global__ static void hc_expand4_add_moesum_vec4_kernel(
        float *out_hc,
        const __half *down_h,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        uint32_t n_embd,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t n_vec = n_embd >> 2u;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (uint64_t)n_tokens * n_vec) return;
    const uint32_t dv = (uint32_t)(gid % n_vec);
    const uint32_t t = (uint32_t)(gid / n_vec);
    const uint32_t d0 = dv * 4u;
    const uint64_t td = (uint64_t)t * n_embd + d0;
    const uint64_t hc_base = (uint64_t)t * 4u * n_embd + d0;
    float4 routed = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t e = 0; e < n_expert; e++) {
        const __half *dp = down_h + ((uint64_t)t * n_expert + e) * n_embd + d0;
        routed.x += __half2float(dp[0]);
        routed.y += __half2float(dp[1]);
        routed.z += __half2float(dp[2]);
        routed.w += __half2float(dp[3]);
    }
    const float4 ba = *reinterpret_cast<const float4 *>(block_add + td);
    float4 bv;
    bv.x = routed.x + ba.x;
    bv.y = routed.y + ba.y;
    bv.z = routed.z + ba.z;
    bv.w = routed.w + ba.w;
    const float4 r0 = *reinterpret_cast<const float4 *>(residual_hc + hc_base + 0u * (uint64_t)n_embd);
    const float4 r1 = *reinterpret_cast<const float4 *>(residual_hc + hc_base + 1u * (uint64_t)n_embd);
    const float4 r2 = *reinterpret_cast<const float4 *>(residual_hc + hc_base + 2u * (uint64_t)n_embd);
    const float4 r3 = *reinterpret_cast<const float4 *>(residual_hc + hc_base + 3u * (uint64_t)n_embd);
    const float *sp = split + (uint64_t)t * 24u;
    const float *post = sp + 4u;
    const float *comb = sp + 8u;
#pragma unroll
    for (uint32_t dst = 0; dst < 4u; dst++) {
        const float p = post[dst];
        const float c0 = comb[0u * 4u + dst];
        const float c1 = comb[1u * 4u + dst];
        const float c2 = comb[2u * 4u + dst];
        const float c3 = comb[3u * 4u + dst];
        float4 acc;
        acc.x = bv.x * p; acc.x += c0 * r0.x; acc.x += c1 * r1.x; acc.x += c2 * r2.x; acc.x += c3 * r3.x;
        acc.y = bv.y * p; acc.y += c0 * r0.y; acc.y += c1 * r1.y; acc.y += c2 * r2.y; acc.y += c3 * r3.y;
        acc.z = bv.z * p; acc.z += c0 * r0.z; acc.z += c1 * r1.z; acc.z += c2 * r2.z; acc.z += c3 * r3.z;
        acc.w = bv.w * p; acc.w += c0 * r0.w; acc.w += c1 * r1.w; acc.w += c2 * r2.w; acc.w += c3 * r3.w;
        *reinterpret_cast<float4 *>(out_hc + hc_base + (uint64_t)dst * n_embd) = acc;
    }
}

__global__ static void hc_expand4_half_kernel(
        float *out_hc,
        const __half *block_out,
        const float *residual_hc,
        const float *split,
        uint32_t n_embd,
        uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    const uint32_t t = gid / n_embd;
    const uint64_t td = (uint64_t)t * n_embd + d;
    const uint64_t hc_base = (uint64_t)t * 4u * n_embd + d;
    const float bv = __half2float(block_out[td]);
    const float r0 = residual_hc[hc_base + 0u * (uint64_t)n_embd];
    const float r1 = residual_hc[hc_base + 1u * (uint64_t)n_embd];
    const float r2 = residual_hc[hc_base + 2u * (uint64_t)n_embd];
    const float r3 = residual_hc[hc_base + 3u * (uint64_t)n_embd];
    const float *sp = split + (uint64_t)t * 24u;
    const float *post = sp + 4u;
    const float *comb = sp + 8u;
#pragma unroll
    for (uint32_t dst = 0; dst < 4u; dst++) {
        float acc = bv * post[dst];
        acc += comb[0u * 4u + dst] * r0;
        acc += comb[1u * 4u + dst] * r1;
        acc += comb[2u * 4u + dst] * r2;
        acc += comb[3u * 4u + dst] * r3;
        out_hc[hc_base + (uint64_t)dst * n_embd] = acc;
    }
}

__global__ static void hc_expand4_add_half_kernel(
        float *out_hc,
        const float *block_out,
        const __half *block_add,
        const float *residual_hc,
        const float *split,
        uint32_t n_embd,
        uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    const uint32_t t = gid / n_embd;
    const uint64_t td = (uint64_t)t * n_embd + d;
    const uint64_t hc_base = (uint64_t)t * 4u * n_embd + d;
    const float bv = block_out[td] + __half2float(block_add[td]);
    const float r0 = residual_hc[hc_base + 0u * (uint64_t)n_embd];
    const float r1 = residual_hc[hc_base + 1u * (uint64_t)n_embd];
    const float r2 = residual_hc[hc_base + 2u * (uint64_t)n_embd];
    const float r3 = residual_hc[hc_base + 3u * (uint64_t)n_embd];
    const float *sp = split + (uint64_t)t * 24u;
    const float *post = sp + 4u;
    const float *comb = sp + 8u;
#pragma unroll
    for (uint32_t dst = 0; dst < 4u; dst++) {
        float acc = bv * post[dst];
        acc += comb[0u * 4u + dst] * r0;
        acc += comb[1u * 4u + dst] * r1;
        acc += comb[2u * 4u + dst] * r2;
        acc += comb[3u * 4u + dst] * r3;
        out_hc[hc_base + (uint64_t)dst * n_embd] = acc;
    }
}

__global__ static void hc_split_weighted_sum_fused_kernel(
        float *out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv) {
    uint32_t t = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();
    /* Four adjacent lanes per thread; see hc_expand4_vec4_kernel for why the
     * scalar form leaves bandwidth on the table. Same per-element sum over the
     * same four slots in the same order, so the output is bit-identical. */
    const uint64_t rbase = (uint64_t)t * 4u * n_embd;
    const uint64_t obase = (uint64_t)t * n_embd;
    if ((n_embd & 3u) == 0u) {
        const float s0 = sp[0], s1 = sp[1], s2 = sp[2], s3 = sp[3];
        for (uint32_t col = d * 4u; col < n_embd; col += blockDim.x * 4u) {
            const float4 v0 = *reinterpret_cast<const float4 *>(residual_hc + rbase + 0u * (uint64_t)n_embd + col);
            const float4 v1 = *reinterpret_cast<const float4 *>(residual_hc + rbase + 1u * (uint64_t)n_embd + col);
            const float4 v2 = *reinterpret_cast<const float4 *>(residual_hc + rbase + 2u * (uint64_t)n_embd + col);
            const float4 v3 = *reinterpret_cast<const float4 *>(residual_hc + rbase + 3u * (uint64_t)n_embd + col);
            float4 acc;
            acc.x = v0.x * s0; acc.x += v1.x * s1; acc.x += v2.x * s2; acc.x += v3.x * s3;
            acc.y = v0.y * s0; acc.y += v1.y * s1; acc.y += v2.y * s2; acc.y += v3.y * s3;
            acc.z = v0.z * s0; acc.z += v1.z * s1; acc.z += v2.z * s2; acc.z += v3.z * s3;
            acc.w = v0.w * s0; acc.w += v1.w * s1; acc.w += v2.w * s2; acc.w += v3.w * s3;
            *reinterpret_cast<float4 *>(out + obase + col) = acc;
        }
        return;
    }
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[rbase + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[obase + col] = acc;
    }
}

__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
        sum += acc * acc;
    }

    __shared__ float partial[256];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        const float v = out[(uint64_t)t * n_embd + col];
        norm_out[(uint64_t)t * n_embd + col] = v * norm_scale * norm_w[col];
    }
}

/* Collapse one layer's hyper-connection streams into the plain hidden state the
 * DSpark drafter consumes, writing into a strided slot of the fused feature
 * row.
 *
 * The drafter's main_proj expects each row's sampled layers concatenated, so
 * capture writes directly at the slot offset rather than building per-layer
 * buffers and gathering them later. The collapse is the unweighted mean over
 * streams: this is a feature extractor for the drafter, not the target's output
 * head, so it deliberately does not use the learned output HC weights. */
__global__ static void dspark_capture_features_kernel(
        float       *out,
        const float *hc,
        uint32_t     out_row_stride,
        uint32_t     n_embd,
        uint32_t     n_hc,
        uint32_t     n_rows) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_embd * n_rows;
    if (gid >= n) return;
    const uint32_t d = (uint32_t)(gid % n_embd);
    const uint32_t t = (uint32_t)(gid / n_embd);
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += hc[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d];
    }
    out[(uint64_t)t * out_row_stride + d] = acc / (float)n_hc;
}

/* Broadcast a plain hidden row into all hyper-connection streams.
 *
 * The DSpark drafter's fused target feature is a single 4096-wide vector, but a
 * stage's attention pre-path consumes hyper-connection form. Replicating the
 * vector across the streams is what lets the injected context row take the same
 * route through the stage as an ordinary token. */
__global__ static void dspark_repeat_hc_kernel(
        float       *out_hc,
        const float *x,
        uint32_t     n_embd,
        uint32_t     n_hc,
        uint32_t     n_rows) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_embd * n_hc * n_rows;
    if (gid >= n) return;
    const uint32_t d = (uint32_t)(gid % n_embd);
    const uint64_t rest = gid / n_embd;
    const uint32_t row = (uint32_t)(rest / n_hc);
    out_hc[gid] = x[(uint64_t)row * n_embd + d];
}
