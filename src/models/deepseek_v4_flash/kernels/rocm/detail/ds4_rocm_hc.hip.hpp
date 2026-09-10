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

/* Fuse HC RMS scaling, mix projection, Sinkhorn split and weighted sum.
 * Each thread retains its columns from all four raw F32 streams, so the final
 * weighted sum needs no second read of the input. Projection weights stay in
 * L2.
 *
 * DeepSeek's official 0731 Block.hc_pre computes F.linear(x.float(), hc_fn)
 * and scales the result by the RMS inverse. Do not round normalized activations
 * to F16: that is an additional approximation in the separate F16 path, not
 * part of the official formula. This kernel receives F16 projection weights. */
struct alignas(16) ds4_hc_half8 {
    __half2 p[4];
};

template <uint32_t TOK, uint32_t TPT, uint32_t COLS>
__global__ static void hc4_norm_mix_split_weighted_sum_kernel(
        float *out,
        float *mix_out,
        float *split_out,
        const float *residual_hc,
        const __half *mix_w,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    constexpr uint32_t MIX_HC = 24u;
    constexpr uint32_t WAVES_PER_TOK = TPT / 32u;
    const uint32_t tl = threadIdx.x / TPT;
    const uint32_t lane = threadIdx.x - tl * TPT;
    const uint32_t wave = lane >> 5u;
    const uint32_t wlane = lane & 31u;
    const uint32_t row = blockIdx.x * TOK + tl;
    const int active = row < n_rows;
    const uint32_t t = active ? row : 0u;
    const uint32_t hc_dim = 4u * n_embd;
    const uint32_t col0 = lane * COLS;

    __shared__ float partial[TOK][WAVES_PER_TOK];
    __shared__ float wavered[TOK][WAVES_PER_TOK][MIX_HC];
    __shared__ float mixbuf[TOK][MIX_HC];
    __shared__ float splitbuf[TOK][MIX_HC];

    /* Four streams x COLS contiguous columns, as float4 loads. */
    float v[4][COLS];
    const float *rrow = residual_hc + (uint64_t)t * hc_dim + col0;
#pragma unroll
    for (uint32_t h = 0; h < 4u; h++) {
        const float *src = rrow + (uint64_t)h * n_embd;
#pragma unroll
        for (uint32_t c = 0; c < COLS; c += 4u) {
            const float4 q = *reinterpret_cast<const float4 *>(src + c);
            v[h][c + 0u] = q.x;
            v[h][c + 1u] = q.y;
            v[h][c + 2u] = q.z;
            v[h][c + 3u] = q.w;
        }
    }

    float sum = 0.0f;
#pragma unroll
    for (uint32_t h = 0; h < 4u; h++) {
#pragma unroll
        for (uint32_t c = 0; c < COLS; c++) sum += v[h][c] * v[h][c];
    }
    /* Shuffle inside the wave, then one combine over the wave totals. A 256-lane
     * LDS tree needs eight barriers and this workgroup is 32 waves wide, which
     * made the two reductions here cost more than the pass they replaced. */
    sum = warp_sum_f32(sum);
    if (wlane == 0u) partial[tl][wave] = sum;
    __syncthreads();
    float rowsum = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < WAVES_PER_TOK; w++) rowsum += partial[tl][w];
    const float nscale = rsqrtf(rowsum / (float)hc_dim + norm_eps);

    /* Apply RMS scaling after projecting the raw F32 row. Reduce one output
     * column before starting the next to limit live registers. */
    for (uint32_t j = 0; j < MIX_HC; j++) {
        const __half *wrow = mix_w + (uint64_t)j * hc_dim + col0;
        float a = 0.0f;
#pragma unroll
        for (uint32_t h = 0; h < 4u; h++) {
            const __half *wsrc = wrow + (uint64_t)h * n_embd;
#pragma unroll
            for (uint32_t c = 0; c < COLS; c += 8u) {
                const ds4_hc_half8 w8 =
                    *reinterpret_cast<const ds4_hc_half8 *>(wsrc + c);
#pragma unroll
                for (uint32_t p = 0; p < 4u; p++) {
                    const float2 wf = __half22float2(w8.p[p]);
                    a += wf.x * v[h][c + 2u * p];
                    a += wf.y * v[h][c + 2u * p + 1u];
                }
            }
        }
        const float s = warp_sum_f32(a);
        if (wlane == 0u) wavered[tl][wave][j] = s;
    }
    __syncthreads();
    if (lane < MIX_HC) {
        float s = 0.0f;
#pragma unroll
        for (uint32_t w = 0; w < WAVES_PER_TOK; w++) s += wavered[tl][w][lane];
        s *= nscale;
        mixbuf[tl][lane] = s;
        if (active) mix_out[(uint64_t)t * MIX_HC + lane] = s;
    }
    __syncthreads();
    if (lane == 0u) {
        hc4_split_one(splitbuf[tl], mixbuf[tl], scale, base, sinkhorn_iters, epsv);
    }
    __syncthreads();
    if (lane < MIX_HC && active) {
        split_out[(uint64_t)t * MIX_HC + lane] = splitbuf[tl][lane];
    }

    if (!active) return;
    const float s0 = splitbuf[tl][0], s1 = splitbuf[tl][1];
    const float s2 = splitbuf[tl][2], s3 = splitbuf[tl][3];
    float *orow = out + (uint64_t)t * n_embd + col0;
#pragma unroll
    for (uint32_t c = 0; c < COLS; c += 4u) {
        float4 q;
        q.x = v[0][c + 0u] * s0; q.x += v[1][c + 0u] * s1; q.x += v[2][c + 0u] * s2; q.x += v[3][c + 0u] * s3;
        q.y = v[0][c + 1u] * s0; q.y += v[1][c + 1u] * s1; q.y += v[2][c + 1u] * s2; q.y += v[3][c + 1u] * s3;
        q.z = v[0][c + 2u] * s0; q.z += v[1][c + 2u] * s1; q.z += v[2][c + 2u] * s2; q.z += v[3][c + 2u] * s3;
        q.w = v[0][c + 3u] * s0; q.w += v[1][c + 3u] * s1; q.w += v[2][c + 3u] * s2; q.w += v[3][c + 3u] * s3;
        *reinterpret_cast<float4 *>(orow + c) = q;
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
