#include "ds4_rocm_norm.hip.hpp"

/* One row of rms_norm_weight_kernel; `partial` is blockDim.x shared floats.
 * orow may alias xr. */
__device__ static void rms_norm_weight_row(float *orow, const float *xr, const float *w, uint32_t n,
                                           float eps, float *partial) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void rms_norm_weight_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partial[256];
    rms_norm_weight_row(out + (uint64_t)row * n, x + (uint64_t)row * n, w, n, eps, partial);
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0);

/* Head RMS norm and the rotated tail in one pass, with the row staged in LDS.
 *
 * Run as two kernels this touches global memory four times per row: read to
 * square, write the scaled head, read the tail back, write the rotated tail.
 * The row is `head_dim` floats -- 2 KiB at the 512 this model uses -- so it
 * fits beside the reduction scratch and the second read disappears entirely.
 *
 * Wide prefill uses this fused route under the model's logit error envelope.
 * Verification retains separate norm and rotation kernels to preserve scalar
 * rounding exactly. */
#define DS4_HEAD_ROPE_LDS_DIM 512u

__global__ static void head_rms_norm_rope_tail_lds_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
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
        float beta_slow,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    const uint32_t t = row / n_head;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float partial[256];
    __shared__ float rowbuf[DS4_HEAD_ROPE_LDS_DIM];
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = xr[i];
        rowbuf[i] = v;
        sum += v * v;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    const uint32_t n_nope = head_dim - n_rot;
    /* Scale the whole row, tail included, exactly as the standalone norm did.
     *
     * Scaling only the head and carrying `rowbuf[n_nope + i] * scale` into the
     * rotation below is algebraically the same, but `-ffast-math` is then free
     * to reassociate `(r * scale) * c` into `r * (scale * c)`, which moved the
     * pinned 128-token trajectory from 116/128 to 113/128 top-1 with worst rank
     * 7. Storing the scaled tail and loading it back keeps the rotation's
     * operands opaque to the optimizer, so this is bit-identical while still
     * saving the standalone norm's second full-row read. */
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        xr[i] = rowbuf[i] * scale;
    }
    __syncthreads();

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        const uint32_t i = pair * 2u;
        const float theta_extrap = (float)(pos0 + t) * powf(theta_scale, (float)pair);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        float *tail = xr + n_nope;
        const float x0 = tail[i];
        const float x1 = tail[i + 1];
        tail[i] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

struct ds4_rope_tail_args {
    uint32_t n_rot;
    uint32_t n_ctx_orig;
    int inverse;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
};

/* Rotates pair `pair` of a RoPE tail in place at position `pos`. */
__device__ __forceinline__ static void rope_tail_rotate_pair(
        float *tail, uint32_t pair, uint32_t pos, const ds4_rope_tail_args &a) {
    const uint32_t n_rot = a.n_rot;
    const uint32_t n_ctx_orig = a.n_ctx_orig;
    const int inverse = a.inverse;
    const float freq_base = a.freq_base;
    const float freq_scale = a.freq_scale;
    const float ext_factor = a.ext_factor;
    const float attn_factor = a.attn_factor;
    const float beta_fast = a.beta_fast;
    const float beta_slow = a.beta_slow;
    uint32_t i = pair * 2;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }

    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    float theta_extrap = (float)pos * powf(theta_scale, (float)pair);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;

    float x0 = tail[i];
    float x1 = tail[i + 1];
    tail[i] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

__global__ static void rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t pos_stride,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    if (gid >= pairs) return;
    uint32_t pair = gid % (n_rot / 2);
    uint32_t tmp = gid / (n_rot / 2);
    uint32_t h = tmp % n_head;
    uint32_t t = tmp / n_head;
    uint32_t n_nope = head_dim - n_rot;
    rope_tail_rotate_pair(x + ((uint64_t)t * n_head + h) * head_dim + n_nope,
                          pair, pos0 + t * pos_stride,
                          ds4_rope_tail_args{n_rot, n_ctx_orig, inverse, freq_base,
                                             freq_scale, ext_factor, attn_factor,
                                             beta_fast, beta_slow});
}

__device__ static float dsv4_e4m3fn_value_dev(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * dsv4_e4m3fn_value_dev(best);
}

/* With kRope the rotated tail follows in the same pass: each row's RoPE reads
 * the normalized values back after a barrier, exactly as the separate
 * rope_tail_kernel launch would. */
template <bool kRope>
__global__ static void head_rms_norm_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps,
                                            uint32_t pos0, ds4_rope_tail_args rope) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) xr[i] *= scale;
    if (!kRope) return;
    __syncthreads();
    /* One pair per thread, as in rope_tail_kernel. A strided pair loop let the
     * compiler hoist and contract the YaRN angle differently, which moved 13%
     * of the rotated values in YaRN layers. */
    if (threadIdx.x < rope.n_rot / 2u) {
        rope_tail_rotate_pair(xr + head_dim - rope.n_rot, threadIdx.x, pos0 + row / n_head, rope);
    }
}

/* RoPE tail of every head of one row, then the row's Q8_0 blocks, in one
 * pass. A 32-value block is either all rotated tail or all untouched head
 * (n_nope % 32 == 0); even lanes rotate their pair exactly as rope_tail_kernel
 * does before the block reads the values back. */
__global__ static void rope_tail_quantize_q8_0_kernel(
        float *x, int8_t *xq, float *xscale, uint32_t head_dim, uint32_t pos,
        ds4_rope_tail_args rope) {
    const uint32_t b = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint64_t i0 = (uint64_t)b * 32u;
    const uint32_t d0 = (uint32_t)(i0 % head_dim);
    const uint32_t n_nope = head_dim - rope.n_rot;
    if (d0 >= n_nope && (lane & 1u) == 0u) {
        rope_tail_rotate_pair(x + (i0 - d0) + n_nope, (d0 - n_nope + lane) / 2u, pos, rope);
    }
    __syncthreads();
    quantize_q8_0_block32(x[i0 + lane], lane, 32u, xq + i0, xscale + b);
}

/* FP8 (E4M3) scale of one 64-value KV group from each lane's |value|. Every
 * thread of the block calls it; `scratch` is this group's 64 shared floats. */
__device__ static float dsv4_fp8_kv_group_scale(float *scratch, uint32_t lane, float absv) {
    scratch[lane] = absv;
    __syncthreads();
    for (uint32_t stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    return exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
}

__device__ static float dsv4_fp8_kv_round(float v, float scale) {
    return dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
}

/* What single-token decode does with the normalized q/kv rows besides the
 * norm: q's Q8_0 form for q_b, and the KV row finished in place (RoPE tail,
 * FP8 rounding of the rest, F16-rounded store to the raw cache). */
struct dsv4_qkv_decode_tail {
    int8_t *q_q8;
    float *q_q8_scale;
    float *raw_row;
    uint32_t pos;
    ds4_rope_tail_args rope;
};

template <bool kDecode>
__global__ static void dsv4_qkv_rms_norm_rows_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        float eps,
        dsv4_qkv_decode_tail tail) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float y = xr[i] * scale * w[i];
        orow[i] = y;
        if (kDecode && which == 0u) {
            quantize_q8_0_block32(y, threadIdx.x & 31u, 32u, tail.q_q8 + (i & ~31u),
                                  tail.q_q8_scale + (i >> 5));
        }
    }
    if (!kDecode || which == 0u) return;

    const uint32_t n_nope = n - tail.rope.n_rot;
    __syncthreads();
    if (threadIdx.x < tail.rope.n_rot / 2u) {
        rope_tail_rotate_pair(orow + n_nope, threadIdx.x, tail.pos, tail.rope);
    }
    __shared__ float fp8_scratch[256];
    const uint32_t lane = threadIdx.x & 63u;
    for (uint32_t g0 = 0; g0 * 64u < n_nope; g0 += blockDim.x / 64u) {
        __syncthreads();
        const uint32_t d = g0 * 64u + threadIdx.x;
        const float v = d < n_nope ? orow[d] : 0.0f;
        const float fp8_scale = dsv4_fp8_kv_group_scale(
                fp8_scratch + (threadIdx.x & ~63u), lane, d < n_nope ? fabsf(v) : 0.0f);
        if (d < n_nope) orow[d] = dsv4_fp8_kv_round(v, fp8_scale);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        tail.raw_row[i] = f16_bits_to_f32(f32_to_f16_bits_hip_round(orow[i]));
    }
}

__device__ static float dsv4_e2m1fn_value_dev(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

__device__ static float dsv4_e2m1fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_value_dev(best);
}

__device__ static float model_ape_value_dev(const void *base, uint64_t offset, uint32_t type,
                                            uint32_t width, uint32_t row, uint32_t col) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[(uint64_t)row * width + col]);
    if (type == 8u) {
        const uint64_t row_bytes = ((uint64_t)width + 31u) / 32u * 34u;
        const unsigned char *blk = (const unsigned char *)p + (uint64_t)row * row_bytes + (uint64_t)(col >> 5) * 34u;
        const float d = q8_0_scale_scalar(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[col & 31u];
        return d * (float)q;
    }
    return ((const float *)p)[(uint64_t)row * width + col];
}

extern "C" int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    if (!hip_tensor_has_elems2(out, n, rows, sizeof(float)) ||
        !hip_tensor_has_elems2(x, n, rows, sizeof(float))) return 0;
    if (n == 0u || rows == 0u) return 1;
    /* The hot caller is the 4-way hyper-connection row, 16,384 floats wide. */
    if (n == 256u * 64u) {
      rms_norm_plain_regs_kernel<64u>
          <<<rows, 256>>>((float*)out->ptr, (const float*)x->ptr, n, rows, eps);
      return hip_ok(hipGetLastError(), "rms_norm_plain regs launch");
    }
    rms_norm_plain_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    return hip_ok(hipGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor* out,
                                             const ds4_gpu_tensor* x,
                                             uint32_t n, float eps) {
  return ds4_gpu_rms_norm_plain_rows_tensor(out, x, n, 1u, eps);
}
extern "C" int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    uint64_t weight_bytes = 0;
    if (!model_map || !hip_u64_mul_checked(n, sizeof(float), &weight_bytes) ||
        !hip_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !hip_tensor_has_f32(out, n) || !hip_tensor_has_f32(x, n)) return 0;
    if (n == 0u) return 1;
    const char *wptr = hip_model_range_ptr(model_map, weight_offset, weight_bytes, "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, 1, eps);
    return hip_ok(hipGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    uint64_t weight_bytes = 0;
    if (!model_map || !hip_u64_mul_checked(n, sizeof(float), &weight_bytes) ||
        !hip_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !hip_tensor_has_elems2(out, n, rows, sizeof(float)) ||
        !hip_tensor_has_elems2(x, n, rows, sizeof(float))) return 0;
    if (n == 0u || rows == 0u) return 1;
    const char *wptr = hip_model_range_ptr(model_map, weight_offset, weight_bytes, "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, rows, eps);
    return hip_ok(hipGetLastError(), "rms_norm_weight launch");
}

extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    uint64_t q_weight_bytes = 0, kv_weight_bytes = 0;
    if (!model_map || !hip_u64_mul_checked(q_n, sizeof(float), &q_weight_bytes) ||
        !hip_u64_mul_checked(kv_n, sizeof(float), &kv_weight_bytes) ||
        !hip_model_range_fits(model_size, q_weight_offset, q_weight_bytes) ||
        !hip_model_range_fits(model_size, kv_weight_offset, kv_weight_bytes) ||
        !hip_tensor_has_elems2(q_out, q_n, rows, sizeof(float)) ||
        !hip_tensor_has_elems2(q, q_n, rows, sizeof(float)) ||
        !hip_tensor_has_elems2(kv_out, kv_n, rows, sizeof(float)) ||
        !hip_tensor_has_elems2(kv, kv_n, rows, sizeof(float))) {
        return 0;
    }
    if ((q_n == 0u && kv_n == 0u) || rows == 0u) return 1;
    const float *q_w = (const float *)hip_model_range_ptr(model_map,
            q_weight_offset, q_weight_bytes, "q_rms_weight");
    const float *kv_w = (const float *)hip_model_range_ptr(model_map,
            kv_weight_offset, kv_weight_bytes, "kv_rms_weight");
    if (!q_w || !kv_w) return 0;
    dim3 grid(rows, 2u, 1u);
    dsv4_qkv_rms_norm_rows_kernel<false><<<grid, 256>>>(
            (float *)q_out->ptr,
            (const float *)q->ptr,
            q_w,
            q_n,
            (float *)kv_out->ptr,
            (const float *)kv->ptr,
            kv_w,
            kv_n,
            rows,
            eps,
            dsv4_qkv_decode_tail{});
    return hip_ok(hipGetLastError(), "dsv4 qkv rms norm rows launch");
}

extern "C" int ds4_gpu_dsv4_qkv_rms_norm_decode_tensor(
        ds4_gpu_tensor       *q_out,
        ds4_gpu_tensor       *q_q8,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_rot,
        uint32_t                pos,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps) {
    const uint64_t q_weight_bytes = (uint64_t)q_n * sizeof(float);
    const uint64_t kv_weight_bytes = (uint64_t)kv_n * sizeof(float);
    dsv4_qkv_decode_tail tail = {};
    if (!model_map || q_n == 0u || kv_n == 0u || q_n % 32u != 0u ||
        (n_rot & 1u) || n_rot > kv_n || n_rot / 2u > 256u || raw_row >= raw_cap ||
        !hip_model_range_fits(model_size, q_weight_offset, q_weight_bytes) ||
        !hip_model_range_fits(model_size, kv_weight_offset, kv_weight_bytes) ||
        !hip_tensor_has_f32(q_out, q_n) || !hip_tensor_has_f32(q, q_n) ||
        !hip_tensor_has_f32(kv_out, kv_n) || !hip_tensor_has_f32(kv, kv_n) ||
        !hip_tensor_has_elems2(raw_cache, raw_cap, kv_n, sizeof(float)) ||
        !hip_q8_0_act(q_q8, 1u, q_n / 32u, &tail.q_q8, &tail.q_q8_scale)) {
        return 0;
    }
    const float *q_w = (const float *)hip_model_range_ptr(model_map,
            q_weight_offset, q_weight_bytes, "q_rms_weight");
    const float *kv_w = (const float *)hip_model_range_ptr(model_map,
            kv_weight_offset, kv_weight_bytes, "kv_rms_weight");
    if (!q_w || !kv_w) return 0;
    tail.raw_row = (float *)raw_cache->ptr + (uint64_t)raw_row * kv_n;
    tail.pos = pos;
    tail.rope = ds4_rope_tail_args{n_rot, n_ctx_orig, 0, freq_base, freq_scale,
                                   ext_factor, attn_factor, beta_fast, beta_slow};
    dsv4_qkv_rms_norm_rows_kernel<true><<<dim3(1u, 2u, 1u), 256>>>(
            (float *)q_out->ptr, (const float *)q->ptr, q_w, q_n,
            (float *)kv_out->ptr, (const float *)kv->ptr, kv_w, kv_n, 1u, eps, tail);
    return hip_ok(hipGetLastError(), "dsv4 qkv rms norm decode launch");
}
extern "C" int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    uint64_t rows64 = 0;
    if (!hip_u64_mul_checked(n_tok, n_head, &rows64) || rows64 > UINT32_MAX ||
        !hip_tensor_has_elems3(x, n_tok, n_head, head_dim, sizeof(float))) return 0;
    if (rows64 == 0u || head_dim == 0u) return 1;
    head_rms_norm_kernel<false><<<(uint32_t)rows64, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, eps,
                                                         0u, ds4_rope_tail_args{});
    return hip_ok(hipGetLastError(), "head_rms_norm launch");
}
extern "C" int ds4_gpu_head_rms_norm_rope_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps) {
    uint64_t rows64 = 0;
    if (n_rot > head_dim || (n_rot & 1u) || n_rot / 2u > 256u ||
        !hip_u64_mul_checked(n_tok, n_head, &rows64) || rows64 > UINT32_MAX ||
        !hip_tensor_has_elems3(x, n_tok, n_head, head_dim, sizeof(float))) return 0;
    if (rows64 == 0u || head_dim == 0u) return 1;
    head_rms_norm_kernel<true><<<(uint32_t)rows64, 256>>>(
            (float *)x->ptr, n_tok, n_head, head_dim, eps, pos0,
            ds4_rope_tail_args{n_rot, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale,
                               ext_factor, attn_factor, beta_fast, beta_slow});
    return hip_ok(hipGetLastError(), "head_rms_norm_rope launch");
}
extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps) {
    uint64_t rows64 = 0;
    if (n_rot > head_dim || (n_rot & 1u) ||
        head_dim == 0u || head_dim > DS4_HEAD_ROPE_LDS_DIM ||
        !hip_u64_mul_checked(n_tok, n_head, &rows64) || rows64 > UINT32_MAX ||
        !hip_tensor_has_elems3(x, n_tok, n_head, head_dim, sizeof(float))) {
        return 0;
    }
    if (rows64 == 0u) return 1;
    // Restrict fusion to the qualified wide-prefill shapes.
    if (n_tok < DS4_ROCM_ATTENTION_WIDE_ROWS) return 0;
    head_rms_norm_rope_tail_lds_kernel<<<(uint32_t)rows64, 256>>>(
            (float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig,
            inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, eps);
    return hip_ok(hipGetLastError(), "head_rms_norm_rope_tail lds launch");
}
extern "C" int ds4_gpu_rope_tail_quantize_q8_0_tensor(ds4_gpu_tensor *x, ds4_gpu_tensor *xq8, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    const uint64_t n = (uint64_t)n_head * head_dim;
    int8_t *xq = NULL;
    float *xscale = NULL;
    if (n_rot > head_dim || (n_rot & 1u) || head_dim % 32u != 0u ||
        (head_dim - n_rot) % 32u != 0u || n == 0u || n / 32u > UINT32_MAX ||
        !hip_tensor_has_f32(x, n) || !hip_q8_0_act(xq8, 1u, n / 32u, &xq, &xscale)) {
        return 0;
    }
    rope_tail_quantize_q8_0_kernel<<<(uint32_t)(n / 32u), 32>>>(
            (float *)x->ptr, xq, xscale, head_dim, pos,
            ds4_rope_tail_args{n_rot, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale,
                               ext_factor, attn_factor, beta_fast, beta_slow});
    return hip_ok(hipGetLastError(), "rope_tail_quantize_q8_0 launch");
}

static int hip_rope_tail_stride_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t pos_stride, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!x || n_rot > head_dim || (n_rot & 1) || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    rope_tail_kernel<<<(pairs + 255) / 256, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, pos_stride, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return hip_ok(hipGetLastError(), "rope_tail launch");
}

extern "C" int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    return hip_rope_tail_stride_tensor(x, n_tok, n_head, head_dim, n_rot, pos0, 1u, n_ctx_orig, inverse, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}
