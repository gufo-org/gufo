// DS4 ROCm common embedding/dense-matmul kernels and device helpers.
//
// Included from ds4_rocm.hip.cpp before more specialized modules; these helpers are
// intentionally kept static in the single translation unit.

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

__device__ static float embed_q8_0_scale(const unsigned char *blk) {
    const uint16_t bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    return __half2float(__ushort_as_half((unsigned short)bits));
}

__global__ static void embed_token_hc_q8_0_kernel(
        float *out,
        const unsigned char *w,
        uint32_t token,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    const uint32_t blocks = (n_embd + 31u) / 32u;
    const uint32_t b = d >> 5u;
    const uint32_t j = d & 31u;
    const unsigned char *blk = w + ((uint64_t)token * blocks + b) * 34u;
    out[gid] = embed_q8_0_scale(blk) * (float)((const int8_t *)(blk + 2u))[j];
}

__global__ static void embed_tokens_hc_q8_0_kernel(
        float *out,
        const int32_t *tokens,
        const unsigned char *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    const uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    const uint32_t blocks = (n_embd + 31u) / 32u;
    const uint32_t b = d >> 5u;
    const uint32_t j = d & 31u;
    const unsigned char *blk = w + ((uint64_t)tok * blocks + b) * 34u;
    out[gid] = embed_q8_0_scale(blk) * (float)((const int8_t *)(blk + 2u))[j];
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

__device__ static float warp_sum_f32(float v);

__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = k0; i < k1; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[tok * out_dim + row] = total;
    }
}

__global__ static void matmul_f16_f32_sharedx_warp_rows_w32_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint32_t in_dim,
        uint64_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const __half *wr = w + row * (uint64_t)in_dim;
    float acc = 0.0f;
    uint32_t i = lane;
    for (; i + 224u < in_dim; i += 256u) {
        acc += __half2float(wr[i]) * shx[i];
        acc += __half2float(wr[i + 32u]) * shx[i + 32u];
        acc += __half2float(wr[i + 64u]) * shx[i + 64u];
        acc += __half2float(wr[i + 96u]) * shx[i + 96u];
        acc += __half2float(wr[i + 128u]) * shx[i + 128u];
        acc += __half2float(wr[i + 160u]) * shx[i + 160u];
        acc += __half2float(wr[i + 192u]) * shx[i + 192u];
        acc += __half2float(wr[i + 224u]) * shx[i + 224u];
    }
    for (; i < in_dim; i += 32u) {
        acc += __half2float(wr[i]) * shx[i];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

__global__ static void matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint32_t in_dim,
        uint64_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const __half *wr0 = w0 + row * (uint64_t)in_dim;
    const __half *wr1 = w1 + row * (uint64_t)in_dim;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint32_t i = lane;
    for (; i + 224u < in_dim; i += 256u) {
        float xv = shx[i];
        acc0 += __half2float(wr0[i]) * xv;
        acc1 += __half2float(wr1[i]) * xv;
        xv = shx[i + 32u];
        acc0 += __half2float(wr0[i + 32u]) * xv;
        acc1 += __half2float(wr1[i + 32u]) * xv;
        xv = shx[i + 64u];
        acc0 += __half2float(wr0[i + 64u]) * xv;
        acc1 += __half2float(wr1[i + 64u]) * xv;
        xv = shx[i + 96u];
        acc0 += __half2float(wr0[i + 96u]) * xv;
        acc1 += __half2float(wr1[i + 96u]) * xv;
        xv = shx[i + 128u];
        acc0 += __half2float(wr0[i + 128u]) * xv;
        acc1 += __half2float(wr1[i + 128u]) * xv;
        xv = shx[i + 160u];
        acc0 += __half2float(wr0[i + 160u]) * xv;
        acc1 += __half2float(wr1[i + 160u]) * xv;
        xv = shx[i + 192u];
        acc0 += __half2float(wr0[i + 192u]) * xv;
        acc1 += __half2float(wr1[i + 192u]) * xv;
        xv = shx[i + 224u];
        acc0 += __half2float(wr0[i + 224u]) * xv;
        acc1 += __half2float(wr1[i + 224u]) * xv;
    }
    for (; i < in_dim; i += 32u) {
        const float xv = shx[i];
        acc0 += __half2float(wr0[i]) * xv;
        acc1 += __half2float(wr1[i]) * xv;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0u) {
        out0[row] = acc0;
        out1[row] = acc1;
    }
}

__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = x[i];
        if (row < out0_dim) sum0 += __half2float(wr0[i]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[i]) * xv;
    }
    partial0[tid] = sum0;
    partial1[tid] = sum1;
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}

__global__ static void repeat_hc_rows_kernel(float *out, const float *rows, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (i >= n) return;

    uint64_t hc_row = (uint64_t)n_hc * n_embd;
    uint64_t tok = i / hc_row;
    uint64_t embd = i % n_embd;
    out[i] = rows[tok * n_embd + embd];
}

__global__ static void pack_slot_rows_f32_kernel(float *out, const float *slots, uint32_t n_rows, uint32_t width, uint32_t n_slots, uint32_t slot_cap) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_rows * n_slots * width;
    if (i >= n) return;

    uint64_t col = i % width;
    uint64_t slot = (i / width) % n_slots;
    uint64_t row = i / ((uint64_t)n_slots * width);
    out[i] = slots[((slot * slot_cap) + row) * width + col];
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

/* Four elements per thread.
 *
 * One element per thread gives a wave 128 B of loads against 64 B of stores, so
 * every store is a partial cache line and the conversion runs at about half of
 * DRAM peak even though it is pure streaming. A `float4` load and a paired-half
 * store per thread makes both sides full lines. Per-element arithmetic is
 * unchanged, and elementwise conversion has no accumulation order. */
__global__ static void f32_to_f16_vec4_kernel(
        __half *out, const float *x, uint64_t groups) {
    const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const float4 v = *(const float4 *)(x + g * 4u);
    __half2 packed[2];
    packed[0] = __floats2half2_rn(v.x, v.y);
    packed[1] = __floats2half2_rn(v.z, v.w);
    *(__half2 *)(out + g * 4u) = packed[0];
    *(__half2 *)(out + g * 4u + 2u) = packed[1];
}

/* Convert `n` floats to halves, vectorized when the extent and both pointers
 * allow it. Callers check hipGetLastError afterwards as before. */
/* Published F16 mirror of one activation buffer.
 *
 * A ratio-4 layer's normalized attention rows feed eight projections -- the
 * compressor KV and gate, the indexer KV, gate and projection, and the QKV
 * pair -- and every F16 route among them converted the same 4,096 x 4,096 rows
 * again, 188 conversions per prefill at 0.475 ms each.
 *
 * The graph publishes the mirror once after producing the rows and clears it at
 * every layer boundary, and a consumer reuses it only when both the pointer and
 * the element count match, so a recycled buffer cannot be served stale: the
 * clear happens before anything can overwrite the published rows. */
static const void *g_f16_input_src = NULL;
static uint64_t     g_f16_input_count = 0;
static __half      *g_f16_input_mirror = NULL;
static uint64_t     g_f16_input_mirror_bytes = 0;

static void hip_f16_input_release(void) {
    if (g_f16_input_mirror) (void)hipFree(g_f16_input_mirror);
    g_f16_input_mirror = NULL;
    g_f16_input_mirror_bytes = 0;
    g_f16_input_src = NULL;
    g_f16_input_count = 0;
}

static void hip_f16_input_clear(void) {
    g_f16_input_src = NULL;
    g_f16_input_count = 0;
}

static __half *hip_f16_input_lookup(const float *x, uint64_t count) {
    if (!g_f16_input_mirror || !g_f16_input_src || count == 0u) return NULL;
    if (g_f16_input_src != (const void *)x || g_f16_input_count != count) {
        return NULL;
    }
    return g_f16_input_mirror;
}

static void hip_launch_f32_to_f16(__half *out, const float *x, uint64_t n) {
    if (n == 0u) return;
    if ((n & 3u) == 0u &&
        ((uintptr_t)x & 15u) == 0u && ((uintptr_t)out & 7u) == 0u) {
        const uint64_t groups = n >> 2u;
        f32_to_f16_vec4_kernel<<<(groups + 255u) / 256u, 256>>>(out, x, groups);
        return;
    }
    f32_to_f16_kernel<<<(n + 255u) / 256u, 256>>>(out, x, n);
}

static int hip_f16_input_publish(const float *x, uint64_t count) {
    hip_f16_input_clear();
    if (!x || count == 0u) return 0;
    uint64_t bytes = 0;
    if (!hip_u64_mul_checked(count, sizeof(__half), &bytes)) return 0;
    if (g_f16_input_mirror_bytes < bytes) {
        if (g_f16_input_mirror) (void)hipFree(g_f16_input_mirror);
        g_f16_input_mirror = NULL;
        g_f16_input_mirror_bytes = 0;
        void *ptr = NULL;
        if (!hip_ok(hipMalloc(&ptr, (size_t)bytes), "f16 input mirror alloc")) {
            (void)hipGetLastError();
            return 0;
        }
        g_f16_input_mirror = (__half *)ptr;
        g_f16_input_mirror_bytes = bytes;
    }
    hip_launch_f32_to_f16(g_f16_input_mirror, x, count);
    if (!hip_ok(hipGetLastError(), "f16 input mirror convert")) return 0;
    g_f16_input_src = (const void *)x;
    g_f16_input_count = count;
    return 1;
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v += __shfl_down(v, offset, 32);
#else
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset, 32);
#endif
    }
    return v;
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v = fmaxf(v, __shfl_down(v, offset, 32));
#else
        v = fmaxf(v, __shfl_down_sync(FULL_WARP_MASK, v, offset, 32));
#endif
    }
    return v;
}

__device__ static uint16_t f32_to_f16_bits_hip_round(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

__device__ static float f16_bits_to_f32(uint16_t bits) {
    return __half2float(__ushort_as_half((unsigned short)bits));
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
