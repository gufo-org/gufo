__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens) {
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    uint32_t t = gid / width;
    uint32_t j = gid - (uint64_t)t * width;
    uint32_t pos_mod = (pos0 + t) % ratio;
    uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
    state_score[(uint64_t)dst_row * width + j] =
        sc[(uint64_t)t * width + j] + model_ape_value_dev(model_map, ape_offset, ape_type, width, pos_mod, j);
}

__global__ static void compressor_set_rows_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t src0,
        uint32_t dst0,
        uint32_t rows) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)rows * width;
    if (gid >= n) return;
    uint32_t r = gid / width;
    uint32_t j = gid - (uint64_t)r * width;
    uint32_t src = src0 + r;
    uint32_t dst = dst0 + r;
    uint32_t phase = (pos0 + src) % ratio;
    state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
    state_score[(uint64_t)dst * width + j] =
        sc[(uint64_t)src * width + j] + model_ape_value_dev(model_map, ape_offset, ape_type, width, phase, j);
}

__global__ static void compressor_prefill_pool_kernel(
        float *comp,
        const float *kv,
        const float *sc,
        const float *state_kv,
        const float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t replay) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y;
    if (d >= head_dim || c >= n_comp) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[DS4_ROCM_COMPRESSOR_MAX_RATIO];
    float scores[DS4_ROCM_COMPRESSOR_MAX_RATIO];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        if (replay && c == 0) {
            for (uint32_t r = 0; r < 4; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * width + d];
                scores[n_cand] = state_score[(uint64_t)r * width + d];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        } else if (c > 0) {
            uint32_t base = (c - 1u) * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                uint32_t t = base + r;
                float ape = model_ape_value_dev(model_map, ape_offset, ape_type, width, (pos0 + t) % ratio, d);
                vals[n_cand] = kv[(uint64_t)t * width + d];
                scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        }
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < 4; r++) {
            uint32_t t = base + r;
            float ape = model_ape_value_dev(model_map, ape_offset, ape_type, width, (pos0 + t) % ratio, head_dim + d);
            vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
            scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < ratio; r++) {
            uint32_t t = base + r;
            float ape = model_ape_value_dev(model_map, ape_offset, ape_type, width, (pos0 + t) % ratio, d);
            vals[n_cand] = kv[(uint64_t)t * width + d];
            scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
}

/* Score-softmax pool of one dimension over the compressor state window. */
__device__ static float compressor_pool_dim(const float *state_kv, const float *state_score,
                                            uint32_t d, uint32_t head_dim, uint32_t ratio) {
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[DS4_ROCM_COMPRESSOR_MAX_RATIO];
    float scores[DS4_ROCM_COMPRESSOR_MAX_RATIO];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n_cand] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    return den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_update_pool_kernel(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    row[d] = compressor_pool_dim(state_kv, state_score, d, head_dim, ratio);
}

__global__ static void compressor_shift_ratio4_kernel(float *state_kv, float *state_score, uint32_t width) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = 4ull * width;
    if (i >= half) return;
    float v = state_kv[half + i];
    float s = state_score[half + i];
    state_kv[i] = v;
    state_score[i] = s;
    state_kv[half + i] = v;
    state_score[half + i] = s;
}

/* One compressor of a single-token decode step. A non-null mirror also gets
 * the F16 copy of an FP8-rounded row. */
struct ds4_compressor_decode_job {
    const float *kv;
    const float *sc;
    float *state_kv;
    float *state_score;
    float *comp_row;
    __half *comp_mirror;
    const void *ape;
    const float *norm_w;
    uint32_t ape_type;
    uint32_t head_dim;
    uint32_t qat;
};

/* Single-token compressor update, one block per job: store this token's
 * kv/score into the state and, on the token that completes a window, pool it,
 * RMS-normalize, rotate the tail, round the row (FP8 for attention rows, the
 * Hadamard FP4 QAT for indexer rows) and shift the ratio-4 state. Each step
 * repeats its standalone kernel's per-element arithmetic. */
__global__ static void compressor_decode_kernel(ds4_compressor_decode_job job0,
                                                ds4_compressor_decode_job job1,
                                                uint32_t ratio, uint32_t pos, uint32_t emit,
                                                ds4_rope_tail_args rope, float rms_eps) {
    const ds4_compressor_decode_job &job = blockIdx.x == 0u ? job0 : job1;
    const uint32_t tid = threadIdx.x;
    const uint32_t head_dim = job.head_dim;
    const uint32_t width = (ratio == 4u ? 2u : 1u) * head_dim;
    const uint32_t pos_mod = pos % ratio;
    const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    for (uint32_t j = tid; j < width; j += blockDim.x) {
        job.state_kv[(uint64_t)dst_row * width + j] = job.kv[j];
        job.state_score[(uint64_t)dst_row * width + j] =
            job.sc[j] + model_ape_value_dev(job.ape, 0, job.ape_type, width, pos_mod, j);
    }
    if (!emit) return;
    __syncthreads();
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        job.comp_row[d] = compressor_pool_dim(job.state_kv, job.state_score, d, head_dim, ratio);
    }
    __syncthreads();
    __shared__ float scratch[256];
    rms_norm_weight_row(job.comp_row, job.comp_row, job.norm_w, head_dim, rms_eps, scratch);
    __syncthreads();
    if (tid < rope.n_rot / 2u) {
        rope_tail_rotate_pair(job.comp_row + head_dim - rope.n_rot, tid, pos + 1u - ratio, rope);
    }
    __syncthreads();
    if (job.qat) {
        __shared__ float absbuf[128];
        indexer_hadamard_fp4_row(job.comp_row, tid, scratch, absbuf);
    } else {
        const uint32_t n_nope = head_dim - rope.n_rot;
        const uint32_t groups = job.comp_mirror ? head_dim / 64u : n_nope / 64u;
        for (uint32_t g0 = 0; g0 < groups; g0 += blockDim.x / 64u) {
            __syncthreads();
            const uint32_t d = g0 * 64u + tid;
            const bool own = d < groups * 64u;
            float v = own ? job.comp_row[d] : 0.0f;
            const float scale = dsv4_fp8_kv_group_scale(
                    scratch + (tid & ~63u), tid & 63u, own && d < n_nope ? fabsf(v) : 0.0f);
            if (own && d < n_nope) {
                v = dsv4_fp8_kv_round(v, scale);
                job.comp_row[d] = v;
            }
            if (own && job.comp_mirror) job.comp_mirror[d] = __float2half(v);
        }
    }
    if (ratio != 4u) return;
    __syncthreads();
    for (uint32_t i = tid; i < 4u * width; i += blockDim.x) {
        const float v = job.state_kv[4u * width + i];
        const float s = job.state_score[4u * width + i];
        job.state_kv[i] = v;
        job.state_score[i] = s;
        job.state_kv[4u * width + i] = v;
        job.state_score[4u * width + i] = s;
    }
}

static uint64_t hip_tensor_2d_bytes(uint32_t type, uint64_t width, uint64_t rows) {
    if (type == 0u) return width * rows * sizeof(float);
    if (type == 1u) return width * rows * sizeof(uint16_t);
    if (type == 8u) return rows * (((width + 31u) / 32u) * 34u);
    return 0;
}

static int hip_ape_type_supported(uint32_t type) {
    return type == 0u || type == 1u || type == 8u;
}

static int hip_compressor_shape_supported(uint32_t head_dim, uint32_t ratio) {
    if (head_dim == 0u || ratio == 0u || ratio > DS4_ROCM_COMPRESSOR_MAX_RATIO) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    return head_dim <= UINT32_MAX / coff;
}

extern "C" int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        !hip_compressor_shape_supported(head_dim, ratio) || n_tokens == 0 ||
        n_tokens > ratio - pos0 % ratio ||
        !hip_ape_type_supported(ape_type)) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    uint64_t kv_bytes = 0, state_bytes = 0;
    const uint64_t ape_bytes = hip_tensor_2d_bytes(ape_type, width, ratio);
    if (!hip_u64_mul3_checked(n_tokens, width, sizeof(float), &kv_bytes) ||
        !hip_u64_mul3_checked(state_rows, width, sizeof(float), &state_bytes) ||
        !hip_model_range_fits(model_size, ape_offset, ape_bytes) ||
        !hip_tensor_has_bytes(kv, kv_bytes) || !hip_tensor_has_bytes(sc, kv_bytes) ||
        !hip_tensor_has_bytes(state_kv, state_bytes) || !hip_tensor_has_bytes(state_score, state_bytes)) {
        return 0;
    }
    const char *ape = hip_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t n = (uint64_t)n_tokens * width;
    compressor_store_kernel<<<(n + 255) / 256, 256>>>(
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (float *)state_kv->ptr,
            (float *)state_score->ptr,
            ape,
            0,
            ape_type,
            head_dim,
            ratio,
            pos0,
            n_tokens);
    return hip_ok(hipGetLastError(), "compressor store launch");
}

extern "C" int ds4_gpu_compressor_decode_tensor(
        const ds4_gpu_compressor_step *steps,
        uint32_t                       n_steps,
        const void                    *model_map,
        uint64_t                       model_size,
        uint32_t                       ratio,
        uint32_t                       pos,
        uint32_t                       n_rot,
        uint32_t                       n_ctx_orig,
        float                          freq_base,
        float                          freq_scale,
        float                          ext_factor,
        float                          attn_factor,
        float                          beta_fast,
        float                          beta_slow,
        float                          rms_eps) {
    if (!steps || n_steps == 0u || n_steps > 2u || !model_map || (n_rot & 1u) != 0u) return 0;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    ds4_compressor_decode_job jobs[2] = {};
    for (uint32_t i = 0; i < n_steps; i++) {
        const ds4_gpu_compressor_step &st = steps[i];
        const uint32_t head_dim = st.head_dim;
        if (!hip_compressor_shape_supported(head_dim, ratio) || !hip_ape_type_supported(st.ape_type) ||
            n_rot > head_dim || (st.qat ? head_dim != 128u : (head_dim - n_rot) % 64u != 0u) ||
            (st.comp_mirror && head_dim % 64u != 0u)) {
            return 0;
        }
        const uint32_t width = (ratio == 4u ? 2u : 1u) * head_dim;
        const uint64_t state_bytes = (uint64_t)width * (ratio == 4u ? 2u : 1u) * ratio * sizeof(float);
        const uint64_t ape_bytes = hip_tensor_2d_bytes(st.ape_type, width, ratio);
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
        const uint64_t rows = (uint64_t)st.comp_row + emit;
        if (!hip_tensor_has_f32(st.kv, width) || !hip_tensor_has_f32(st.sc, width) ||
            !hip_tensor_has_bytes(st.state_kv, state_bytes) ||
            !hip_tensor_has_bytes(st.state_score, state_bytes) ||
            !hip_model_range_fits(model_size, st.ape_offset, ape_bytes) ||
            !hip_model_range_fits(model_size, st.norm_offset, norm_bytes) ||
            (emit && !hip_tensor_has_elems2(st.comp_cache, rows, head_dim, sizeof(float))) ||
            (emit && st.comp_mirror && !hip_tensor_has_elems2(st.comp_mirror, rows, head_dim, sizeof(__half)))) {
            return 0;
        }
        ds4_compressor_decode_job &job = jobs[i];
        job.ape = hip_model_range_ptr(model_map, st.ape_offset, ape_bytes, "compressor_ape");
        job.norm_w = (const float *)hip_model_range_ptr(model_map, st.norm_offset, norm_bytes, "compressor_norm");
        if (!job.ape || !job.norm_w) return 0;
        job.kv = (const float *)st.kv->ptr;
        job.sc = (const float *)st.sc->ptr;
        job.state_kv = (float *)st.state_kv->ptr;
        job.state_score = (float *)st.state_score->ptr;
        job.comp_row = st.comp_cache ? (float *)st.comp_cache->ptr + (uint64_t)st.comp_row * head_dim : NULL;
        job.comp_mirror = st.comp_mirror ? (__half *)st.comp_mirror->ptr + (uint64_t)st.comp_row * head_dim : NULL;
        job.ape_type = st.ape_type;
        job.head_dim = head_dim;
        job.qat = st.qat ? 1u : 0u;
    }
    compressor_decode_kernel<<<n_steps, 256>>>(
            jobs[0], jobs[1], ratio, pos, emit,
            ds4_rope_tail_args{n_rot, n_ctx_orig, 0, freq_base, freq_scale, ext_factor,
                               attn_factor, beta_fast, beta_slow},
            rms_eps);
    return hip_ok(hipGetLastError(), "compressor decode launch");
}

extern "C" int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        bool                    state_already_stored,
        bool                    decode_one_token,
        bool                    defer_finalize) {
    (void)decode_one_token;
    (void)defer_finalize;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || !hip_compressor_shape_supported(head_dim, ratio) ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        !hip_ape_type_supported(ape_type) || norm_type != 0u) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    uint64_t kv_bytes = 0, state_bytes = 0, comp_bytes = 0, norm_bytes = 0;
    const uint64_t ape_bytes = hip_tensor_2d_bytes(ape_type, width, ratio);
    if (!hip_u64_mul_checked(width, sizeof(float), &kv_bytes) ||
        !hip_u64_mul3_checked(state_rows, width, sizeof(float), &state_bytes) ||
        !hip_u64_mul3_checked((uint64_t)comp_row + (emit ? 1u : 0u), head_dim, sizeof(float), &comp_bytes) ||
        !hip_u64_mul_checked(head_dim, sizeof(float), &norm_bytes) ||
        !hip_model_range_fits(model_size, ape_offset, ape_bytes) ||
        !hip_model_range_fits(model_size, norm_offset, norm_bytes) ||
        !hip_tensor_has_bytes(kv_cur, kv_bytes) || !hip_tensor_has_bytes(sc_cur, kv_bytes) ||
        !hip_tensor_has_bytes(state_kv, state_bytes) || !hip_tensor_has_bytes(state_score, state_bytes) ||
        (emit && !hip_tensor_has_bytes(comp_cache, comp_bytes))) {
        return 0;
    }
    if (!state_already_stored) {
        if (!ds4_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                     model_map, model_size, ape_offset, ape_type,
                                                     head_dim, ratio, pos, 1)) {
            return 0;
        }
    }
    if (!emit) return 1;
    ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;
    compressor_update_pool_kernel<<<(head_dim + 255) / 256, 256>>>(
            (float *)comp_row_view->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            head_dim,
            ratio);
    int ok = hip_ok(hipGetLastError(), "compressor update pool launch");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                       model_map, model_size, norm_offset,
                                                       head_dim, 1, rms_eps);
    if (ok) ok = ds4_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,
                                            pos + 1u - ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow);
    ds4_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        uint64_t half = 4ull * width;
        compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr, width);
        ok = hip_ok(hipGetLastError(), "compressor ratio4 shift launch");
    }
    return ok;
}
extern "C" int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        ds4_gpu_tensor       *comp_mirror_f16) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        !hip_compressor_shape_supported(head_dim, ratio) || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        !hip_ape_type_supported(ape_type) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    uint64_t kv_bytes = 0, state_bytes = 0, comp_bytes = 0, norm_bytes = 0;
    const uint64_t ape_bytes = hip_tensor_2d_bytes(ape_type, width, ratio);

    if (!hip_u64_mul3_checked(n_tokens, width, sizeof(float), &kv_bytes) ||
        !hip_u64_mul3_checked(state_rows, width, sizeof(float), &state_bytes) ||
        !hip_u64_mul3_checked(n_comp, head_dim, sizeof(float), &comp_bytes) ||
        !hip_u64_mul_checked(head_dim, sizeof(float), &norm_bytes) ||
        !hip_model_range_fits(model_size, ape_offset, ape_bytes) ||
        !hip_model_range_fits(model_size, norm_offset, norm_bytes) ||
        !hip_tensor_has_bytes(kv, kv_bytes) || !hip_tensor_has_bytes(sc, kv_bytes) ||
        !hip_tensor_has_bytes(state_kv, state_bytes) || !hip_tensor_has_bytes(state_score, state_bytes) ||
        (n_comp && !hip_tensor_has_bytes(comp_cache, comp_bytes)) ||
        (comp_mirror_f16 &&
         !hip_tensor_has_elems2(
                 comp_mirror_f16, n_comp, head_dim, sizeof(half)))) {
        return 0;
    }
    const char *ape = hip_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!hip_ok(hipMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!hip_ok(hipGetLastError(), "compressor state score fill launch")) return 0;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            uint32_t prev_start = cutoff - ratio;
            uint64_t n = (uint64_t)ratio * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    prev_start, 0, ratio);
            if (!hip_ok(hipGetLastError(), "compressor prefill prev state launch")) return 0;
        }
        if (rem != 0) {
            uint64_t n = (uint64_t)rem * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    cutoff, ratio, rem);
            if (!hip_ok(hipGetLastError(), "compressor prefill rem state launch")) return 0;
        }
    } else if (rem != 0) {
        uint64_t n = (uint64_t)rem * width;
        compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv->ptr, (const float *)sc->ptr,
                ape, 0, ape_type, width, ratio, pos0,
                cutoff, 0, rem);
        if (!hip_ok(hipGetLastError(), "compressor prefill rem state launch")) return 0;
    }
    if (n_comp != 0) {
        dim3 grid((head_dim + 255) / 256, n_comp, 1);
        compressor_prefill_pool_kernel<<<grid, 256>>>(
                (float *)comp_cache->ptr,
                (const float *)kv->ptr,
                (const float *)sc->ptr,
                (const float *)state_kv->ptr,
                (const float *)state_score->ptr,
                ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 0);
        if (!hip_ok(hipGetLastError(), "compressor prefill pool launch")) return 0;
        if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, n_comp, rms_eps)) return 0;
        if (n_rot != 0 && !hip_rope_tail_stride_tensor(comp_cache, n_comp, 1, head_dim,
                                                         n_rot, pos0, ratio, n_ctx_orig, false,
                                                         freq_base, freq_scale, ext_factor,
                                                         attn_factor, beta_fast, beta_slow)) return 0;
        if (quantize_fp8) {
            const int quantized = comp_mirror_f16
                ? ds4_gpu_dsv4_fp8_kv_quantize_mirror_f16_tensor(
                        comp_cache, comp_mirror_f16, n_comp, head_dim, n_rot)
                : ds4_gpu_dsv4_fp8_kv_quantize_tensor(
                        comp_cache, n_comp, head_dim, n_rot);
            if (!quantized) return 0;
        } else if (comp_mirror_f16) {
            return 0;
        }
    }
    return 1;
}
extern "C" int ds4_gpu_compressor_prefill_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        ds4_gpu_tensor       *comp_mirror_f16) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        !hip_compressor_shape_supported(head_dim, ratio) || n_tokens == 0 ||
        n_tokens % ratio != 0 || pos0 % ratio != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        !hip_ape_type_supported(ape_type) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    uint64_t kv_bytes = 0, state_bytes = 0, comp_bytes = 0, norm_bytes = 0;
    const uint64_t ape_bytes = hip_tensor_2d_bytes(ape_type, width, ratio);
    if (!hip_u64_mul3_checked(n_tokens, width, sizeof(float), &kv_bytes) ||
        !hip_u64_mul3_checked(state_rows, width, sizeof(float), &state_bytes) ||
        !hip_u64_mul3_checked(n_comp, head_dim, sizeof(float), &comp_bytes) ||
        !hip_u64_mul_checked(head_dim, sizeof(float), &norm_bytes) ||
        !hip_model_range_fits(model_size, ape_offset, ape_bytes) ||
        !hip_model_range_fits(model_size, norm_offset, norm_bytes) ||
        !hip_tensor_has_bytes(kv, kv_bytes) || !hip_tensor_has_bytes(sc, kv_bytes) ||
        !hip_tensor_has_bytes(state_kv, state_bytes) || !hip_tensor_has_bytes(state_score, state_bytes) ||
        !hip_tensor_has_bytes(comp_cache, comp_bytes) ||
        (comp_mirror_f16 &&
         !hip_tensor_has_elems2(
                 comp_mirror_f16, n_comp, head_dim, sizeof(half)))) {
        return 0;
    }
    const char *ape = hip_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    dim3 grid((head_dim + 255) / 256, n_comp, 1);
    compressor_prefill_pool_kernel<<<grid, 256>>>(
            (float *)comp_cache->ptr,
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 1);
    if (!hip_ok(hipGetLastError(), "compressor replay pool launch")) return 0;
    if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                               model_map, model_size, norm_offset,
                                               head_dim, n_comp, rms_eps)) return 0;
    if (n_rot != 0 && !hip_rope_tail_stride_tensor(comp_cache, n_comp, 1, head_dim,
                                                     n_rot, pos0, ratio, n_ctx_orig, false,
                                                     freq_base, freq_scale, ext_factor,
                                                     attn_factor, beta_fast, beta_slow)) return 0;
    if (quantize_fp8) {
        const int quantized = comp_mirror_f16
            ? ds4_gpu_dsv4_fp8_kv_quantize_mirror_f16_tensor(
                    comp_cache, comp_mirror_f16, n_comp, head_dim, n_rot)
            : ds4_gpu_dsv4_fp8_kv_quantize_tensor(
                    comp_cache, n_comp, head_dim, n_rot);
        if (!quantized) return 0;
    } else if (comp_mirror_f16) {
        return 0;
    }

    // Leave the state as per-window updates would: store the last window,
    // then for ratio 4 shift it into the previous-window rows.
    const uint64_t last = (uint64_t)(n_tokens - ratio) * width;
    const uint64_t n = (uint64_t)ratio * width;
    compressor_store_kernel<<<(n + 255) / 256, 256>>>(
            (const float *)kv->ptr + last, (const float *)sc->ptr + last,
            (float *)state_kv->ptr, (float *)state_score->ptr,
            ape, 0, ape_type, head_dim, ratio, pos0 + n_tokens - ratio, ratio);
    if (!hip_ok(hipGetLastError(), "compressor replay state launch")) return 0;
    if (ratio != 4u) return 1;
    compressor_shift_ratio4_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr, width);
    return hip_ok(hipGetLastError(), "compressor replay shift launch");
}
extern "C" int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || !hip_ape_type_supported(ape_type)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    uint64_t tail_bytes = 0, state_bytes = 0;
    const uint64_t ape_bytes = hip_tensor_2d_bytes(ape_type, width, ratio);
    if (!hip_u64_mul3_checked(ratio, width, sizeof(float), &tail_bytes) ||
        !hip_u64_mul3_checked(state_rows, width, sizeof(float), &state_bytes) ||
        !hip_model_range_fits(model_size, ape_offset, ape_bytes) ||
        !hip_tensor_has_bytes(kv_tail, tail_bytes) || !hip_tensor_has_bytes(sc_tail, tail_bytes) ||
        !hip_tensor_has_bytes(state_kv, state_bytes) || !hip_tensor_has_bytes(state_score, state_bytes)) {
        return 0;
    }
    const char *ape = hip_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t state_n = (uint64_t)state_rows * width;
    if (!hip_ok(hipMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!hip_ok(hipGetLastError(), "compressor state score fill launch")) return 0;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            0, 0, ratio);
    return hip_ok(hipGetLastError(), "compressor state set launch");
}
