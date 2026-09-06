extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim);
extern "C" int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot) &&
           ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, head_dim);
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    store_raw_kv_batch_kernel<<<(head_dim + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, row, 1, head_dim);
    return hip_ok(hipGetLastError(), "store_raw_kv launch");
}
extern "C" int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    store_raw_kv_batch_kernel<<<(n + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, pos0, n_tokens, head_dim);
    return hip_ok(hipGetLastError(), "store_raw_kv_batch launch");
}
extern "C" int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)hip_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    const ds4_rocm_runtime_config *cfg = hip_runtime_config();
    if (cfg->oldhip_attention_decode) {
        const uint32_t rows = n_raw + n_comp;
        const size_t shmem = (size_t)(rows ? rows : 1u) * sizeof(float);
        attention_decode_mixed_one_fast_oldhip_kernel<<<(unsigned)n_head, 256, shmem>>>(
                (float *)heads->ptr,
                (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : NULL,
                use_mask ? (const float *)comp_mask->ptr : NULL,
                sinks,
                n_raw,
                raw_cap,
                raw_start,
                n_comp,
                use_mask,
                n_head,
                head_dim,
                (uint32_t)((head_dim & 3u) == 0u));
        return hip_ok(hipGetLastError(), "attention decode oldhip fast launch");
    }
    if (!hip_attention_score_buffer_fits(n_comp)) {
        if (!use_mask && head_dim == 512u) {
            dim3 online_grid(1, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              1,
                                                                              n_raw - 1u,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              0,
                                                                              0,
                                                                              n_head,
                                                                              head_dim);
            return hip_ok(hipGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    dim3 grid(1, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_mask,
                                                 1, 0, n_raw, raw_cap, raw_start, n_comp,
                                                 0, 0, n_head, head_dim);
    return hip_ok(hipGetLastError(), "attention decode launch");
}
extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        window > 256) return 0;
    const float *sinks = (const float *)hip_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (n_tokens > 1 && head_dim == 512 &&
        ((window != 0u ? window : n_tokens) <= 768u)) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   0,
                                                                   window,
                                                                   1,
                                                                   n_head,
                                                                   head_dim);
        return hip_ok(hipGetLastError(), "attention raw window launch");
    }
    if (g_hipblas_ready && n_tokens > 1 && head_dim == 512) {
        const uint32_t n_keys = n_tokens;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = (score_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)hip_tmp_alloc(tmp_bytes, "attention raw hipblas");
        if (!tmp) return 0;
        float *scores = tmp;
        float *out_tmp = (float *)((char *)tmp + out_offset);
        const float alpha = 1.0f / sqrtf((float)head_dim);
        const float beta = 0.0f;
        hipblasStatus_t st = hipblasSgemmStridedBatched(g_hipblas,
                                                      HIPBLAS_OP_T,
                                                      HIPBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      (const float *)raw_kv->ptr,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!hipblas_ok(st, "attention raw score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_raw_softmax_kernel<<<sgrid, 256>>>(scores, sinks, n_tokens, window, n_keys);
        if (!hip_ok(hipGetLastError(), "attention raw softmax launch")) return 0;
        const float one = 1.0f;
        st = hipblasSgemmStridedBatched(g_hipblas,
                                       HIPBLAS_OP_N,
                                       HIPBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       (const float *)raw_kv->ptr,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!hipblas_ok(st, "attention raw value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return hip_ok(hipGetLastError(), "attention raw unpack launch");
    }
    if (window == 0u && n_tokens > 256u) return 0;
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_raw_kernel<<<grid, 128>>>((float *)heads->ptr,
                                                sinks,
                                                (const float *)q->ptr,
                                                (const float *)raw_kv->ptr,
                                                n_tokens, window, n_head, head_dim);
    return hip_ok(hipGetLastError(), "attention_prefill_raw launch");
}
static int attention_decode_batch_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    if (n_comp != 0 && ratio == 0) return 0;
    const float *sinks = (const float *)hip_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    constexpr int fast_window_attention = 1;
    if (!hip_attention_score_buffer_fits(n_comp)) {
        if (!use_comp_mask && head_dim == 512u) {
            dim3 online_grid(n_tokens, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              window,
                                                                              ratio,
                                                                              n_head,
                                                                              head_dim);
            return hip_ok(hipGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    /*
     * A prompt longer than one chunk resumes through this entry rather than
     * attention_prefill_mixed_launch, and the ratio-128 layers then landed on
     * the scalar F32 producer below -- the same shape the rocWMMA producer
     * takes from 916 ms to 316 ms on the first chunk. Only the ring parameters
     * differ, and the kernel already consumes raw_cap/raw_start/pos0, so the
     * gap was launch plumbing.
     *
     * Held to the same conditions as the zero-prefix route: it converts Q and
     * KV to F16 rather than reordering a reduction, so it stays at prompt-chunk
     * width.
     */
    if (!use_comp_mask &&
        g_rocm_gfx1151 && n_tokens >= DS4_ROCM_ATTENTION_WIDE_ROWS &&
        n_comp != 0u && comp_kv && n_head == 64u && head_dim == 512u &&
        window != 0u && window <= 256u && ratio != 0u) {
        const dim3 grid(n_tokens, n_head / 32u, 1u);
        attention_mixed_heads32_wmma_kernel<false, false><<<grid, 1024>>>(
                (float *)heads->ptr,
                sinks,
                (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                (const float *)comp_kv->ptr,
                NULL,
                NULL,
                0u,
                n_tokens,
                pos0,
                n_raw,
                raw_cap,
                raw_start,
                n_comp,
                0u,
                window,
                ratio,
                n_head,
                head_dim);
        if (hip_ok(hipGetLastError(), "attention batch mixed window wmma launch")) {
            static int notice_printed = 0;
            if (!notice_printed) {
                notice_printed = 1;
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX
                        "resumed mixed-window chunk using wave32 rocWMMA\n");
            }
            return 1;
        }
        return 0;
    }
#endif
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        fast_window_attention) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   raw_cap,
                                                                   raw_start,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return hip_ok(hipGetLastError(), "attention decode window launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_comp_mask, n_tokens, pos0, n_raw, raw_cap,
                                                 raw_start, n_comp, window, ratio, n_head, head_dim);
    return hip_ok(hipGetLastError(), "attention decode batch launch");
}

extern "C" int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, NULL, NULL, 0, n_tokens, pos0,
                                      n_raw, raw_cap, raw_start, 0, window, 1,
                                      n_head, head_dim);
}

extern "C" int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                                      n_tokens, pos0, n_raw, raw_cap, raw_start,
                                      n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        comp_kv->bytes < (uint64_t)n_comp * head_dim *
                             (comp_kv_f16 ? sizeof(half) : sizeof(float)) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) {
        return 0;
    }
    if (top_k > DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP) return 0;
    const float *sinks = (const float *)hip_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    const int32_t *topk_ptr = (const int32_t *)topk->ptr;
    const bool wmma_supported =
        n_tokens >= 128u && n_head == 64u && head_dim == 512u &&
        top_k == 512u && window <= 256u;
    if (comp_kv_f16 && !wmma_supported) return 0;
    const ds4_rocm_runtime_config *cfg = hip_runtime_config();
    if (n_tokens == 1u && cfg->oldhip_attention_decode) {
        const uint32_t rows = n_raw + (top_k < n_comp ? top_k : n_comp);
        const size_t shmem = (size_t)(rows ? rows : 1u) * sizeof(float);
        attention_decode_indexed_mixed_one_fast_oldhip_kernel<<<(unsigned)n_head, 256, shmem>>>(
                (float *)heads->ptr,
                (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                (const float *)comp_kv->ptr,
                topk_ptr,
                sinks,
                n_raw,
                raw_cap,
                raw_start,
                n_comp,
                top_k,
                pos0,
                ratio,
                n_head,
                head_dim,
                (uint32_t)((head_dim & 3u) == 0u));
        return hip_ok(hipGetLastError(), "attention indexed decode oldhip fast launch");
    }
    /* The single-pass rocWMMA kernel visits each KV row block once, so the score
     * cache that let the old second pass skip its QK matrix multiply has no
     * reader. It cost up to 1 GiB of the shared scratch buffer. */
    float *const wmma_score_cache = nullptr;
    const uint32_t wmma_score_stride = 0u;
    if (n_tokens > 1u && top_k == 512u) {
        const uint64_t sort_bytes = (uint64_t)n_tokens * top_k * sizeof(int32_t);
        char *scratch = (char *)hip_tmp_alloc(
            sort_bytes, "indexed attention topk sort");
        int32_t *sorted = (int32_t *)scratch;
        if (!sorted) return 0;
        indexed_topk_sort_512_asc_kernel<<<n_tokens, 512>>>(sorted, topk_ptr, n_tokens);
        if (!hip_ok(hipGetLastError(), "indexed attention topk sort launch")) return 0;
        topk_ptr = sorted;
    }
    if (n_tokens > 1 &&
        head_dim == 512 &&
        top_k <= DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (wmma_supported) {
            const dim3 grid(n_tokens, n_head / 32u, 1u);
            if (comp_kv_f16) {
                attention_mixed_heads32_wmma_kernel<true, true>
                    <<<grid, 1024>>>(
                        (float *)heads->ptr,
                        sinks,
                        (const float *)q->ptr,
                        (const float *)raw_kv->ptr,
                        comp_kv->ptr,
                        topk_ptr,
                        wmma_score_cache,
                        wmma_score_stride,
                        n_tokens,
                        pos0,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp,
                        top_k,
                        window,
                        ratio,
                        n_head,
                        head_dim);
            } else {
                attention_mixed_heads32_wmma_kernel<true, false>
                    <<<grid, 1024>>>(
                        (float *)heads->ptr,
                        sinks,
                        (const float *)q->ptr,
                        (const float *)raw_kv->ptr,
                        comp_kv->ptr,
                        topk_ptr,
                        wmma_score_cache,
                        wmma_score_stride,
                        n_tokens,
                        pos0,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp,
                        top_k,
                        window,
                        ratio,
                        n_head,
                        head_dim);
            }
            return hip_ok(
                hipGetLastError(),
                "attention indexed wave32 wmma launch");
        }
        if (n_head <= 64u) {
            dim3 grid(n_tokens, (n_head + 31u) / 32u, 1);
            attention_indexed_mixed_heads8_online_kernel<8, 32><<<grid, 1024>>>((float *)heads->ptr,
                                                                                sinks,
                                                                                (const float *)q->ptr,
                                                                                (const float *)raw_kv->ptr,
                                                                                (const float *)comp_kv->ptr,
                                                                                topk_ptr,
                                                                                n_tokens,
                                                                                pos0,
                                                                                n_raw,
                                                                                raw_cap,
                                                                                raw_start,
                                                                                n_comp,
                                                                                top_k,
                                                                                window,
                                                                                ratio,
                                                                                n_head,
                                                                                head_dim);
            return hip_ok(hipGetLastError(), "attention indexed online heads32 launch");
        }
#endif
        dim3 grid(n_tokens, (n_head + 15u) / 16u, 1);
        attention_indexed_mixed_heads8_online_kernel<8, 16><<<grid, 512>>>((float *)heads->ptr,
                                                                           sinks,
                                                                           (const float *)q->ptr,
                                                                           (const float *)raw_kv->ptr,
                                                                           (const float *)comp_kv->ptr,
                                                                           topk_ptr,
                                                                           n_tokens,
                                                                           pos0,
                                                                           n_raw,
                                                                           raw_cap,
                                                                           raw_start,
                                                                           n_comp,
                                                                           top_k,
                                                                           window,
                                                                           ratio,
                                                                           n_head,
                                                                           head_dim);
        return hip_ok(hipGetLastError(), "attention indexed online launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_indexed_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  (const float *)comp_kv->ptr,
                                                  topk_ptr,
                                                  n_tokens,
                                                  pos0,
                                                  n_raw,
                                                  raw_cap,
                                                  raw_start,
                                                  n_comp,
                                                  top_k,
                                                  window,
                                                  ratio,
                                                  n_head,
                                                  head_dim);
    return hip_ok(hipGetLastError(), "attention indexed mixed launch");
}

static uint64_t attention_mixed_hipblas_tmp_bytes(
        uint32_t n_keys,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint64_t kv_count = (uint64_t)n_keys * head_dim;
    const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
    const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
    const uint64_t kv_bytes = kv_count * sizeof(float);
    const uint64_t score_offset = (kv_bytes + 255u) & ~255ull;
    const uint64_t score_bytes = score_count * sizeof(float);
    const uint64_t out_offset = score_offset + ((score_bytes + 255u) & ~255ull);
    return out_offset + out_count * sizeof(float);
}

static int attention_prefill_mixed_hipblas_tiled(
        ds4_gpu_tensor       *heads,
        const float          *sinks,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    const uint32_t n_keys = n_tokens + n_comp;
    uint32_t tile_tokens = n_tokens;
    const uint64_t tile_cap = 4ull * 1024ull * 1024ull * 1024ull;
    while (tile_tokens > 1u &&
           attention_mixed_hipblas_tmp_bytes(n_keys, tile_tokens, n_head, head_dim) > tile_cap) {
        tile_tokens = (tile_tokens + 1u) >> 1u;
    }
    const uint64_t kv_count = (uint64_t)n_keys * head_dim;
    const uint64_t kv_bytes = kv_count * sizeof(float);
    const uint64_t score_offset = (kv_bytes + 255u) & ~255ull;
    const uint64_t score_bytes = (uint64_t)n_head * tile_tokens * n_keys * sizeof(float);
    const uint64_t out_offset = score_offset + ((score_bytes + 255u) & ~255ull);
    const uint64_t tmp_bytes = out_offset + (uint64_t)n_head * tile_tokens * head_dim * sizeof(float);
    float *tmp = (float *)hip_tmp_alloc(tmp_bytes, "attention mixed hipblas tiled");
    if (!tmp) return 0;
    float *kv = tmp;
    float *scores = (float *)((char *)tmp + score_offset);
    float *out_tmp = (float *)((char *)tmp + out_offset);
    attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
            kv,
            (const float *)raw_kv->ptr,
            n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
            n_tokens,
            n_comp,
            head_dim);
    if (!hip_ok(hipGetLastError(), "attention mixed tiled kv pack launch")) return 0;

    const float alpha = 1.0f / sqrtf((float)head_dim);
    const float beta = 0.0f;
    const float one = 1.0f;
    for (uint32_t t0 = 0; t0 < n_tokens; t0 += tile_tokens) {
        const uint32_t nt = (t0 + tile_tokens <= n_tokens) ? tile_tokens : (n_tokens - t0);
        const float *q_tile = (const float *)q->ptr + (uint64_t)t0 * n_head * head_dim;
        hipblasStatus_t st = hipblasSgemmStridedBatched(g_hipblas,
                                                      HIPBLAS_OP_T,
                                                      HIPBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)nt,
                                                      (int)head_dim,
                                                      &alpha,
                                                      kv,
                                                      (int)head_dim,
                                                      0,
                                                      q_tile,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * nt,
                                                      (int)n_head);
        if (!hipblas_ok(st, "attention mixed tiled score gemm")) return 0;
        dim3 sgrid(nt, n_head, 1);
        attention_prefill_mixed_softmax_tile_kernel<<<sgrid, 256>>>(
                scores,
                sinks,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                use_comp_mask,
                n_tokens,
                t0,
                nt,
                n_comp,
                window,
                ratio,
                n_keys);
        if (!hip_ok(hipGetLastError(), "attention mixed tiled softmax launch")) return 0;
        st = hipblasSgemmStridedBatched(g_hipblas,
                                       HIPBLAS_OP_N,
                                       HIPBLAS_OP_N,
                                       (int)head_dim,
                                       (int)nt,
                                       (int)n_keys,
                                       &one,
                                       kv,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * nt,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * nt,
                                       (int)n_head);
        if (!hipblas_ok(st, "attention mixed tiled value gemm")) return 0;
        const uint64_t n = (uint64_t)nt * n_head * head_dim;
        float *heads_tile = (float *)heads->ptr + (uint64_t)t0 * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>(
                heads_tile,
                out_tmp,
                nt,
                n_head,
                head_dim);
        if (!hip_ok(hipGetLastError(), "attention mixed tiled unpack launch")) return 0;
    }
    return 1;
}

static int attention_prefill_mixed_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || ratio == 0 ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)hip_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    /* Mixed-window prompt chunks can reach the same wave32 rocWMMA producer the
     * indexed layers use. This launcher is the zero-prefix route, so the raw
     * cache is the chunk's own linear KV: positions start at zero, the ring is
     * the chunk itself, and `window <= 256` keeps `raw_count` inside the
     * kernel's row table.
     *
     * Worth 599 ms per 4,096-token chunk (915.82 -> 316.35 ms). It is the one
     * accelerated route that lowers precision rather than reordering a
     * reduction -- the scalar kernel below keeps the whole mixed-window score
     * and value pass in F32, while the rocWMMA producer converts Q and KV to
     * F16 -- so it is held to the prompt-chunk width like the rest. The indexed
     * layers already use this producer. */
    if (!use_comp_mask && g_rocm_gfx1151 &&
        n_tokens >= DS4_ROCM_ATTENTION_WIDE_ROWS && n_comp != 0u &&
        n_head == 64u && head_dim == 512u && window != 0u && window <= 256u) {
        const dim3 grid(n_tokens, n_head / 32u, 1u);
        attention_mixed_heads32_wmma_kernel<false, false><<<grid, 1024>>>(
                (float *)heads->ptr,
                sinks,
                (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                (const float *)comp_kv->ptr,
                NULL,
                NULL,
                0u,
                n_tokens,
                0u,
                n_tokens,
                n_tokens,
                0u,
                n_comp,
                0u,
                window,
                ratio,
                n_head,
                head_dim);
        static int notice_printed = 0;
        if (!notice_printed) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "mixed-window prefill using wave32 rocWMMA "
                    "(heads=32, rows=16, single-pass)\n");
            notice_printed = 1;
        }
        return hip_ok(hipGetLastError(),
                      "attention mixed-window wave32 wmma launch");
    }
#endif
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        ((window != 0u ? window : n_tokens) + n_comp <= 768u)) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return hip_ok(hipGetLastError(), "attention mixed window launch");
    }
    if (g_hipblas_ready && n_tokens > 1 && head_dim == 512) {
        const uint32_t n_keys = n_tokens + n_comp;
        const uint64_t kv_count = (uint64_t)n_keys * head_dim;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t kv_bytes = kv_count * sizeof(float);
        const uint64_t score_offset = (kv_bytes + 255u) & ~255ull;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = score_offset + ((score_bytes + 255u) & ~255ull);
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)hip_tmp_alloc(tmp_bytes, "attention mixed hipblas");
        if (!tmp) {
            return attention_prefill_mixed_hipblas_tiled(heads,
                                                        sinks,
                                                        q,
                                                        raw_kv,
                                                        comp_kv,
                                                        comp_mask,
                                                        use_comp_mask,
                                                        n_tokens,
                                                        n_comp,
                                                        window,
                                                        ratio,
                                                        n_head,
                                                        head_dim);
        }
        float *kv = tmp;
        float *scores = (float *)((char *)tmp + score_offset);
        float *out_tmp = (float *)((char *)tmp + out_offset);
        attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                kv,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                n_tokens,
                n_comp,
                head_dim);
        if (!hip_ok(hipGetLastError(), "attention mixed kv pack launch")) return 0;
        const float alpha = 1.0f / sqrtf((float)head_dim);
        const float beta = 0.0f;
        hipblasStatus_t st = hipblasSgemmStridedBatched(g_hipblas,
                                                      HIPBLAS_OP_T,
                                                      HIPBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      kv,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!hipblas_ok(st, "attention mixed score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_mixed_softmax_kernel<<<sgrid, 256>>>(
                scores,
                sinks,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                use_comp_mask,
                n_tokens,
                n_comp,
                window,
                ratio,
                n_keys);
        if (!hip_ok(hipGetLastError(), "attention mixed softmax launch")) return 0;
        const float one = 1.0f;
        st = hipblasSgemmStridedBatched(g_hipblas,
                                       HIPBLAS_OP_N,
                                       HIPBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!hipblas_ok(st, "attention mixed value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return hip_ok(hipGetLastError(), "attention mixed unpack launch");
    }
    const uint32_t max_raw = (window != 0u && window < n_tokens) ? window : n_tokens;
    if ((uint64_t)max_raw + n_comp > DS4_ROCM_ATTENTION_PREFILL_MIXED_SCORE_CAP) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "attention mixed scalar fallback unsupported for %llu scores "
                "(cap=%u, tokens=%u, comp=%u, window=%u)\n",
                (unsigned long long)((uint64_t)max_raw + n_comp),
                DS4_ROCM_ATTENTION_PREFILL_MIXED_SCORE_CAP,
                n_tokens,
                n_comp,
                window);
        return 0;
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                  use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                  use_comp_mask, n_tokens, n_comp, window, ratio,
                                                  n_head, head_dim);
    return hip_ok(hipGetLastError(), "attention prefill mixed launch");
}

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, NULL, 0, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}

/* `rope` non-null asks for the inverse rotated tail to be folded into the F16
 * group pack. Only the packed hipBLAS/rocWMMA route can absorb it, so every
 * other route returns 0 and the caller keeps its separate rope_tail launch. */
static int attention_output_q8_batch_launch(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens,
        const ds4_attn_pack_rope *rope) {
    (void)group_tmp;
    (void)low_tmp;
    if (rope && !ds4_attn_pack_rope_eligible(rope, (uint32_t)group_dim)) return 0;
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t blocks_b = (low_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    const uint64_t out_b_bytes = out_dim * blocks_b * 34;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            hip_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    const unsigned char *out_b = reinterpret_cast<const unsigned char *>(
            hip_model_range_ptr(model_map, out_b_offset, out_b_bytes, "attn_out_b"));
    if (!out_a || !out_b) return 0;

    /*
     * The hipBLAS output route packs heads to F16 and runs a batched GEMM whose
     * macro tile is far wider than a DSpark verification block, so at these
     * widths the grouped Q8 kernels win outright.
     */
    const uint32_t attn_small_batch_rows = ds4_rocm_dense_small_batch_rows();
    const int attn_output_hipblas =
        hip_runtime_config()->attention_output_hipblas_all &&
        !(n_tokens > 1u && n_tokens <= attn_small_batch_rows);
    if (rope && !attn_output_hipblas) return 0;
    if (!attn_output_hipblas) {
        if (hip_q8_prequant_decode_enabled() &&
            (group_dim & 31u) == 0u &&
            rank <= UINT32_MAX && n_tokens <= 8u) {
            const uint64_t x_rows = (uint64_t)n_tokens * n_groups;
            const uint64_t xq_bytes = x_rows * blocks_a * 32u;
            const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
            const uint64_t tmp_bytes =
                scale_offset + x_rows * blocks_a * sizeof(float);
            void *tmp = hip_tmp_alloc(
                tmp_bytes, "attention output a session batch prequant");
            if (!tmp) return 0;
            int8_t *xq = (int8_t *)tmp;
            float *xscale = (float *)((char *)tmp + scale_offset);
            dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1u);
            quantize_q8_0_f32_kernel<<<qgrid, 32u>>>(
                xq,
                xscale,
                (const float *)heads->ptr,
                group_dim,
                blocks_a);
            if (!hip_ok(
                    hipGetLastError(),
                    "attention output a session batch quantize launch")) {
                return 0;
            }

            const uint32_t rows_per_block =
                n_tokens == 2u ? hip_runtime_config()->attn_q8_batch_rpb : 4u;
            const unsigned grid =
                (unsigned)((low_dim + rows_per_block - 1u) /
                           rows_per_block);
            const unsigned threads = rows_per_block * 32u;
            if (n_tokens == 2u) {
                grouped_q8_0_a_preq_batch_reuse_w32_kernel<2u, true>
                    <<<grid, threads>>>(
                        (float *)low->ptr,
                        out_a,
                        xq,
                        xscale,
                        group_dim,
                        rank,
                        n_groups,
                        n_tokens,
                        blocks_a,
                        rows_per_block);
            } else if (n_tokens == 3u) {
              grouped_q8_0_a_preq_batch_reuse_w32_kernel<3u, true><<<grid, threads>>>(
                  (float*)low->ptr, out_a, xq, xscale, group_dim, rank,
                  n_groups, n_tokens, blocks_a, rows_per_block);
            } else if (n_tokens == 4u) {
              grouped_q8_0_a_preq_batch_reuse_w32_kernel<4u, true><<<grid, threads>>>(
                  (float*)low->ptr, out_a, xq, xscale, group_dim, rank,
                  n_groups, n_tokens, blocks_a, rows_per_block);
            } else if (n_tokens == 5u) {
              grouped_q8_0_a_preq_batch_reuse_w32_kernel<5u, true><<<grid, threads>>>(
                  (float*)low->ptr, out_a, xq, xscale, group_dim, rank,
                  n_groups, n_tokens, blocks_a, rows_per_block);
            } else if (n_tokens == 6u) {
              grouped_q8_0_a_preq_batch_reuse_w32_kernel<6u, true><<<grid, threads>>>(
                  (float*)low->ptr, out_a, xq, xscale, group_dim, rank,
                  n_groups, n_tokens, blocks_a, rows_per_block);
            } else if (n_tokens == 7u) {
              grouped_q8_0_a_preq_batch_reuse_w32_kernel<7u, true><<<grid, threads>>>(
                  (float*)low->ptr, out_a, xq, xscale, group_dim, rank,
                  n_groups, n_tokens, blocks_a, rows_per_block);
            } else if (n_tokens == 8u) {
              grouped_q8_0_a_preq_batch_reuse_w32_kernel<8u, true><<<grid, threads>>>(
                  (float*)low->ptr, out_a, xq, xscale, group_dim, rank,
                  n_groups, n_tokens, blocks_a, rows_per_block);
            } else {
              grouped_q8_0_a_preq_batch_reuse_w32_kernel<2u, false><<<grid, threads>>>(
                  (float*)low->ptr, out_a, xq, xscale, group_dim, rank,
                  n_groups, n_tokens, blocks_a, rows_per_block);
            }
        } else if ((group_dim & 31u) == 0u && rank <= UINT32_MAX &&
                   n_tokens <= UINT32_MAX) {
            const uint32_t rows_per_block = 32u;
            /* Token tile matched to the rows present, as in the dense Q8 path. */
            const uint32_t tile = n_tokens >= 32u ? 32u
                                : n_tokens > 8u   ? 16u
                                : n_tokens > 4u   ? 8u
                                : n_tokens > 2u   ? 4u
                                                  : 2u;
            const uint32_t block_tile = 16u;
            hip_launch_grouped_q8_a_sharedx((float *)low->ptr,
                                             out_a,
                                             (const float *)heads->ptr,
                                             n_tokens,
                                             n_groups,
                                             (uint32_t)blocks_a,
                                             (uint32_t)rank,
                                             blocks_a * 34u,
                                             rows_per_block,
                                             tile,
                                             block_tile);
        } else {
            dim3 grid_a(((unsigned)low_dim + 7u) / 8u, (unsigned)n_tokens, 1);
            grouped_q8_0_a_f32_batch_warp8_kernel<<<grid_a, 256>>>(
                    (float *)low->ptr,
                    out_a,
                    (const float *)heads->ptr,
                    group_dim,
                    rank,
                    n_groups,
                    n_tokens,
                    blocks_a);
        }
        if (!hip_ok(
                hipGetLastError(),
                "attention_output_q8_a narrow batch launch")) {
            return 0;
        }
        return hip_matmul_q8_0_tensor_labeled(out,
                                               model_map,
                                               model_size,
                                               out_b_offset,
                                               low_dim,
                                               out_dim,
                                               low,
                                               n_tokens,
                                               "attn_output_b");
    }

    const __half *out_a_f16 = NULL;
    if (g_hipblas_ready &&
        n_tokens >= 2u) {
        out_a_f16 = hip_q8_f16_ptr(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, "attn_output_a");
    }
    if (out_a_f16) {
        if (hip_runtime_config()->attention_output_hipblas_all) {
            const int interleaved_b = 1;
            const __half *out_b_f16_t = hip_q8_f16_transpose_ptr(model_map, out_b_offset, out_b_bytes,
                                                                  low_dim, out_dim, "attn_output_b");
            const __half *out_b_f16 = out_b_f16_t
                ? NULL
                : hip_q8_f16_ptr(model_map, out_b_offset, out_b_bytes,
                                  low_dim, out_dim, "attn_output_b");
            if (out_b_f16 || out_b_f16_t) {
                const uint64_t heads_h_count = (uint64_t)n_groups * n_tokens * group_dim;
                const uint64_t low_h_count = (uint64_t)n_groups * n_tokens * rank;
                /* The B projection's macro tile covers whole column blocks, so
                 * stage the activations and check the result for a width rounded
                 * up to that block. The pad columns are zeroed below, computed
                 * from zeros, and never read back.
                 *
                 * Held to prompt-chunk width like every other accelerated
                 * prefill route: below it the fallback stays, which keeps the
                 * pinned trajectory's own twenty-token prefill on the library
                 * this envelope was measured against. */
                uint64_t b_pad_n = ds4_gemm_f16_wmma_pad_n(n_tokens);
                if (out_b_f16_t == NULL ||
                    n_tokens < DS4_ROCM_ATTENTION_WIDE_ROWS ||
                    !ds4_gemm_f16_wmma_padded_eligible(out_dim, b_pad_n, low_dim) ||
                    out->bytes < b_pad_n * out_dim * sizeof(float)) {
                    b_pad_n = n_tokens;
                }
                const uint64_t low_h_pad_count = b_pad_n * low_dim;
                const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
                const uint64_t low_h_offset = (heads_h_bytes + 255u) & ~255ull;
                const uint64_t tmp_bytes = low_h_offset + low_h_pad_count * sizeof(__half);
                void *tmp = hip_tmp_alloc(tmp_bytes, "attention output packed b hipblas");
                if (!tmp) return 0;
                __half *heads_h = (__half *)tmp;
                __half *low_h = (__half *)((char *)tmp + low_h_offset);
                /* Async so the pass keeps one command stream: the shared scratch
                 * arena is also the routed MoE's, so the pad cannot be zeroed
                 * once per chunk, and a blocking clear would be a host
                 * round-trip per layer half. */
                if (b_pad_n != n_tokens &&
                    !hip_ok(hipMemsetAsync(low_h + low_h_count, 0,
                                            (low_h_pad_count - low_h_count) *
                                                    sizeof(__half),
                                            0),
                             "attention output b activation pad clear")) {
                    return 0;
                }
                if (rope) {
                    if (!hip_launch_attention_pack_group_heads_rope_f16(
                                heads_h,
                                (const float *)heads->ptr,
                                n_tokens,
                                n_groups,
                                group_dim,
                                heads_h_count,
                                rope)) {
                        return 0;
                    }
                } else {
                    hip_launch_attention_pack_group_heads_f16(
                            heads_h,
                            (const float *)heads->ptr,
                            n_tokens,
                            n_groups,
                            group_dim,
                            heads_h_count);
                }
                if (!hip_ok(hipGetLastError(), "attention_output_q8 packed heads pack launch")) return 0;
                const float alpha = 1.0f;
                const float beta0 = 0.0f;
                const float beta1 = 1.0f;
                int a_wmma_ok = 0;
#ifdef __HIP_PLATFORM_AMD__
                /* rocBLAS reaches 2.7 TFLOP/s on this strided-batched shape;
                 * see the batched launcher. Only the interleaved-B layout puts
                 * each group's rank block at `g * rank` of a `low_dim` row,
                 * which is what the kernel's `ldc` and C stride assume. */
                a_wmma_ok = interleaved_b &&
                            ds4_gemm_f16_wmma_batched_eligible(
                                    rank, n_tokens, group_dim, n_groups) &&
                            ds4_gemm_f16_wmma_batched_launch(
                                    low_h, out_a_f16, heads_h, rank, n_tokens,
                                    group_dim, low_dim, n_groups,
                                    "attention output a batched wmma launch");
#endif
                hipblasStatus_t st = HIPBLAS_STATUS_SUCCESS;
                if (!a_wmma_ok) st = hipblasGemmStridedBatchedEx(g_hipblas,
                                                               HIPBLAS_OP_T,
                                                               HIPBLAS_OP_N,
                                                               (int)rank,
                                                               (int)n_tokens,
                                                               (int)group_dim,
                                                               &alpha,
                                                               out_a_f16,
                                                               HIPBLAS_R_16F,
                                                               (int)group_dim,
                                                               (long long)rank * group_dim,
                                                               heads_h,
                                                               HIPBLAS_R_16F,
                                                               (int)group_dim,
                                                               (long long)n_tokens * group_dim,
                                                               &beta0,
                                                               low_h,
                                                               HIPBLAS_R_16F,
                                                               interleaved_b ? (int)low_dim : (int)rank,
                                                               interleaved_b ? (long long)rank : (long long)rank * n_tokens,
                                                               (int)n_groups,
                                                               HIPBLAS_COMPUTE_32F,
                                                               HIPBLAS_GEMM_DEFAULT);
                if (st == HIPBLAS_STATUS_SUCCESS && interleaved_b) {
                    const __half *b_ptr = out_b_f16_t ? out_b_f16_t : out_b_f16;
                    const auto b_op = out_b_f16_t ? HIPBLAS_OP_N : HIPBLAS_OP_T;
                    const int b_lda = out_b_f16_t ? (int)out_dim : (int)low_dim;
#ifdef __HIP_PLATFORM_AMD__
                    /* The rocWMMA tile moves 3.22 GB against Tensile's 5.7 GB on
                     * this shape; see the kernel comment. Needs the transposed
                     * weight cache, so opA is N. `b_pad_n` is n_tokens unless a
                     * tile is available for a rounded-up width. */
                    if (out_b_f16_t &&
                        ds4_gemm_f16_wmma_padded_eligible(out_dim, b_pad_n, low_dim) &&
                        ds4_gemm_f16_wmma_padded_launch<false, float>(
                                (float *)out->ptr, out_b_f16_t, low_h,
                                out_dim, b_pad_n, low_dim,
                                "attention output b wmma launch")) {
                        return 1;
                    }
                    if (hipblaslt_route_enabled(DS4_ROCM_LT_ROUTE_ATTN_B) &&
                        hipblaslt_gemm_f16(out->ptr, b_ptr, low_h,
                                           (uint32_t)out_dim, n_tokens,
                                           (uint32_t)low_dim, b_op, HIP_R_32F,
                                           "attention output b")) {
                        return 1;
                    }
#endif
                    st = hipblasGemmEx(g_hipblas,
                                      b_op,
                                      HIPBLAS_OP_N,
                                      (int)out_dim,
                                      (int)n_tokens,
                                      (int)low_dim,
                                      &alpha,
                                      b_ptr,
                                      HIPBLAS_R_16F,
                                      b_lda,
                                      low_h,
                                      HIPBLAS_R_16F,
                                      (int)low_dim,
                                      &beta0,
                                      out->ptr,
                                      HIPBLAS_R_32F,
                                      (int)out_dim,
                                      HIPBLAS_COMPUTE_32F,
                                      HIPBLAS_GEMM_DEFAULT);
                    if (st == HIPBLAS_STATUS_SUCCESS) return 1;
                    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " attention output interleaved B failed: status %d; falling back\n", (int)st);
                } else if (st == HIPBLAS_STATUS_SUCCESS) {
                    int ok_packed_b = 1;
                    for (uint32_t g = 0; g < n_groups; g++) {
                        const float *beta = (g == 0u) ? &beta0 : &beta1;
                        st = hipblasGemmEx(g_hipblas,
                                           HIPBLAS_OP_T,
                                           HIPBLAS_OP_N,
                                           (int)out_dim,
                                           (int)n_tokens,
                                           (int)rank,
                                           &alpha,
                                           out_b_f16 + (uint64_t)g * rank,
                                           HIPBLAS_R_16F,
                                           (int)low_dim,
                                           low_h + (uint64_t)g * rank * n_tokens,
                                           HIPBLAS_R_16F,
                                           (int)rank,
                                           beta,
                                           out->ptr,
                                           HIPBLAS_R_32F,
                                           (int)out_dim,
                                           HIPBLAS_COMPUTE_32F,
                                           HIPBLAS_GEMM_DEFAULT);
                        if (st != HIPBLAS_STATUS_SUCCESS) {
                            ok_packed_b = 0;
                            break;
                        }
                    }
                    if (ok_packed_b) return 1;
                    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " attention output packed B failed: status %d; falling back\n", (int)st);
                } else {
                    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " attention output packed A failed: status %d; falling back\n", (int)st);
                }
            }
        }
        if (rope) return 0;
        const uint64_t heads_h_count = (uint64_t)n_groups * n_tokens * group_dim;
        const uint64_t low_tmp_count = (uint64_t)n_groups * n_tokens * rank;
        const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
        const uint64_t low_tmp_offset = (heads_h_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = low_tmp_offset + low_tmp_count * sizeof(float);
        void *tmp = hip_tmp_alloc(tmp_bytes, "attention output a hipblas");
        if (!tmp) return 0;
        __half *heads_h = (__half *)tmp;
        float *low_packed = (float *)((char *)tmp + low_tmp_offset);
        hip_launch_attention_pack_group_heads_f16(
                heads_h,
                (const float *)heads->ptr,
                n_tokens,
                n_groups,
                group_dim,
                heads_h_count);
        if (!hip_ok(hipGetLastError(), "attention_output_q8_a pack launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        hipblasStatus_t st = hipblasGemmStridedBatchedEx(g_hipblas,
                                                       HIPBLAS_OP_T,
                                                       HIPBLAS_OP_N,
                                                       (int)rank,
                                                       (int)n_tokens,
                                                       (int)group_dim,
                                                       &alpha,
                                                       out_a_f16,
                                                       HIPBLAS_R_16F,
                                                       (int)group_dim,
                                                       (long long)rank * group_dim,
                                                       heads_h,
                                                       HIPBLAS_R_16F,
                                                       (int)group_dim,
                                                       (long long)n_tokens * group_dim,
                                                       &beta,
                                                       low_packed,
                                                       HIPBLAS_R_32F,
                                                       (int)rank,
                                                       (long long)rank * n_tokens,
                                                       (int)n_groups,
                                                       HIPBLAS_COMPUTE_32F,
                                                       HIPBLAS_GEMM_DEFAULT);
        if (!hipblas_ok(st, "attention output a gemm")) return 0;
        attention_unpack_group_low_kernel<<<(low_tmp_count + 255) / 256, 256>>>(
                (float *)low->ptr,
                low_packed,
                n_tokens,
                n_groups,
                rank);
        if (!hip_ok(hipGetLastError(), "attention_output_q8_a unpack launch")) return 0;
    } else {
        if (rope) return 0;
        const uint64_t x_rows = (uint64_t)n_tokens * n_groups;
        const uint64_t xq_bytes = x_rows * blocks_a * 32u;
        const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
        const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
        void *tmp = hip_tmp_alloc(tmp_bytes, "attention output a q8 prequant");
        if (!tmp) return 0;
        int8_t *xq = (int8_t *)tmp;
        float *xscale = (float *)((char *)tmp + scale_offset);
        const int use_dp4a = 1;
        dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
        quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq,
                                                xscale,
                                                (const float *)heads->ptr,
                                                group_dim,
                                                blocks_a);
        if (!hip_ok(hipGetLastError(), "attention_output_q8_a prequant launch")) return 0;
        dim3 grid_a(((unsigned)low_dim + 7u) / 8u, (unsigned)n_tokens, 1);
        grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr,
                                                          out_a,
                                                          xq,
                                                          xscale,
                                                          group_dim,
                                                          rank,
                                                          n_groups,
                                                          n_tokens,
                                                          blocks_a,
                                                          use_dp4a);
        if (!hip_ok(hipGetLastError(), "attention_output_q8_a preq launch")) return 0;
    }

    if (attn_output_hipblas) {
        if (hip_matmul_q8_0_tensor_f16_gemm(out,
                                             model_map,
                                             model_size,
                                             out_b_offset,
                                             low_dim,
                                             out_dim,
                                             low,
                                             n_tokens,
                                             "attn_output_b")) {
            return 1;
        }
    }
    return hip_matmul_q8_0_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           out_b_offset,
                                           low_dim,
                                           out_dim,
                                           low,
                                           n_tokens,
                                           "attn_output_b");
}

extern "C" int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    return attention_output_q8_batch_launch(out, low, group_tmp, low_tmp,
                                            model_map, model_size, out_a_offset,
                                            out_b_offset, group_dim, rank,
                                            n_groups, out_dim, heads, n_tokens,
                                            NULL);
}

extern "C" int ds4_gpu_attention_output_q8_batch_inv_rope_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens,
        uint32_t                head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow) {
    /* Held to prompt-chunk width: the rotation moves into registers ahead of an
     * F16 conversion, which is the same value the separate pass produced, but
     * this backend builds with -ffast-math and the surrounding expression
     * changes, so the narrow verification widths keep the separate launches. */
    if (n_tokens < DS4_ROCM_ATTENTION_WIDE_ROWS) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    const ds4_attn_pack_rope rope = {
        head_dim, n_rot, pos0, n_ctx_orig, /*inverse=*/1,
        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
    };
    return attention_output_q8_batch_launch(out, low, group_tmp, low_tmp,
                                            model_map, model_size, out_a_offset,
                                            out_b_offset, group_dim, rank,
                                            n_groups, out_dim, heads, n_tokens,
                                            &rope);
#else
    (void)out; (void)low; (void)group_tmp; (void)low_tmp; (void)model_map;
    (void)model_size; (void)out_a_offset; (void)out_b_offset; (void)group_dim;
    (void)rank; (void)n_groups; (void)out_dim; (void)heads; (void)head_dim;
    (void)n_rot; (void)pos0; (void)n_ctx_orig; (void)freq_base;
    (void)freq_scale; (void)ext_factor; (void)attn_factor; (void)beta_fast;
    (void)beta_slow;
    return 0;
#endif
}

extern "C" int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    if (out_a_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < low_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            hip_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    if (!out_a) return 0;
    /* Match the production HIP decode path for the CyberNeurova attention-output
     * A projection.  The full-row Q8 reduction is numerically close but crosses
     * FP8 KV midpoints in downstream layers; split-K16x8 preserves the same
     * accumulation shape used by the old-HIP backend and is also cache-friendly. */
    if (!hip_runtime_config()->disable_splitk_attn_out_low &&
        group_dim == 4096u && rank == 1024u && n_groups == 8u && blocks_a == 128u) {
        const uint32_t n_splits = 8u;
        float *partial = (float *)hip_tmp_alloc((uint64_t)n_splits * low_dim * sizeof(float), "attention output low splitk");
        if (!partial) return 0;
        grouped_q8_0_a_partial16_w32_kernel<<<dim3((unsigned)((low_dim + 31u) / 32u), 8u),
                                              1024u, 512u * sizeof(float)>>>(
                partial,
                out_a,
                (const float *)heads->ptr,
                n_groups,
                (uint32_t)rank,
                blocks_a * 34u);
        if (!hip_ok(hipGetLastError(), "attention_output_low_q8 splitk8 partial launch")) return 0;
        q8_partial_sum8_kernel<<<(unsigned)((low_dim + 255u) / 256u), 256>>>(
                (float *)low->ptr,
                partial,
                (uint32_t)low_dim);
        return hip_ok(hipGetLastError(), "attention_output_low_q8 splitk sum launch");
    }
    if (!hip_q8_prequant_decode_enabled()) {
        if ((group_dim & 31u) == 0u && group_dim <= 4096u &&
            (rank % 64u) == 0u) {
            const unsigned rows_per_block = 64u;
            grouped_q8_0_a_f32_sharedx_rows_w32_2row_kernel<<<
                    (unsigned)((low_dim + rows_per_block - 1u) /
                               rows_per_block),
                    1024u,
                    (size_t)group_dim * sizeof(float)>>>(
                    (float *)low->ptr,
                    out_a,
                    (const float *)heads->ptr,
                    n_groups,
                    (uint32_t)blocks_a,
                    rank,
                    blocks_a * 34u);
            return hip_ok(hipGetLastError(),
                           "attention_output_low_q8 f32 sharedx launch");
        }
        grouped_q8_0_a_f32_warp8_kernel<<<
                ((unsigned)low_dim + 7u) / 8u, 256>>>(
                (float *)low->ptr,
                out_a,
                (const float *)heads->ptr,
                group_dim,
                rank,
                n_groups,
                blocks_a);
        return hip_ok(hipGetLastError(),
                       "attention_output_low_q8 f32 launch");
    }

    const uint64_t x_rows = (uint64_t)n_groups;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes =
        scale_offset + x_rows * blocks_a * sizeof(float);
    void *tmp = hip_tmp_alloc(tmp_bytes,
                               "attention output low q8 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const ds4_rocm_runtime_config *cfg = hip_runtime_config();
    const int use_dp4a = 1;
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
            xq,
            xscale,
            (const float *)heads->ptr,
            group_dim,
            blocks_a);
    if (!hip_ok(hipGetLastError(),
                 "attention_output_low_q8 prequant launch")) {
        return 0;
    }
    const uint32_t rows_per_block = cfg->attn_out_low_decode_rpb;
    dim3 grid_a(
            ((unsigned)low_dim + rows_per_block - 1u) / rows_per_block,
            1,
            1);
    grouped_q8_0_a_preq_warp8_kernel<<<grid_a,
                                       rows_per_block * 32u>>>(
            (float *)low->ptr,
            out_a,
            xq,
            xscale,
            group_dim,
            rank,
            n_groups,
            1,
            blocks_a,
            use_dp4a);
    return hip_ok(hipGetLastError(),
                   "attention_output_low_q8 launch");
}

/* Non-causal block attention for the DSpark drafter.
 *
 * The drafted block is produced in one pass, so every block row must see the
 * injected target KV *and* every other block row. That is the opposite of the
 * target's causal window, hence a dedicated launcher rather than a mask
 * variation on the causal kernels. */
extern "C" int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map ||
        n_tokens == 0u || n_head == 0u || head_dim == 0u ||
        n_raw == 0u || raw_cap == 0u || n_raw > raw_cap ||
        raw_start >= raw_cap ||
        !hip_model_range_fits(model_size, sinks_offset,
                              (uint64_t)n_head * sizeof(float)) ||
        !hip_tensor_has_elems3(heads, n_tokens, n_head, head_dim, sizeof(float)) ||
        !hip_tensor_has_elems3(q, n_tokens, n_head, head_dim, sizeof(float)) ||
        !hip_tensor_has_elems2(raw_kv, raw_cap, head_dim, sizeof(float))) {
        return 0;
    }
    /* The kernel stages one score per visible key in shared memory. */
    const uint64_t shared_bytes = (uint64_t)n_raw * sizeof(float);
    if (shared_bytes > 32768u) return 0;
    const float *sinks = (const float *)hip_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "dspark_attn_sinks");
    if (!sinks) return 0;
    dim3 grid(n_tokens, n_head, 1);
    attention_noncausal_raw_batch_heads_kernel<<<grid, 256, (size_t)shared_bytes>>>(
            (float *)heads->ptr,
            sinks,
            (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_tokens,
            n_raw,
            raw_cap,
            raw_start,
            n_head,
            head_dim);
    return hip_ok(hipGetLastError(), "dspark noncausal attention launch");
}
