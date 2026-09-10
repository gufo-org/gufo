/* The vec4 expand kernels read and write four adjacent embedding lanes as one
 * 16-byte access, which needs the row pitch to be a multiple of four and every
 * base 16-byte aligned. hipMalloc over-aligns, but these tensors can be views
 * into a larger arena, so check rather than assume. */
static inline bool hip_hc_vec4_ok(uint32_t n_embd, const void *a, const void *b,
                                  const void *c) {
    return (n_embd & 3u) == 0u &&
           (((uintptr_t)a | (uintptr_t)b | (uintptr_t)c) & 15u) == 0u;
}

static int hip_hc_flat_token_count(const ds4_gpu_tensor *out, uint32_t n_embd, uint64_t *n_tokens) {
    if (!out || n_embd == 0u || !n_tokens) return 0;
    uint64_t row_bytes = 0;
    if (!hip_u64_mul3_checked(n_embd, 1u, sizeof(float), &row_bytes) ||
        row_bytes == 0u || out->bytes < row_bytes || (out->bytes % row_bytes) != 0u) return 0;
    *n_tokens = out->bytes / row_bytes;
    return *n_tokens != 0u && *n_tokens <= UINT32_MAX;
}

static int hip_hc_hc_token_count(const ds4_gpu_tensor *out_hc, uint32_t n_embd, uint32_t n_hc, uint64_t *n_tokens) {
    if (!out_hc || n_embd == 0u || n_hc == 0u || !n_tokens) return 0;
    uint64_t row_elems = 0, row_bytes = 0;
    if (!hip_u64_mul_checked(n_hc, n_embd, &row_elems) ||
        !hip_u64_mul_checked(row_elems, sizeof(float), &row_bytes) ||
        row_bytes == 0u || out_hc->bytes < row_bytes || (out_hc->bytes % row_bytes) != 0u) return 0;
    *n_tokens = out_hc->bytes / row_bytes;
    return *n_tokens != 0u && *n_tokens <= UINT32_MAX;
}

static int hip_hc_mix_width(uint32_t n_hc, uint64_t *mix_hc) {
    if (n_hc == 0u || !mix_hc) return 0;
    const uint64_t h = (uint64_t)n_hc;
    const uint64_t mix = 2ull * h + h * h;
    if (mix > UINT32_MAX) return 0;
    *mix_hc = mix;
    return 1;
}

extern "C" int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, residual_bytes = 0, weights_bytes = 0;
    if (!out || !residual_hc || !weights || n_hc == 0u ||
        !hip_hc_flat_token_count(out, n_embd, &n_tokens64) ||
        !hip_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &residual_bytes) ||
        !hip_u64_mul3_checked(n_tokens64, n_hc, sizeof(float), &weights_bytes) ||
        residual_hc->bytes < residual_bytes || weights->bytes < weights_bytes) return 0;
    uint32_t n_tokens = (uint32_t)n_tokens64;
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)weights->ptr,
        n_embd, n_hc, n_tokens, n_hc);
    return hip_ok(hipGetLastError(), "hc_weighted_sum launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)hip_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)hip_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    hc_split_weighted_sum_fused_kernel<<<(uint32_t)n_rows, 256>>>(
            (float *)out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const float *)residual_hc->ptr,
            scale,
            base,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps);
    return hip_ok(hipGetLastError(), "hc split weighted sum launch");
}
/* Fused HC pre-block computation for the supported 4x4096 row and F16 GGUF
 * weights. Invalid storage or a failed launch is an error; do not silently
 * change the numerical path. Narrow verification retains scalar arithmetic. */
extern "C" int ds4_gpu_hc_norm_mix_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mix,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                mix_weight_offset,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                n_rows,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    /* Four tokens per workgroup, 256 lanes each, sixteen columns per lane.
     * More tokens amortize the 786 KiB projection weight over more rows and
     * fewer leave register room for a second resident workgroup; at 152 VGPRs
     * and no scratch this kernel gets one workgroup per CU either way, and two
     * and eight tokens measured 494.0 and 493.1 tok/s against four's 498.6. */
    constexpr uint32_t TOK = 4u, TPT = 256u, COLS = 16u;
    if (!out || !mix || !split || !residual_hc || !model_map ||
        n_hc != 4u || n_rows == 0u || n_embd != TPT * COLS) {
        return 0;
    }
    const uint64_t hc_dim = 4ull * n_embd;
    const uint64_t mix_hc = 24ull;
    uint64_t out_bytes = 0, residual_bytes = 0, mix_bytes = 0, weight_bytes = 0;
    if (!hip_u64_mul3_checked(n_rows, n_embd, sizeof(float), &out_bytes) ||
        !hip_u64_mul3_checked(n_rows, hc_dim, sizeof(float), &residual_bytes) ||
        !hip_u64_mul3_checked(n_rows, mix_hc, sizeof(float), &mix_bytes) ||
        !hip_u64_mul3_checked(mix_hc, hc_dim, sizeof(uint16_t), &weight_bytes) ||
        out->bytes < out_bytes || residual_hc->bytes < residual_bytes ||
        mix->bytes < mix_bytes || split->bytes < mix_bytes ||
        scale_offset > model_size ||
        3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size ||
        mix_hc * sizeof(float) > model_size - base_offset ||
        mix_weight_offset > model_size ||
        weight_bytes > model_size - mix_weight_offset) {
        return 0;
    }
    const float *scale = (const float *)hip_model_range_ptr(
            model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)hip_model_range_ptr(
            model_map, base_offset, mix_hc * sizeof(float), "hc_base");
    const __half *mix_w = (const __half *)hip_model_range_ptr(
            model_map, mix_weight_offset, weight_bytes, "hc_mix_fused");
    if (!scale || !base || !mix_w) return 0;
    /* The weight is read eight halves at a time and the row in float4. */
    if (((uintptr_t)mix_w & 15u) != 0u ||
        ((uintptr_t)residual_hc->ptr & 15u) != 0u ||
        ((uintptr_t)out->ptr & 15u) != 0u) {
        return 0;
    }
    hc4_norm_mix_split_weighted_sum_kernel<TOK, TPT, COLS>
            <<<(n_rows + TOK - 1u) / TOK, TOK * TPT>>>(
            (float *)out->ptr,
            (float *)mix->ptr,
            (float *)split->ptr,
            (const float *)residual_hc->ptr,
            mix_w,
            scale,
            base,
            n_embd,
            n_rows,
            sinkhorn_iters,
            eps,
            norm_eps);
    return hip_ok(hipGetLastError(), "hc norm mix split weighted sum launch");
}

extern "C" int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        norm_out->bytes < out->bytes ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset ||
        norm_weight_offset > model_size ||
        (uint64_t)n_embd * sizeof(float) > model_size - norm_weight_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)hip_model_range_ptr(model_map, scale_offset,
            3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)hip_model_range_ptr(model_map, base_offset,
            mix_bytes, "hc_base");
    const float *norm_w = (const float *)hip_model_range_ptr(model_map, norm_weight_offset,
            (uint64_t)n_embd * sizeof(float), "hc_norm_weight");
    if (!scale || !base || !norm_w) return 0;
    hc_split_weighted_sum_norm_fused_kernel<<<(uint32_t)n_rows, 256>>>(
            (float *)out->ptr,
            (float *)norm_out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const float *)residual_hc->ptr,
            scale,
            base,
            norm_w,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
    return hip_ok(hipGetLastError(), "hc split weighted sum norm launch");
}
extern "C" int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const float *scale = (const float *)hip_model_range_ptr(model_map, scale_offset, sizeof(float), "output_hc_scale");
    const float *base = (const float *)hip_model_range_ptr(model_map, base_offset, row_bytes, "output_hc_base");
    if (!scale || !base) return 0;
    uint64_t n = n_tokens * n_hc;
    output_hc_weights_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const float *)pre->ptr,
            scale,
            base,
            n_hc,
            (uint32_t)n_tokens,
            eps);
    return hip_ok(hipGetLastError(), "output hc weights launch");
}
extern "C" int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, hc_bytes = 0, split_bytes = 0, mix_hc64 = 0;
    if (!out_hc || !block_out || !residual_hc || !split ||
        !hip_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !hip_hc_mix_width(n_hc, &mix_hc64) ||
        !hip_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !hip_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !hip_u64_mul3_checked(n_tokens64, mix_hc64, sizeof(float), &split_bytes) ||
        block_out->bytes < flat_bytes || residual_hc->bytes < hc_bytes || split->bytes < split_bytes) return 0;
    uint32_t n_tokens = (uint32_t)n_tokens64;
    if (n_hc == 4u) {
        const uint64_t n = (uint64_t)n_tokens * n_embd;
        if (hip_hc_vec4_ok(n_embd, out_hc->ptr, block_out->ptr, residual_hc->ptr)) {
            hc_expand4_vec4_kernel<<<((n >> 2u) + 255) / 256, 256>>>(
                    (float *)out_hc->ptr, (const float *)block_out->ptr,
                    (const float *)residual_hc->ptr, (const float *)split->ptr,
                    n_embd, n_tokens);
            return hip_ok(hipGetLastError(), "hc_expand_split4 vec4 launch");
        }
        hc_expand4_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    (const float *)split->ptr,
                                                    n_embd,
                                                    n_tokens);
        return hip_ok(hipGetLastError(), "hc_expand_split4 launch");
    }
    uint32_t mix_hc = (uint32_t)mix_hc64;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 0);
    return hip_ok(hipGetLastError(), "hc_expand_split launch");
}
extern "C" int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, hc_bytes = 0, split_bytes = 0, mix_hc64 = 0;
    if (!out_hc || !block_out || !block_add || !residual_hc || !split ||
        !hip_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !hip_hc_mix_width(n_hc, &mix_hc64) ||
        !hip_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !hip_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !hip_u64_mul3_checked(n_tokens64, mix_hc64, sizeof(float), &split_bytes) ||
        block_out->bytes < flat_bytes || block_add->bytes < flat_bytes ||
        residual_hc->bytes < hc_bytes || split->bytes < split_bytes) return 0;
    uint32_t n_tokens = (uint32_t)n_tokens64;
    if (n_hc == 4u) {
        const uint64_t n = (uint64_t)n_tokens * n_embd;
        hc_expand4_add_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                        (const float *)block_out->ptr,
                                                        (const float *)block_add->ptr,
                                                        (const float *)residual_hc->ptr,
                                                        (const float *)split->ptr,
                                                        n_embd,
                                                        n_tokens);
        return hip_ok(hipGetLastError(), "hc_expand_add_split4 launch");
    }
    uint32_t mix_hc = (uint32_t)mix_hc64;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1);
    return hip_ok(hipGetLastError(), "hc_expand_add_split launch");
}
/* Fused form of ds4_gpu_hc_expand_add_split_tensor: `down_h` holds the routed
 * per-expert F16 rows and the 6-way sum happens here. Returns 0 when the shape
 * is unsupported so the caller keeps the separate sum plus expand. */
extern "C" int ds4_gpu_hc_expand_add_split_moesum_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *down_h,
        const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc,
        uint32_t n_expert, uint32_t n_tokens) {
    uint64_t flat_bytes = 0, hc_bytes = 0, split_bytes = 0, mix_hc64 = 0,
             down_bytes = 0;
    if (!out_hc || !down_h || !block_add || !residual_hc || !split ||
        n_hc != 4u || n_embd == 0u || n_expert == 0u || n_tokens == 0u ||
        !hip_hc_mix_width(n_hc, &mix_hc64) ||
        !hip_u64_mul3_checked(n_tokens, n_embd, sizeof(float), &flat_bytes) ||
        !hip_u64_mul3_checked(n_tokens, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !hip_u64_mul3_checked(n_tokens, mix_hc64, sizeof(float), &split_bytes) ||
        !hip_u64_mul3_checked((uint64_t)n_tokens * n_expert, n_embd, sizeof(__half), &down_bytes) ||
        block_add->bytes < flat_bytes || residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes || down_h->bytes < down_bytes ||
        out_hc->bytes < hc_bytes) {
        return 0;
    }
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (hip_hc_vec4_ok(n_embd, out_hc->ptr, block_add->ptr, residual_hc->ptr) &&
        ((uintptr_t)down_h->ptr & 7u) == 0u) {
        hc_expand4_add_moesum_vec4_kernel<<<((n >> 2u) + 255) / 256, 256>>>(
                (float *)out_hc->ptr, (const __half *)down_h->ptr,
                (const float *)block_add->ptr, (const float *)residual_hc->ptr,
                (const float *)split->ptr, n_embd, n_expert, n_tokens);
        return hip_ok(hipGetLastError(),
                      "hc_expand_add_split4 moesum vec4 launch");
    }
    hc_expand4_add_moesum_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out_hc->ptr, (const __half *)down_h->ptr,
            (const float *)block_add->ptr, (const float *)residual_hc->ptr,
            (const float *)split->ptr, n_embd, n_expert, n_tokens);
    return hip_ok(hipGetLastError(), "hc_expand_add_split4 moesum launch");
}

extern "C" int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return hip_matmul_q8_0_hc_expand_tensor_labeled(out_hc, shared_out,
                                                    model_map, model_size,
                                                    weight_offset,
                                                    in_dim, out_dim,
                                                    shared_mid,
                                                    routed_out,
                                                    residual_hc,
                                                    split,
                                                    n_embd, n_hc,
                                                    "shared_down_hc_expand");
}

extern "C" int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return hip_matmul_q8_0_hc_expand_tensor_labeled(out_hc, block_out,
                                                    model_map, model_size,
                                                    weight_offset,
                                                    in_dim, out_dim,
                                                    x,
                                                    NULL,
                                                    residual_hc,
                                                    split,
                                                    n_embd, n_hc,
                                                    "q8_hc_expand");
}

extern "C" int ds4_gpu_spec_row_argmax_tensor(
        ds4_gpu_tensor       *out_index,
        const ds4_gpu_tensor *logits,
        uint32_t                vocab,
        uint32_t                n_rows) {
    if (vocab == 0u || n_rows == 0u ||
        !hip_tensor_has_elems(out_index, n_rows, sizeof(int32_t)) ||
        !hip_tensor_has_elems2(logits, n_rows, vocab, sizeof(float))) {
        return 0;
    }
    spec_row_argmax_kernel<<<n_rows, 256>>>((int32_t *)out_index->ptr,
                                            (const float *)logits->ptr,
                                            vocab,
                                            n_rows);
    return hip_ok(hipGetLastError(), "spec row argmax launch");
}

extern "C" int ds4_gpu_dspark_capture_features_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *hc,
        uint32_t                out_row_stride,
        uint32_t                slot_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                n_rows) {
    if (!out || !hc || n_embd == 0u || n_hc == 0u || n_rows == 0u ||
        out_row_stride < slot_offset + n_embd ||
        !hip_tensor_has_elems2(out, n_rows, out_row_stride, sizeof(float)) ||
        !hip_tensor_has_elems3(hc, n_rows, n_hc, n_embd, sizeof(float))) {
        return 0;
    }
    const uint64_t n = (uint64_t)n_embd * n_rows;
    dspark_capture_features_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out->ptr + slot_offset,
            (const float *)hc->ptr,
            out_row_stride,
            n_embd,
            n_hc,
            n_rows);
    return hip_ok(hipGetLastError(), "dspark capture features launch");
}
