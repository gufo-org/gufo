extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (n_rot > head_dim || !hip_tensor_has_elems2(x, n_tok, head_dim, sizeof(float))) return 0;
    if (n_tok == 0u || head_dim == 0u) return 1;
    const uint32_t n_nope = head_dim - n_rot;
    if (n_nope == 0) return 1;
    const uint32_t groups = (n_nope + 63u) / 64u;
    fp8_kv_quantize_kernel<<<dim3(n_tok, groups), 64>>>(
            (float *)x->ptr, nullptr, n_tok, head_dim, n_rot);
    return hip_ok(hipGetLastError(), "fp8_kv_quantize launch");
}

extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_mirror_f16_tensor(
        ds4_gpu_tensor *x,
        ds4_gpu_tensor *mirror_f16,
        uint32_t n_tok,
        uint32_t head_dim,
        uint32_t n_rot) {
    if (n_rot > head_dim ||
        !hip_tensor_has_elems2(x, n_tok, head_dim, sizeof(float)) ||
        !hip_tensor_has_elems2(
                mirror_f16, n_tok, head_dim, sizeof(half))) {
        return 0;
    }
    if (n_tok == 0u || head_dim == 0u) return 1;
    const uint32_t groups = (head_dim + 63u) / 64u;
    fp8_kv_quantize_kernel<<<dim3(n_tok, groups), 64>>>(
            (float *)x->ptr,
            (half *)mirror_f16->ptr,
            n_tok,
            head_dim,
            n_rot);
    return hip_ok(
            hipGetLastError(), "fp8_kv_quantize mirror f16 launch");
}

extern "C" int ds4_gpu_tensor_convert_f32_to_f16(
        ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src,
        uint64_t count) {
    if (!dst || !src ||
        count > dst->bytes / sizeof(half) ||
        count > src->bytes / sizeof(float)) {
        return 0;
    }
    if (count == 0u) return 1;
    hip_launch_f32_to_f16(
            (half *)dst->ptr,
            (const float *)src->ptr,
            count);
    return hip_ok(hipGetLastError(), "tensor f32 to f16 launch");
}
