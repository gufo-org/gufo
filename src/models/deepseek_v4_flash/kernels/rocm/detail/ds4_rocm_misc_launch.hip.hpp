extern "C" int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!hip_tensor_has_f32(out, n) || !hip_tensor_has_f32(a, n) || !hip_tensor_has_f32(b, n)) return 0;
    if (n == 0u) return 1;
    add_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return hip_ok(hipGetLastError(), "add launch");
}

/* See hip_f16_input_publish: publish the normalized attention rows once so the
 * layer's projections share one F16 mirror instead of each converting them. The
 * graph clears the mirror at every layer boundary. Returns 0 when the mirror
 * could not be produced, which is not an error -- consumers convert locally. */
extern "C" int ds4_gpu_publish_f16_input_tensor(const ds4_gpu_tensor *x,
                                                 uint64_t count) {
    if (!x || count == 0u) {
        hip_f16_input_clear();
        return 0;
    }
    uint64_t bytes = 0;
    if (!hip_u64_mul_checked(count, sizeof(float), &bytes) || x->bytes < bytes) {
        hip_f16_input_clear();
        return 0;
    }
    return hip_f16_input_publish((const float *)x->ptr, count);
}

extern "C" void ds4_gpu_clear_f16_input(void) { hip_f16_input_clear(); }
