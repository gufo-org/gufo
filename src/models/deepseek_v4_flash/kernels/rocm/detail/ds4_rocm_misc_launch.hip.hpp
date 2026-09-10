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
