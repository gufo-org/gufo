extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!hip_tensor_has_f32(out, n) || !hip_tensor_has_f32(gate, n) || !hip_tensor_has_f32(up, n)) return 0;
    if (n == 0u) return 1;
    swiglu_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return hip_ok(hipGetLastError(), "swiglu launch");
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
    ds4_gpu_tensor* gate, ds4_gpu_tensor* up, ds4_gpu_tensor* mid,
    const void* model_map, uint64_t model_size, uint64_t gate_offset,
    uint64_t up_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor* x, float clamp, uint32_t n_tok) {
  if (!gate || !up || !mid || !model_map || !x || in_dim == 0u ||
      out_dim == 0u || n_tok == 0u || out_dim > UINT32_MAX / n_tok ||
      in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
    return 0;
  }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t x_bytes = 0;
    uint64_t out_bytes = 0;
    if (!hip_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !hip_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !hip_u64_mul3_checked(in_dim, n_tok, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(out_dim, n_tok, sizeof(float), &out_bytes) ||
        !hip_tensor_has_bytes(x, x_bytes) ||
        !hip_tensor_has_bytes(gate, out_bytes) ||
        !hip_tensor_has_bytes(up, out_bytes) ||
        !hip_tensor_has_bytes(mid, out_bytes)) {
      return 0;
    }

    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up, model_map, model_size,
                                           gate_offset, up_offset, in_dim,
                                           out_dim, out_dim, x, n_tok) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)(out_dim * n_tok),
                                 clamp, 1.0f);
}
