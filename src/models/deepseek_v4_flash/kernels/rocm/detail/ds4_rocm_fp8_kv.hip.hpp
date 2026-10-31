// DS4 ROCm FP8 KV quantization and raw-cache store kernels.
//
// This file is included from ds4_rocm.hip.cpp (same translation unit) so these
// kernels can reuse the backend's existing device helpers without HIP device
// linking or behavior changes.

__global__ static void fp8_kv_quantize_kernel(
        float *x,
        half *mirror_f16,
        uint32_t n_tok,
        uint32_t head_dim,
        uint32_t n_rot) {
    const uint32_t row = blockIdx.x;
    const uint32_t grp = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t off = grp * 64u;
    if (row >= n_tok || off >= head_dim) return;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
    float v = 0.0f;
    if (tid < 64u && off + tid < head_dim) v = xr[off + tid];
    const float scale = dsv4_fp8_kv_group_scale(
            scratch, tid, (tid < 64u && off + tid < n_nope) ? fabsf(v) : 0.0f);
    if (tid < 64u && off + tid < head_dim) {
        const uint32_t d = off + tid;
        if (d < n_nope) {
            v = dsv4_fp8_kv_round(v, scale);
            xr[d] = v;
        }
        if (mirror_f16) {
            mirror_f16[(uint64_t)row * head_dim + d] = __float2half(v);
        }
    }
}

__global__ static void store_raw_kv_batch_kernel(float *raw, const float *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t t = gid / head_dim;
    uint32_t row = (pos0 + t) % raw_cap;
    const uint16_t hb = f32_to_f16_bits_hip_round(kv[(uint64_t)t * head_dim + d]);
    raw[(uint64_t)row * head_dim + d] = f16_bits_to_f32(hb);
}
