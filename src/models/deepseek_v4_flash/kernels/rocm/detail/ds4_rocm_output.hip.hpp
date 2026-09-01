// DS4 ROCm output/pointwise kernels.
//
// Included from ds4_rocm.hip.cpp in the same translation unit; API launch glue stays
// in ds4_rocm.hip.cpp for now.

__global__ static void output_hc_weights_kernel(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_tokens * n_hc;
    if (gid >= n) return;
    uint32_t h = gid % n_hc;
    float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
}


__global__ static void swiglu_kernel(float *out, const float *gate, const float *up, uint32_t n, float clamp, float weight) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    float s = g / (1.0f + expf(-g));
    out[i] = s * u * weight;
}

__global__ static void add_kernel(float *out, const float *a, const float *b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

__global__ static void add3_kernel(
        float       *out,
        const float *a,
        const float *b,
        const float *c,
        uint32_t     n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i] + c[i];
}

__global__ static void zero_kernel(float *out, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0f;
}

/* Per-row vocabulary argmax for the DSpark speculative verifier.
 *
 * One block owns one verification row.  Speculative decoding compares the
 * target's top token after every accepted draft position, so the whole point
 * is to leave the 129280-wide rows on the device and read back one index per
 * row instead of megabytes of logits.  Ties resolve to the lower token index
 * so the result matches the host argmax used by ordinary decode. */
__global__ static void spec_row_argmax_kernel(
        int32_t     *out_index,
        const float *logits,
        uint32_t     vocab,
        uint32_t     n_rows) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const float *values = logits + (uint64_t)row * vocab;

    float best_value = -INFINITY;
    uint32_t best_index = 0;
    for (uint32_t i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float value = values[i];
        if (value > best_value || (value == best_value && i < best_index)) {
            best_value = value;
            best_index = i;
        }
    }

    __shared__ float shared_value[256];
    __shared__ uint32_t shared_index[256];
    shared_value[threadIdx.x] = best_value;
    shared_index[threadIdx.x] = best_index;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            const float other_value = shared_value[threadIdx.x + stride];
            const uint32_t other_index = shared_index[threadIdx.x + stride];
            const float current_value = shared_value[threadIdx.x];
            const uint32_t current_index = shared_index[threadIdx.x];
            if (other_value > current_value ||
                (other_value == current_value && other_index < current_index)) {
                shared_value[threadIdx.x] = other_value;
                shared_index[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) out_index[row] = (int32_t)shared_index[0];
}
