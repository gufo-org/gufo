extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!hip_tensor_has_f32(out, n) || !hip_tensor_has_f32(gate, n) || !hip_tensor_has_f32(up, n)) return 0;
    if (n == 0u) return 1;
    swiglu_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return hip_ok(hipGetLastError(), "swiglu launch");
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
  if (!gate || !up || !mid || !model_map || !x || in_dim == 0u ||
      out_dim == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
    return 0;
  }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t x_bytes = 0;
    uint64_t out_bytes = 0;
    if (!hip_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !hip_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !hip_u64_mul3_checked(in_dim, 1u, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(out_dim, 1u, sizeof(float), &out_bytes) ||
        !hip_tensor_has_bytes(x, x_bytes) ||
        !hip_tensor_has_bytes(gate, out_bytes) ||
        !hip_tensor_has_bytes(up, out_bytes) ||
        !hip_tensor_has_bytes(mid, out_bytes)) {
      return 0;
    }

    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                             model_map, model_size,
                                             gate_offset, up_offset,
                                             in_dim, out_dim, out_dim,
                                             x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}

static int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

static hipStream_t g_shared_gate_up_stream = NULL;
static hipEvent_t g_shared_gate_up_ready_event = NULL;
static void *g_shared_gate_up_tmp = NULL;
static uint64_t g_shared_gate_up_tmp_bytes = 0;
static int g_shared_gate_up_pending = 0;

static int hip_shared_gate_up_async_wait_internal(void) {
    if (!g_shared_gate_up_pending) return 1;
    hipError_t err = hipStreamSynchronize(g_shared_gate_up_stream);
    g_shared_gate_up_pending = 0;
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async wait failed: %s\n", hipGetErrorString(err));
        (void)hipGetLastError();
        return 0;
    }
    return 1;
}

static void *hip_shared_gate_up_async_tmp_alloc(uint64_t bytes) {
    if (bytes == 0) return NULL;
    if (g_shared_gate_up_tmp_bytes >= bytes) return g_shared_gate_up_tmp;
    if (g_shared_gate_up_tmp) {
        (void)hip_shared_gate_up_async_wait_internal();
        (void)hipFree(g_shared_gate_up_tmp);
        g_shared_gate_up_tmp = NULL;
        g_shared_gate_up_tmp_bytes = 0;
    }
    void *ptr = NULL;
    hipError_t err = hipMalloc(&ptr, (size_t)bytes);
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async temp alloc failed (%.2f MiB): %s\n",
                (double)bytes / 1048576.0, hipGetErrorString(err));
        (void)hipGetLastError();
        return NULL;
    }
    g_shared_gate_up_tmp = ptr;
    g_shared_gate_up_tmp_bytes = bytes;
    return g_shared_gate_up_tmp;
}

static void hip_shared_gate_up_async_cleanup(void) {
    if (g_shared_gate_up_stream) {
        (void)hip_shared_gate_up_async_wait_internal();
    }
    if (g_shared_gate_up_tmp) {
        (void)hipFree(g_shared_gate_up_tmp);
        g_shared_gate_up_tmp = NULL;
        g_shared_gate_up_tmp_bytes = 0;
    }
    if (g_shared_gate_up_ready_event) {
        (void)hipEventDestroy(g_shared_gate_up_ready_event);
        g_shared_gate_up_ready_event = NULL;
    }
    if (g_shared_gate_up_stream) {
        (void)hipStreamDestroy(g_shared_gate_up_stream);
        g_shared_gate_up_stream = NULL;
    }
}

static int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    uint64_t x_bytes = 0, out_bytes = 0;
    if (!gate || !up || !mid || !model_map || !x || n_tok == 0 ||
        (in_dim & 31u) != 0u || in_dim == 0u || out_dim == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        !hip_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || gate->bytes < out_bytes ||
        up->bytes < out_bytes || mid->bytes < out_bytes) {
      return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0;
    if (!hip_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !hip_u64_mul_checked(out_dim, row_bytes, &weight_bytes))
      return 0;
    if (gate_offset > model_size || up_offset > model_size ||
        weight_bytes > model_size - gate_offset ||
        weight_bytes > model_size - up_offset) {
      return 0;
    }
    const char *wg = hip_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8_batch");
    const char *wu = hip_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8_batch");
    if (!wg || !wu) return 0;

    const uint32_t rows_per_block = 32u;
    const uint32_t tile = 16u;
    const uint32_t block_tile = 16u;
    const dim3 grid((uint32_t)((out_dim + rows_per_block - 1u) / rows_per_block),
                    (uint32_t)((n_tok + tile - 1u) / tile),
                    1u);
    const size_t shmem = (size_t)tile * block_tile * 32u * sizeof(float);
    constexpr int store_gate_up = 0;
#define DS4_LAUNCH_SHARED_GU_BATCH(TT, BT) \
    shared_gate_up_swiglu_q8_0_batch_sharedx_w32_kernel<TT, BT><<<grid, rows_per_block * 32u, shmem>>>( \
            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr, \
            reinterpret_cast<const unsigned char *>(wg), reinterpret_cast<const unsigned char *>(wu), \
            (const float *)x->ptr, (uint32_t)blocks, (uint32_t)out_dim, (uint32_t)n_tok, row_bytes, store_gate_up)
    DS4_LAUNCH_SHARED_GU_BATCH(16u, 16u);
#undef DS4_LAUNCH_SHARED_GU_BATCH
    return hip_ok(hipGetLastError(), "shared gate/up fused q8 batch launch");
}
