static const void *g_model_host_base;
static uint64_t g_model_registered_size;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static hipStream_t g_model_upload_stream;
static hipblasHandle_t g_hipblas;
static int g_hipblas_ready;
/* Strix Halo detection. The optimized prefill routes below are scored on
 * gfx1151 only; every one of them keeps its generic fallback so an unexpected
 * device still runs. */
static int g_rocm_gfx1151;
/* Vendored llama.cpp MMQ tier availability (kernels/rocm/mmq). */
static int g_rocm_mmq_ready;
/*
 * Minimum widths for the independently retained prompt routes.
 *
 * Routed MMQ is accurate and faster from 32 rows. Dense MMQ and hipBLASLt
 * start at 64. The mixed-attention, fused norm/rope, padded output-B, and
 * inverse-rope routes remain at 96 because enabling them at 64 exceeded the
 * conversational maximum-error envelope. The 32/64/96 batch-versus-sequential
 * logit gate in deepseek_v4_flash_engine_test covers these boundaries.
 */
#define DS4_ROCM_WIDE_PREFILL_ROWS 64u
#define DS4_ROCM_ATTENTION_WIDE_ROWS 96u
#define DS4_ROCM_ROUTED_MMQ_ROWS 32u

/*
 * Retained hipBLASLt routing subset; see hipblaslt_route_mask.
 *
 * 23 is every site except the attention output-B fallback (bit 8). Each site
 * was measured against the pinned trajectory on its own: bits 1, 2, 4 and 16
 * each hold it at 116/128 rank sum 142, and bit 8 alone drops it to 114/128
 * rank sum 150. Bit 8 only fires when the 256x128 rocWMMA tile rejects the
 * shape, which at prompt-chunk width means an `n` that is not a multiple of
 * 128 -- so excluding it costs nothing at a 4,096-token prompt and removes the
 * drift entirely.
 */
#ifndef DS4_ROCM_LT_ROUTE_DEFAULT_MASK
#define DS4_ROCM_LT_ROUTE_DEFAULT_MASK 23
#endif
#ifdef __HIP_PLATFORM_AMD__
#include "ds4_rocm_hipblaslt.hip.hpp"
#endif
enum {
    DS4_ROCM_N_EXPERT = 256u,
    DS4_ROCM_N_EXPERT_USED = 6u,
    DS4_ROCM_COMPRESSOR_MAX_RATIO = 128u
};
#define DS4_ROCM_EXPERT_WEIGHT_SCALE 1.5f
#define DS4_ROCM_EXPERT_WEIGHT_SCALE_TOL 1.0e-6f

struct hip_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
};

struct hip_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct hip_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct hip_q8_f16_transpose_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

static std::vector<hip_model_range> g_model_ranges;
static std::vector<hip_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;

/*
 * Optional DSpark support-model mapping.
 *
 * The primary 80 GiB target GGUF owns g_model_host_base and the staged
 * O_DIRECT copier. The DSpark draft model is a second, much smaller file whose
 * tensors are addressed by the same (host_base, absolute offset) pair, so it
 * gets its own registered base and managed arena instead of displacing the
 * target's ranges. On Strix Halo this keeps the 5.58 GiB support copy out of
 * the fixed local-memory headroom used by prompt processing while preserving
 * ordinary GPU pointers and warm device bandwidth for drafting.
 */
/*
 * Verification-block dispatch mode.
 *
 * Prefill, decode, and resumed prefill all keep the kernel selection they were
 * tuned and baselined with. The small-batch routes are enabled only while a
 * DSpark verification block is being encoded, so enabling speculative decoding
 * cannot change any retained path's output, kernel choice, or throughput.
 */
static int g_small_batch_mode;

static uint32_t ds4_rocm_small_batch_limit(uint32_t configured) {
    return g_small_batch_mode ? configured : 0u;
}

static bool ds4_rocm_verifier_batch_mode() {
    return g_small_batch_mode == 1;
}

static bool ds4_rocm_support_batch_mode() {
    return g_small_batch_mode == 2;
}

static const void *g_support_host_base;
static uint64_t g_support_registered_size;
static char *g_support_arena;
static uint64_t g_support_arena_bytes;
static uint64_t g_support_arena_used;
static bool g_support_managed;
static std::vector<hip_model_range> g_support_ranges;
static std::vector<hip_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<hip_q8_f16_transpose_range> g_q8_f16_transpose_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_transpose_by_offset;
static uint64_t g_model_range_bytes;
static uint64_t g_q8_f16_bytes;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_budget_notice_printed;
static uint64_t g_model_load_progress_next;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;
static void *g_hip_tmp;
static uint64_t g_hip_tmp_bytes;
/*
 * One worker per common 528-672 MiB tensor span keeps the SN7100 near its
 * useful direct-I/O queue depth. The chunk is sized per span below, so sixteen
 * slots retain the same roughly 1 GiB peak staging budget for larger ranges.
 */
enum { DS4_ROCM_MODEL_STAGE_COUNT = 16 };
static void *g_model_stage_raw[DS4_ROCM_MODEL_STAGE_COUNT];
static void *g_model_stage[DS4_ROCM_MODEL_STAGE_COUNT];
static hipEvent_t g_model_stage_event[DS4_ROCM_MODEL_STAGE_COUNT];
static uint64_t g_model_stage_bytes;

static int hip_ok(hipError_t err, const char *what);

static int hip_u64_mul_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return 0;
    if (a != 0u && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int hip_u64_mul3_checked(uint64_t a, uint64_t b, uint64_t c, uint64_t *out) {
    uint64_t tmp = 0;
    return hip_u64_mul_checked(a, b, &tmp) && hip_u64_mul_checked(tmp, c, out);
}

static int hip_model_range_fits(uint64_t model_size, uint64_t offset, uint64_t bytes) {
    return offset <= model_size && bytes <= model_size - offset;
}

static int hip_tensor_has_bytes(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && t->bytes >= bytes;
}

static int hip_tensor_has_elems(const ds4_gpu_tensor *t, uint64_t elems, uint64_t elem_size) {
    uint64_t bytes = 0;
    return hip_u64_mul_checked(elems, elem_size, &bytes) && hip_tensor_has_bytes(t, bytes);
}

static int hip_tensor_has_elems2(const ds4_gpu_tensor *t, uint64_t a, uint64_t b, uint64_t elem_size) {
    uint64_t bytes = 0;
    return hip_u64_mul3_checked(a, b, elem_size, &bytes) && hip_tensor_has_bytes(t, bytes);
}

static int hip_tensor_has_elems3(const ds4_gpu_tensor *t, uint64_t a, uint64_t b, uint64_t c, uint64_t elem_size) {
    uint64_t ab = 0, elems = 0, bytes = 0;
    return hip_u64_mul_checked(a, b, &ab) &&
           hip_u64_mul_checked(ab, c, &elems) &&
           hip_u64_mul_checked(elems, elem_size, &bytes) &&
           hip_tensor_has_bytes(t, bytes);
}

static int hip_tensor_has_f32(const ds4_gpu_tensor *t, uint64_t elems) {
    return hip_tensor_has_elems(t, elems, sizeof(float));
}

static int hip_tensor_has_i32(const ds4_gpu_tensor *t, uint64_t elems) {
    return hip_tensor_has_elems(t, elems, sizeof(int32_t));
}

static int hip_tensor_has_f16(const ds4_gpu_tensor *t, uint64_t elems) {
    return hip_tensor_has_elems(t, elems, sizeof(__half));
}

static int hip_tensor_has_u16(const ds4_gpu_tensor *t, uint64_t elems) {
    return hip_tensor_has_elems(t, elems, sizeof(uint16_t));
}

static const char *hip_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
/* Tile geometry for dequant_q8_0_to_f16_transpose_tiled_kernel; see its
 * definition in ds4_rocm_q8.hip.hpp for why the transpose is tiled. */
#define DS4_Q8_T_TILE_I 32u
#define DS4_Q8_T_TILE_ROW 64u
#define DS4_Q8_T_LDS_PITCH (DS4_Q8_T_TILE_ROW + 8u)
__global__ static void dequant_q8_0_to_f16_transpose_tiled_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

static void hip_shared_gate_up_async_cleanup(void);

static void *hip_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_hip_tmp_bytes >= bytes) return g_hip_tmp;
    if (g_hip_tmp) {
        (void)hipFree(g_hip_tmp);
        g_hip_tmp = NULL;
        g_hip_tmp_bytes = 0;
    }
    void *ptr = NULL;
    hipError_t err = hipMalloc(&ptr, (size_t)bytes);
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "temp alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, hipGetErrorString(err));
        (void)hipGetLastError();
        return NULL;
    }
    g_hip_tmp = ptr;
    g_hip_tmp_bytes = bytes;
    return g_hip_tmp;
}

static int hip_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_ROCM_ATTENTION_SCORE_CAP - DS4_ROCM_ATTENTION_RAW_SCORE_CAP;
}

static const char *hip_support_range_ptr(const void *support_map, uint64_t offset, uint64_t bytes) {
    const uint64_t end = offset + bytes;
    if (end < offset) return NULL;
    for (const hip_model_range &r : g_support_ranges) {
        if (r.host_base == support_map && offset >= r.offset && end <= r.offset + r.bytes) {
            return r.device_ptr + (offset - r.offset);
        }
    }
    return NULL;
}

static const char *hip_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    if (bytes == 0) return (const char *)model_map + offset;

    if (g_support_host_base != NULL && model_map == g_support_host_base) {
        const char *support = hip_support_range_ptr(model_map, offset, bytes);
        if (support) return support;
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "support range rejected for %s: offset %llu "
                "(%llu bytes) is not resident\n",
                what ? what : "weights",
                (unsigned long long)offset,
                (unsigned long long)bytes);
        return NULL;
    }

    if (model_map != g_model_host_base) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range rejected for %s: unregistered model mapping\n",
                what ? what : "weights");
        return NULL;
    }

    const uint64_t end = offset + bytes;
    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const hip_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && end >= offset && bytes <= r.bytes) return r.device_ptr;
    }
    for (const hip_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && offset >= r.offset && end >= offset && end <= r.offset + r.bytes) {
            return r.device_ptr + (offset - r.offset);
        }
    }

    return hip_model_range_ptr_from_fd(model_map, offset, bytes, what);
}

static int hip_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return 1;

    const uint64_t end = offset + bytes;
    if (end < offset) return 0;
    for (const hip_model_range &r : g_model_ranges) {
        if (r.host_base == model_map &&
            offset >= r.offset &&
            end <= r.offset + r.bytes) {
            return 1;
        }
    }
    return 0;
}

static void hip_q8_f16_cache_release_all(void) {
    for (const hip_q8_f16_transpose_range &r : g_q8_f16_transpose_ranges) {
        (void)hipFree(r.device_ptr);
    }
    for (const hip_q8_f16_range &r : g_q8_f16_ranges) {
        (void)hipFree(r.device_ptr);
    }
    g_q8_f16_transpose_ranges.clear();
    g_q8_f16_transpose_by_offset.clear();
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f16_bytes = 0;
}

static int hip_env_present(const char *env) {
    if (env != NULL) return env[0] != '\0' && strcmp(env, "0") != 0;
    return 0;
}

static uint32_t hip_rows_per_block_or_default(uint32_t v, uint32_t def) {
    return (v == 1u || v == 2u || v == 4u || v == 8u || v == 16u || v == 32u) ? v : def;
}

struct ds4_rocm_runtime_config {
    int initialized;
    int q8_prequant_decode;
    int disable_splitk_attn_out_low;
    int disable_shared_gate_up_fused_w32;
    int attention_output_hipblas_all;
    int shared_down_hipblas;
    int q8_decode_sharedx_64k;
    uint32_t q8_decode_rpb;
    uint32_t q8_hc_decode_rpb;
    uint32_t attn_out_low_decode_rpb;
    uint32_t moe_decode_rpb;
    uint32_t moe_decode_gate_rpb;
    uint32_t moe_decode_down_rpb;
    int oldhip_attention_decode;
};

static ds4_rocm_runtime_config g_rocm_cfg;

static const ds4_rocm_runtime_config *hip_runtime_config(void) {
    if (!g_rocm_cfg.initialized) {
        g_rocm_cfg.q8_prequant_decode = 1;
        g_rocm_cfg.disable_splitk_attn_out_low = 1;
        g_rocm_cfg.disable_shared_gate_up_fused_w32 = 1;
        g_rocm_cfg.attention_output_hipblas_all = 1;
        g_rocm_cfg.shared_down_hipblas = 1;
        g_rocm_cfg.q8_decode_sharedx_64k = 1;
        g_rocm_cfg.q8_decode_rpb = 1u;
        g_rocm_cfg.q8_hc_decode_rpb = 16u;
        g_rocm_cfg.attn_out_low_decode_rpb = 32u;
        g_rocm_cfg.moe_decode_rpb = 1u;
        g_rocm_cfg.moe_decode_gate_rpb = g_rocm_cfg.moe_decode_rpb;
        g_rocm_cfg.moe_decode_down_rpb = g_rocm_cfg.moe_decode_rpb;
        g_rocm_cfg.oldhip_attention_decode = 1;
        g_rocm_cfg.initialized = 1;
    }
    return &g_rocm_cfg;
}

static bool hip_q8_prequant_decode_enabled(void) {
    return hip_runtime_config()->q8_prequant_decode;
}

static uint64_t hip_q8_f16_cache_limit_bytes(void) {
    return UINT64_MAX;
}

static uint64_t hip_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) {
        return 512ull * 1048576ull;
    }

    /* The expanded Q8->F16 cache is only an acceleration path.  Keep enough
     * device memory free for cuBLAS workspaces, transient graph buffers, and
     * driver bookkeeping instead of letting optional cached weights consume the
     * last few GiB on 96 GiB cards. */
    const uint64_t min_reserve = 4096ull * 1048576ull;
    const uint64_t pct_reserve = total_bytes / 20u; /* 5% */
    return pct_reserve > min_reserve ? pct_reserve : min_reserve;
}

static void hip_q8_f16_cache_budget_notice(
        const char *reason,
        uint64_t request_bytes,
        uint64_t free_bytes,
        uint64_t total_bytes,
        uint64_t reserve_bytes,
        uint64_t limit_bytes) {
    if (g_q8_f16_budget_notice_printed) return;
    g_q8_f16_budget_notice_printed = 1;
    if (limit_bytes != UINT64_MAX && free_bytes == 0 && total_bytes == 0 && reserve_bytes == 0) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0);
    } else if (limit_bytes == UINT64_MAX) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    }
}

static int hip_q8_f16_cache_has_budget(uint64_t request_bytes, const char *label) {
    (void)label;
    const uint64_t limit = hip_q8_f16_cache_limit_bytes();
    if (limit == 0) return 0;
    if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
        hip_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
        return 0;
    }

    size_t free_b = 0;
    size_t total_b = 0;
    hipError_t err = hipMemGetInfo(&free_b, &total_b);
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 cache memory query failed: %s; using q8 kernels\n",
                hipGetErrorString(err));
        (void)hipGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = hip_q8_f16_cache_reserve_bytes(total_bytes);
    if (request_bytes > free_bytes ||
        free_bytes - request_bytes < reserve_bytes) {
        hip_q8_f16_cache_budget_notice("budget exhausted", request_bytes,
                                        free_bytes, total_bytes,
                                        reserve_bytes, limit);
        return 0;
    }
    return 1;
}

static void hip_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes) {
    if (!g_q8_f16_disabled_after_oom) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache disabled after %s "
                "(request=%.2f MiB cached=%.2f GiB); using q8 kernels\n",
                what ? what : "allocation failure",
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    g_q8_f16_disabled_after_oom = 1;
    if (!g_q8_f16_ranges.empty()) {
        (void)hipDeviceSynchronize();
        hip_q8_f16_cache_release_all();
    }
    (void)hipGetLastError();
}

static int hip_q8_f16_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (g_q8_f16_disabled_after_oom) return 0;
    if (!label) return 0;
    if (strstr(label, "attn_output_a") != NULL ||
        strstr(label, "attn_output_b") != NULL ||
        strstr(label, "attention_output_a") != NULL ||
        strstr(label, "attention_output_b") != NULL) {
        return 1;
    }
    if (strstr(label, "attn_q_b") != NULL) {
        return 1;
    }
    if (strstr(label, "ffn_gate_shexp") != NULL ||
        strstr(label, "ffn_up_shexp") != NULL ||
        strstr(label, "ffn_down_shexp") != NULL) {
        return 1;
    }
    return (in_dim == 4096u && out_dim == 2048u) ||
           (in_dim == 2048u && out_dim == 4096u) ||
           (in_dim == 4096u && out_dim == 1024u) ||
           (in_dim == 4096u && out_dim == 512u) ||
           (in_dim == 1024u && out_dim == 32768u);
}

static int hip_q8_label_is_attention_output(const char *label) {
    return label &&
           (strstr(label, "attn_output_a") != NULL ||
            strstr(label, "attn_output_b") != NULL ||
            strstr(label, "attention_output_a") != NULL ||
            strstr(label, "attention_output_b") != NULL);
}

static const __half *hip_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_by_offset.find(offset);
    if (exact != g_q8_f16_by_offset.end()) {
        const hip_q8_f16_range &r = g_q8_f16_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    for (const hip_q8_f16_range &r : g_q8_f16_ranges) {
        if (r.host_base == model_map && r.offset == offset &&
            r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!hip_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = hip_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;

    uint64_t out_bytes = 0;
    if (in_dim == 0u || out_dim == 0u ||
        !hip_u64_mul3_checked(in_dim, out_dim, sizeof(__half), &out_bytes)) return NULL;
    if (!hip_q8_f16_cache_has_budget(out_bytes, label)) return NULL;

    __half *dev = NULL;
    hipError_t err = hipMalloc(&dev, (size_t)out_bytes);
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, hipGetErrorString(err));
        hip_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!hip_ok(hipGetLastError(), "q8 fp16 dequant launch")) {
        (void)hipFree(dev);
        hip_q8_f16_cache_disable_after_failure("dequant launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    return dev;
}

static const __half *hip_q8_f16_transpose_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_transpose_by_offset.find(offset);
    if (exact != g_q8_f16_transpose_by_offset.end()) {
        const hip_q8_f16_transpose_range &r = g_q8_f16_transpose_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    for (const hip_q8_f16_transpose_range &r : g_q8_f16_transpose_ranges) {
        if (r.host_base == model_map && r.offset == offset &&
            r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!hip_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;
    const char *q8 = hip_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;
    uint64_t out_bytes = 0;
    if (in_dim == 0u || out_dim == 0u ||
        !hip_u64_mul3_checked(in_dim, out_dim, sizeof(__half), &out_bytes)) return NULL;
    if (!hip_q8_f16_cache_has_budget(out_bytes, label)) return NULL;
    __half *dev = NULL;
    hipError_t err = hipMalloc(&dev, (size_t)out_bytes);
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 transpose cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, hipGetErrorString(err));
        hip_q8_f16_cache_disable_after_failure("transpose allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const dim3 grid(
            (unsigned)((out_dim + DS4_Q8_T_TILE_ROW - 1u) / DS4_Q8_T_TILE_ROW),
            (unsigned)blocks,
            1u);
    dequant_q8_0_to_f16_transpose_tiled_kernel<<<grid, 256>>>(
            dev,
            (const unsigned char *)q8,
            in_dim,
            out_dim,
            blocks);
    if (!hip_ok(hipGetLastError(), "q8 fp16 transpose dequant launch")) {
        (void)hipFree(dev);
        hip_q8_f16_cache_disable_after_failure("transpose launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_transpose_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_transpose_by_offset[offset] = g_q8_f16_transpose_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    return dev;
}

static int hip_ok(hipError_t err, const char *what) {
    if (err == hipSuccess) return 1;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "%s failed: %s\n", what, hipGetErrorString(err));
    return 0;
}

static double hip_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int hip_model_load_progress_enabled(void) {
    return 1;
}

static void hip_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}

static void hip_model_load_progress_note(uint64_t cached_bytes) {
    if (!hip_model_load_progress_enabled()) return;

    const double now = hip_wall_sec();
    if (!g_model_load_progress_started) {
        g_model_load_progress_started = 1;
        g_model_load_progress_tty = isatty(STDERR_FILENO) != 0;
        g_model_load_progress_next = (g_model_load_progress_tty ? 2ull : 16ull) *
                                     1024ull * 1024ull * 1024ull;
        g_model_load_progress_last = now;
        if (g_model_load_progress_tty) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors into device cache: 0.00 GiB");
        } else {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors into device cache\n");
        }
    }

    if (cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (g_model_load_progress_tty ? 2.0 : 10.0)) {
        return;
    }

    if (g_model_load_progress_tty) {
        fprintf(stderr, "\r" DS4_GPU_LOG_PREFIX "loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors %.2f GiB cached\n",
                (double)cached_bytes / 1073741824.0);
    }
    fflush(stderr);
    g_model_load_progress_last = now;
    const uint64_t step = (g_model_load_progress_tty ? 2ull : 16ull) *
                          1024ull * 1024ull * 1024ull;
    while (g_model_load_progress_next <= cached_bytes) {
        g_model_load_progress_next += step;
    }
}

static uint64_t hip_model_copy_chunk_bytes(uint64_t range_bytes) {
    const uint64_t mib = 1048576ull;
    uint64_t bytes =
        (range_bytes + DS4_ROCM_MODEL_STAGE_COUNT - 1u) /
        DS4_ROCM_MODEL_STAGE_COUNT;
    bytes = ((bytes + mib - 1u) / mib) * mib;
    if (bytes < 16u * mib) bytes = 16u * mib;
    if (bytes > 64u * mib) bytes = 64u * mib;
    return bytes;
}

static void hip_model_discard_source_pages(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (!model_map || bytes == 0 || offset > model_size) return;
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t h1 = h0 + bytes;
    const uintptr_t p0 = h0 & ~(uintptr_t)(page_sz - 1u);
    const uintptr_t p1 = (h1 + page_sz - 1u) & ~(uintptr_t)(page_sz - 1u);
    if (p1 > p0) (void)posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED);
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
#endif
}

static void hip_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd < 0 || bytes == 0) return;
    (void)posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)offset;
    (void)bytes;
#endif
}

static uint64_t hip_round_down(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    return (v / align) * align;
}

static uint64_t hip_round_up(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    const uint64_t rem = v % align;
    return rem == 0 ? v : v + (align - rem);
}

static void *hip_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static void hip_model_stage_pool_free_buffers(void) {
    for (size_t i = 0; i < DS4_ROCM_MODEL_STAGE_COUNT; i++) {
        if (g_model_stage_event[i]) {
            (void)hipEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)hipFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
}

static int hip_model_stage_pool_alloc(uint64_t bytes) {
    if (g_model_stage_bytes >= bytes) return 1;
    hip_model_stage_pool_free_buffers();
    if (!g_model_upload_stream) {
        hipError_t err = hipStreamCreateWithFlags(&g_model_upload_stream, hipStreamNonBlocking);
        if (err != hipSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model upload stream creation failed: %s\n", hipGetErrorString(err));
            (void)hipGetLastError();
            return 0;
        }
    }
    uint64_t alloc_bytes = bytes;
    if (g_model_direct_align > 1u) {
        const uint64_t pad = g_model_direct_align - 1u;
        if (alloc_bytes > UINT64_MAX - pad) return 0;
        alloc_bytes += pad;
    }
    if (alloc_bytes > (uint64_t)SIZE_MAX) return 0;
    for (size_t i = 0; i < DS4_ROCM_MODEL_STAGE_COUNT; i++) {
        hipError_t err = hipMallocHost(&g_model_stage_raw[i], (size_t)alloc_bytes);
        if (err != hipSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "pinned model staging allocation failed: %s\n", hipGetErrorString(err));
            (void)hipGetLastError();
            return 0;
        }
        g_model_stage[i] = hip_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = hipEventCreateWithFlags(&g_model_stage_event[i], hipEventDisableTiming);
        if (err != hipSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging event creation failed: %s\n", hipGetErrorString(err));
            (void)hipGetLastError();
            return 0;
        }
    }
    g_model_stage_bytes = bytes;
    return 1;
}

static void hip_model_staging_release(void) {
    if (g_model_upload_stream) {
        (void)hipStreamSynchronize(g_model_upload_stream);
    }
    hip_model_stage_pool_free_buffers();
    if (g_model_upload_stream) {
        (void)hipStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = NULL;
    }
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
}

static int hip_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n_req = (bytes - done > (uint64_t)SSIZE_MAX) ? (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int hip_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    const int direct_fd = g_model_direct_fd;
    const uint64_t direct_align = g_model_direct_align;
    if (direct_fd >= 0 && direct_align > 1 && g_model_file_size != 0) {
        const uint64_t aligned_off = hip_round_down(offset, direct_align);
        const uint64_t delta = offset - aligned_off;
        uint64_t read_size = hip_round_up(delta + bytes, direct_align);
        if (aligned_off <= g_model_file_size &&
            read_size <= stage_bytes &&
            read_size <= g_model_file_size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (hip_pread_full(direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return 1;
            }
        }
    }
#else
    (void)stage_bytes;
#endif
    return hip_pread_full(g_model_fd, stage, bytes, offset);
}

struct hip_model_stage_read_task {
    void *stage;
    uint64_t stage_bytes;
    uint64_t offset;
    uint64_t bytes;
    const char *payload;
    int ok;
    int error;
};

static void *hip_model_stage_read_worker(void *opaque) {
    hip_model_stage_read_task *task =
        static_cast<hip_model_stage_read_task *>(opaque);
    task->ok = hip_model_stage_read(
        task->stage,
        task->stage_bytes,
        task->offset,
        task->bytes,
        &task->payload);
    task->error = task->ok ? 0 : errno;
    return NULL;
}

static uint64_t hip_model_cache_limit_bytes(void) {
    return UINT64_MAX;
}

static uint64_t hip_model_arena_chunk_bytes(uint64_t need) {
    uint64_t bytes = 1792ull * 1048576ull;
    if (bytes < need) {
        const uint64_t align = 256ull * 1048576ull;
        bytes = (need + align - 1u) & ~(align - 1u);
    }
    return bytes;
}

static char *hip_model_arena_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_model_cache_full) return NULL;
    const uint64_t align = 256u;
    const uint64_t aligned = (bytes + align - 1u) & ~(align - 1u);

    for (hip_model_arena &a : g_model_arenas) {
        const uint64_t used = (a.used + align - 1u) & ~(align - 1u);
        if (used <= a.bytes && aligned <= a.bytes - used) {
            char *ptr = a.device_ptr + used;
            a.used = used + aligned;
            return ptr;
        }
    }

    const uint64_t limit = hip_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;

    const uint64_t chunk = hip_model_arena_chunk_bytes(aligned);
    void *dev = NULL;
    hipError_t err = hipMalloc(&dev, (size_t)chunk);
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)chunk / 1048576.0,
                hipGetErrorString(err));
        (void)hipGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    return (char *)dev;
}

static const char *hip_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (g_model_fd < 0 || bytes == 0) return NULL;
    if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) return NULL;
    const uint64_t limit = hip_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        return NULL;
    }

    char *dev = hip_model_arena_alloc(bytes, what);
    if (!dev) {
        return NULL;
    }
    hipError_t err = hipSuccess;

    const uint64_t chunk = hip_model_copy_chunk_bytes(bytes);
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!hip_model_stage_pool_alloc(stage_bytes)) return NULL;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        hip_model_stage_read_task tasks[DS4_ROCM_MODEL_STAGE_COUNT] = {};
        pthread_t threads[DS4_ROCM_MODEL_STAGE_COUNT] = {};
        int thread_started[DS4_ROCM_MODEL_STAGE_COUNT] = {};
        uint64_t batch_bytes = 0;
        size_t batch_count = 0;
        while (batch_count < DS4_ROCM_MODEL_STAGE_COUNT &&
               copied + batch_bytes < bytes) {
            const uint64_t remaining = bytes - copied - batch_bytes;
            const uint64_t n = remaining < chunk ? remaining : chunk;
            const size_t bi = batch_count;
            if (chunk_idx >= DS4_ROCM_MODEL_STAGE_COUNT) {
                err = hipEventSynchronize(g_model_stage_event[bi]);
                if (err != hipSuccess) {
                    fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging wait failed for %s: %s\n",
                            what ? what : "weights", hipGetErrorString(err));
                    (void)hipGetLastError();
                    return NULL;
                }
            }
            tasks[bi] = {
                g_model_stage[bi],
                g_model_stage_bytes,
                offset + copied + batch_bytes,
                n,
                NULL,
                0,
                0,
            };
            if (pthread_create(
                    &threads[bi],
                    NULL,
                    hip_model_stage_read_worker,
                    &tasks[bi]) == 0) {
                thread_started[bi] = 1;
            } else {
                (void)hip_model_stage_read_worker(&tasks[bi]);
            }
            batch_bytes += n;
            batch_count++;
            chunk_idx++;
        }
        for (size_t bi = 0; bi < batch_count; bi++) {
            if (thread_started[bi]) {
                (void)pthread_join(threads[bi], NULL);
            }
            if (!tasks[bi].ok) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "model range read failed for %s at %.2f MiB: %s\n",
                        what ? what : "weights",
                        (double)(tasks[bi].offset - offset) / 1048576.0,
                        strerror(tasks[bi].error));
                return NULL;
            }
        }
        for (size_t bi = 0; bi < batch_count; bi++) {
            const uint64_t n = tasks[bi].bytes;
            const uint64_t relative_offset = tasks[bi].offset - offset;
            err = hipMemcpyAsync(
                dev + relative_offset,
                tasks[bi].payload,
                (size_t)n,
                hipMemcpyHostToDevice,
                g_model_upload_stream);
            if (err != hipSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "model range copy failed for %s at %.2f MiB: %s\n",
                        what ? what : "weights",
                        (double)relative_offset / 1048576.0,
                        hipGetErrorString(err));
                (void)hipGetLastError();
                return NULL;
            }
            err = hipEventRecord(g_model_stage_event[bi], g_model_upload_stream);
            if (err != hipSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging record failed for %s: %s\n",
                        what ? what : "weights", hipGetErrorString(err));
                (void)hipGetLastError();
                return NULL;
            }
            hip_model_drop_file_pages(tasks[bi].offset, n);
            hip_model_discard_source_pages(
                model_map,
                g_model_registered_size,
                tasks[bi].offset,
                n);
            hip_model_load_progress_note(
                g_model_range_bytes + relative_offset + n);
        }
        copied += batch_bytes;
    }
    err = hipStreamSynchronize(g_model_upload_stream);
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range upload sync failed for %s: %s\n",
                what ? what : "weights", hipGetErrorString(err));
        (void)hipGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    hip_model_load_progress_note(g_model_range_bytes);
    return (const char *)dev;
}

static void hip_model_range_release_all(void) {
    for (const hip_model_arena &a : g_model_arenas) {
        if (a.device_ptr) (void)hipFree(a.device_ptr);
    }
    g_model_arenas.clear();
    g_model_ranges.clear();
    g_model_range_by_offset.clear();
    g_model_range_bytes = 0;
    hip_model_load_progress_reset();
}

static int hipblas_ok(hipblasStatus_t st, const char *what) {
    if (st == HIPBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " %s failed: status %d\n", what, (int)st);
    return 0;
}


extern "C" int ds4_gpu_init(void) {
    int dev = 0;
    if (!hip_ok(hipSetDevice(dev), "set device")) return 0;
    g_rocm_gfx1151 = 0;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) == hipSuccess) {
        g_rocm_gfx1151 = prop.major == 11 && prop.minor == 5;
        fprintf(stderr, DS4_GPU_LOG_PREFIX "backend initialized on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
    }
    if (!g_hipblas_ready) {
        if (!hipblas_ok(hipblasCreate(&g_hipblas), "create handle")) return 0;
        (void)hipblasSetMathMode(g_hipblas, HIPBLAS_TF32_TENSOR_OP_MATH);
        g_hipblas_ready = 1;
    }
#ifdef __HIP_PLATFORM_AMD__
    if (!g_hipblaslt_ready) {
        if (hipblaslt_ok(hipblasLtCreate(&g_hipblaslt), "create handle")) {
            g_hipblaslt_ready = 1;
        }
    }
#endif
    /* The vendored MMQ tier is the largest prefill win available here, and it
     * is scoped to DS4_ROCM_WIDE_PREFILL_ROWS so decode, DSpark verification
     * and short resumed batches keep the kernels the pinned trajectory was
     * recorded against. */
    g_rocm_mmq_ready = g_rocm_gfx1151 && ds4_mmq_init(dev) == 0;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "native MMQ %s\n",
            g_rocm_mmq_ready ? "enabled" : "unavailable");
    return 1;
}

extern "C" void ds4_gpu_release_support_map(void);

extern "C" void ds4_gpu_cleanup(void) {
    (void)hipDeviceSynchronize();
    ds4_mmq_cleanup();
    ds4_gpu_release_support_map();
    hip_shared_gate_up_async_cleanup();
#ifdef __HIP_PLATFORM_AMD__
    hipblaslt_gemm_plan_clear();
#endif
    if (g_hipblas_ready) {
        (void)hipblasDestroy(g_hipblas);
        g_hipblas_ready = 0;
        g_hipblas = NULL;
    }
#ifdef __HIP_PLATFORM_AMD__
    if (g_hipblaslt_ready) {
        (void)hipblasLtDestroy(g_hipblaslt);
        g_hipblaslt_ready = 0;
        g_hipblaslt = NULL;
    }
#endif
    hip_model_range_release_all();
    hip_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    if (g_hip_tmp) {
        (void)hipFree(g_hip_tmp);
        g_hip_tmp = NULL;
        g_hip_tmp_bytes = 0;
    }
    hip_model_staging_release();
    g_model_host_base = NULL;
    g_model_registered_size = 0;
    g_model_fd = -1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!hip_ok(hipMalloc(&t->ptr, (size_t)bytes), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!hip_ok(hipMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

static uint64_t hip_managed_kv_reserve_bytes(uint64_t total_bytes) {
    const uint64_t min_reserve = 8ull * 1073741824ull;
    const uint64_t max_reserve = 40ull * 1073741824ull;
    uint64_t reserve = total_bytes / 4u;
    if (reserve < min_reserve) reserve = min_reserve;
    if (reserve > max_reserve) reserve = max_reserve;
    return reserve;
}

extern "C" int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;

    /* Very large KV caches are where device-only hipMalloc() can make a
     * unified-memory machine unresponsive.  Managed memory restores the old
     * demand-paged behavior for this one long-lived allocation class only. */
    const uint64_t huge_kv = 8ull * 1073741824ull;
    if (kv_cache_bytes >= huge_kv) return 1;

    const uint64_t large_context = 8ull * 1073741824ull;
    if (context_bytes < large_context) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    hipError_t err = hipMemGetInfo(&free_b, &total_b);
    if (err != hipSuccess) {
        (void)hipGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = hip_managed_kv_reserve_bytes(total_bytes);
    if (context_bytes > free_bytes) return 1;
    return free_bytes - context_bytes < reserve_bytes;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    return t;
}

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) (void)hipFree(tensor->ptr);
    free(tensor);
}

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    fill_f32_kernel<<<(count + 255u) / 256u, 256>>>((float *)tensor->ptr, count, value);
    return hip_ok(hipGetLastError(), "tensor fill f32 launch");
}

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return hip_ok(hipMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, hipMemcpyHostToDevice), "tensor write");
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return hip_ok(hipMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, hipMemcpyDeviceToHost), "tensor read");
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    return hip_ok(hipMemcpy((char *)dst->ptr + dst_offset,
                              (const char *)src->ptr + src_offset,
                              (size_t)bytes,
                              hipMemcpyDeviceToDevice),
                   "tensor copy");
}

extern "C" int ds4_gpu_begin_commands(void) { return 1; }
extern "C" int ds4_gpu_flush_commands(void) { return hip_ok(hipDeviceSynchronize(), "flush"); }
extern "C" int ds4_gpu_end_commands(void) {
    return hip_ok(hipDeviceSynchronize(), "end commands");
}
extern "C" int ds4_gpu_synchronize(void) { return hip_ok(hipDeviceSynchronize(), "synchronize"); }

static int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;
    hip_model_range_release_all();
    hip_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    g_model_host_base = model_map;
    g_model_registered_size = model_size;
    g_model_cache_full = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }

    /* Strix Halo uses the staged full-copy path in ds4_gpu_set_model_map_range().
     * Avoid host-registering the mmap here: that would make the staged copier
     * believe the model is already device-resident. */
    return 1;
}

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 ||
        map_offset > model_size ||
        map_size > model_size - map_offset) {
        return 0;
    }
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    /*
     * Do not eagerly copy a contiguous model image here.  On Strix Halo the
     * caller immediately follows with accelerator_cache_model_tensors(), which
     * prepares the exact tensor spans selected by --layers.  Copying here would
     * either allocate the whole GGUF image or, for sparse span sets, an oversized
     * envelope before the precise tensor-span cache gets a chance to run.
     */
    return 1;
}

extern "C" int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    g_model_fd_host_base = g_model_host_base;
    g_model_file_size = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_model_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) g_model_direct_align = (uint64_t)st.st_blksize;
        }
#if defined(__linux__) && defined(O_DIRECT)
        {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
            }
        }
#endif
    }
    return 1;
}

extern "C" void ds4_gpu_release_model_staging(void) {
    hip_model_staging_release();
}

extern "C" int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    if (!hip_model_range_ptr(model_map, offset, bytes, label ? label : "model_tensor")) return 0;
    return hip_model_range_is_cached(model_map, offset, bytes);
}

/*
 * Reserve one contiguous arena for the DSpark support model and copy the
 * requested tensor spans into it. Managed storage is the Strix Halo default:
 * a device-only 5.58 GiB duplicate reduces prompt-processing headroom, while
 * read-only host registration makes support kernels bandwidth-bound.
 */
extern "C" int ds4_gpu_reserve_support_map(const void *support_map,
                                           uint64_t support_size,
                                           uint64_t arena_bytes) {
    if (!support_map || support_size == 0 || arena_bytes == 0) return 0;
    if (g_support_host_base != NULL && g_support_host_base != support_map) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "a different support model is already registered\n");
        return 0;
    }
    if (g_support_arena != NULL) {
        return g_support_arena_bytes >= arena_bytes ? 1 : 0;
    }

    void *device = NULL;
    const hipError_t alloc_err = hipMallocManaged(
        &device, (size_t)arena_bytes, hipMemAttachGlobal);
    if (!hip_ok(alloc_err, "managed support arena alloc")) {
        return 0;
    }
    g_support_arena = (char *)device;
    g_support_arena_bytes = arena_bytes;
    g_support_arena_used = 0;
    g_support_managed = true;
    g_support_host_base = support_map;
    g_support_registered_size = support_size;
    g_support_ranges.clear();
    return 1;
}

extern "C" int ds4_gpu_cache_support_range(const void *support_map,
                                           uint64_t support_size,
                                           uint64_t offset,
                                           uint64_t bytes,
                                           const char *label) {
    if (!support_map || bytes == 0) return 1;
    if (support_map != g_support_host_base || !g_support_arena) return 0;
    if (offset > support_size || bytes > support_size - offset) return 0;
    if (hip_support_range_ptr(support_map, offset, bytes) != NULL) return 1;
    /* Align each span so quantized block loads stay naturally aligned. */
    const uint64_t aligned_used = hip_round_up(g_support_arena_used, 256u);
    if (aligned_used > g_support_arena_bytes ||
        bytes > g_support_arena_bytes - aligned_used) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "support arena exhausted for %s (%.2f MiB requested, "
                "%.2f MiB free)\n",
                label ? label : "support_tensor",
                (double)bytes / 1048576.0,
                (double)(g_support_arena_bytes - aligned_used) / 1048576.0);
        return 0;
    }
    char *device = g_support_arena + aligned_used;
    if (g_support_managed) {
        memcpy(device, (const char *)support_map + offset, (size_t)bytes);
    } else {
        if (!hip_ok(hipMemcpy(device,
                              (const char *)support_map + offset,
                              (size_t)bytes,
                              hipMemcpyHostToDevice),
                    "support range copy")) {
            return 0;
        }
    }
    g_support_arena_used = aligned_used + bytes;
    g_support_ranges.push_back({support_map, offset, bytes, device});
    return 1;
}

extern "C" void ds4_gpu_set_small_batch_mode(int mode) {
    g_small_batch_mode = mode >= 0 && mode <= 2 ? mode : 0;
}

extern "C" void ds4_gpu_release_support_map(void) {
    if (g_support_arena) (void)hipFree(g_support_arena);
    g_support_arena = NULL;
    g_support_arena_bytes = 0;
    g_support_arena_used = 0;
    g_support_host_base = NULL;
    g_support_registered_size = 0;
    g_support_managed = false;
    g_support_ranges.clear();
}
