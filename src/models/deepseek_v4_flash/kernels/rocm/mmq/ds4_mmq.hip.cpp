// SPDX-License-Identifier: MIT
// ds4_mmq.hip.cpp - host wrapper around llama.cpp's vendored mul_mat_q kernels.
//
// Implements the public ds4_mmq_* entry points and explicitly instantiates
// the mul_mat_q_case<T> template for each quant type they use.
//
#include "ds4_mmq.h"

#include "common.hip.hpp"
#include "mmq.hip.hpp"
#include "quantize.hip.hpp"
#include "mmid.hip.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------

// Reusable Q8_1 scratch supplied by the graph owner.
static void  *g_aligned_q81_scratch_ptr = nullptr;
static size_t g_aligned_q81_scratch_bytes = 0;
static int    g_routed_max_expert_rows = 0;
static int    g_routed_tile_cols = 0;

struct ds4_hip_persistent_scratch {
    void *ptr = nullptr;
    size_t bytes = 0;
    std::mutex mutex;
};

static ds4_hip_persistent_scratch g_hip_dense_q81_scratch;
static ds4_hip_persistent_scratch g_hip_pair_map_scratch;

static void *ds4_hip_persistent_scratch_reserve_locked(
        ds4_hip_persistent_scratch *scratch,
        size_t bytes,
        size_t granularity,
        hipStream_t stream,
        const char *tag) {
    if (!scratch || bytes == 0 || stream != (hipStream_t)0) return nullptr;
    if (scratch->ptr && scratch->bytes >= bytes) return scratch->ptr;
    if (granularity == 0 || bytes > SIZE_MAX - (granularity - 1u)) {
        return nullptr;
    }

    const size_t reserve =
        ((bytes + granularity - 1u) / granularity) * granularity;
    hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: scratch resize synchronization failed: %s\n",
                tag, hipGetErrorString(err));
        return nullptr;
    }

    void *next = nullptr;
    err = hipMalloc(&next, reserve);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: persistent scratch allocation of %zu bytes failed: %s\n",
                tag, reserve, hipGetErrorString(err));
        (void)hipGetLastError();
        return nullptr;
    }
    if (scratch->ptr) (void)hipFree(scratch->ptr);
    scratch->ptr = next;
    scratch->bytes = reserve;
    return scratch->ptr;
}

extern "C" void ds4_mmq_set_aligned_q81_scratch(void *ptr, size_t bytes) {
    g_aligned_q81_scratch_ptr = ptr;
    g_aligned_q81_scratch_bytes = ptr ? bytes : 0;
}

extern "C" void ds4_mmq_set_routed_max_expert_rows(int rows) {
    g_routed_max_expert_rows = rows > 0 ? rows : 0;
}

extern "C" void ds4_mmq_set_routed_tile_cols(int cols) {
    g_routed_tile_cols = cols > 0 ? cols : 0;
}

extern "C" int ds4_mmq_routed_tile_cols_for_counts(
        const unsigned int *counts, int n_experts) {
    /* Pick the column-tile width that minimizes what the routed gate/up pass
     * actually issues. A tile is executed whole, so expert e with c_e rows
     * costs `ceil(c_e / w)` tiles of `w` columns each, and every one of those
     * tiles also re-reads e's weight panel. The padding term therefore falls
     * with w while the weight term rises, and the balance depends on the
     * distribution rather than on its maximum: at a 4,096-token chunk the mean
     * bucket is 96 rows and 80 is right, at 512 tokens it is twelve and 80
     * spends 85% of its issue on padding.
     *
     * `kPanel` prices one weight-panel reload in column-equivalents. It was
     * fitted, not assumed: the real per-expert counts were dumped for every
     * layer at 256, 512, 1,024 and 2,048 tokens and scored against a
     * forced-width sweep of the same four chunks. Over P in 0..128, P = 16 is
     * the single constant closest to the per-width optimum -- 0.85% mean and
     * 3.4% worst-case shortfall, against 1.7% mean at P = 8 or 32 and 2.6% at
     * P = 48. No constant reproduces the whole curve: the optimum is 16 columns
     * at 512 tokens and 48 at 1,024, and this cost function is linear in
     * `tiles * (w + P)`, so it cannot separate those two. The caller therefore
     * runs it only where it is validated; see
     * DS4_ROCM_ROUTED_TILE_MODEL_ROWS. */
    if (!counts || n_experts <= 0) return 0;
    constexpr int kPanel = 16;
    int best_cols = 0;
    long long best_cost = 0;
    for (int w = 16; w <= 80; w += 16) {
        long long cost = 0;
        for (int e = 0; e < n_experts; e++) {
            const unsigned int c = counts[e];
            if (c == 0u) continue;
            cost += (long long)((c + (unsigned int)w - 1u) / (unsigned int)w) *
                    (long long)(w + kPanel);
        }
        if (best_cols == 0 || cost < best_cost) {
            best_cols = w;
            best_cost = cost;
        }
    }
    return best_cols;
}

extern "C" int ds4_mmq_init(int device) {
    if (device < 0) {
        fprintf(stderr, "ds4_mmq_init: invalid device %d\n", device);
        return -1;
    }
    ds4_ggml_hip_set_device(device);
    // Trigger lazy population of the device-info singleton.
    const auto & info = ds4_ggml_hip_info();
    if (info.device_count == 0) {
        fprintf(stderr, "ds4_mmq_init: no HIP devices found\n");
        return -1;
    }
    if (device >= info.device_count) {
        fprintf(stderr, "ds4_mmq_init: device %d out of range (have %d)\n",
                device, info.device_count);
        return -1;
    }

    return 0;
}

extern "C" void ds4_mmq_cleanup(void) {
    (void)hipDeviceSynchronize();
    {
        std::lock_guard<std::mutex> lock(g_hip_dense_q81_scratch.mutex);
        if (g_hip_dense_q81_scratch.ptr) {
            (void)hipFree(g_hip_dense_q81_scratch.ptr);
            g_hip_dense_q81_scratch.ptr = nullptr;
            g_hip_dense_q81_scratch.bytes = 0;
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_hip_pair_map_scratch.mutex);
        if (g_hip_pair_map_scratch.ptr) {
            (void)hipFree(g_hip_pair_map_scratch.ptr);
            g_hip_pair_map_scratch.ptr = nullptr;
            g_hip_pair_map_scratch.bytes = 0;
        }
    }

    g_aligned_q81_scratch_ptr = nullptr;
    g_aligned_q81_scratch_bytes = 0;
}

// ----------------------------------------------------------------------------
// Gating: when should the caller choose mmq over dequant+cublas?
//
// Body lifted verbatim from llama.cpp's ggml/src/ggml-cuda/mmq.cu:267-372
// (we do not vendor mmq.cu itself, since its other half talks to ggml_tensor
// and ggml_backend internals we don't carry over).
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Dense matmul implementation, shared across all three quant types.
//
// Computes  out[col, row] = sum_k W[row, k] * X[k, col]   with W in the
// type-specific block layout and X / out in F32 (X K-innermost row-major,
// out column-major out[col*M + row]).
//
// Mirrors upstream mmq.cu:154-159 (the no-ids branch) but builds mmq_args
// from plain pointers + shape ints instead of ggml_tensor introspection.
// ----------------------------------------------------------------------------

// Per-device singleton context. Owns the pool for stream-K fixup scratch.
// Phase 4 will make this per-stream as well; for now a single context per
// device is sufficient for the dense path.
namespace {

__global__ static void ds4_mmq_sanitize_f32_kernel(float *p, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = p[i];
    if (!isfinite(v)) p[i] = 0.0f;
}

static void ds4_mmq_sanitize_f32(float *p, uint64_t n, hipStream_t stream) {
    if (!p || n == 0) return;
    ds4_mmq_sanitize_f32_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, stream>>>(p, n);
}

ds4_ggml_hip_context * get_ctx_for_device(int device) {
    static ds4_ggml_hip_context * cached[DS4_GGML_HIP_MAX_DEVICES] = {};
    if (device < 0 || device >= DS4_GGML_HIP_MAX_DEVICES) return nullptr;
    if (!cached[device]) {
        cached[device] = new ds4_ggml_hip_context(device);
    }
    return cached[device];
}

template <ggml_type type>
int ds4_mmq_dense_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        hipStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (K <= 0 || M <= 0 || N <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        // mmq requires K to be a multiple of the largest super-block size
        // it sees during the inner tile loop, which is QK_K=256.
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ds4_ggml_hip_get_device();
    const int cc  = ds4_ggml_hip_info().devices[dev].cc;

    ds4_ggml_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    /* Order the pool's hipMallocAsync/hipFreeAsync on the SAME
     * stream the kernels below launch on.  The pool defaults to
     * hipStreamPerThread; with kernels on the legacy stream the RAII free is
     * ordered on an EMPTY stream, so the driver can recycle/remap the scratch
     * while the in-flight quantize/GEMM still reads it -> intermittent illegal
     * access under shape churn (the batched-draft early-step crash).  The vec
     * impls already do this (graph-capture fix); the batched impls were missed.
     */
    ds4_pool_set_stream(stream);

    // 1. Quantize F32 activations into the MMQ Q8_1 format.
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = N;
    const int64_t ne12         = 1;
    const int64_t ne13         = 1;

    const size_t y_block_size = sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne13 * ne12 * ne11 * ne10_padded * y_block_size /
            y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    ds4_ggml_hip_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1 = nullptr;
    std::unique_lock<std::mutex> dense_scratch_lock;
    if (stream == (hipStream_t)0) {
        dense_scratch_lock =
            std::unique_lock<std::mutex>(g_hip_dense_q81_scratch.mutex);
        src1_q8_1 = (char *)ds4_hip_persistent_scratch_reserve_locked(
            &g_hip_dense_q81_scratch,
            nbytes_src1_q8_1,
            64u * 1024u * 1024u,
            stream,
            tag);
    }
    if (!src1_q8_1) {
        src1_q8_1 = src1_q8_1_pool.alloc(
            ctx->pool(), nbytes_src1_q8_1);
    }

    // S1.1a fix: the mmq Y (activation) buffer is over-allocated for the kernel's
    // tail-tile reads (the +mmq_x_max blocks above), and ne11 columns may not fill
    // the final column tile -- but ds4_quantize_mmq_q8_1_hip only writes the ne11 valid
    // columns.  The mmq kernel (mmq.hip.hpp:3528) unconditionally loads the full column
    // tile, reading the never-written tail.  Pool allocs reuse stale device memory,
    // so that tail is non-deterministic: any allocator/stream perturbation (e.g. an
    // MTP draft's hipMalloc) changes it and flips a near-threshold argmax in the
    // batched forward (confirmed by compute-sanitizer --tool initcheck on a PRO6000
    // / sm_120: 4-byte uninitialized __global__ read in mul_mat_q_process_tile).
    // The tail's dot-products are masked out by write_back, so only their
    // non-determinism matters; zero the buffer so the tail is a deterministic zero
    // (a zero q8_1 block contributes 0 to the dot product).

    ds4_quantize_mmq_q8_1_hip(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1,
        type, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/ne12, /*ne3=*/ne13,
        stream);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, hipGetErrorString(err));
        return -2;
    }

    // 2. Build mmq_args. stride_row_x is in WEIGHT BLOCKS per row, which
    //    is K / blck_size(type). Q8_0 has block size 32; Q2_K and IQ2_XXS
    //    are K-quants with block size 256.
    const int64_t blck   = ggml_blck_size(type);
    const int64_t s01    = (int64_t)K / blck;
    const int64_t s1     = (int64_t)M;
    const int64_t s12    = ne11 * ne10_padded * y_block_size /
                           (y_values_per_block * sizeof(int));
    const int64_t s13    = ne12 * s12;

    const bool use_stream_k =
        (DS4_GGML_HIP_CC_IS_NVIDIA(cc) && ds4_ggml_hip_highest_compiled_arch(cc) >= DS4_GGML_HIP_CC_VOLTA) ||
        DS4_GGML_HIP_CC_IS_CDNA(cc);

    mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1,
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,    /*nrows_x=*/(int64_t)M,    /*ncols_dst=*/ne11,
        /*stride_row_x=*/s01,/*ncols_y=*/ne11,          /*nrows_dst=*/s1,
        /*nchannels_x=*/1,   /*nchannels_y=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/s12, /*stride_channel_dst=*/0,
        /*nsamples_x=*/1,    /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/s13, /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne11,
    };
    const bool inline_sanitize =
        type == GGML_TYPE_Q8_0 && !use_stream_k;
    args.sanitize_output = inline_sanitize;

    mul_mat_q_case<type>(*ctx, args, stream);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n", tag, hipGetErrorString(err));
        return -3;
    }
    if (!inline_sanitize) {
        ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)N, stream);
    }
    return 0;
}

} // anonymous namespace

extern "C" int ds4_mmq_q8_0_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, hipStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q8_0>("ds4_mmq_q8_0_dense", W, X, out, M, N, K, stream);
}

// ----------------------------------------------------------------------------
// Paired MoE matmul implementation.
//
// Mirrors upstream mmq.cu:163-222 (the ids != nullptr branch).  Caller
// provides:
//   - per-expert weights stacked contiguously
//   - per-token activations [n_tokens, K]
//   - routing table ids[t, s] = expert id
// The wrapper invokes:
//   1. ds4_ggml_hip_launch_mm_ids_helper to build (ids_src1, ids_dst,
//      expert_bounds) - permutations that sort assignments by expert.
//   2. ds4_quantize_mmq_q8_1_hip with ids_src1 - gathers and quantizes the
//      activation into the expert-major flat layout.
//   3. mul_mat_q_case<type> with ids_dst + expert_bounds - the matmul.
// ----------------------------------------------------------------------------

namespace {

struct ds4_mmq_swiglu_epilogue {
    const float * router_weights;
    float       * mid_f32;
    half        * mid_f16;
    float         clamp;
};

// Paired IQ2_XXS MoE: one helper + one quantize covers both weights; the
// second mul_mat_q_case launch reuses the activation and routing map with a
// fresh (x, dst) pair. Each token's experts are unique, so no expert bucket
// exceeds n_tokens rows. With swiglu_epilogue, the second launch combines its
// result with out_a (the gate) into the weighted SwiGLU mid outputs.
int ds4_mmq_moe_pair_impl(
    const char* tag, const void* W_a, const void* W_b, const float* X_f32,
    const int32_t* ids, float* out_a, float* out_b, int M, int K, int n_tokens,
    int n_experts, int n_expert_used, hipStream_t stream,
    const ds4_mmq_swiglu_epilogue* swiglu_epilogue = nullptr) {
  constexpr ggml_type type = GGML_TYPE_IQ2_XXS;
  if (!W_a || !W_b || !X_f32 || !ids ||
      !out_a || (!out_b && !swiglu_epilogue)) {
    fprintf(stderr, "%s: null pointer\n", tag);
    return -1;
  }
  if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 ||
      n_expert_used <= 0) {
    fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n", tag,
            M, K, n_tokens, n_experts, n_expert_used);
    return -1;
  }
  if (K % 256 != 0) {
    fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
    return -1;
  }
  if (n_expert_used > n_experts) {
    fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used,
            n_experts);
    return -1;
  }
  if (swiglu_epilogue &&
      (!swiglu_epilogue->router_weights || !swiglu_epilogue->mid_f32)) {
    fprintf(stderr, "%s: invalid weighted SwiGLU epilogue\n", tag);
    return -1;
  }

  const int dev = ds4_ggml_hip_get_device();
  const int cc = ds4_ggml_hip_info().devices[dev].cc;

  ds4_ggml_hip_context* ctx = get_ctx_for_device(dev);
  if (!ctx) {
    fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
    return -1;
  }

  ds4_pool_set_stream(stream); /* task #22: pool ops must be stream-ordered with
                                  the kernels (see ds4_mmq_dense_impl) */

  const int64_t ne_get_rows = (int64_t)n_tokens * n_expert_used;
  const int64_t ne00 = K;
  const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
  const int64_t ne11 = 1;
  const int64_t ne12 = n_tokens;
  const int64_t blck = ggml_blck_size(type);
  const int64_t s01 = (int64_t)K / blck;
  const int64_t s02 = (int64_t)M * s01;

  ds4_ggml_hip_pool_alloc<int32_t> ids_src1_alloc;
  ds4_ggml_hip_pool_alloc<int32_t> ids_dst_alloc;
  ds4_ggml_hip_pool_alloc<int32_t> expert_bounds_alloc;
  int32_t* ids_src1 = nullptr;
  int32_t* ids_dst = nullptr;
  int32_t* expert_bounds = nullptr;
    std::unique_lock<std::mutex> pair_map_scratch_lock;

    const size_t y_block_size = sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * y_block_size / y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    if (stream == (hipStream_t)0 &&
        (uint64_t)ne_get_rows <=
            (SIZE_MAX / sizeof(int32_t) - (size_t)n_experts - 1u) / 2u) {
        pair_map_scratch_lock =
            std::unique_lock<std::mutex>(g_hip_pair_map_scratch.mutex);
        const size_t map_elems =
            2u * (size_t)ne_get_rows + (size_t)n_experts + 1u;
        int32_t *map = (int32_t *)
            ds4_hip_persistent_scratch_reserve_locked(
                &g_hip_pair_map_scratch,
                map_elems * sizeof(int32_t),
                1024u * 1024u,
                stream,
                tag);
        if (map) {
            ids_src1 = map;
            ids_dst = map + ne_get_rows;
            expert_bounds = ids_dst + ne_get_rows;
        }
    }
    if (!ids_src1) {
        ids_src1 = ids_src1_alloc.alloc(ctx->pool(), ne_get_rows);
        ids_dst = ids_dst_alloc.alloc(ctx->pool(), ne_get_rows);
        expert_bounds =
            expert_bounds_alloc.alloc(ctx->pool(), n_experts + 1);
    }

    const int si1  = n_expert_used;
    const int sis1 = 1;

    hipError_t err = hipSuccess;
    {
      // Zero the ID maps so entries dropped by mm_ids_helper never expose
      // stale pool memory.
      hipMemsetAsync(ids_src1, 0, ne_get_rows * sizeof(int32_t), stream);
      hipMemsetAsync(ids_dst, 0, ne_get_rows * sizeof(int32_t), stream);
      ds4_ggml_hip_launch_mm_ids_helper(
          ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens,
          n_expert_used, /*nchannels_y=*/(int)ne11, si1, sis1, stream);

      err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag,
                hipGetErrorString(err));
        return -2;
      }
    }

    const bool use_stream_k =
        (DS4_GGML_HIP_CC_IS_NVIDIA(cc) && ds4_ggml_hip_highest_compiled_arch(cc) >= DS4_GGML_HIP_CC_VOLTA) ||
        DS4_GGML_HIP_CC_IS_CDNA(cc);
    if (swiglu_epilogue && use_stream_k) {
        return -13;
    }
    /* True top-k assignments cannot select one expert twice for a token, so
     * no expert bucket can exceed n_tokens rows. */
    const int64_t routed_ncols_max = (int64_t)n_tokens;
    /* The caller may know the largest expert bucket (see
     * ds4_mmq_set_routed_max_expert_rows). ncols_max keeps selecting the tile
     * shape from the chunk width; only the column-tile grid shrinks. */
    const int64_t routed_ncols_grid =
        g_routed_max_expert_rows > 0 &&
        (int64_t)g_routed_max_expert_rows < routed_ncols_max
            ? (int64_t)g_routed_max_expert_rows
            : 0;

    {
    ds4_ggml_hip_pool_alloc<char> src1_q8_1_alloc;
    char *src1_q8_1 = nullptr;
    if (g_aligned_q81_scratch_ptr &&
        g_aligned_q81_scratch_bytes >= nbytes_src1_q8_1) {
        src1_q8_1 = (char *)g_aligned_q81_scratch_ptr;
    } else {
        src1_q8_1 = src1_q8_1_alloc.alloc(ctx->pool(), nbytes_src1_q8_1);
    }

    // S1.1a fix (same as the dense path): zero the over-allocated mmq Y buffer
    // so the kernel's unconditional masked-out tail-tile read (mmq.hip.hpp:3528) returns
    // a deterministic zero instead of allocator-perturbation-dependent stale memory.
    const int64_t s11_src = (int64_t)K;
    const int64_t s12_src = (int64_t)K * ne11;
    const int64_t s13_src = (int64_t)K * ne11 * ne12;
    ds4_quantize_mmq_q8_1_hip(X_f32, ids_src1,
                           (void*)src1_q8_1, type, /*ne00=*/K, s11_src,
                           s12_src, s13_src,
                           /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows,
                           /*ne2=*/1, /*ne3=*/1, stream);

    err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr, "%s: MMQ activation quantize failed: %s\n", tag,
              hipGetErrorString(err));
      return -3;
    }

    const int64_t s1      = (int64_t)M;
    const int64_t s12_mmq = ne11 * ne10_padded * y_block_size /
                            (y_values_per_block * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    mmq_args args = {
        /*x=*/(const char *)W_a,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1,
        /*ids_dst=*/ids_dst,
        /*expert_bounds=*/expert_bounds,
        /*dst=*/out_a,
        /*ncols_x=*/ne00,
        /*nrows_x=*/(int64_t)M,
        /*ncols_dst=*/ne_get_rows,
        /*stride_row_x=*/s01,
        /*ncols_y=*/ne_get_rows,
        /*nrows_dst=*/s1,
        /*nchannels_x=*/(int64_t)n_experts,
        /*nchannels_y=*/(int64_t)n_experts,
        /*stride_channel_x=*/s02,
        /*stride_channel_y=*/s12_mmq,
        /*stride_channel_dst=*/(int64_t)0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/s13_mmq,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/routed_ncols_max,
        /*x_soa=*/nullptr,
        /*soa_blocks=*/0,
    };
    args.ncols_grid_max = routed_ncols_grid;
    args.mmq_x_request = g_routed_tile_cols;

    {
      mul_mat_q_case<type>(*ctx, args, stream);
      err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (pair a) launch failed: %s\n", tag,
                hipGetErrorString(err));
        return -4;
      }
    }

    // Second matmul over the same activation buffer and same routing map.
    args.x     = (const char *)W_b;
    args.dst   = out_b;
    if (swiglu_epilogue) {
        args.epilogue_gate = out_a;
        args.epilogue_weights = swiglu_epilogue->router_weights;
        args.epilogue_mid = swiglu_epilogue->mid_f32;
        args.epilogue_mid_h = swiglu_epilogue->mid_f16;
        args.epilogue_clamp = swiglu_epilogue->clamp;
    }
    {
      mul_mat_q_case<type>(*ctx, args, stream);
      err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (pair b) launch failed: %s\n", tag,
                hipGetErrorString(err));
        return -5;
      }
    }
    }

    if (!swiglu_epilogue) {
        ds4_mmq_sanitize_f32(out_a, (uint64_t)M * (uint64_t)ne_get_rows, stream);
        ds4_mmq_sanitize_f32(out_b, (uint64_t)M * (uint64_t)ne_get_rows, stream);
    }
    return 0;
}

} // anonymous namespace

extern "C" int ds4_mmq_iq2_xxs_moe_pair_token_bound(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        hipStream_t stream) {
  return ds4_mmq_moe_pair_impl(
      "ds4_mmq_iq2_xxs_moe_pair_token_bound", W_a, W_b, X, ids, out_a, out_b, M,
      K, n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair_token_bound_swiglu(
        const void * W_gate, const void * W_up,
        const float * X, const int32_t * ids, const float * router_weights,
        float * gate, float * discard, float * mid_f32, void * mid_f16,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, hipStream_t stream) {
    const ds4_mmq_swiglu_epilogue epilogue = {
        router_weights,
        mid_f32,
        (half *)mid_f16,
        clamp,
    };
    return ds4_mmq_moe_pair_impl(
        "ds4_mmq_iq2_xxs_moe_pair_token_bound_swiglu", W_gate, W_up, X, ids,
        gate, discard, M, K, n_tokens, n_experts, n_expert_used, stream,
        &epilogue);
}

// Explicit instantiations, one per quant type the entry points use.
// Each instantiation drags in the load_tiles_<type> + vec_dot_<type>_*
// device functions from mmq.hip.hpp, so the .o objects below contain everything
// needed to link against the public C entries.
template void mul_mat_q_case<GGML_TYPE_Q8_0>(
    ds4_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ2_XXS>(
    ds4_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream);
