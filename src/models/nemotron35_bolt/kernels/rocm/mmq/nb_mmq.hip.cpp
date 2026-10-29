// SPDX-License-Identifier: MIT
// Raw-pointer adapter for Nemotron 3.5 Bolt's llama.cpp-derived quantized
// matrix kernels (see VENDOR.md and nb_mmq.h).
//
// Dispatch mirrors llama.cpp's ggml_cuda_mul_mat at
// the vendored commit:
//   dense: N <= MMVQ_MAX_BATCH_SIZE (8) uses the MMVQ vector kernels, larger
//          N the MMQ tiled kernels.
// Activations are quantized to Q8_1 exactly as upstream: canonical
// block_q8_1 rows for MMVQ, block_q8_1_mmq tiles for MMQ.

#include "nb_mmq_prelude.h"

#include "nb_mmq.h"

namespace nb_mmq {

#include "common.hip.hpp"
#include "mmq.hip.hpp"
#include "mmvq.hip.hpp"
#include "quantize.hip.hpp"

namespace {

std::mutex g_ctx_mutex;
nb_ggml_hip_context * g_ctx[NB_GGML_HIP_MAX_DEVICES] = {};

nb_ggml_hip_context * get_ctx_for_device(int device) {
    if (device < 0 || device >= NB_GGML_HIP_MAX_DEVICES) return nullptr;
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    if (!g_ctx[device]) {
        g_ctx[device] = new nb_ggml_hip_context(device);
    }
    return g_ctx[device];
}

bool supported_type(int weight_type) {
    return weight_type == GGML_TYPE_Q8_0 || weight_type == GGML_TYPE_IQ2_XXS ||
           weight_type == GGML_TYPE_NVFP4;
}

int check_launch(const char * tag, const char * stage) {
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: %s failed: %s\n", tag, stage, hipGetErrorString(err));
        return -2;
    }
    return 0;
}

void mul_mat_q_switch_type(nb_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q8_0:    mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);    break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream); break;
        case GGML_TYPE_NVFP4:   mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);   break;
        default:                GGML_ABORT("unsupported MMQ type %d", (int) args.type_x);
    }
}

bool use_stream_k(int cc) {
    return (NB_GGML_HIP_CC_IS_NVIDIA(cc) &&
            nb_ggml_hip_highest_compiled_arch(cc) >= NB_GGML_HIP_CC_VOLTA) ||
           NB_GGML_HIP_CC_IS_CDNA(cc);
}

// Size of the MMQ activation buffer for `rows` rows, plus the tail slack the
// kernel reads for a partial last column tile. Only the tail is cleared: the
// quantizer writes every other byte, and the tail feeds lanes the write-back
// masks out.
struct mmq_y_buffer {
    size_t body;
    size_t total;
};

mmq_y_buffer mmq_y_bytes(int64_t rows, int64_t ne10_padded, int cc) {
    const size_t body = (size_t) rows * ne10_padded * sizeof(block_q8_1_mmq) / (4 * QK8_1);
    return {body, body + (size_t) get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq)};
}

// Dense, N <= MMVQ_MAX_BATCH_SIZE: canonical Q8_1 rows + mul_mat_vec_q.
int dense_vec(const char * tag, nb_ggml_hip_context & ctx, ggml_type type, const void * W,
              const float * X, float * out, int M, int N, int K, hipStream_t stream) {
    const int64_t ne10_padded = GGML_PAD((int64_t) K, MATRIX_ROW_PADDING);
    nb_ggml_hip_pool_alloc<char> x_q8(ctx.pool(), (size_t) N * ne10_padded * sizeof(block_q8_1) / QK8_1);

    // src1 [K, N, 1, 1], K innermost.
    nb_quantize_row_q8_1_hip(X, nullptr, x_q8.get(), type, K,
                             /*s01=*/K, /*s02=*/(int64_t) K * N, /*s03=*/(int64_t) K * N,
                             ne10_padded, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1, stream);
    if (check_launch(tag, "Q8_1 quantize") != 0) return -2;

    // Upstream's no-ids stride mapping (ggml_cuda_mul_mat_vec_q).
    const int64_t s01 = K / ggml_blck_size(type);
    const int64_t s11 = ne10_padded / QK8_1;
    const nb_ggml_hip_mm_fusion_args_device fusion = {};
    mul_mat_vec_q_switch_type(W, type, x_q8.get(), /*ids=*/nullptr, fusion, out,
                              /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/N,
                              /*stride_row_x=*/(int) s01, /*stride_col_y=*/(int) s11,
                              /*stride_col_dst=*/M,
                              /*nchannels_x=*/1, /*nchannels_y=*/1, /*nchannels_dst=*/1,
                              /*stride_channel_x=*/0, /*stride_channel_y=*/(int) (N * s11),
                              /*stride_channel_dst=*/0,
                              /*nsamples_x=*/1, /*nsamples_dst=*/1, /*stride_sample_x=*/0,
                              /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
                              /*ids_stride=*/0, stream);
    return check_launch(tag, "mul_mat_vec_q");
}

// Dense, small N, activations already in block_q8_1 (K a multiple of 512, so
// no row padding), an output scale and an optional per-output addend.
int dense_vec_q8_1(const char * tag, ggml_type type, const void * W, const void * Xq,
                   float scale, const float * bias, float * out, int M, int N, int K,
                   hipStream_t stream) {
    const int64_t s01 = K / ggml_blck_size(type);
    const int64_t s11 = K / QK8_1;
    nb_ggml_hip_mm_fusion_args_device fusion = {};
    fusion.dst_scale = scale;
    fusion.x_bias = bias;
    mul_mat_vec_q_switch_type(W, type, Xq, /*ids=*/nullptr, fusion, out,
                              /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/N,
                              /*stride_row_x=*/(int) s01, /*stride_col_y=*/(int) s11,
                              /*stride_col_dst=*/M,
                              /*nchannels_x=*/1, /*nchannels_y=*/1, /*nchannels_dst=*/1,
                              /*stride_channel_x=*/0, /*stride_channel_y=*/(int) (N * s11),
                              /*stride_channel_dst=*/0,
                              /*nsamples_x=*/1, /*nsamples_dst=*/1, /*stride_sample_x=*/0,
                              /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
                              /*ids_stride=*/0, stream);
    return check_launch(tag, "mul_mat_vec_q");
}

// Dense, larger N: block_q8_1_mmq tiles + mul_mat_q.
int dense_tiled(const char * tag, nb_ggml_hip_context & ctx, int cc, ggml_type type, const void * W,
                const float * X, float * out, int M, int N, int K, hipStream_t stream) {
    const int64_t ne10_padded = GGML_PAD((int64_t) K, MATRIX_ROW_PADDING);
    const mmq_y_buffer bytes = mmq_y_bytes(N, ne10_padded, cc);
    nb_ggml_hip_pool_alloc<char> x_q8(ctx.pool(), bytes.total);
    NB_HIP_CHECK(hipMemsetAsync(x_q8.get() + bytes.body, 0, bytes.total - bytes.body, stream));

    nb_quantize_mmq_q8_1_hip(X, nullptr, x_q8.get(), type, K,
                             /*s01=*/K, /*s02=*/(int64_t) K * N, /*s03=*/(int64_t) K * N,
                             ne10_padded, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1, stream);
    if (check_launch(tag, "MMQ Q8_1 quantize") != 0) return -2;

    const int64_t s01 = K / ggml_blck_size(type);
    const int64_t s12 = (int64_t) N * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const mmq_args args = {
        /*x=*/(const char *) W, /*type_x=*/type, /*y=*/(const int *) x_q8.get(),
        /*ids_dst=*/nullptr, /*expert_bounds=*/nullptr, /*dst=*/out,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/N,
        /*stride_row_x=*/s01, /*ncols_y=*/N, /*nrows_dst=*/M,
        /*nchannels_x=*/1, /*nchannels_y=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/s12, /*stride_channel_dst=*/0,
        /*nsamples_x=*/1, /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/s12, /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k(cc), /*ncols_max=*/N,
    };
    mul_mat_q_switch_type(ctx, args, stream);
    return check_launch(tag, "mul_mat_q");
}

}  // namespace

}  // namespace nb_mmq

extern "C" int nb_mmq_init(int device) {
    using namespace nb_mmq;
    const auto & info = nb_ggml_hip_info();
    if (device < 0 || device >= info.device_count) {
        fprintf(stderr, "nb_mmq_init: device %d out of range (have %d)\n", device,
                info.device_count);
        return -1;
    }
    nb_ggml_hip_set_device(device);
    return get_ctx_for_device(device) ? 0 : -1;
}

extern "C" void nb_mmq_cleanup(void) {
    using namespace nb_mmq;
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    for (auto *& ctx : g_ctx) {
        delete ctx;
        ctx = nullptr;
    }
}

extern "C" int nb_mmq_bind_stream(hipStream_t stream) {
    using namespace nb_mmq;
    nb_ggml_hip_context * ctx = get_ctx_for_device(nb_ggml_hip_get_device());
    return ctx != nullptr && nb_pool_reserve_stream(ctx->pool(), stream) == hipSuccess ? 0 : -1;
}

extern "C" void nb_mmq_release_stream(hipStream_t stream) {
    using namespace nb_mmq;
    nb_ggml_hip_context * ctx = get_ctx_for_device(nb_ggml_hip_get_device());
    if (ctx != nullptr) nb_pool_release_stream(ctx->pool(), stream);
}

extern "C" int nb_mmq_dense(int weight_type, const void * W, const float * X, float * out, int M,
                            int N, int K, hipStream_t stream) {
    using namespace nb_mmq;
    static const char * tag = "nb_mmq_dense";
    if (!supported_type(weight_type)) {
        fprintf(stderr, "%s: unsupported weight type %d\n", tag, weight_type);
        return -1;
    }
    if (!W || !X || !out || M <= 0 || N <= 0 || K <= 0 || K % 256 != 0) {
        fprintf(stderr, "%s: bad arguments M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    const int dev = nb_ggml_hip_get_device();
    nb_ggml_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) return -1;
    const int cc = nb_ggml_hip_info().devices[dev].cc;
    const ggml_type type = (ggml_type) weight_type;

    nb_pool_set_stream(stream);
    if (N <= MMVQ_MAX_BATCH_SIZE) {
        return dense_vec(tag, *ctx, type, W, X, out, M, N, K, stream);
    }
    return dense_tiled(tag, *ctx, cc, type, W, X, out, M, N, K, stream);
}

extern "C" int nb_mmq_dense_q8_1(int weight_type, const void * W, const void * Xq,
                                  float scale, const float * bias, float * out, int M, int N,
                                  int K, hipStream_t stream) {
    using namespace nb_mmq;
    static const char * tag = "nb_mmq_dense_q8_1";
    if (!supported_type(weight_type)) {
        fprintf(stderr, "%s: unsupported weight type %d\n", tag, weight_type);
        return -1;
    }
    // Upstream fuses epilogues only into single-column products.
    if (!W || !Xq || !out || M <= 0 || N <= 0 || N > MMVQ_MAX_BATCH_SIZE || K <= 0 ||
        K % MATRIX_ROW_PADDING != 0 || ((bias != nullptr || scale != 1.0f) && N != 1)) {
        fprintf(stderr, "%s: bad arguments M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    return dense_vec_q8_1(tag, (ggml_type) weight_type, W, Xq, scale, bias, out, M, N, K,
                          stream);
}
