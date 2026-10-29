// SPDX-License-Identifier: MIT
// Adapted from llama.cpp ggml/src/ggml-cuda (MIT, The ggml authors); see
// VENDOR.md.
#pragma once
#include "common.hip.hpp"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// Direct switch over the per-type templated mmvq dispatch. Was
// `static` upstream; promoted so nb_mmq.hip.cpp can call it without
// constructing ggml_tensor structs.  Args match the original signature
// exactly.
void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const nb_ggml_hip_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, hipStream_t stream);
