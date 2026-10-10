#pragma once

#include "common.hpp"
#include "mmq.hpp"

#include <cstdint>

#define HIP_QUANTIZE_BLOCK_SIZE     256
#define HIP_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    HIP_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*HIP_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

void quantize_row_q8_1_hip(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, hipStream_t stream);

// quantize_row_q8_1_hip for `slots` one-expert rows, written in expert
// order: row r of vy is slot slot_of_rank[r], whose expert is sorted_ids[r].
void quantize_row_q8_1_by_expert_hip(
        const float * x, const int32_t * ids, void * vy, int32_t * sorted_ids,
        int32_t * slot_of_rank, int64_t ne00, int64_t s01, int64_t ne0,
        int slots, hipStream_t stream);

void quantize_mmq_q8_1_hip(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, hipStream_t stream);
