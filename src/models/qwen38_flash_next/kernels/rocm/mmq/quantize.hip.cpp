#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "quantize.hpp"
#include <cstdint>

__launch_bounds__(HIP_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

// quantize_q8_1 for one-expert slot rows (the decode down projection's
// view). Slot i1 is written at its rank among the slots ordered by expert
// and then slot, so slots that share an expert are adjacent; the first
// column block also records each rank's expert and source slot. Every row
// keeps quantize_q8_1's values; only its position changes.
__launch_bounds__(HIP_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1_by_expert(
        const float * __restrict__ x, const int32_t * __restrict__ ids,
        void * __restrict__ vy, int32_t * __restrict__ sorted_ids,
        int32_t * __restrict__ slot_of_rank, const int64_t ne00,
        const int64_t s01, const int64_t ne0, const int slots) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
    const int i1 = blockIdx.y;
    const int lane = threadIdx.x % 32;
    const int32_t expert = ids[i1];
    int rank = 0;
    for (int base = 0; base < slots; base += 32) {
        const int j = base + lane;
        const int32_t other = j < slots ? ids[j] : 0;
        rank += __popc(static_cast<uint32_t>(__ballot(
            j < slots && (other < expert || (other == expert && j < i1)))));
    }

    if (i0 >= ne0) {
        return;
    }

    const int64_t i_cont = (int64_t)rank * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i1*s01 + i0] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (i0 == 0) {
        sorted_ids[rank] = expert;
        slot_of_rank[rank] = i1;
    }

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor(sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

void quantize_row_q8_1_hip(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, hipStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + HIP_QUANTIZE_BLOCK_SIZE - 1) / HIP_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(HIP_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

void quantize_row_q8_1_by_expert_hip(
        const float * x, const int32_t * ids, void * vy, int32_t * sorted_ids,
        int32_t * slot_of_rank, const int64_t ne00, const int64_t s01,
        const int64_t ne0, const int slots, hipStream_t stream) {
    GGML_ASSERT(ne0 % QK8_1 == 0 && slots > 0);
    const int64_t block_num_x = (ne0 + HIP_QUANTIZE_BLOCK_SIZE - 1) / HIP_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, slots, 1);
    const dim3 block_size(HIP_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1_by_expert<<<num_blocks, block_size, 0, stream>>>(
        x, ids, vy, sorted_ids, slot_of_rank, ne00, s01, ne0, slots);
}

void quantize_mmq_q8_1_hip(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, hipStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the HIP grid:
    const int64_t block_num_y = (ne0 + 4*HIP_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*HIP_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(HIP_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

}  // namespace qfn_mmq
