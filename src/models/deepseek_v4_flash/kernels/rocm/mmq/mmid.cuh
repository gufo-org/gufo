#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, cudaStream_t stream);

// ds4 local (P5): whether the large-n global-memory mm_ids path is enabled
// (unconditional; callers no longer refuse past-cap shapes).
