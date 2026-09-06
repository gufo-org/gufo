/* Dynamic LDS for the Q2-down WMMA kernel. The epilogue aliases its
 * float C staging over the half A/B staging buffers, so allocation needs the
 * larger region rather than their sum. */
static size_t ds4_rocm_q2_down_wmma_shmem(uint32_t mtiles, uint32_t bm,
                                          uint32_t bn, uint32_t bk) {
    const size_t ab = ((size_t)mtiles * bm * bk + 2u * (size_t)bk * bn) *
                      sizeof(__half);
    const size_t c = (2u * (size_t)mtiles * bm * bn) * sizeof(float);
    return ab > c ? ab : c;
}

/* Dynamic LDS for the wide-N variant.
 *
 * The A and B staging halves, then the raw Q2_K slab window that the staged
 * dequantizer reads. The epilogue's single-fragment C page aliases A and B, so
 * only the larger of those two counts, but the raw window has to sit beyond
 * both because the K loop still needs it. */
static size_t ds4_rocm_q2_down_wide_shmem(uint32_t mtiles, uint32_t bm,
                                          uint32_t bn, uint32_t bk,
                                          uint32_t nfrag) {
    /* Two mid-tile buffers: the kernel prefetches the next K step's tile while
     * the current one still feeds the matrix ops. */
    const size_t ab = (2u * (size_t)mtiles * bm * bk +
                       (size_t)nfrag * bk * bn) * sizeof(__half);
    const size_t c = ((size_t)mtiles * bm * bn) * sizeof(float);
    const size_t raw = (size_t)nfrag * bn * (84u / sizeof(uint32_t)) *
                       sizeof(uint32_t);
    return (ab > c ? ab : c) + raw;
}

/* Smallest row group used by the wide Q2-down kernel. Scratch is sized for
 * this case; the wider 64-row route therefore only over-allocates. */
#define DS4_ROCM_WIDE_DOWN_MIN_TILE_M 32u

/* Chunk width above which the routed column-tile cost model stands aside and the
 * vendored selector's own rule runs. A forced-width sweep put the default within
 * 1.1% of the best measured width from here up; below it the default gives up 10
 * to 14%. The table is at the call site. */
#define DS4_ROCM_ROUTED_TILE_MODEL_ROWS 1024u

/* The wide Q2-down pair tile normally uses four row fragments. A later pp128
 * profile isolated the narrow exception: two fragments reduced this kernel
 * from 361.27 to 335.27 ms and the full span from about 2,048 to 2,025 ms.
 * Widths at and above 256 retain four fragments because halving the rows per
 * tile repeats enough Q2_K dequantization to lose throughput. */

/* Deferred routed expert sum.
 *
 * The 6-way sum over the per-expert F16 down rows writes a float buffer that the
 * hyper-connection expansion immediately reads back, one 128 MiB round trip per
 * layer. When the caller asks to defer it and the F16 down route is the one that
 * runs, the sum is skipped here and folded into the expansion instead. The
 * caller must read ds4_gpu_routed_sum_deferred() afterwards: only the F16 route
 * can defer, so the request is advisory. */
static int g_routed_defer_sum_request = 0;
static int g_routed_defer_sum_taken = 0;

extern "C" void ds4_gpu_set_routed_defer_sum(int enabled) {
    g_routed_defer_sum_request = enabled ? 1 : 0;
    g_routed_defer_sum_taken = 0;
}

extern "C" int ds4_gpu_routed_sum_deferred(void) { return g_routed_defer_sum_taken; }

/*
 * Row count below which a routed-MoE call is treated as a speculative
 * verification block rather than a prompt chunk.
 *
 * The expert-tile routes size their grids by the 256-entry expert table, so a
 * six-row block launches two orders of magnitude more blocks than it needs. The
 * decode routes are indexed by (row, expert) pair instead, which both matches
 * the work and reproduces one-token decode's per-row arithmetic.
 */
static uint32_t ds4_rocm_moe_small_batch_rows(void) {
    return ds4_rocm_small_batch_limit(16u);
}

static uint32_t ds4_rocm_compact_down_rows_per_block(void) {
    return 32u;
}

/* Mixed IQ2_XXS-gate/Q2_K-down models already compute routed mid activations
 * as float.  Reuse the newer Q2_K expert-batch/WMMA down kernels instead of
 * re-quantizing mid to Q8_K and taking the older qwarp down path.  This keeps
 * the CyberNeurova all-Q2 path untouched while giving the standard IQ2 mix the
 * same fast Q2 down projection used by q2k_path. */
static int routed_moe_q2_float_down_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *down,
        const ds4_gpu_tensor *mid,
        const __half *mid_h_hot,
        int hot_mid_f16,
        const char *down_w,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *sorted_pairs,
        uint32_t *hot_experts_dev,
        uint32_t n_tokens,
        uint32_t n_expert,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        const uint32_t *h_counts_in,
        uint32_t *wide_tile_map_dev) {
    if (!out || !down || !mid || !down_w || !counts || !offsets || !sorted_pairs ||
        n_tokens == 0u || n_expert != DS4_ROCM_N_EXPERT_USED ||
        (expert_mid_dim % ROCM_QK_K) != 0u || expert_mid_dim == 0u || out_dim == 0u ||
        !hip_tensor_has_elems3(mid, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !hip_tensor_has_elems3(down, n_tokens, n_expert, out_dim, sizeof(float)) ||
        !hip_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }

    uint32_t h_counts[DS4_ROCM_N_EXPERT] = {0};
    if (h_counts_in) {
        memcpy(h_counts, h_counts_in, sizeof(h_counts));
    } else if (!hip_ok(hipMemcpy(
                    h_counts,
                    counts,
                    sizeof(h_counts),
                    hipMemcpyDeviceToHost),
                "routed_moe iq2/q2 float-down counts copy")) {
        return 0;
    }

    /* Pairs a wave carries per dequantization of its weight row.
     *
     * At 32 tokens the mean live expert bucket is below two assignments. A
     * two-pair tile avoids staging two empty activation rows for most experts
     * while preserving each pair's K accumulation and F16 storage order.
     *
     * Eight looks free -- this route only serves experts below the hot
     * threshold, so every bucket would fit one tile instead of the two that a
     * five-to-seven-row bucket pays at four -- and it is 64% slower: 132.6 to
     * 218.0 ms at kernel level. The mid staging loop runs over the whole tile
     * whatever the bucket holds, so eight stages 2,048 floats per K block for
     * three real rows, and that costs more than the second weight pass it saves. */
    const uint32_t down_tile = n_tokens <= 128u ? 2u : 4u;
    /* Output rows per workgroup on the scalar route, one wave each.
     *
     * Swept because this kernel takes two barriers per 256-value K slab with
     * however many waves the block holds, so sixteen looked like it might be
     * paying for synchronization it did not need. It is the other way round --
     * the waves share one staged mid tile, so fewer of them means the tile is
     * staged more often for less work:
     *
     *   rows/block   pp256   pp512  pp1024  pp2048
     *            2  225.15  321.08  397.01  478.04
     *            4  242.77  337.54  409.62  486.08
     *            8  254.89  348.86  416.84  490.54
     *           16  355.20 / 354.92 either side of the sweep
     *           32  264.90  357.61  422.24  494.46
     *
     * Monotone in the number of waves, so the barriers are not the cost. The
     * 1,024-thread block is worth about +0.7% at 512 over sixteen, which is one
     * sample inside the spread of the two bracketing sixteen arms, so sixteen
     * stays; revisit only with a proper interleaved pair. */
    const uint32_t down_rpb = 16u;
    const uint32_t down_threads = down_rpb * 32u;
    const size_t down_shmem = (size_t)down_tile * 256u * sizeof(float);
    const int use_f16_down = (out_dim & 1u) == 0u;
    __half *down_h = use_f16_down ? (__half *)down->ptr : NULL;

    uint32_t hot_count = 0u;
    uint32_t hot_max = 0u;
    /*
     * The scalar/WMMA crossover shifts with prompt width. At pp512, thresholds
     * 2/3/4/6/8/10 measured 357.88/359.70/361.05/358.64/355.71/352.63
     * tok/s after a pp4096 warmup. Four also improves pp256 through pp2048.
     * Keep sub-64-row state initialization and wide prompts on the established
     * eight-row policy; short-seed arithmetic affects the pinned trajectory.
     */
    const uint32_t hot_threshold =
        n_tokens >= 64u && n_tokens <= 2048u ? 4u : 8u;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    const int use_wmma_hot = hot_experts_dev &&
        (expert_mid_dim % 16u) == 0u && (out_dim % 16u) == 0u;
#else
    const int use_wmma_hot = 0;
#endif
    uint32_t h_active[256] = {0};
    if (use_wmma_hot) {
        for (uint32_t e = 0; e < 256u; e++) {
            const uint32_t c = h_counts[e];
            if (c >= hot_threshold) {
                h_active[hot_count++] = e;
                if (c > hot_max) hot_max = c;
            }
        }
    }

    const uint32_t scalar_max = hot_count != 0u ? hot_threshold : 0u;
    /* Keep the hot experts first and append the populated scalar experts. The
     * same short H2D copy feeds both kernels. At 32 tokens fewer than half the
     * expert table is live, so this avoids tens of thousands of empty scalar
     * workgroups without changing pair order or arithmetic. */
    uint32_t singleton_count = 0u;
    for (uint32_t e = 0; e < DS4_ROCM_N_EXPERT; e++) {
        const uint32_t count = h_counts[e];
        if (count == 1u && (scalar_max == 0u || count < scalar_max)) {
            h_active[hot_count + singleton_count++] = e;
        }
    }
    uint32_t cold_multi_count = 0u;
    for (uint32_t e = 0; e < DS4_ROCM_N_EXPERT; e++) {
        const uint32_t count = h_counts[e];
        if (count > 1u && (scalar_max == 0u || count < scalar_max)) {
            h_active[hot_count + singleton_count + cold_multi_count++] = e;
        }
    }
    const uint32_t cold_count = singleton_count + cold_multi_count;
    const uint32_t active_count = hot_count + cold_count;
    const bool compact_experts = hot_experts_dev != NULL;
    if (compact_experts && active_count != 0u &&
        !hip_ok(hipMemcpy(hot_experts_dev, h_active,
                         active_count * sizeof(uint32_t), hipMemcpyHostToDevice),
                "routed_moe iq2/q2 float-down active copy")) {
        return 0;
    }
    const uint32_t *singleton_experts_dev =
        compact_experts ? hot_experts_dev + hot_count : NULL;
    const uint32_t *cold_multi_experts_dev =
        compact_experts ? singleton_experts_dev + singleton_count : NULL;
    const uint32_t singleton_rpb = 32u;
    const uint32_t singleton_threads = singleton_rpb * 32u;
    const dim3 singleton_grid(
        (out_dim + singleton_rpb - 1u) / singleton_rpb,
        compact_experts ? singleton_count : DS4_ROCM_N_EXPERT,
        1u);
    const dim3 down_grid(
        (out_dim + down_rpb - 1u) / down_rpb,
        compact_experts ? cold_multi_count : DS4_ROCM_N_EXPERT,
        1u);
    if (singleton_count != 0u && use_f16_down) {
        moe_down_q2K_expert_batch_sharedmid_kernel<1,false,true><<<
                singleton_grid, singleton_threads, 256u * sizeof(float)>>>(
                NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, 2u, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, 0u, singleton_experts_dev);
    } else if (singleton_count != 0u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<1><<<
                singleton_grid, singleton_threads, 256u * sizeof(float)>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, 2u, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, 0u, singleton_experts_dev);
    }
    if (cold_multi_count == 0u) {
        /* nothing to do */
    } else if (use_f16_down) {
        if (down_tile == 2u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<2,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 2u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, 0u, cold_multi_experts_dev);
        } else if (down_tile == 4u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 2u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, 0u, cold_multi_experts_dev);
        } else if (down_tile == 8u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<8,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 2u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, 0u, cold_multi_experts_dev);
        } else {
            moe_down_q2K_expert_batch_sharedmid_kernel<16,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 2u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, 0u, cold_multi_experts_dev);
        }
    } else if (down_tile == 2u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<2><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 2u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, 0u, cold_multi_experts_dev);
    } else if (down_tile == 4u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<4><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 2u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, 0u, cold_multi_experts_dev);
    } else if (down_tile == 8u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<8><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 2u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, 0u, cold_multi_experts_dev);
    } else {
        moe_down_q2K_expert_batch_sharedmid_kernel<16><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 2u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, 0u, cold_multi_experts_dev);
    }
    if (!hip_ok(hipGetLastError(), "routed_moe iq2/q2 float-down scalar launch")) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (use_wmma_hot && hot_count != 0u) {
        constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
        const int no_n2 = 0;
        const uint32_t wmma_mtiles = 4u;
        /* Four column fragments. Eight halves how often the 64-row mid tile is
         * re-read per column block and measured neutral both before and after
         * the per-slab weight staging (1,913 / 1,900 ms, then 415.4 / 417.9
         * tok/s), which is what ruled out mid re-read bytes as this kernel's
         * bound. Software-pipelining the mid load through registers was also
         * noise (+0.7%), and double-buffering both tiles so one barrier per K
         * step suffices cost 2% because the extra 4 KiB dropped residency from
         * six workgroups per CU to four. Counters put the kernel at 9 to 18%
         * instruction-issue utilisation with 96 VGPRs and no scratch, so it is
         * latency-bound, and on this tile layout occupancy is the lever that
         * matters rather than barrier count. */
        /* Four column fragments. Narrow chunks use two row tiles; wider
         * chunks retain four.
         *
         * Eight column fragments halve how often the 64-row mid tile is
         * re-staged and measured 417.69 / 415.46 tok/s against four's 406.53 /
         * 406.54, but prefetching the mid tile instead reaches the same 421.0
         * with 4 KiB less LDS, and stacking both gives 417.69 / 415.46 -- the
         * two address the same stall, so only one of them pays. Eight row tiles
         * halve weight dequantization per output but leave about half of a mean
         * expert bucket empty at this router skew, and cancelled exactly
         * (377.14 against 377.41 tok/s). */
        constexpr uint32_t wide_nfrag = 4u;
        const uint32_t wide_mtiles = n_tokens <= 128u ? 2u : 4u;
        if (wmma_mtiles == 4u && use_f16_down && hot_mid_f16 && mid_h_hot &&
            (out_dim % (wide_nfrag * bn)) == 0u) {
            const uint32_t wide_tile_m = wide_mtiles * bm;
            const dim3 block(32u * wide_mtiles, 1u, 1u);
            /*
             * A rectangular (row group, hot expert) grid has to be sized by the
             * largest bucket, and the router skew at a 4,096-token chunk puts
             * that near 3,500 rows while the mean bucket is 96. Every block past
             * its own expert's bucket exits immediately, but it still reserves
             * the full dynamic LDS tile, and that reservation is what caps
             * residency. Compact the pairs that have rows into one list instead.
             */
            uint32_t *tile_map = NULL;
            uint32_t wide_tiles = 0u;
            if (wide_tile_map_dev) {
                const uint32_t cap =
                    n_tokens * n_expert / DS4_ROCM_WIDE_DOWN_MIN_TILE_M +
                    DS4_ROCM_N_EXPERT + 1u;
                static std::vector<uint32_t> h_map;
                h_map.clear();
                h_map.reserve(cap);
                for (uint32_t h = 0; h < hot_count && h_map.size() < cap; h++) {
                    const uint32_t groups =
                        (h_counts[h_active[h]] + wide_tile_m - 1u) /
                        wide_tile_m;
                    for (uint32_t g = 0; g < groups && h_map.size() < cap; g++) {
                        h_map.push_back((h << 16) | g);
                    }
                }
                if (!h_map.empty() &&
                    hip_ok(hipMemcpy(wide_tile_map_dev, h_map.data(),
                                       h_map.size() * sizeof(uint32_t),
                                       hipMemcpyHostToDevice),
                            "routed_moe wide down tile map copy")) {
                    tile_map = wide_tile_map_dev;
                    wide_tiles = (uint32_t)h_map.size();
                }
            }
            const dim3 grid(out_dim / (wide_nfrag * bn),
                            tile_map ? wide_tiles
                                     : (hot_max + wide_tile_m - 1u) / wide_tile_m,
                            tile_map ? 1u : hot_count);
            const size_t shmem =
                ds4_rocm_q2_down_wide_shmem(
                    wide_mtiles, bm, bn, bk, wide_nfrag);
            if (wide_mtiles == 2u) {
                moe_down_q2K_hotlist_wmma_wide_kernel<
                    2, 16, 16, 16, wide_nfrag, true, true>
                        <<<grid, block, shmem>>>(
                        NULL, down_h, down_w, NULL, mid_h_hot,
                        counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes,
                        0u, tile_map);
            } else {
                moe_down_q2K_hotlist_wmma_wide_kernel<
                    4, 16, 16, 16, wide_nfrag, true, true>
                        <<<grid, block, shmem>>>(
                        NULL, down_h, down_w, NULL, mid_h_hot,
                        counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes,
                        0u, tile_map);
            }
        } else if (!no_n2) {
            if (wmma_mtiles == 4u) {
                constexpr uint32_t mt = 4u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 =
                ds4_rocm_q2_down_wmma_shmem(mt, bm, bn, bk);
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
            } else if (wmma_mtiles == 16u) {
                constexpr uint32_t mt = 16u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 =
                    ds4_rocm_q2_down_wmma_shmem(mt, bm, bn, bk);
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
            } else {
                constexpr uint32_t mt = 8u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 =
                    ds4_rocm_q2_down_wmma_shmem(mt, bm, bn, bk);
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
            }
        } else if (wmma_mtiles == 16u) {
            constexpr uint32_t mt = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(__half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<16,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        } else if (wmma_mtiles == 4u) {
            constexpr uint32_t mt = 4u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(__half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<4,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        } else {
            constexpr uint32_t mt = 8u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(__half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<8,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        }
        if (!hip_ok(hipGetLastError(), "routed_moe iq2/q2 float-down wmma launch")) return 0;
    }
#endif

    const uint64_t n = (uint64_t)n_tokens * out_dim;
    if (use_f16_down && g_routed_defer_sum_request) {
        /* The per-expert F16 rows stay in `down` for the fused expansion. */
        g_routed_defer_sum_taken = 1;
        return 1;
    }
    if (use_f16_down && (out_dim & 1u) == 0u) {
        const uint64_t n2 = (uint64_t)n_tokens * (out_dim >> 1u);
        moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                (float *)out->ptr, down_h, out_dim, n_expert, n_tokens);
    } else if (use_f16_down) {
        moe_sum_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                (float *)out->ptr, down_h, out_dim, n_expert, n_tokens);
    } else {
        moe_sum_kernel<<<(n + 255u) / 256u, 256>>>(
                (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
    }
    return hip_ok(hipGetLastError(), "routed_moe iq2/q2 float-down sum launch");
}

typedef struct {
    int q4k_path;
    int iq2_path;
    int q2k_path;
    uint64_t gate_bytes;
    uint64_t down_bytes;
} routed_moe_launch_plan;

static int routed_moe_build_plan(
        const ds4_gpu_tensor *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        routed_moe_launch_plan *plan) {
    if (!plan) return 0;
    memset(plan, 0, sizeof(*plan));
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0u ||
        n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT_USED ||
        expert_in_dim == 0u || expert_mid_dim == 0u || out_dim == 0u ||
        expert_in_dim % ROCM_QK_K != 0 || expert_mid_dim % ROCM_QK_K != 0 ||
        !hip_tensor_has_elems2(x, n_tokens, expert_in_dim, sizeof(float)) ||
        !hip_tensor_has_elems2(selected, n_tokens, n_expert, sizeof(int32_t)) ||
        !hip_tensor_has_elems2(weights, n_tokens, n_expert, sizeof(float)) ||
        !hip_tensor_has_elems3(gate, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !hip_tensor_has_elems3(up, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !hip_tensor_has_elems3(mid, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !hip_tensor_has_elems3(down, n_tokens, n_expert, out_dim, sizeof(float)) ||
        !hip_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }
    plan->q4k_path = (gate_type == 12u && down_type == 12u);
    plan->iq2_path = (gate_type == 16u && down_type == 10u);
    plan->q2k_path = (gate_type == 10u && down_type == 10u);
    if (!plan->q4k_path && !plan->iq2_path && !plan->q2k_path) return 0;
    if (!hip_u64_mul_checked(n_total_expert, gate_expert_bytes, &plan->gate_bytes) ||
        !hip_u64_mul_checked(n_total_expert, down_expert_bytes, &plan->down_bytes) ||
        !hip_model_range_fits(model_size, gate_offset, plan->gate_bytes) ||
        !hip_model_range_fits(model_size, up_offset, plan->gate_bytes) ||
        !hip_model_range_fits(model_size, down_offset, plan->down_bytes)) {
        return 0;
    }
    return 1;
}

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens) {
    routed_moe_launch_plan plan;
    if (!routed_moe_build_plan(out, gate, up, mid, down, model_map, model_size,
                               gate_offset, up_offset, down_offset, gate_type, down_type,
                               gate_expert_bytes, down_expert_bytes, expert_in_dim,
                               expert_mid_dim, out_dim, selected, weights, n_total_expert, n_expert, x,
                               n_tokens, &plan)) {
        return 0;
    }
    const int q4k_path = plan.q4k_path;
    const int iq2_path = plan.iq2_path;
    const int q2k_path = plan.q2k_path;
    const uint64_t gate_bytes = plan.gate_bytes;
    const uint64_t down_bytes = plan.down_bytes;
    const char *gate_w = hip_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
    const char *up_w = hip_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
    const char *down_w = hip_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / ROCM_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / ROCM_QK_K;
    const uint64_t xq_count = (uint64_t)n_tokens * xq_blocks;
    const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
    const uint64_t xq_bytes = xq_count * sizeof(hip_block_q8_K);
    const uint64_t midq_bytes = midq_count * sizeof(hip_block_q8_K);
    if (!q2k_path && down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        hip_block_q8_K *xq = (hip_block_q8_K *)down->ptr;
        hip_block_q8_K *midq = (hip_block_q8_K *)gate->ptr;
        const uint32_t pair_count = n_tokens * n_expert;
        /*
         * A DSpark verification block is a handful of rows, not a prompt chunk.
         * The sorted-pair expert-tile route pays a grid sized by the expert
         * table rather than by the work, so at these widths it costs far more
         * than running each (row, expert) pair through the decode route. The
         * decode route also computes each row independently, which is what
         * makes a verified row's logits match ordinary one-token decode.
         */
        const uint32_t small_batch = n_tokens > 1u &&
            n_tokens <= ds4_rocm_moe_small_batch_rows();
        const uint32_t grouped_gate =
            (ds4_rocm_verifier_batch_mode() ||
             ds4_rocm_support_batch_mode()) &&
            iq2_path && !q4k_path && n_expert == 6u &&
            n_tokens > 1u && n_tokens <= 8u;
        const uint32_t grouped_gate_only =
            grouped_gate && n_tokens > 6u;
        /*
         * DSpark rows share enough routed experts to amortize sorting and load
         * each selected gate/up matrix once. The verifier's Q2 down projection
         * is compacted separately over the live (row, slot) pairs below. W7
         * and W8 use sorting only for gate/up, retaining the qualified decode
         * down route and reduction order.
         */
        const uint32_t use_sorted_pairs =
            grouped_gate ||
            (n_tokens > 1u && !small_batch &&
             (!q4k_path || n_tokens >= 32u));
        const uint32_t use_expert_tiles = use_sorted_pairs;
        const uint32_t expert_tile_m = grouped_gate ? 4u : 8u;
        const uint32_t write_gate_up = 0u;
        /*
         * The vendored MMQ tier owns routed IQ2 gate/up for prompt chunks. It
         * reaches the integer matrix cores through `mul_mat_q`, which the native
         * hotlist WMMA kernel below cannot, and bounds each expert by
         * `n_tokens` rather than `n_tokens * top_k`.
         *
         * It is excluded from every DSpark route. MMQ's reduction order differs
         * from the native kernels', and speculative verification is only
         * lossless while a batched row's argmax matches ordinary decode's.
         */
        const uint32_t use_mmq_gateup =
            g_rocm_mmq_ready && iq2_path && !q4k_path &&
            n_tokens >= DS4_ROCM_ROUTED_MMQ_ROWS && !g_small_batch_mode;
        const uint32_t use_p2_sorted = 0u;
        const uint32_t use_atomic_down = use_expert_tiles && n_tokens >= 128u;
        const uint32_t use_gate_row2048 = !q4k_path && use_expert_tiles && n_tokens >= 128u;
        const uint32_t use_down_tile16 = !q4k_path && use_atomic_down && n_tokens >= 128u;
        const uint32_t use_decode_lut_gate =
            (n_tokens == 1u || (small_batch && !grouped_gate)) &&
            xq_blocks <= 16u;
        const uint32_t gate_row_span = 1024u;
        const uint32_t down_row_span = 2048u;
        const uint32_t use_down_row2048 = !q4k_path && use_atomic_down && use_down_tile16;
        const uint32_t use_compact_float_down =
            ds4_rocm_verifier_batch_mode() &&
            grouped_gate && n_expert == DS4_ROCM_N_EXPERT_USED &&
            n_tokens > 1u && n_tokens <= 6u;
        const uint32_t use_direct_down_sum6 =
            n_expert == 6u && n_tokens == 1u;
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t *iq2_gate_hot_dev = NULL;
        uint32_t *wide_tile_map_dev = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        /* Per-expert assignment counts, read back once and shared by every
         * consumer below: the MMQ column-tile bound, the Q2 down hot list, and
         * the cold-expert launch skip. */
        uint32_t h_sorted_counts[DS4_ROCM_N_EXPERT] = {0};
        int h_sorted_counts_valid = 0;
        uint32_t h_sorted_counts_max = 0;
        if (!use_mmq_gateup) {
            dim3 xq_grid(xq_blocks, n_tokens, 1);
            q8_K_quantize_kernel<<<xq_grid, 256>>>(
                    xq, (const float *)x->ptr, expert_in_dim, n_tokens);
            ok = hip_ok(hipGetLastError(), "routed_moe x quantize launch");
        }
        if (ok && use_sorted_pairs) {
            const uint64_t counts_bytes = 256ull * sizeof(uint32_t);
            const uint64_t offsets_bytes = 257ull * sizeof(uint32_t);
            const uint64_t cursors_bytes = 256ull * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            /*
             * One tile per full group plus at most one partial tile per distinct
             * expert. A batch cannot touch more experts than it has pairs, so the
             * old flat +256 slack made a narrow batch launch two orders of
             * magnitude more tiles than it could ever fill.
             */
            const uint32_t tile_slack =
                pair_count < DS4_ROCM_N_EXPERT ? pair_count : DS4_ROCM_N_EXPERT;
            tile_capacity =
                (pair_count + expert_tile_m - 1u) / expert_tile_m + tile_slack;
            tile16_capacity = use_down_tile16 ? ((pair_count + 15u) / 16u + 256u) : 0u;
            const uint64_t tile_offsets_bytes = 257ull * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? 257ull * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t iq2_gate_hot_off = tile16_starts_off + tile16_starts_bytes;
            const uint64_t iq2_gate_hot_bytes = 256ull * sizeof(uint32_t);
            /* One entry per (hot expert, row group) the wide Q2 down kernel has
             * rows for, replacing its rectangular hot_max-by-hot_count grid. */
            const uint64_t wide_tile_map_off = iq2_gate_hot_off + iq2_gate_hot_bytes;
            const uint64_t wide_tile_map_bytes =
                ((uint64_t)pair_count / DS4_ROCM_WIDE_DOWN_MIN_TILE_M +
                 DS4_ROCM_N_EXPERT + 1ull) * sizeof(uint32_t);
            const uint64_t scratch_bytes = wide_tile_map_off + wide_tile_map_bytes;
            uint8_t *scratch = (uint8_t *)hip_tmp_alloc(scratch_bytes,
                                                         "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                iq2_gate_hot_dev = (uint32_t *)(scratch + iq2_gate_hot_off);
                wide_tile_map_dev = (uint32_t *)(scratch + wide_tile_map_off);
                ok = hip_ok(hipMemset(counts, 0, counts_bytes), "routed_moe sorted counts clear");
                if (ok) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        (const int32_t *)selected->ptr,
                        pair_count);
                    ok = hip_ok(hipGetLastError(), "routed_moe sorted count launch");
                }
                /*
                 * The compact verifier/session down path consumes the device
                 * counts directly. Avoid synchronizing all 256 counters back
                 * to the host on every layer when no later route reads them.
                 */
                if (ok && !use_compact_float_down && !grouped_gate_only) {
                  ok = hip_ok(
                      hipMemcpy(h_sorted_counts, counts,
                                sizeof(h_sorted_counts), hipMemcpyDeviceToHost),
                      "routed_moe sorted counts copy");
                  if (ok) {
                    h_sorted_counts_valid = 1;
                    for (uint32_t e = 0; e < DS4_ROCM_N_EXPERT; e++) {
                      if (h_sorted_counts[e] > h_sorted_counts_max) {
                        h_sorted_counts_max = h_sorted_counts[e];
                      }
                    }
                  }
                }
                if (ok) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts);
                    ok = hip_ok(hipGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok) {
                    moe_scatter_sorted_pairs_deterministic_kernel<<<
                            256u, 256u, 256u * sizeof(uint32_t)>>>(
                        sorted_pairs,
                        offsets,
                        (const int32_t *)selected->ptr,
                        pair_count);
                    ok = hip_ok(hipGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, expert_tile_m);
                    ok = hip_ok(hipGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tiles_kernel<<<1, 256>>>(tile_experts, tile_starts, tile_offsets, counts, expert_tile_m);
                    ok = hip_ok(hipGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, 16u);
                    ok = hip_ok(hipGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<1, 256>>>(tile16_experts, tile16_starts, tile16_offsets, counts, 16u);
                    ok = hip_ok(hipGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        int mmq_gateup_done = 0;
        int mmq_hot_mid_f16 = 0;
        __half *mmq_mid_h = NULL;
        if (ok && use_mmq_gateup) {
            const uint64_t pair_count64 = (uint64_t)n_tokens * n_expert;
            const uint64_t mid_count = pair_count64 * expert_mid_dim;
            uint64_t down_h_bytes = 0;
            uint64_t mid_h_bytes = 0;
            /* The wide Q2 down kernel reads its activations as F16, so carve the
             * mirror out of the down scratch behind the F16 down result. */
            if ((out_dim & 1u) == 0u &&
                hip_u64_mul3_checked(
                    pair_count64, out_dim, sizeof(__half), &down_h_bytes) &&
                hip_u64_mul_checked(mid_count, sizeof(__half), &mid_h_bytes) &&
                down_h_bytes <= down->bytes &&
                mid_h_bytes <= down->bytes - down_h_bytes) {
                mmq_mid_h = (__half *)((char *)down->ptr + down_h_bytes);
            }

            const int use_fused_swiglu = g_rocm_gfx1151 && mmq_mid_h != NULL;
            int rc = -1;
            /* The routed grid covers ncols_max column tiles for all 256
             * experts; without a bound that is the whole chunk, of which one
             * bucket holds about n_tokens * n_expert / 256 rows. */
            ds4_mmq_set_routed_max_expert_rows(
                h_sorted_counts_valid ? (int)h_sorted_counts_max : 0);
            /* The bound above removes empty column tiles; this sizes the ones
             * that remain. A narrow chunk spreads its pairs so thinly that the
             * default 80-column tile is mostly padding, and only the whole
             * count array says how thinly. See ds4_mmq_set_routed_tile_cols.
             *
             * Scoped to chunks under DS4_ROCM_ROUTED_TILE_MODEL_ROWS because a
             * forced-width sweep says the default is already right above it.
             * Whole-chunk tok/s, one warm process, this build with the width
             * pinned:
             *
             *   width   pp256   pp512  pp1024  pp2048
             *      16  261.47  354.44  426.59  472.36
             *      32  243.66  343.55  420.91  476.07
             *      48  240.51  346.71  435.80  499.01
             *      64  228.26  332.74  429.85  496.93
             *      80  219.74  320.62  420.49  493.31
             *  default  228.80  321.43  421.62  493.80
             *
             * The default rule yields 80 at every one of these widths, so it
             * gives up 14% at 256 and 10% at 512 and is within 1.1% of the best
             * measured width at 1,024 and 2,048. The cost model reproduces the
             * narrow optimum and mis-ranks 32 against 48 at the wide end. A
             * fixed 48-column tile measured +3.4% at 1,024 and +1.1% at 2,048,
             * so use it only across that measured interval and leave 4,096 and
             * larger chunks on the vendored selector. */
            int routed_tile_cols = 0;
            if (h_sorted_counts_valid &&
                n_tokens < DS4_ROCM_ROUTED_TILE_MODEL_ROWS) {
              routed_tile_cols = ds4_mmq_routed_tile_cols_for_counts(
                  h_sorted_counts, (int)DS4_ROCM_N_EXPERT);
            } else if (h_sorted_counts_valid && n_tokens <= 2048u) {
              routed_tile_cols = 48;
            }
            ds4_mmq_set_routed_tile_cols(routed_tile_cols);
            if (routed_tile_cols != 0) {
                static int logged_tile_cols = 0;
                if (logged_tile_cols != routed_tile_cols) {
                    logged_tile_cols = routed_tile_cols;
                    uint32_t live = 0;
                    for (uint32_t e = 0; e < DS4_ROCM_N_EXPERT; e++) {
                        if (h_sorted_counts[e] != 0u) live++;
                    }
                    fprintf(stderr,
                            DS4_GPU_LOG_PREFIX
                            "routed gate/up column tile %d (tokens=%u, "
                            "%u live experts, max bucket %u)\n",
                            routed_tile_cols,
                            n_tokens,
                            live,
                            h_sorted_counts_max);
                }
            }
            if (use_fused_swiglu) {
                ds4_mmq_set_aligned_q81_scratch(up->ptr, (size_t)up->bytes);
                rc = ds4_mmq_iq2_xxs_moe_pair_token_bound_swiglu(
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    (float *)gate->ptr,
                    (float *)down->ptr,
                    (float *)mid->ptr,
                    mmq_mid_h,
                    (int)expert_mid_dim,
                    (int)expert_in_dim,
                    (int)n_tokens,
                    (int)n_total_expert,
                    (int)n_expert,
                    clamp,
                    (hipStream_t)0);
            }
            const int fused_swiglu_done = use_fused_swiglu && rc == 0;
            if (rc != 0) {
                ds4_mmq_set_aligned_q81_scratch(down->ptr, (size_t)down->bytes);
                rc = ds4_mmq_iq2_xxs_moe_pair_token_bound(
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected->ptr,
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (int)expert_mid_dim,
                    (int)expert_in_dim,
                    (int)n_tokens,
                    (int)n_total_expert,
                    (int)n_expert,
                    (hipStream_t)0);
            }
            ds4_mmq_set_aligned_q81_scratch(NULL, 0);
            ds4_mmq_set_routed_max_expert_rows(0);
            ds4_mmq_set_routed_tile_cols(0);
            if (rc == 0 && !fused_swiglu_done) {
                moe_mmq_swiglu_weighted_clamp_kernel<<<
                        (uint32_t)((mid_count + 255u) / 256u), 256>>>(
                        (float *)mid->ptr,
                        mmq_mid_h,
                        (const float *)gate->ptr,
                        (const float *)up->ptr,
                        (const float *)weights->ptr,
                        expert_mid_dim,
                        n_tokens,
                        n_expert,
                        clamp);
                rc = hip_ok(hipGetLastError(),
                             "routed_moe HIP MMQ swiglu launch") ? 0 : -1;
            }
            if (rc == 0 && mmq_mid_h) mmq_hot_mid_f16 = 1;
            if (rc == 0) {
                mmq_gateup_done = 1;
                static int logged = 0;
                if (!logged) {
                    logged = 1;
                    fprintf(stderr,
                            DS4_GPU_LOG_PREFIX
                            "routed MoE using HIP MMQ gate/up%s with native Q2 down\n",
                            fused_swiglu_done ? " fused SwiGLU" : "");
                }
            } else {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "gate/up MMQ returned %d "
                        "(tokens=%u); falling back\n",
                        rc,
                        n_tokens);
                dim3 xq_grid(xq_blocks, n_tokens, 1);
                q8_K_quantize_kernel<<<xq_grid, 256>>>(
                        xq, (const float *)x->ptr, expert_in_dim, n_tokens);
                ok = hip_ok(hipGetLastError(),
                             "routed_moe MMQ fallback x quantize launch");
            }
        }
        uint32_t iq2_gate_hot_count = 0u;
        uint32_t iq2_gate_hot_max = 0u;
        const uint32_t iq2_gate_hot_threshold = 8u;
        const uint32_t iq2_down_hot_threshold = 8u;
        uint32_t h_iq2_gate_hot[256] = {0};
        const uint32_t use_iq2_gate_wmma =
            ok && !mmq_gateup_done &&
            iq2_path && n_tokens >= iq2_gate_hot_threshold &&
            n_expert == 6u && !write_gate_up && !grouped_gate_only &&
            sorted_pairs && sorted_offsets && sorted_counts && tile_experts && iq2_gate_hot_dev && use_expert_tiles &&
            (expert_in_dim % 16u) == 0u && (expert_mid_dim % 16u) == 0u;
        if (use_iq2_gate_wmma) {
            uint32_t h_counts[256] = {0};
            if (!hip_ok(hipMemcpy(h_counts, sorted_counts, sizeof(h_counts), hipMemcpyDeviceToHost),
                         "routed_moe iq2 gate wmma counts copy")) {
                ok = 0;
            } else {
                for (uint32_t e = 0; e < 256u; e++) {
                    const uint32_t c = h_counts[e];
                    if (c >= iq2_gate_hot_threshold) {
                        h_iq2_gate_hot[iq2_gate_hot_count++] = e;
                        if (c > iq2_gate_hot_max) iq2_gate_hot_max = c;
                    }
                }
                if (iq2_gate_hot_count != 0u &&
                    !hip_ok(hipMemcpy(iq2_gate_hot_dev, h_iq2_gate_hot,
                                        iq2_gate_hot_count * sizeof(uint32_t), hipMemcpyHostToDevice),
                             "routed_moe iq2 gate hot copy")) {
                    ok = 0;
                }
            }
        }
        const uint32_t iq2_gate_scalar_max = iq2_gate_hot_count != 0u ? iq2_gate_hot_threshold : 0u;
        const int use_iq2_hot_f16_mid =
            mmq_hot_mid_f16 ||
            (use_iq2_gate_wmma && iq2_gate_hot_count != 0u &&
             iq2_gate_hot_threshold == iq2_down_hot_threshold &&
             (out_dim & 1u) == 0u);
        __half *iq2_hot_mid_h = mmq_mid_h
            ? mmq_mid_h
            : (use_iq2_hot_f16_mid ? (__half *)gate->ptr : NULL);
        const int use_iq2_x_f16 = use_iq2_gate_wmma && iq2_gate_hot_count != 0u &&
            up->bytes >= (uint64_t)n_tokens * expert_in_dim * sizeof(__half);
        __half *iq2_x_h = use_iq2_x_f16 ? (__half *)up->ptr : NULL;
        if (ok && use_iq2_x_f16) {
            const uint64_t xh_count = (uint64_t)n_tokens * expert_in_dim;
            hip_launch_f32_to_f16(iq2_x_h, (const float *)x->ptr, xh_count);
            ok = hip_ok(hipGetLastError(), "routed_moe iq2 gate x f16 launch");
        }
        if (ok && !mmq_gateup_done) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, n_tokens * n_expert, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (q4k_path) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    if (expert_tile_m == 8u) {
                        moe_gate_up_mid_q4K_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else {
                        moe_gate_up_mid_q4K_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    }
                } else if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        iq2_gate_scalar_max, write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    if (n_tokens <= 8u || xq_blocks > 16u) {
                        /* At native C2-C8 decode widths, direct xq reads leave
                         * MALL to absorb four-pair reuse and avoid 18,688 bytes
                         * of LDS. A matched C8 profile dropped this kernel from
                         * 5,235.69 to 4,198.79 ms. */
                        moe_gate_up_mid_expert_tile4_row32_kernel<false><<<
                                tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr,
                            (float *)mid->ptr, gate_w, up_w, xq,
                            sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts,
                            (const float *)weights->ptr, gate_expert_bytes,
                            gate_row_bytes, xq_blocks, expert_mid_dim,
                            n_expert, iq2_gate_scalar_max, write_gate_up,
                            clamp);
                    } else {
                        /* Prompt batches have enough pair reuse to repay the
                         * LDS copy; pp128 regressed when direct reads were used
                         * here as well. */
                        moe_gate_up_mid_expert_tile4_row32_kernel<true><<<
                                tgrid, 256,
                                4u * 16u * sizeof(hip_block_q8_K)>>>(
                            (float *)gate->ptr, (float *)up->ptr,
                            (float *)mid->ptr, gate_w, up_w, xq,
                            sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts,
                            (const float *)weights->ptr, gate_expert_bytes,
                            gate_row_bytes, xq_blocks, expert_mid_dim,
                            n_expert, iq2_gate_scalar_max, write_gate_up,
                            clamp);
                    }
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && sorted_pairs) {
                if (q4k_path) {
                    moe_gate_up_mid_q4K_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        sorted_pairs,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                } else {
                    moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        sorted_pairs,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, n_tokens * n_expert, 1);
                if (q4k_path) {
                    moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_lut_gate) {
                  moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                      (float*)gate->ptr, (float*)up->ptr, (float*)mid->ptr,
                      gate_w, up_w, xq, (const int32_t*)selected->ptr,
                      (const float*)weights->ptr, gate_expert_bytes,
                      gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                      write_gate_up, clamp);
                } else {
                  moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                      (float*)gate->ptr, (float*)up->ptr, (float*)mid->ptr,
                      gate_w, up_w, xq, (const int32_t*)selected->ptr,
                      (const float*)weights->ptr, gate_expert_bytes,
                      gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                      clamp);
                }
            }
            ok = hip_ok(hipGetLastError(), "routed_moe gate/up launch");
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
            if (ok && use_iq2_gate_wmma && iq2_gate_hot_count != 0u) {
                constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
                const uint32_t wmma_mtiles = 4u;
                if (wmma_mtiles == 4u) {
                    constexpr uint32_t mt = 4u;
                    const dim3 block(32u * mt, 1u, 1u);
                    const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                    (iq2_gate_hot_max + mt * bm - 1u) / (mt * bm),
                                    iq2_gate_hot_count);
                    const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(__half) +
                                            (4u * mt * bm * bn) * sizeof(float);
                    if (use_iq2_hot_f16_mid && use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_hot_f16_mid) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    }
                } else {
                    constexpr uint32_t mt = 8u;
                    const dim3 block(32u * mt, 1u, 1u);
                    const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                    (iq2_gate_hot_max + mt * bm - 1u) / (mt * bm),
                                    iq2_gate_hot_count);
                    const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(__half) +
                                            (4u * mt * bm * bn) * sizeof(float);
                    if (use_iq2_hot_f16_mid && use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_hot_f16_mid) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    }
                }
                ok = hip_ok(hipGetLastError(), "routed_moe iq2 wmma hot gate/up launch");
            }
#endif
        }
        const uint32_t use_iq2_q2_float_down =
            ok && iq2_path && n_tokens > 1u && n_expert == 6u &&
            !use_direct_down_sum6 && !use_compact_float_down &&
            !grouped_gate_only &&
            sorted_pairs && sorted_offsets && sorted_counts && tile_experts;
        if (ok && !use_iq2_q2_float_down && !use_compact_float_down) {
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
            ok = hip_ok(hipGetLastError(), "routed_moe mid quantize launch");
        }
        if (ok) {
            if (use_compact_float_down) {
                const uint32_t rows_per_block =
                    ds4_rocm_compact_down_rows_per_block();
                const uint32_t pair_count = n_tokens * n_expert;
                dim3 compact_grid(
                    (out_dim + rows_per_block - 1u) / rows_per_block,
                    pair_count,
                    1u);
                moe_down_q2K_pair_float_batch_warp32_kernel
                    <<<compact_grid, rows_per_block * 32u>>>(
                        (__half *)down->ptr,
                        down_w,
                        (const float *)mid->ptr,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        expert_mid_dim,
                        out_dim,
                        pair_count);
                ok = hip_ok(
                    hipGetLastError(),
                    "routed_moe compact float down launch");
                if (ok) {
                    const uint64_t n2 =
                        (uint64_t)n_tokens * (out_dim >> 1u);
                    moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                        (float *)out->ptr,
                        (const __half *)down->ptr,
                        out_dim,
                        n_expert,
                        n_tokens);
                    ok = hip_ok(
                        hipGetLastError(),
                        "routed_moe compact float down sum launch");
                }
            } else if (use_iq2_q2_float_down) {
                ok = routed_moe_q2_float_down_launch(
                        out, down, mid, iq2_hot_mid_h, use_iq2_hot_f16_mid, down_w,
                        sorted_counts, sorted_offsets, sorted_pairs, tile_experts,
                        n_tokens, n_expert, expert_mid_dim, out_dim,
                        down_expert_bytes, down_row_bytes,
                        h_sorted_counts_valid ? h_sorted_counts : NULL,
                        wide_tile_map_dev);
            } else {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens * n_expert, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, n_tokens, 1);
                if (q4k_path) {
                    moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                } else {
                    moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = hip_ok(hipGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum6) {
                /* The direct decode kernel writes the final token row. */
            } else if (!grouped_gate_only && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (q4k_path) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    if (expert_tile_m == 8u) {
                        moe_down_q4K_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        moe_down_q4K_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (!grouped_gate_only && sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (!grouped_gate_only && sorted_pairs) {
                if (q4k_path) {
                    moe_down_q4K_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        sorted_pairs,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        sorted_pairs,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            } else {
                if (q4k_path) {
                    moe_down_q4K_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            }
            ok = hip_ok(hipGetLastError(), "routed_moe down launch");
            }
        }
        if (ok && !use_atomic_down && !use_direct_down_sum6 &&
            !use_compact_float_down && !use_iq2_q2_float_down) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = hip_ok(hipGetLastError(), "routed_moe sum launch");
        }
        return ok;
    }

    if (q2k_path && n_expert == 6u && n_tokens >= 32u) {
        const uint32_t pair_count = n_tokens * n_expert;
        const uint64_t counts_bytes = 256ull * sizeof(uint32_t);
        const uint64_t offsets_bytes = 257ull * sizeof(uint32_t);
        const uint64_t cursors_bytes = 256ull * sizeof(uint32_t);
        const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
        const uint64_t hot_gate_bytes = 256ull * sizeof(uint32_t);
        const uint64_t hot_down_bytes = 256ull * sizeof(uint32_t);
        const uint64_t f16_low_gate_bytes = 256ull * sizeof(uint32_t);
        const uint64_t f16_low_down_bytes = 256ull * sizeof(uint32_t);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        const int moe_wmma_hot = expert_in_dim % 16u == 0u &&
                                 expert_mid_dim % 16u == 0u &&
                                 out_dim % 16u == 0u;
#else
        const int moe_wmma_hot = 0;
#endif
        const uint64_t f16_mid_bytes = moe_wmma_hot ? (uint64_t)pair_count * expert_mid_dim * sizeof(__half) : 0ull;
        const uint64_t f16_down_bytes = moe_wmma_hot ? (uint64_t)pair_count * out_dim * sizeof(__half) : 0ull;
        const uint64_t wmma_x_bytes = moe_wmma_hot ? (uint64_t)n_tokens * expert_in_dim * sizeof(__half) : 0ull;
        auto align256 = [](uint64_t v) -> uint64_t { return (v + 255ull) & ~255ull; };
        const uint64_t base_scratch_end = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes +
                                          hot_gate_bytes + hot_down_bytes +
                                          f16_low_gate_bytes + f16_low_down_bytes;
        const uint64_t f16_mid_off = align256(base_scratch_end);
        const uint64_t f16_down_off = align256(f16_mid_off + f16_mid_bytes);
        const uint64_t wmma_x_off = align256(f16_down_off + f16_down_bytes);
        const uint64_t scratch_bytes = align256(wmma_x_off + wmma_x_bytes);
        uint8_t *scratch = (uint8_t *)hip_tmp_alloc(scratch_bytes, "routed_moe q2 expert batch buckets");
        if (!scratch) return 0;
        uint32_t *counts = (uint32_t *)scratch;
        uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
        uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
        uint32_t *sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
        const uint64_t wmma_list_base = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
        uint32_t *wmma_gate_hot_dev = (uint32_t *)(scratch + wmma_list_base);
        uint32_t *wmma_down_hot_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes);
        uint32_t *wmma_gate_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes);
        uint32_t *wmma_down_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes + f16_low_gate_bytes);
        __half *wmma_mid_h = moe_wmma_hot ? (__half *)(scratch + f16_mid_off) : NULL;
        __half *wmma_down_h = moe_wmma_hot ? (__half *)(scratch + f16_down_off) : NULL;
        __half *wmma_x_h = moe_wmma_hot ? (__half *)(scratch + wmma_x_off) : NULL;
        ok = hip_ok(hipMemset(counts, 0, counts_bytes), "routed_moe q2 expert counts clear");
        if (ok) {
            moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                    counts,
                    (const int32_t *)selected->ptr,
                    pair_count);
            ok = hip_ok(hipGetLastError(), "routed_moe q2 expert count launch");
        }
        if (ok) {
            moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts);
            ok = hip_ok(hipGetLastError(), "routed_moe q2 expert prefix launch");
        }
        if (ok) {
            moe_scatter_sorted_pairs_deterministic_kernel<<<
                    256u, 256u, 256u * sizeof(uint32_t)>>>(
                    sorted_pairs,
                    offsets,
                    (const int32_t *)selected->ptr,
                    pair_count);
            ok = hip_ok(hipGetLastError(), "routed_moe q2 expert scatter launch");
        }
        if (ok && moe_wmma_hot) {
            const uint64_t xh_count = (uint64_t)n_tokens * expert_in_dim;
            hip_launch_f32_to_f16(wmma_x_h, (const float *)x->ptr, xh_count);
            ok = hip_ok(hipGetLastError(), "routed_moe q2 wmma x f16 launch");
        }
        if (!ok) return 0;

        uint32_t wmma_f16_hot_count = 0u, wmma_f16_hot_max = 0u;
        uint32_t wmma_f16_low_count = 0u, wmma_f16_low_max = 0u;
        uint32_t h_counts[256] = {0};
        uint32_t h_f16_hot[256] = {0};
        uint32_t h_f16_low[256] = {0};
        const uint32_t wmma_hot_threshold = 8u;
        const uint32_t wmma_f16_low_threshold = 64u;
        if (moe_wmma_hot) {
            if (!hip_ok(hipMemcpy(h_counts, counts, 256u * sizeof(uint32_t), hipMemcpyDeviceToHost),
                         "routed_moe q2 wmma counts copy")) return 0;
            for (uint32_t e = 0; e < 256u; e++) {
                const uint32_t c = h_counts[e];
                if (c >= wmma_hot_threshold) {
                    if (c < wmma_f16_low_threshold) {
                        h_f16_low[wmma_f16_low_count++] = e;
                        if (c > wmma_f16_low_max) wmma_f16_low_max = c;
                    } else {
                        h_f16_hot[wmma_f16_hot_count++] = e;
                        if (c > wmma_f16_hot_max) wmma_f16_hot_max = c;
                    }
                }
            }
        }
        const uint32_t gate_rpb = 16u;
        const uint32_t down_rpb = 16u;
        const uint32_t gate_threads = gate_rpb * 32u;
        const uint32_t down_threads = down_rpb * 32u;
        const size_t gate_shmem = 4u * 256u * sizeof(float);
        const size_t down_shmem = 4u * 256u * sizeof(float);
        const uint32_t scalar_max = moe_wmma_hot && (wmma_f16_low_count != 0u || wmma_f16_hot_count != 0u)
            ? wmma_hot_threshold : 0u;
        dim3 gate_grid((expert_mid_dim + gate_rpb - 1u) / gate_rpb, 256u, 1);
        moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<4><<<gate_grid, gate_threads, gate_shmem>>>(
                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_in_dim, expert_mid_dim,
                gate_expert_bytes, gate_row_bytes, clamp);
        if (!hip_ok(hipGetLastError(), "routed_moe q2 expert gate/up launch")) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_hot && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 = (mt4 * bm * bk + 4u * bk * bn) * sizeof(__half) +
                                    (4u * mt4 * bm * bn) * sizeof(float);
            if (!hip_ok(hipMemcpy(wmma_gate_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), hipMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low hot copy")) return 0;
            moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                    counts, offsets, sorted_pairs, wmma_gate_f16_low_dev, wmma_f16_low_count,
                    expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
            if (!hip_ok(hipGetLastError(), "routed_moe q2 wmma f16-low gate/up launch")) return 0;
        }
        if (moe_wmma_hot && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(__half) +
                                    (4u * mt * bm * bn) * sizeof(float);
            if (!hip_ok(hipMemcpy(wmma_gate_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), hipMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid hot copy")) return 0;
            moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                    counts, offsets, sorted_pairs, wmma_gate_hot_dev, wmma_f16_hot_count,
                    expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
            if (!hip_ok(hipGetLastError(), "routed_moe q2 wmma f16-mid gate/up launch")) return 0;
        }
#endif
        dim3 down_grid((out_dim + down_rpb - 1u) / down_rpb, 256u, 1);
        if (moe_wmma_hot) {
            moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes);
        } else {
            moe_down_q2K_expert_batch_sharedmid_kernel<4><<<down_grid, down_threads, down_shmem>>>(
                    (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes);
        }
        if (!hip_ok(hipGetLastError(), "routed_moe q2 expert down launch")) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_hot && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 =
                ds4_rocm_q2_down_wmma_shmem(mt4, bm, bn, bk);
            if (!hip_ok(hipMemcpy(wmma_down_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), hipMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low down hot copy")) return 0;
            moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                    counts, offsets, sorted_pairs, wmma_down_f16_low_dev, wmma_f16_low_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
            if (!hip_ok(hipGetLastError(), "routed_moe q2 wmma f16-low down launch")) return 0;
        }
        if (moe_wmma_hot && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 =
                    ds4_rocm_q2_down_wmma_shmem(mt, bm, bn, bk);
            if (!hip_ok(hipMemcpy(wmma_down_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), hipMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid down hot copy")) return 0;
            moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                    counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_f16_hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
            if (!hip_ok(hipGetLastError(), "routed_moe q2 wmma f16-mid down launch")) return 0;
        }
#endif
        const uint64_t n = (uint64_t)n_tokens * out_dim;
        if (moe_wmma_hot) {
            if ((out_dim & 1u) == 0u) {
                const uint64_t n2 = n >> 1u;
                moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            } else {
                moe_sum_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            }
        } else {
            moe_sum_kernel<<<(n + 255u) / 256u, 256>>>(
                    (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        }
        ok = hip_ok(hipGetLastError(), "routed_moe q2 expert sum launch");
        return ok;
    }

    const ds4_rocm_runtime_config *cfg = hip_runtime_config();
    if (q2k_path && n_expert == 6u) {
        uint32_t rows_per_block = cfg->moe_decode_rpb;
        const uint32_t threads = rows_per_block * 32u;
        constexpr int store_gate_up = 0;
        const uint64_t xq_gate_bytes = (uint64_t)n_tokens * xq_blocks * sizeof(hip_block_q8_K);
        const int q8k_gateup = n_tokens == 1u &&
            n_expert == 6u && down->bytes >= xq_gate_bytes;
        int ok_gateup = 1;
        if (q8k_gateup) {
            hip_block_q8_K *xq_gate = (hip_block_q8_K *)down->ptr;
            dim3 xq_grid(xq_blocks, n_tokens, 1);
            q8_K_quantize_kernel<<<xq_grid, 256>>>(xq_gate, (const float *)x->ptr, expert_in_dim, n_tokens);
            ok_gateup = hip_ok(hipGetLastError(), "routed_moe q2 oldhip q8k gate input quantize launch");
            if (ok_gateup) {
                dim3 gate_grid((expert_mid_dim + 255u) / 256u, n_tokens * n_expert, 1);
                moe_gate_up_mid_q2K_decode_q8_qwarp32_kernel<<<gate_grid, 256u>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq_gate,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        (uint32_t)store_gate_up,
                        clamp);
                ok_gateup = hip_ok(hipGetLastError(), "routed_moe q2 oldhip q8k gate/up launch");
            }
        } else if (rows_per_block == 1u) {
            dim3 gate_grid(expert_mid_dim, n_tokens * n_expert, 1);
            moe_gate_up_mid_q2K_rows_rpb1_w32_kernel<<<gate_grid, 32u>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    clamp,
                    store_gate_up);
            ok_gateup = hip_ok(hipGetLastError(), "routed_moe q2 oldhip rows gate/up launch");
        } else {
            dim3 gate_grid((expert_mid_dim + rows_per_block - 1u) / rows_per_block, n_tokens * n_expert, 1);
            moe_gate_up_mid_q2K_rows_w32_kernel<<<gate_grid, threads>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    clamp,
                    store_gate_up);
            ok_gateup = hip_ok(hipGetLastError(), "routed_moe q2 oldhip rows gate/up launch");
        }
        if (!ok_gateup) return 0;
        int ok_decode_moe = 1;
        const uint64_t midq_bytes = (uint64_t)n_tokens * n_expert * midq_blocks * sizeof(hip_block_q8_K);
        const int q8k_down = n_tokens == 1u &&
            n_expert == 6u && down->bytes >= midq_bytes;
        if (q8k_down) {
            hip_block_q8_K *midq = (hip_block_q8_K *)down->ptr;
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
            ok_decode_moe = hip_ok(hipGetLastError(), "routed_moe q2 oldhip q8k mid quantize launch");
            if (ok_decode_moe) {
                moe_down_sum6_qwarp32_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                ok_decode_moe = hip_ok(hipGetLastError(), "routed_moe q2 oldhip q8k down launch");
            }
        } else {
            dim3 down_grid((out_dim + rows_per_block - 1u) / rows_per_block, n_tokens, 1);
            moe_down_q2K_sum_rows_w32_kernel<<<down_grid, threads>>>(
                    (float *)out->ptr,
                    down_w,
                    (const float *)mid->ptr,
                    (const int32_t *)selected->ptr,
                    n_tokens,
                    expert_mid_dim,
                    out_dim,
                    down_expert_bytes,
                    down_row_bytes);
            ok_decode_moe = hip_ok(hipGetLastError(), "routed_moe q2 oldhip rows down launch");
        }
        return ok_decode_moe;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
        if (q2k_path) {
            moe_gate_up_mid_q2K_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        } else {
            moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        }
        ok = hip_ok(hipGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, n_tokens * n_expert, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            (const int32_t *)selected->ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = hip_ok(hipGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = hip_ok(hipGetLastError(), "routed_moe sum launch");
    }
    return ok;
}

extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, const ds4_gpu_tensor *add_in, uint32_t layer_index, bool force_resident) {
    (void)add_in;
    (void)layer_index;
    (void)force_resident;
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, 1);
}

extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16, bool force_resident) {
    (void)layer_index;
    (void)force_resident;
    if (mid_is_f16) *mid_is_f16 = false;
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, n_tokens);
}
