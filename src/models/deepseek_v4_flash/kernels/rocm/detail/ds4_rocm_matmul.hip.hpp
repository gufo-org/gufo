#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <type_traits>
#include <rocwmma/rocwmma.hpp>

/* Blocked F16 GEMM for the attention output B projection.
 *
 * The engine sends this shape (`m=4096 n=chunk k=8192`, `opA=N`, F16 operands
 * and an F32 result) to hipBLASLt, which picks a `MT96x96x32` Tensile kernel.
 * That tile re-reads each panel `m/96` and `n/96` times, about 5.7 GB for a
 * 4,096-token chunk, and a kernel trace prices the call at 25.58 ms, which is
 * roughly DRAM peak for that traffic. A 256x128 macro tile moves 3.22 GB
 * instead. `rocm/tools/f16_gemm_bench.cpp` scored every geometry with three
 * rotating operand copies so that the 32 MB MALL could not hide the re-reads:
 * 15.82 ms for the production candidate against 11.80 ms here.
 *
 * This is the only projection shape worth taking off hipBLASLt. Tensile wins
 * every `opA=T` shape in the same harness, by 1.4x to 2.6x.
 *
 * Layout: A is the transposed weight cache `Wt[k][m]` row-major, so the LDS
 * tile is `[k][m]` with a `col_major` fragment and both the global read and the
 * LDS write stay contiguous in `m`. B is `low[n][k]` row-major, staged `[n][k]`
 * with a `col_major` matrix_b. C is `out[n][m]` row-major, which is column-major
 * `[m][n]` with leading dimension `m`.
 *
 * Accumulators live in registers for the whole K loop, so each output element
 * is a single ascending walk over K: deterministic, though not hipBLASLt's
 * order, so this needs a likelihood gate rather than an exact logit hash. */
template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t WMF, uint32_t WNF,
          uint32_t SWZ, bool A_ROWMAJOR, typename OutT>
__global__ __launch_bounds__((BM / 16u / WMF) * (BN / 16u / WNF) * 32u) void
ds4_gemm_f16_wmma_kernel(OutT *__restrict__ out,
                         const __half *__restrict__ weight,
                         const __half *__restrict__ act,
                         uint32_t m,
                         uint32_t n,
                         uint32_t k,
                         uint32_t ldc,
                         uint64_t stride_a,
                         uint64_t stride_b,
                         uint64_t stride_c) {
    /* `blockIdx.z` walks independent problems for the strided-batched form; a
     * plain GEMM passes a z extent of one and zero strides. `ldc` lets the
     * result land in a wider matrix, which the attention output A GEMM needs
     * because each group writes its own `rank` block of a `low_dim` row. */
    const __half *__restrict__ weight_t = weight + (uint64_t)blockIdx.z * stride_a;
    const __half *__restrict__ low_h = act + (uint64_t)blockIdx.z * stride_b;
    out += (uint64_t)blockIdx.z * stride_c;
    constexpr uint32_t WAVES_M = BM / 16u / WMF;
    constexpr uint32_t WAVES_N = BN / 16u / WNF;
    constexpr uint32_t THREADS = WAVES_M * WAVES_N * 32u;
    /* A's LDS tile follows the source's contiguous axis: [m][k] for a row-major
     * W[m][k] (opA=T) and [k][m] for a row-major Wt[k][m] (opA=N), so both the
     * global read and the LDS write stay contiguous either way. */
    constexpr uint32_t A_PITCH = A_ROWMAJOR ? BK : BM;
    constexpr uint32_t A_LINES = A_ROWMAJOR ? BM : BK;
    __shared__ __half shA[A_LINES * A_PITCH];
    __shared__ __half shB[BN * BK];
    /* Narrowing epilogue scratch, one 16x16 F32 tile per wave. Only the F16
     * result path needs it; the F32 path stores straight to global. */
    __shared__ float shEpi[std::is_same<OutT, __half>::value
                                   ? WAVES_M * WAVES_N * 256u
                                   : 1u];

    uint32_t bm, bn;
    if (SWZ <= 1u) {
        bm = blockIdx.x;
        bn = blockIdx.y;
    } else {
        const uint32_t tiles_m = gridDim.x;
        const uint32_t group = SWZ * tiles_m;
        const uint32_t linear = blockIdx.y * tiles_m + blockIdx.x;
        const uint32_t gid = linear / group;
        const uint32_t within = linear - gid * group;
        const uint32_t cols = min(SWZ, gridDim.y - gid * SWZ);
        bm = within / cols;
        bn = gid * SWZ + (within - bm * cols);
    }
    const uint32_t m0 = bm * BM;
    const uint32_t n0 = bn * BN;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t wm = wave % WAVES_M;
    const uint32_t wn = wave / WAVES_M;

    using frag_a = rocwmma::fragment<
            rocwmma::matrix_a, 16, 16, 16, __half,
            typename std::conditional<A_ROWMAJOR, rocwmma::row_major,
                                      rocwmma::col_major>::type>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half,
                                     rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    frag_c acc[WMF][WNF];
    for (uint32_t i = 0; i < WMF; i++)
        for (uint32_t j = 0; j < WNF; j++) rocwmma::fill_fragment(acc[i][j], 0.0f);

    constexpr uint32_t A_GROUPS = BM * BK / 8u;
    constexpr uint32_t B_GROUPS = BN * BK / 8u;
    constexpr uint32_t A_RUN = A_PITCH / 8u;
    constexpr uint32_t B_RUN = BK / 8u;

    for (uint32_t k0 = 0; k0 < k; k0 += BK) {
        for (uint32_t g = tid; g < A_GROUPS; g += THREADS) {
            const uint32_t run = g % A_RUN;
            const uint32_t line = g / A_RUN;
            const uint64_t src = A_ROWMAJOR
                    ? (uint64_t)(m0 + line) * k + k0 + run * 8u
                    : (uint64_t)(k0 + line) * m + m0 + run * 8u;
            *reinterpret_cast<uint4 *>(&shA[line * A_PITCH + run * 8u]) =
                    *reinterpret_cast<const uint4 *>(&weight_t[src]);
        }
        for (uint32_t g = tid; g < B_GROUPS; g += THREADS) {
            const uint32_t run = g % B_RUN;
            const uint32_t line = g / B_RUN;
            *reinterpret_cast<uint4 *>(&shB[line * BK + run * 8u]) =
                    *reinterpret_cast<const uint4 *>(
                            &low_h[(uint64_t)(n0 + line) * k + k0 + run * 8u]);
        }
        __syncthreads();
        for (uint32_t kk = 0; kk < BK; kk += 16u) {
            frag_a fa[WMF];
            frag_b fb[WNF];
            for (uint32_t i = 0; i < WMF; i++) {
                const uint32_t mm = (wm * WMF + i) * 16u;
                if (A_ROWMAJOR) {
                    rocwmma::load_matrix_sync(fa[i], &shA[mm * A_PITCH + kk],
                                              A_PITCH);
                } else {
                    rocwmma::load_matrix_sync(fa[i], &shA[kk * A_PITCH + mm],
                                              A_PITCH);
                }
            }
            for (uint32_t j = 0; j < WNF; j++) {
                rocwmma::load_matrix_sync(
                        fb[j], &shB[(wn * WNF + j) * 16u * BK + kk], BK);
            }
            for (uint32_t i = 0; i < WMF; i++)
                for (uint32_t j = 0; j < WNF; j++)
                    rocwmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }

    if (std::is_same<OutT, float>::value) {
        for (uint32_t i = 0; i < WMF; i++) {
            for (uint32_t j = 0; j < WNF; j++) {
                const uint32_t mo = m0 + (wm * WMF + i) * 16u;
                const uint32_t no = n0 + (wn * WNF + j) * 16u;
                rocwmma::store_matrix_sync(
                        (float *)out + mo + (uint64_t)no * ldc, acc[i][j], ldc,
                        rocwmma::mem_col_major);
            }
        }
        return;
    }
    /* An F16 result needs a narrowing epilogue that rocWMMA's F32 accumulator
     * cannot store directly, so page each wave's tiles back through LDS. */
    float *tile = shEpi + (uint64_t)wave * 256u;
    const uint32_t lane = tid & 31u;
    for (uint32_t i = 0; i < WMF; i++) {
        for (uint32_t j = 0; j < WNF; j++) {
            rocwmma::store_matrix_sync(tile, acc[i][j], 16u,
                                       rocwmma::mem_col_major);
            const uint32_t mo = m0 + (wm * WMF + i) * 16u;
            const uint32_t no = n0 + (wn * WNF + j) * 16u;
            for (uint32_t e = lane; e < 256u; e += 32u) {
                const uint32_t c = e >> 4u;
                const uint32_t r = e & 15u;
                ((__half *)out)[(uint64_t)(no + c) * ldc + mo + r] =
                        __float2half(tile[c * 16u + r]);
            }
        }
    }
}

/* 256x128 with four m-fragments and two n-fragments per wave, 512 threads, and
 * a two-column swizzle scored best across every shape in
 * `rocm/tools/f16_gemm_bench.cpp`. Bigger macro tiles spill: 256x256 needs 128
 * accumulator VGPRs and measured 1.8x slower. */
#define DS4_WMMA_GEMM_BM 256u
#define DS4_WMMA_GEMM_BN 128u
#define DS4_WMMA_GEMM_BK 32u

/* A 256x128 tile only pays when the grid can fill the 40 CUs several times
 * over. Routing every `opA=T` projection through it cost +456 ms per two-chunk
 * trace, because `m=256` and `m=512` leave 32 and 64 workgroups for 40 CUs while
 * Tensile's MT96x96 and MT32x32 tiles keep hundreds in flight. Six workgroups
 * per CU is the floor that kept only the shapes this kernel wins. */
#define DS4_WMMA_GEMM_MIN_BLOCKS 240u

static int ds4_gemm_f16_wmma_eligible(uint64_t m, uint64_t n, uint64_t k) {
    /* Static LDS is BM*BK + BN*BK halves = 24 KiB, so two workgroups per CU. */
    if (!g_rocm_gfx1151 || m % DS4_WMMA_GEMM_BM != 0u ||
        n % DS4_WMMA_GEMM_BN != 0u || k % DS4_WMMA_GEMM_BK != 0u ||
        m > UINT32_MAX || n > UINT32_MAX || k > UINT32_MAX) {
        return 0;
    }
    return (m / DS4_WMMA_GEMM_BM) * (n / DS4_WMMA_GEMM_BN) >=
           DS4_WMMA_GEMM_MIN_BLOCKS;
}

template <bool A_ROWMAJOR, typename OutT>
static int ds4_gemm_f16_wmma_launch(OutT *out,
                                    const __half *weight,
                                    const __half *act,
                                    uint64_t m,
                                    uint64_t n,
                                    uint64_t k,
                                    const char *what) {
    const dim3 grid((uint32_t)(m / DS4_WMMA_GEMM_BM),
                    (uint32_t)(n / DS4_WMMA_GEMM_BN),
                    1u);
    ds4_gemm_f16_wmma_kernel<DS4_WMMA_GEMM_BM, DS4_WMMA_GEMM_BN,
                             DS4_WMMA_GEMM_BK, 4u, 2u, 2u, A_ROWMAJOR, OutT>
            <<<grid, 512u>>>(out, weight, act, (uint32_t)m, (uint32_t)n,
                             (uint32_t)k, (uint32_t)m, 0, 0, 0);
    return hip_ok(hipGetLastError(), what);
}

/* Narrow-n companion tile, and the padding rule that lets an arbitrary chunk
 * width reach either of them.
 *
 * `DS4_WMMA_GEMM_MIN_BLOCKS` was calibrated to keep this kernel off the small-m
 * `opA=T` projections, where 32 to 64 workgroups lose to Tensile. It also
 * excluded the one shape the kernel was written for whenever the prompt chunk
 * was under 2,048 rows -- which is every conversational turn -- and there the
 * fallback is not Tensile's good `opA=T` tile but rocBLAS on `opA=N`.
 * `tools/bench/dsv4_attn_out_gemm_bench.hip` at `m=4096 k=8192`:
 *
 *   n      rocBLAS  256x128 W4x2  128x128 W2x2 SWZ8
 *    128  1.533 ms      0.934 ms           0.653 ms
 *    256  2.879 ms      1.018 ms           0.949 ms
 *    512  5.202 ms      2.094 ms           1.887 ms
 *   1024 10.521 ms      3.505 ms           4.042 ms
 *
 * rocBLAS holds 5.6-6.6 TFLOP/s at every width while both tiles reach 16-18, so
 * the block floor was giving away a factor of three; 128x128 wins below n=1024
 * and 256x128 above it. Every WMMA geometry in that harness agrees bit for bit:
 * the macro tile only partitions the output, and each element still walks K once
 * in ascending order, so this is a scheduling choice with no numerical
 * consequence. Only leaving rocBLAS is a numerical change, and at these widths
 * the two disagree by 3e-4.
 *
 * A width that is not a multiple of the tile is rounded up. The caller owns
 * both the activation staging buffer and the result, so it sizes them for the
 * padded width and zeroes the activation pad; the extra result columns are
 * computed from zeros and never read. */
#define DS4_WMMA_GEMM_NARROW_BM 128u
#define DS4_WMMA_GEMM_NARROW_BN 128u
#define DS4_WMMA_GEMM_WIDE_MIN_N 1024u

/* Column granularity both tiles share, hence the padding the caller must hold. */
static inline uint64_t ds4_gemm_f16_wmma_pad_n(uint64_t n) {
    const uint64_t bn = DS4_WMMA_GEMM_NARROW_BN;
    return (n + bn - 1u) / bn * bn;
}

/* Whether either tile can serve `m x n_pad x k`, with `n_pad` already rounded. */
static int ds4_gemm_f16_wmma_padded_eligible(uint64_t m, uint64_t n_pad,
                                             uint64_t k) {
    if (!g_rocm_gfx1151 || n_pad % DS4_WMMA_GEMM_NARROW_BN != 0u ||
        k % DS4_WMMA_GEMM_BK != 0u || m > UINT32_MAX || n_pad > UINT32_MAX ||
        k > UINT32_MAX) {
        return 0;
    }
    return m % DS4_WMMA_GEMM_NARROW_BM == 0u;
}

template <bool A_ROWMAJOR, typename OutT>
static int ds4_gemm_f16_wmma_padded_launch(OutT *out,
                                           const __half *weight,
                                           const __half *act,
                                           uint64_t m,
                                           uint64_t n_pad,
                                           uint64_t k,
                                           const char *what) {
    if (n_pad >= DS4_WMMA_GEMM_WIDE_MIN_N && m % DS4_WMMA_GEMM_BM == 0u &&
        n_pad % DS4_WMMA_GEMM_BN == 0u) {
        return ds4_gemm_f16_wmma_launch<A_ROWMAJOR, OutT>(out, weight, act, m,
                                                          n_pad, k, what);
    }
    constexpr uint32_t kNarrowWmf = 2u;
    constexpr uint32_t kNarrowWnf = 2u;
    constexpr uint32_t kNarrowThreads =
            (DS4_WMMA_GEMM_NARROW_BM / 16u / kNarrowWmf) *
            (DS4_WMMA_GEMM_NARROW_BN / 16u / kNarrowWnf) * 32u;
    const dim3 grid((uint32_t)(m / DS4_WMMA_GEMM_NARROW_BM),
                    (uint32_t)(n_pad / DS4_WMMA_GEMM_NARROW_BN),
                    1u);
    ds4_gemm_f16_wmma_kernel<DS4_WMMA_GEMM_NARROW_BM, DS4_WMMA_GEMM_NARROW_BN,
                             DS4_WMMA_GEMM_BK, kNarrowWmf, kNarrowWnf, 8u,
                             A_ROWMAJOR, OutT>
            <<<grid, kNarrowThreads>>>(out, weight, act, (uint32_t)m,
                                       (uint32_t)n_pad, (uint32_t)k,
                                       (uint32_t)m, 0, 0, 0);
    return hip_ok(hipGetLastError(), what);
}

/* Strided-batched form for the attention output A GEMM.
 *
 * `hipblasGemmStridedBatchedEx` sends `rank x n_tokens x group_dim` once per
 * head group to rocBLAS, which picks `Cijk_Alik_Bljk_HHS_MT128x128x32` and
 * reaches 2.7 TFLOP/s: 12.88 ms per call for 34.4 GFLOP. Each group's A panel
 * is only `rank * group_dim * 2` bytes, so it stays in the MALL across the whole
 * n sweep and the call should be bound by B, which is read once.
 *
 * A 128-row macro tile matches `rank` exactly, so `m` needs no padding, and the
 * output leading dimension is `low_dim` because group `g` owns the `rank` block
 * at offset `g * rank` of every token's row.
 *
 * `m` must equal the tile: relaxing it to any multiple, which lets `rank=1024`
 * through, measured 855.99 ms per 4,096-token chunk against rocBLAS's 563.72 ms
 * for the same call. Each group's A panel then no longer stays MALL-resident
 * across the n sweep, which is the whole premise of the tile. */
#define DS4_WMMA_BATCH_BM 128u
#define DS4_WMMA_BATCH_BN 128u

static int ds4_gemm_f16_wmma_batched_eligible(uint64_t m,
                                              uint64_t n,
                                              uint64_t k,
                                              uint64_t batches) {
    return g_rocm_gfx1151 && m == DS4_WMMA_BATCH_BM &&
           n % DS4_WMMA_BATCH_BN == 0u && k % DS4_WMMA_GEMM_BK == 0u &&
           batches > 0u && batches <= UINT16_MAX && m <= UINT32_MAX &&
           n <= UINT32_MAX && k <= UINT32_MAX &&
           (n / DS4_WMMA_BATCH_BN) * batches >= DS4_WMMA_GEMM_MIN_BLOCKS;
}

static int ds4_gemm_f16_wmma_batched_launch(__half *out,
                                            const __half *weight,
                                            const __half *act,
                                            uint64_t m,
                                            uint64_t n,
                                            uint64_t k,
                                            uint64_t ldc,
                                            uint64_t batches,
                                            const char *what) {
    const dim3 grid((uint32_t)(m / DS4_WMMA_BATCH_BM),
                    (uint32_t)(n / DS4_WMMA_BATCH_BN),
                    (uint32_t)batches);
    ds4_gemm_f16_wmma_kernel<DS4_WMMA_BATCH_BM, DS4_WMMA_BATCH_BN,
                             DS4_WMMA_GEMM_BK, 2u, 2u, 1u, true, __half>
            <<<grid, 512u>>>(out, weight, act, (uint32_t)m, (uint32_t)n,
                             (uint32_t)k, (uint32_t)ldc, m * k, n * k, m);
    return hip_ok(hipGetLastError(), what);
}
#endif
__global__ static void matmul_f16_tiny_batch_wave_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_tok) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (row >= out_dim || tok >= n_tok) return;

    const __half *wr = w + (uint64_t)row * in_dim;
    const float *xr = x + (uint64_t)tok * in_dim;
    float sum = 0.0f;
    for (uint32_t i = lane; i < in_dim; i += 32u) {
        const float xv = __half2float(__float2half(xr[i]));
        sum += __half2float(wr[i]) * xv;
    }
    sum = warp_sum_f32(sum);
    if (lane == 0u) {
        out[(uint64_t)tok * out_dim + row] = sum;
    }
}

template <uint32_t BT>
static void hip_launch_q8_batch_sharedx_bt(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes,
        dim3 grid,
        uint32_t rows_per_block,
        uint32_t tile) {
    const size_t shmem = (size_t)tile * BT * 32u * sizeof(float);
    if (tile == 2u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<2u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 4u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<4u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 8u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<8u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 16u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<16u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<32u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    }
}

static void hip_launch_q8_batch_sharedx(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes,
        uint32_t rows_per_block,
        uint32_t tile,
        uint32_t block_tile) {
    const dim3 grid((out_dim + rows_per_block - 1u) / rows_per_block,
                    (n_tok + tile - 1u) / tile,
                    1u);
    if (block_tile == 8u) {
        hip_launch_q8_batch_sharedx_bt<8u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    } else if (block_tile == 32u) {
        hip_launch_q8_batch_sharedx_bt<32u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    } else {
        hip_launch_q8_batch_sharedx_bt<16u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    }
}

template <uint32_t BT>
static void hip_launch_grouped_q8_a_sharedx_bt(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint64_t row_bytes,
        dim3 grid,
        uint32_t rows_per_block,
        uint32_t tile) {
    const size_t shmem = (size_t)tile * BT * 32u * sizeof(float);
    if (tile == 2u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<2u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 4u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<4u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 8u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<8u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 16u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<16u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<32u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    }
}

static void hip_launch_grouped_q8_a_sharedx(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint64_t row_bytes,
        uint32_t rows_per_block,
        uint32_t tile,
        uint32_t block_tile) {
    const uint32_t row_blocks = (rank + rows_per_block - 1u) / rows_per_block;
    const dim3 grid(n_groups * row_blocks,
                    (n_tokens + tile - 1u) / tile,
                    1u);
    if (block_tile == 8u) {
        hip_launch_grouped_q8_a_sharedx_bt<8u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    } else if (block_tile == 32u) {
        hip_launch_grouped_q8_a_sharedx_bt<32u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    } else {
        hip_launch_grouped_q8_a_sharedx_bt<16u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    }
}

template <uint32_t BT>
static void hip_launch_grouped_q8_a_sharedx_strided_bt(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint32_t x_token_stride,
        uint32_t x_group_stride,
        uint64_t row_bytes,
        dim3 grid,
        uint32_t rows_per_block,
        uint32_t tile) {
    const size_t shmem = (size_t)tile * BT * 32u * sizeof(float);
    if (tile == 2u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<2u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else if (tile == 4u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<4u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else if (tile == 8u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<8u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else if (tile == 16u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<16u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<32u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    }
}

static void hip_launch_grouped_q8_a_sharedx_strided(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint32_t x_token_stride,
        uint32_t x_group_stride,
        uint64_t row_bytes,
        uint32_t rows_per_block,
        uint32_t tile,
        uint32_t block_tile) {
    const uint32_t row_blocks =
        (rank + rows_per_block - 1u) / rows_per_block;
    const dim3 grid(n_groups * row_blocks,
                    (n_tokens + tile - 1u) / tile,
                    1u);
    if (block_tile == 8u) {
        hip_launch_grouped_q8_a_sharedx_strided_bt<8u>(
            low, w, heads, n_tokens, n_groups, n_blocks, rank,
            x_token_stride, x_group_stride, row_bytes, grid,
            rows_per_block, tile);
    } else if (block_tile == 32u) {
        hip_launch_grouped_q8_a_sharedx_strided_bt<32u>(
            low, w, heads, n_tokens, n_groups, n_blocks, rank,
            x_token_stride, x_group_stride, row_bytes, grid,
            rows_per_block, tile);
    } else {
        hip_launch_grouped_q8_a_sharedx_strided_bt<16u>(
            low, w, heads, n_tokens, n_groups, n_blocks, rank,
            x_token_stride, x_group_stride, row_bytes, grid,
            rows_per_block, tile);
    }
}


static int hip_launch_q8_batch_reuse(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint32_t n_tok,
        uint32_t rows_per_block) {
    const unsigned grid =
        (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block);
    const unsigned threads = rows_per_block * 32u;
    if (n_tok == 2u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<2, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 3u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<3, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 4u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<4, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 5u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<5, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 6u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<6, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 7u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<7, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 8u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<8, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 10u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<10, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 12u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<12, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 16u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<16, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 18u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<18, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 24u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<24, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else if (n_tok == 32u) {
        matmul_q8_0_preq_batch_reuse_w32_kernel<32, true><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    } else {
        matmul_q8_0_preq_batch_reuse_w32_kernel<16, false><<<grid, threads>>>(
                out, w, xq, xscale, in_dim, out_dim, blocks, n_tok,
                rows_per_block);
    }
    return hip_ok(hipGetLastError(), "matmul_q8_0 batch reuse launch");
}

static int hip_launch_q8_group_pairs(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        const uint32_t *group_offsets,
        uint32_t group_count,
        uint32_t rows_per_block) {
    if (!group_offsets || group_count < 2u || group_count > 8u ||
        (group_count & 1u) != 0u) {
        return 0;
    }
    const unsigned grid =
        (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block);
    const unsigned threads = rows_per_block * 32u;
    const uint32_t equal_rows = group_offsets[1u] - group_offsets[0u];
    bool all_equal = equal_rows != 0u && equal_rows <= 6u;
    for (uint32_t group = 1u; all_equal && group < group_count; ++group) {
        all_equal =
            group_offsets[group + 1u] - group_offsets[group] == equal_rows;
    }
    if (all_equal) {
        const uint32_t total_rows = group_offsets[group_count];
#define DS4_LAUNCH_Q8_ALL_ROWS(NT)                                      \
        matmul_q8_0_preq_all_rows_exact_w32_kernel<NT>                  \
            <<<grid, threads>>>(out, w, xq, xscale, in_dim, out_dim,    \
                                blocks, rows_per_block)
        switch (total_rows) {
            case 6u:  DS4_LAUNCH_Q8_ALL_ROWS(6u);  break;
            case 8u:  DS4_LAUNCH_Q8_ALL_ROWS(8u);  break;
            case 12u: DS4_LAUNCH_Q8_ALL_ROWS(12u); break;
            case 16u: DS4_LAUNCH_Q8_ALL_ROWS(16u); break;
            case 18u: DS4_LAUNCH_Q8_ALL_ROWS(18u); break;
            case 24u: DS4_LAUNCH_Q8_ALL_ROWS(24u); break;
            case 32u: DS4_LAUNCH_Q8_ALL_ROWS(32u); break;
            default: break;
        }
#undef DS4_LAUNCH_Q8_ALL_ROWS
        if (total_rows == 6u || total_rows == 8u || total_rows == 12u ||
            total_rows == 16u || total_rows == 18u || total_rows == 24u ||
            total_rows == 32u) {
            return hip_ok(hipGetLastError(),
                          "matmul_q8_0 all-row exact launch");
        }

        const dim3 pair_grid(grid, group_count / 2u, 1u);
#define DS4_LAUNCH_Q8_EQUAL_GROUP_PAIRS(NT)                                \
        matmul_q8_0_preq_equal_group_pairs_w32_kernel<NT>                  \
            <<<pair_grid, threads>>>(out, w, xq, xscale, in_dim, out_dim,  \
                                     blocks, rows_per_block)
        switch (equal_rows) {
            case 1u: DS4_LAUNCH_Q8_EQUAL_GROUP_PAIRS(1u); break;
            case 2u: DS4_LAUNCH_Q8_EQUAL_GROUP_PAIRS(2u); break;
            case 3u: DS4_LAUNCH_Q8_EQUAL_GROUP_PAIRS(3u); break;
            case 4u: DS4_LAUNCH_Q8_EQUAL_GROUP_PAIRS(4u); break;
            case 5u: DS4_LAUNCH_Q8_EQUAL_GROUP_PAIRS(5u); break;
            case 6u: DS4_LAUNCH_Q8_EQUAL_GROUP_PAIRS(6u); break;
            default: return 0;
        }
#undef DS4_LAUNCH_Q8_EQUAL_GROUP_PAIRS
        return hip_ok(hipGetLastError(),
                      "matmul_q8_0 equal grouped-pairs launch");
    }

    for (uint32_t group = 0; group < group_count; group += 2u) {
        const uint32_t rows0 =
            group_offsets[group + 1u] - group_offsets[group];
        const uint32_t rows1 =
            group_offsets[group + 2u] - group_offsets[group + 1u];
        if (rows0 == 0u || rows0 > 6u ||
            rows1 == 0u || rows1 > 6u) {
            return 0;
        }
#define DS4_LAUNCH_Q8_GROUP_PAIR(NT)                                    \
        matmul_q8_0_preq_group_pair_w32_kernel<NT><<<grid, threads>>>(  \
            out, w, xq, xscale, in_dim, out_dim, blocks,                \
            group_offsets[group], group_offsets[group + 1u],            \
            rows_per_block)
        if (rows0 == rows1) {
            switch (rows0) {
                case 1u: DS4_LAUNCH_Q8_GROUP_PAIR(1u); break;
                case 2u: DS4_LAUNCH_Q8_GROUP_PAIR(2u); break;
                case 3u: DS4_LAUNCH_Q8_GROUP_PAIR(3u); break;
                case 4u: DS4_LAUNCH_Q8_GROUP_PAIR(4u); break;
                case 5u: DS4_LAUNCH_Q8_GROUP_PAIR(5u); break;
                case 6u: DS4_LAUNCH_Q8_GROUP_PAIR(6u); break;
                default: return 0;
            }
#undef DS4_LAUNCH_Q8_GROUP_PAIR
        } else {
#define DS4_LAUNCH_Q8_RAGGED_GROUP_PAIR(NT0, NT1)                       \
        matmul_q8_0_preq_ragged_group_pair_w32_kernel<NT0, NT1>        \
            <<<grid, threads>>>(                                        \
                out, w, xq, xscale, in_dim, out_dim, blocks,           \
                group_offsets[group], group_offsets[group + 1u],       \
                rows_per_block)
#define DS4_DISPATCH_Q8_RAGGED_SECOND(NT0)                              \
        switch (rows1) {                                                \
            case 1u: DS4_LAUNCH_Q8_RAGGED_GROUP_PAIR(NT0, 1u); break; \
            case 2u: DS4_LAUNCH_Q8_RAGGED_GROUP_PAIR(NT0, 2u); break; \
            case 3u: DS4_LAUNCH_Q8_RAGGED_GROUP_PAIR(NT0, 3u); break; \
            case 4u: DS4_LAUNCH_Q8_RAGGED_GROUP_PAIR(NT0, 4u); break; \
            case 5u: DS4_LAUNCH_Q8_RAGGED_GROUP_PAIR(NT0, 5u); break; \
            case 6u: DS4_LAUNCH_Q8_RAGGED_GROUP_PAIR(NT0, 6u); break; \
            default: return 0;                                          \
        }
            switch (rows0) {
                case 1u: DS4_DISPATCH_Q8_RAGGED_SECOND(1u); break;
                case 2u: DS4_DISPATCH_Q8_RAGGED_SECOND(2u); break;
                case 3u: DS4_DISPATCH_Q8_RAGGED_SECOND(3u); break;
                case 4u: DS4_DISPATCH_Q8_RAGGED_SECOND(4u); break;
                case 5u: DS4_DISPATCH_Q8_RAGGED_SECOND(5u); break;
                case 6u: DS4_DISPATCH_Q8_RAGGED_SECOND(6u); break;
                default: return 0;
            }
#undef DS4_DISPATCH_Q8_RAGGED_SECOND
#undef DS4_LAUNCH_Q8_RAGGED_GROUP_PAIR
        }
        if (!hip_ok(hipGetLastError(),
                    "matmul_q8_0 grouped-pair launch")) {
            return 0;
        }
    }
    return 1;
}

/*
 * Row count below which dense projections take the per-row decode kernels
 * instead of a batched GEMM.
 *
 * Prompt chunks are hundreds to thousands of rows and amortize hipBLAS macro
 * tiles; a DSpark verification block is a handful of rows and does not. Zero
 * disables the small-batch route entirely, which is how an A/B run isolates it.
 */
static uint32_t ds4_rocm_dense_small_batch_rows(void) {
    return ds4_rocm_small_batch_limit(24u);
}

static int hip_matmul_q8_0_tensor_f16_gemm(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        const char *label) {
    if (!g_hipblas_ready || !out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !hip_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !hip_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !hip_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const __half *w_f16 = hip_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w_f16) return 0;
    const uint64_t xh_count = n_tok * in_dim;
    /* See hip_f16_input_publish: several projections in a layer share these rows. */
    __half *xh = hip_f16_input_lookup((const float *)x->ptr, xh_count);
    if (!xh) {
        xh = (__half *)hip_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
        if (!xh) return 0;
        hip_launch_f32_to_f16(xh, (const float *)x->ptr, xh_count);
        if (!hip_ok(hipGetLastError(), "q8 f16 activation convert launch")) return 0;
    }
#ifdef __HIP_PLATFORM_AMD__
    if (n_tok >= DS4_ROCM_WIDE_PREFILL_ROWS &&
        hipblaslt_route_enabled(DS4_ROCM_LT_ROUTE_Q8_F32) &&
        hipblaslt_gemm_f16(out->ptr, w_f16, xh, (uint32_t)out_dim,
                           (uint32_t)n_tok, (uint32_t)in_dim, HIPBLAS_OP_T,
                           HIP_R_32F, label ? label : "q8 f16 projection")) {
        return 1;
    }
#endif
    const float alpha = 1.0f;
    const float beta = 0.0f;
    hipblasStatus_t st = hipblasGemmEx(g_hipblas,
                                     HIPBLAS_OP_T,
                                     HIPBLAS_OP_N,
                                     (int)out_dim,
                                     (int)n_tok,
                                     (int)in_dim,
                                     &alpha,
                                     w_f16,
                                     HIPBLAS_R_16F,
                                     (int)in_dim,
                                     xh,
                                     HIPBLAS_R_16F,
                                     (int)in_dim,
                                     &beta,
                                     out->ptr,
                                     HIPBLAS_R_32F,
                                     (int)out_dim,
                                     HIPBLAS_COMPUTE_32F,
                                     HIPBLAS_GEMM_DEFAULT);
    if (st == HIPBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16 matmul failed: status %d\n", (int)st);
    hip_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16 matmul failure",
                                            in_dim * out_dim * sizeof(__half));
    return 0;
}

static int hip_matmul_q8_0_tensor_f16_gemm_out_half(
        ds4_gpu_tensor *out_h,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        const char *label) {
    if (!g_hipblas_ready || !out_h || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !hip_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !hip_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !hip_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(n_tok, out_dim, sizeof(__half), &out_bytes) ||
        x->bytes < x_bytes || out_h->bytes < out_bytes) return 0;
    const __half *w_f16 = hip_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w_f16) return 0;
    const uint64_t xh_count = n_tok * in_dim;
    __half *xh = hip_f16_input_lookup((const float *)x->ptr, xh_count);
    if (!xh) {
        xh = (__half *)hip_tmp_alloc(xh_count * sizeof(__half), "q8 f16-out gemm activations");
        if (!xh) return 0;
        hip_launch_f32_to_f16(xh, (const float *)x->ptr, xh_count);
        if (!hip_ok(hipGetLastError(), "q8 f16-out activation convert launch")) return 0;
    }
#ifdef __HIP_PLATFORM_AMD__
    if (n_tok >= DS4_ROCM_WIDE_PREFILL_ROWS &&
        hipblaslt_route_enabled(DS4_ROCM_LT_ROUTE_Q8_F16) &&
        hipblaslt_gemm_f16(out_h->ptr, w_f16, xh, (uint32_t)out_dim,
                           (uint32_t)n_tok, (uint32_t)in_dim, HIPBLAS_OP_T,
                           HIP_R_16F,
                           label ? label : "q8 f16-out projection")) {
        return 1;
    }
#endif
    const float alpha = 1.0f;
    const float beta = 0.0f;
    hipblasStatus_t st = hipblasGemmEx(g_hipblas,
                                     HIPBLAS_OP_T,
                                     HIPBLAS_OP_N,
                                     (int)out_dim,
                                     (int)n_tok,
                                     (int)in_dim,
                                     &alpha,
                                     w_f16,
                                     HIPBLAS_R_16F,
                                     (int)in_dim,
                                     xh,
                                     HIPBLAS_R_16F,
                                     (int)in_dim,
                                     &beta,
                                     out_h->ptr,
                                     HIPBLAS_R_16F,
                                     (int)out_dim,
                                     HIPBLAS_COMPUTE_32F,
                                     HIPBLAS_GEMM_DEFAULT);
    if (st == HIPBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16-out matmul failed: status %d\n", (int)st);
    hip_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16-out matmul failure",
                                            in_dim * out_dim * sizeof(__half));
    return 0;
}

static int hip_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !hip_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !hip_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !hip_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    /* The shared-expert hipBLAS route wins for prompt chunks and loses badly for
     * verification blocks, for the same macro-tile padding reason. */
    if (n_tok > ds4_rocm_dense_small_batch_rows() &&
        hip_runtime_config()->shared_down_hipblas && in_dim == 2048u && out_dim == 4096u &&
        hip_matmul_q8_0_tensor_f16_gemm(out, model_map, model_size, weight_offset,
                                         in_dim, out_dim, x, n_tok, label ? label : "shared_expert")) {
        return 1;
    }
    const char *wptr = hip_model_range_ptr(model_map, weight_offset, weight_bytes, "q8_0");
    if (!wptr) return 0;
    /* Dense Q8 prefill through the vendored MMQ tier. Excluded from DSpark
     * verification blocks so a verified row keeps decode's reduction order. */
    /*
     * Dense MMQ at pp32 is a quality-sensitive policy, not a blanket size
     * threshold.  The bit assignments let profiler runs isolate projection
     * families.  Only Q-B (bit 2) is enabled by default: it improved pp32
     * throughput while preserving the sequential greedy choice.  KV and Q-A
     * changed that choice, and enabling more individually-safe families
     * provided no repeatable gain.
     */
    uint32_t dense_mmq_shape_bit = 32u;
    if (out_dim == 512u) {
      dense_mmq_shape_bit = 1u;
    } else if (out_dim == 1024u) {
      dense_mmq_shape_bit = 2u;
    } else if (out_dim == 32768u) {
      dense_mmq_shape_bit = 4u;
    } else if (out_dim == 2048u) {
      dense_mmq_shape_bit = 8u;
    } else if (out_dim == 4096u) {
      dense_mmq_shape_bit = 16u;
    }
    const bool dense_mmq_shape_enabled =
        n_tok >= DS4_ROCM_WIDE_PREFILL_ROWS ||
        (hip_runtime_config()->dense_mmq_mask & dense_mmq_shape_bit) != 0u;
    if (n_tok >= hip_runtime_config()->dense_mmq_rows && !g_small_batch_mode &&
        dense_mmq_shape_enabled && g_rocm_mmq_ready &&
        ds4_mmq_q8_0_dense(wptr, (const float*)x->ptr, (float*)out->ptr,
                           (int)out_dim, (int)n_tok, (int)in_dim,
                           (hipStream_t)0) == 0) {
      static int notice_printed = 0;
      if (!notice_printed) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "dense Q8 prefill using native HIP MMQ\n");
        notice_printed = 1;
      }
      return 1;
    }
    if (n_tok == 1 && !hip_q8_prequant_decode_enabled()) {
        const bool extended_sharedx =
            in_dim > 8192u &&
            in_dim <= 16384u &&
            hip_runtime_config()->q8_decode_sharedx_64k;
        if ((in_dim & 31u) == 0u &&
            (in_dim <= 8192u || extended_sharedx)) {
            const unsigned rows_per_block = 32u;
            const unsigned threads = rows_per_block * 32u;
            matmul_q8_0_f32_sharedx_warp_rows_w32_kernel<<<
                    (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out_dim,
                    blocks * 34u);
            const hipError_t launch_err = hipGetLastError();
            if (launch_err == hipSuccess) {
                if (extended_sharedx) {
                    static int notice_printed = 0;
                    if (!notice_printed) {
                        fprintf(stderr,
                                DS4_GPU_LOG_PREFIX
                                "Q8 one-token shared-input kernel enabled "
                                "through 64 KiB LDS (in_dim=%llu)\n",
                                (unsigned long long)in_dim);
                        notice_printed = 1;
                    }
                }
                return 1;
            }
            if (!extended_sharedx) {
                return hip_ok(launch_err,
                               "matmul_q8_0 f32 sharedx launch");
            }
            static int fallback_notice_printed = 0;
            if (!fallback_notice_printed) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX
                        "Q8 64 KiB shared-input launch unavailable "
                        "(%s); falling back to the warp-row kernel\n",
                        hipGetErrorString(launch_err));
                fallback_notice_printed = 1;
            }
        }
        matmul_q8_0_f32_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                in_dim,
                out_dim,
                blocks);
        return hip_ok(hipGetLastError(), "matmul_q8_0 f32 warp launch");
    }
    if (n_tok > 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if ((in_dim % 32u) == 0u &&
            out_dim >= 1024u &&
            n_tok >= 256u &&
            in_dim <= UINT32_MAX && out_dim <= UINT32_MAX && n_tok <= UINT32_MAX) {
            const dim3 grid((uint32_t)((out_dim + 63u) / 64u),
                            (uint32_t)((n_tok + 63u) / 64u),
                            1u);
            matmul_q8_0_f32_batch_wmma_4w_kernel<<<grid, 128u>>>(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)n_tok,
                    (uint32_t)in_dim,
                    (uint32_t)out_dim,
                    blocks * 34u);
            return hip_ok(hipGetLastError(), "matmul_q8_0 f32 batch wmma 4w launch");
        }
#endif
        const uint32_t small_batch_rows = ds4_rocm_dense_small_batch_rows();
        if ((in_dim & 31u) == 0u && n_tok <= small_batch_rows && n_tok <= 16u) {
            /*
             * Verification-block width: quantize the activations once and reuse
             * each weight block across the batch, in the decode kernel's access
             * order. This both moves the weights once and keeps the result
             * bitwise equal to ordinary decode.
             */
            const uint64_t xq_bytes = (uint64_t)n_tok * blocks * 32u;
            const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
            const uint64_t tmp_bytes =
                scale_offset + (uint64_t)n_tok * blocks * sizeof(float);
            void *tmp = hip_tmp_alloc(tmp_bytes, "q8_0 narrow batch prequant");
            if (tmp) {
                int8_t *xq = (int8_t *)tmp;
                float *xscale = (float *)((char *)tmp + scale_offset);
                dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
                quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
                        xq, xscale, (const float *)x->ptr, in_dim, blocks);
                if (hip_ok(hipGetLastError(),
                           "matmul_q8_0 narrow batch quantize launch")) {
                  return hip_launch_q8_batch_reuse(
                      (float*)out->ptr,
                      reinterpret_cast<const unsigned char*>(wptr), xq, xscale,
                      in_dim, out_dim, blocks, (uint32_t)n_tok,
                      n_tok == 2u ? hip_runtime_config()->q8_batch_rpb
                                  : hip_runtime_config()->q8_decode_rpb);
                }
            }
        }
        if ((in_dim & 31u) == 0u && out_dim <= UINT32_MAX && n_tok <= UINT32_MAX) {
            const uint32_t rows_per_block = 32u;
            /*
             * Match the token tile to the rows actually present. The tile is the
             * unit of shared-activation staging and of the token loop inside the
             * kernel, so a 32-wide tile on a six-row verification block does
             * more than five times the necessary work. Prompt chunks are far
             * wider than any tile and keep the 32-wide choice.
             */
            const uint32_t tile = n_tok >= 32u ? 32u
                                : n_tok > 8u   ? 16u
                                : n_tok > 4u   ? 8u
                                : n_tok > 2u   ? 4u
                                               : 2u;
            const uint32_t block_tile = 16u;
            hip_launch_q8_batch_sharedx((float *)out->ptr,
                                         reinterpret_cast<const unsigned char *>(wptr),
                                         (const float *)x->ptr,
                                         (uint32_t)blocks,
                                         (uint32_t)out_dim,
                                         (uint32_t)n_tok,
                                         blocks * 34u,
                                         rows_per_block,
                                         tile,
                                         block_tile);
            return hip_ok(hipGetLastError(), "matmul_q8_0 f32 batch sharedx launch");
        }
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_f32_batch_warp8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                in_dim,
                out_dim,
                n_tok,
                blocks);
        return hip_ok(hipGetLastError(), "matmul_q8_0 f32 batch warp launch");
    }
    if (g_hipblas_ready && n_tok > 1) {
        const __half *w_f16 = hip_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f16) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = hip_f16_input_lookup((const float *)x->ptr, xh_count);
            if (!xh) {
                xh = (__half *)hip_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
                if (!xh) return 0;
                hip_launch_f32_to_f16(xh, (const float *)x->ptr, xh_count);
                if (!hip_ok(hipGetLastError(), "q8 f16 activation convert launch")) return 0;
            }
            const float alpha = 1.0f;
            const float beta = 0.0f;
            hipblasStatus_t st = hipblasGemmEx(g_hipblas,
                                             HIPBLAS_OP_T,
                                             HIPBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w_f16,
                                             HIPBLAS_R_16F,
                                             (int)in_dim,
                                             xh,
                                             HIPBLAS_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out->ptr,
                                             HIPBLAS_R_32F,
                                             (int)out_dim,
                                             HIPBLAS_COMPUTE_32F,
                                             HIPBLAS_GEMM_DEFAULT);
            if (st == HIPBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16 matmul failed: status %d\n", (int)st);
            hip_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16 matmul failure",
                                                    in_dim * out_dim * sizeof(__half));
            /* The F16 expansion cache is only an optimization.  If hipBLAS
             * rejects the cached path under memory pressure, retry the same
             * operation through the native Q8 kernels below. */
        }
    }
    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = hip_tmp_alloc(tmp_bytes, "q8_0 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const ds4_rocm_runtime_config *cfg = hip_runtime_config();
    const int use_dp4a = 1;
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!hip_ok(hipGetLastError(), "matmul_q8_0 quantize launch")) return 0;
    if (n_tok == 1) {
        const uint32_t rows_per_block = cfg->q8_decode_rpb;
        matmul_q8_0_preq_rows_w32_kernel<<<
                ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                rows_per_block * 32u>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                blocks,
                rows_per_block,
                use_dp4a);
        return hip_ok(hipGetLastError(), "matmul_q8_0 rows launch");
    }
    if (blocks <= 32u) {
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_preq_batch_warp8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return hip_ok(hipGetLastError(), "matmul_q8_0 batch warp launch");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_q8_0_preq_kernel<<<grid, 256>>>((float *)out->ptr,
                                           reinterpret_cast<const unsigned char *>(wptr),
                                           xq,
                                           xscale,
                                           in_dim, out_dim, n_tok, blocks,
                                           use_dp4a);
    return hip_ok(hipGetLastError(), "matmul_q8_0 launch");
}

extern "C" int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return hip_matmul_q8_0_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "q8_0");
}

extern "C" int ds4_gpu_matmul_q8_0_group_pairs_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        const uint32_t *group_offsets,
        uint32_t group_count) {
    if (!out || !x || !model_map || !group_offsets || in_dim == 0u ||
        out_dim == 0u || n_tok == 0u || (in_dim & 31u) != 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_tok > UINT32_MAX || group_count < 2u || group_count > 8u ||
        (group_count & 1u) != 0u || group_offsets[0] != 0u ||
        group_offsets[group_count] != n_tok) {
        return 0;
    }
    for (uint32_t group = 0; group < group_count; ++group) {
        if (group_offsets[group + 1u] <= group_offsets[group]) {
            return 0;
        }
    }
    const uint64_t blocks = in_dim / 32u;
    uint64_t row_bytes = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t x_bytes = 0u;
    uint64_t out_bytes = 0u;
    if (weight_offset > model_size ||
        !hip_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !hip_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !hip_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) {
        return 0;
    }
    const unsigned char *w = reinterpret_cast<const unsigned char *>(
        hip_model_range_ptr(
            model_map, weight_offset, weight_bytes, "q8_0_group_pairs"));
    if (!w) return 0;

    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes =
        scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = hip_tmp_alloc(tmp_bytes, "q8_0 grouped-pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1u);
    quantize_q8_0_f32_kernel<<<qgrid, 32u>>>(
        xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!hip_ok(hipGetLastError(),
                "matmul_q8_0 grouped-pair quantize launch")) {
        return 0;
    }
    return hip_launch_q8_group_pairs(
        (float *)out->ptr, w, xq, xscale, in_dim, out_dim, blocks,
        group_offsets, group_count, hip_runtime_config()->q8_decode_rpb);
}

static int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map ||
        in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out0_dim > UINT32_MAX || out1_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    if (n_tok != 1) {
        return hip_matmul_q8_0_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                               in_dim, out0_dim, x, n_tok, "q8_0_pair0") &&
               hip_matmul_q8_0_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                               in_dim, out1_dim, x, n_tok, "q8_0_pair1");
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight0_bytes = 0, weight1_bytes = 0;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        !hip_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !hip_u64_mul_checked(out0_dim, row_bytes, &weight0_bytes) ||
        !hip_u64_mul_checked(out1_dim, row_bytes, &weight1_bytes)) {
        return 0;
    }
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) {
        return 0;
    }
    const char *w0 = hip_model_range_ptr(model_map, weight0_offset, weight0_bytes, "q8_0_pair0");
    const char *w1 = hip_model_range_ptr(model_map, weight1_offset, weight1_bytes, "q8_0_pair1");
    if (!w0 || !w1) return 0;
    if (!hip_q8_prequant_decode_enabled()) {
        const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
        if ((in_dim & 31u) == 0u && in_dim <= 8192u) {
            const unsigned rows_per_block = 32u;
            const unsigned threads = rows_per_block * 32u;
            matmul_q8_0_pair_f32_sharedx_warp_rows_w32_kernel<<<
                    (unsigned)((max_out + rows_per_block - 1u) / rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out0->ptr,
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    reinterpret_cast<const unsigned char *>(w1),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out0_dim,
                    out1_dim,
                    blocks * 34u);
            return hip_ok(hipGetLastError(), "matmul_q8_0 pair f32 sharedx launch");
        }
        matmul_q8_0_pair_f32_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
                (float *)out0->ptr,
                (float *)out1->ptr,
                reinterpret_cast<const unsigned char *>(w0),
                reinterpret_cast<const unsigned char *>(w1),
                (const float *)x->ptr,
                in_dim,
                out0_dim,
                out1_dim,
                blocks);
        return hip_ok(hipGetLastError(), "matmul_q8_0 pair f32 warp launch");
    }

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = hip_tmp_alloc(tmp_bytes, "q8_0 pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = 1;
    dim3 qgrid((unsigned)blocks, 1, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!hip_ok(hipGetLastError(), "matmul_q8_0 pair quantize launch")) {
        return 0;
    }
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    matmul_q8_0_pair_preq_warp8_kernel<<<
            ((unsigned)max_out + 7u) / 8u, 256>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            xq,
            xscale,
            in_dim,
            out0_dim,
            out1_dim,
            blocks,
            use_dp4a);
    return hip_ok(hipGetLastError(), "matmul_q8_0 pair warp launch");
}

static int hip_matmul_q8_0_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float))) {
        return 0;
    }
    const char *wptr = hip_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;
    if (!hip_q8_prequant_decode_enabled()) {
        if ((in_dim & 31u) == 0u && in_dim <= 8192u) {
            const unsigned rows_per_block = 32u;
            const unsigned threads = rows_per_block * 32u;
            matmul_q8_0_hc_expand_f32_sharedx_warp_rows_w32_kernel<<<
                    (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out_hc->ptr,
                    (float *)block_out->ptr,
                    block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
                    (const float *)residual_hc->ptr,
                    (const float *)split->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out_dim,
                    blocks * 34u,
                    n_embd,
                    n_hc,
                    block_add ? 1 : 0);
            return hip_ok(hipGetLastError(), "matmul_q8_0_hc_expand f32 sharedx launch");
        }
        matmul_q8_0_hc_expand_f32_warp8_kernel<<<
                ((unsigned)out_dim + 7u) / 8u, 256>>>(
                (float *)out_hc->ptr,
                (float *)block_out->ptr,
                block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
                (const float *)residual_hc->ptr,
                (const float *)split->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                in_dim,
                out_dim,
                n_embd,
                n_hc,
                blocks,
                block_add ? 1 : 0);
        return hip_ok(hipGetLastError(), "matmul_q8_0_hc_expand f32 launch");
    }

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = hip_tmp_alloc(tmp_bytes, "q8_0 hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const ds4_rocm_runtime_config *cfg = hip_runtime_config();
    const int use_dp4a = 1;
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!hip_ok(hipGetLastError(), "matmul_q8_0_hc_expand quantize launch")) {
        return 0;
    }
    const uint32_t rows_per_block = cfg->q8_hc_decode_rpb;
    matmul_q8_0_hc_expand_preq_rows_w32_kernel<<<
            ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
            rows_per_block * 32u>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            xq,
            xscale,
            in_dim,
            out_dim,
            n_embd,
            n_hc,
            blocks,
            rows_per_block,
            block_add ? 1 : 0,
            use_dp4a);
    return hip_ok(hipGetLastError(), "matmul_q8_0_hc_expand rows launch");
}

extern "C" int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !hip_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !hip_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const char *wptr = hip_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    const int ordered_decode = n_tok == 1u;
    if (ds4_rocm_support_batch_mode() &&
        n_tok > 1u && n_tok <= 8u &&
        in_dim == 16384u && out_dim == 24u) {
        const dim3 grid((uint32_t)out_dim, (uint32_t)n_tok, 1u);
        matmul_f16_tiny_batch_wave_kernel<<<grid, 32u>>>(
            (float *)out->ptr,
            w,
            (const float *)x->ptr,
            (uint32_t)in_dim,
            (uint32_t)out_dim,
            (uint32_t)n_tok);
        return hip_ok(
            hipGetLastError(),
            "f16 DSpark support tiny-batch wave launch");
    }
    const bool f16_decode_router_shape = (in_dim == 4096u && out_dim == 256u);
    const bool f16_decode_sharedx_shape =
        !f16_decode_router_shape &&
        in_dim <= 8192u &&
        in_dim * sizeof(float) <= 65536u;
    if (n_tok > 1u && n_tok <= 6u &&
        n_tok <= ds4_rocm_dense_small_batch_rows() &&
        !f16_decode_sharedx_shape) {
        const dim3 grid((uint32_t)out_dim, 1u, 1u);
        if (n_tok == 2u) {
            matmul_f16_ordered_batch_reuse_kernel<2u><<<grid, 32u>>>(
                (float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim);
        } else if (n_tok == 3u) {
            matmul_f16_ordered_batch_reuse_kernel<3u><<<grid, 32u>>>(
                (float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim);
        } else if (n_tok == 4u) {
            matmul_f16_ordered_batch_reuse_kernel<4u><<<grid, 32u>>>(
                (float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim);
        } else if (n_tok == 5u) {
            matmul_f16_ordered_batch_reuse_kernel<5u><<<grid, 32u>>>(
                (float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim);
        } else if (n_tok == 6u) {
            matmul_f16_ordered_batch_reuse_kernel<6u><<<grid, 32u>>>(
                (float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim);
        }
        return hip_ok(
            hipGetLastError(),
            "f16 ordered narrow-batch reuse launch");
    }
    if (n_tok == 8u &&
        n_tok <= ds4_rocm_dense_small_batch_rows() &&
        !f16_decode_sharedx_shape) {
        const dim3 grid((uint32_t)out_dim, 1u, 1u);
        matmul_f16_ordered_batch_reuse_kernel<4u><<<grid, 32u>>>(
            (float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim);
        if (!hip_ok(
                hipGetLastError(),
                "f16 ordered first half-batch reuse launch")) {
            return 0;
        }
        matmul_f16_ordered_batch_reuse_kernel<4u><<<grid, 32u>>>(
            (float *)out->ptr + 4u * out_dim,
            w,
            (const float *)x->ptr + 4u * in_dim,
            in_dim,
            out_dim);
        return hip_ok(
            hipGetLastError(),
            "f16 ordered second half-batch reuse launch");
    }
    /*
     * A speculative verification block has a handful of rows. hipBLAS pads M to
     * its macro-tile, so a six-row projection costs about what a 128-row one
     * would; replaying the single-row warp kernel per row is both far cheaper at
     * this width and bitwise identical to ordinary decode, which is what lets a
     * verified row's greedy choice match autoregressive decode.
     */
    if (n_tok > 1u && n_tok <= ds4_rocm_dense_small_batch_rows()) {
        /*
         * Replay the single-row decode kernel per row rather than widening the
         * reduction across a token grid.
         *
         * Widening is about 10 ms/pass cheaper, but it changes the accumulation
         * order of the 16384-wide hyper-connection projection, and that alone
         * flips the target's greedy choice at the very first verified row. The
         * first row is the one position speculative decoding cannot afford to get
         * wrong, so this path keeps decode's order exactly.
         */
        for (uint64_t t = 0; t < n_tok; t++) {
            ds4_gpu_tensor x_row = {
                (void *)((char *)x->ptr + t * in_dim * sizeof(float)),
                in_dim * sizeof(float),
                0};
            ds4_gpu_tensor out_row = {
                (void *)((char *)out->ptr + t * out_dim * sizeof(float)),
                out_dim * sizeof(float),
                0};
            if (!ds4_gpu_matmul_f16_tensor(&out_row, model_map, model_size,
                                           weight_offset, in_dim, out_dim,
                                           &x_row, 1u)) {
                return 0;
            }
        }
        return 1;
    }
    if (g_hipblas_ready && n_tok > 1) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = hip_f16_input_lookup((const float *)x->ptr, xh_count);
        if (!xh) {
            xh = (__half *)hip_tmp_alloc(xh_count * sizeof(__half), "f16 gemm activations");
            if (!xh) return 0;
            hip_launch_f32_to_f16(xh, (const float *)x->ptr, xh_count);
            if (!hip_ok(hipGetLastError(), "f16 activation convert launch")) return 0;
        }
#ifdef __HIP_PLATFORM_AMD__
        if (n_tok >= DS4_ROCM_WIDE_PREFILL_ROWS &&
        hipblaslt_route_enabled(DS4_ROCM_LT_ROUTE_F16) &&
            hipblaslt_gemm_f16(out->ptr, w, xh, (uint32_t)out_dim,
                               (uint32_t)n_tok, (uint32_t)in_dim, HIPBLAS_OP_T,
                               HIP_R_32F, "f16 projection")) {
            return 1;
        }
#endif
        const float alpha = 1.0f;
        const float beta = 0.0f;
        hipblasStatus_t st = hipblasGemmEx(g_hipblas,
                                         HIPBLAS_OP_T,
                                         HIPBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         HIPBLAS_R_16F,
                                         (int)in_dim,
                                         xh,
                                         HIPBLAS_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         HIPBLAS_R_32F,
                                         (int)out_dim,
                                         HIPBLAS_COMPUTE_32F,
                                         HIPBLAS_GEMM_DEFAULT);
        return hipblas_ok(st, "f16 matmul");
    }
    /* The 4096x256 F16 router projection is latency-bound and the ordered
     * 32-thread row kernel is at least as fast on gfx1151; keep shared-X for
     * compressor/indexer F16 decode where reusing x across rows is the win. */
    if (n_tok == 1u && !f16_decode_router_shape) {
        if (in_dim <= 8192u && in_dim * sizeof(float) <= 65536u) {
            const uint32_t rows_per_block = 32u;
            matmul_f16_f32_sharedx_warp_rows_w32_kernel<<<
                    ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                    rows_per_block * 32u,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)in_dim, out_dim);
            return hip_ok(hipGetLastError(), "matmul_f16 sharedx launch");
        }
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (ordered_decode) {
        matmul_f16_ordered_chunks_kernel<<<grid, 32>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return hip_ok(hipGetLastError(), "matmul_f16_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return hip_ok(hipGetLastError(), "matmul_f16 launch");
}

extern "C" int ds4_gpu_matmul_f16_group_pairs_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        const uint32_t *group_offsets,
        uint32_t group_count) {
    if (!out || !x || !model_map || !group_offsets ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_tok > UINT32_MAX || group_count < 2u || group_count > 8u ||
        (group_count & 1u) != 0u || group_offsets[0] != 0u ||
        group_offsets[group_count] != n_tok) {
        return 0;
    }
    for (uint32_t group = 0u; group < group_count; ++group) {
        if (group_offsets[group + 1u] <= group_offsets[group]) {
            return 0;
        }
    }
    uint64_t weight_bytes = 0u;
    uint64_t x_bytes = 0u;
    uint64_t out_bytes = 0u;
    if (weight_offset > model_size ||
        !hip_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t),
                              &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !hip_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) {
        return 0;
    }
    const char *wptr = hip_model_range_ptr(
        model_map, weight_offset, weight_bytes, "f16_group_pairs");
    if (!wptr) return 0;
    const __half *w = reinterpret_cast<const __half *>(wptr);

    const uint32_t equal_rows = group_offsets[1u] - group_offsets[0u];
    bool all_equal = equal_rows != 0u && equal_rows <= 6u;
    for (uint32_t group = 1u; all_equal && group < group_count; ++group) {
        all_equal =
            group_offsets[group + 1u] - group_offsets[group] == equal_rows;
    }
    if (all_equal && in_dim <= 8192u &&
        2u * in_dim * sizeof(float) <= 65536u) {
        const uint32_t rows_per_block = 32u;
        const dim3 grid(
            (static_cast<uint32_t>(out_dim) + rows_per_block - 1u) /
                rows_per_block,
            (group_count / 2u) * equal_rows, 1u);
        matmul_f16_f32_equal_group_pairs_sharedx_warp_rows_w32_kernel<<<
            grid, rows_per_block * 32u,
            2u * static_cast<size_t>(in_dim) * sizeof(float)>>>(
                reinterpret_cast<float *>(out->ptr), w,
                reinterpret_cast<const float *>(x->ptr),
                static_cast<uint32_t>(in_dim), out_dim, equal_rows);
        return hip_ok(hipGetLastError(),
                      "matmul_f16 equal grouped-pairs sharedx launch");
    }
    if (all_equal) {
        const dim3 grid(static_cast<uint32_t>(out_dim),
                        (group_count / 2u) * equal_rows, 1u);
        matmul_f16_ordered_equal_group_pairs_exact_kernel<<<grid, 32u>>>(
            reinterpret_cast<float *>(out->ptr), w,
            reinterpret_cast<const float *>(x->ptr), in_dim, out_dim,
            equal_rows);
        return hip_ok(hipGetLastError(),
                      "matmul_f16 equal grouped-pairs launch");
    }

    for (uint32_t group = 0u; group < group_count; group += 2u) {
        const uint32_t row0 = group_offsets[group];
        const uint32_t row1 = group_offsets[group + 1u];
        const uint32_t rows0 = row1 - row0;
        const uint32_t rows1 =
            group_offsets[group + 2u] - group_offsets[group + 1u];
        if (rows0 == 0u || rows0 > 6u ||
            rows1 == 0u || rows1 > 6u) {
            return 0;
        }
        const uint32_t pair_rows =
            rows0 > rows1 ? rows0 : rows1;
        if (in_dim <= 8192u &&
            2u * in_dim * sizeof(float) <= 65536u) {
            const uint32_t rows_per_block = 32u;
            const dim3 grid(
                (static_cast<uint32_t>(out_dim) + rows_per_block - 1u) /
                    rows_per_block,
                pair_rows, 1u);
            matmul_f16_f32_ragged_group_pair_sharedx_warp_rows_w32_kernel<<<
                grid, rows_per_block * 32u,
                2u * static_cast<size_t>(in_dim) * sizeof(float)>>>(
                    reinterpret_cast<float *>(out->ptr), w,
                    reinterpret_cast<const float *>(x->ptr),
                    static_cast<uint32_t>(in_dim), out_dim,
                    row0, row1, rows0, rows1);
        } else {
            const dim3 grid(
                static_cast<uint32_t>(out_dim), pair_rows, 1u);
            matmul_f16_ordered_ragged_group_pair_exact_kernel<<<
                grid, 32u>>>(
                    reinterpret_cast<float *>(out->ptr), w,
                    reinterpret_cast<const float *>(x->ptr),
                    in_dim, out_dim, row0, row1, rows0, rows1);
        }
        if (!hip_ok(hipGetLastError(),
                    "matmul_f16 grouped-pair launch")) {
            return 0;
        }
    }
    return 1;
}

/* F16 projection over activations that are already F16.
 *
 * The paired form below feeds two weight matrices from one activation, so the
 * F32-to-F16 conversion has no reason to run twice. */
static int hip_matmul_f16_f16_input_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const __half *x_h,
        uint64_t x_bytes_available,
        uint64_t n_tok) {
    if (!out || !x_h || !model_map || !g_hipblas_ready || n_tok < 2u ||
        in_dim == 0u || out_dim == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !hip_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !hip_u64_mul3_checked(n_tok, in_dim, sizeof(__half), &x_bytes) ||
        !hip_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x_bytes_available < x_bytes || out->bytes < out_bytes) {
        return 0;
    }
    const char *wptr = hip_model_range_ptr(
            model_map, weight_offset, weight_bytes, "f16_half_input");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
#ifdef __HIP_PLATFORM_AMD__
    if (n_tok >= DS4_ROCM_WIDE_PREFILL_ROWS &&
        hipblaslt_route_enabled(DS4_ROCM_LT_ROUTE_F16_PAIR) &&
        hipblaslt_gemm_f16(out->ptr, w, x_h, (uint32_t)out_dim,
                           (uint32_t)n_tok, (uint32_t)in_dim, HIPBLAS_OP_T,
                           HIP_R_32F, "f16 paired projection")) {
        return 1;
    }
#endif
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipblasStatus_t st = hipblasGemmEx(g_hipblas,
                                     HIPBLAS_OP_T,
                                     HIPBLAS_OP_N,
                                     (int)out_dim,
                                     (int)n_tok,
                                     (int)in_dim,
                                     &alpha,
                                     w,
                                     HIPBLAS_R_16F,
                                     (int)in_dim,
                                     x_h,
                                     HIPBLAS_R_16F,
                                     (int)in_dim,
                                     &beta,
                                     out->ptr,
                                     HIPBLAS_R_32F,
                                     (int)out_dim,
                                     HIPBLAS_COMPUTE_32F,
                                     HIPBLAS_GEMM_DEFAULT);
    return st == HIPBLAS_STATUS_SUCCESS;
}

extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    if (n_tok >= 128u && g_hipblas_ready) {
        /* One activation conversion for both weights. */
        uint64_t x_bytes = 0, xh_bytes = 0;
        if (hip_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) &&
            hip_u64_mul3_checked(n_tok, in_dim, sizeof(__half), &xh_bytes) &&
            x->bytes >= x_bytes) {
            __half *xh = (__half *)hip_tmp_alloc(
                xh_bytes, "f16 pair gemm activations");
            if (xh) {
                const uint64_t xh_count = n_tok * in_dim;
                hip_launch_f32_to_f16(
                    xh, (const float *)x->ptr, xh_count);
                if (hip_ok(hipGetLastError(),
                            "f16 pair activation convert launch") &&
                    hip_matmul_f16_f16_input_tensor(
                        out0, model_map, model_size, weight0_offset,
                        in_dim, out_dim, xh, xh_bytes, n_tok) &&
                    hip_matmul_f16_f16_input_tensor(
                        out1, model_map, model_size, weight1_offset,
                        in_dim, out_dim, xh, xh_bytes, n_tok)) {
                    return 1;
                }
            }
        }
    }
    if (n_tok != 1) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    uint64_t weight_bytes = 0;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        !hip_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes)) {
        return 0;
    }
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out_dim * sizeof(float) ||
        out1->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const __half *w0 = (const __half *)hip_model_range_ptr(model_map, weight0_offset, weight_bytes, "f16_pair0");
    const __half *w1 = (const __half *)hip_model_range_ptr(model_map, weight1_offset, weight_bytes, "f16_pair1");
    if (!w0 || !w1) return 0;
    if (in_dim <= 8192u && in_dim * sizeof(float) <= 65536u) {
        const uint32_t rows_per_block = 32u;
        matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel<<<
                ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                rows_per_block * 32u,
                (size_t)in_dim * sizeof(float)>>>(
                (float *)out0->ptr, (float *)out1->ptr, w0, w1,
                (const float *)x->ptr, (uint32_t)in_dim, out_dim);
        return hip_ok(hipGetLastError(), "matmul_f16_pair sharedx launch");
    }
    matmul_f16_pair_ordered_chunks_kernel<<<(unsigned)out_dim, 32>>>(
        (float *)out0->ptr,
        (float *)out1->ptr,
        w0,
        w1,
        (const float *)x->ptr,
        in_dim,
        out_dim,
        out_dim);
    return hip_ok(hipGetLastError(), "matmul_f16_pair_ordered_chunks launch");
}

extern "C" int ds4_gpu_matmul_f16_pair_narrow_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0u ||
        out_dim == 0u || n_tok < 2u || n_tok > 8u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }
    uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        !hip_u64_mul3_checked(
            out_dim, in_dim, sizeof(uint16_t), &weight_bytes) ||
        weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        !hip_u64_mul3_checked(
            n_tok, in_dim, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(
            n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out0->bytes < out_bytes ||
        out1->bytes < out_bytes) {
        return 0;
    }
    const __half *w0 = (const __half *)hip_model_range_ptr(
        model_map, weight0_offset, weight_bytes, "f16_pair_narrow0");
    const __half *w1 = (const __half *)hip_model_range_ptr(
        model_map, weight1_offset, weight_bytes, "f16_pair_narrow1");
    if (!w0 || !w1) return 0;
    constexpr uint32_t rows_per_block = 32u;
    const unsigned grid =
        (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block);
    const unsigned threads = rows_per_block * 32u;
#define DS4_LAUNCH_F16_PAIR_NARROW(BATCH, OUT0, OUT1, X)                  \
    matmul_f16_pair_batch_reuse_warp_rows_w32_kernel<BATCH>              \
        <<<grid, threads>>>(                                              \
            (OUT0), (OUT1), w0, w1, (X), (uint32_t)in_dim, out_dim,      \
            rows_per_block)
    if (n_tok == 2u) {
        DS4_LAUNCH_F16_PAIR_NARROW(
            2u, (float *)out0->ptr, (float *)out1->ptr,
            (const float *)x->ptr);
    } else if (n_tok == 3u) {
        DS4_LAUNCH_F16_PAIR_NARROW(
            3u, (float *)out0->ptr, (float *)out1->ptr,
            (const float *)x->ptr);
    } else if (n_tok == 4u) {
        DS4_LAUNCH_F16_PAIR_NARROW(
            4u, (float *)out0->ptr, (float *)out1->ptr,
            (const float *)x->ptr);
    } else if (n_tok == 5u) {
        DS4_LAUNCH_F16_PAIR_NARROW(
            5u, (float *)out0->ptr, (float *)out1->ptr,
            (const float *)x->ptr);
    } else if (n_tok == 6u) {
        DS4_LAUNCH_F16_PAIR_NARROW(
            6u, (float *)out0->ptr, (float *)out1->ptr,
            (const float *)x->ptr);
    } else if (n_tok == 7u) {
        DS4_LAUNCH_F16_PAIR_NARROW(
            7u, (float *)out0->ptr, (float *)out1->ptr,
            (const float *)x->ptr);
    } else {
        DS4_LAUNCH_F16_PAIR_NARROW(
            8u, (float *)out0->ptr, (float *)out1->ptr,
            (const float *)x->ptr);
    }
#undef DS4_LAUNCH_F16_PAIR_NARROW
    return hip_ok(
        hipGetLastError(), "f16 ordered paired narrow-batch launch");
}

extern "C" int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !hip_u64_mul3_checked(out_dim, in_dim, sizeof(float), &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !hip_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const char *wptr = hip_model_range_ptr(model_map, weight_offset, weight_bytes, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    if (g_hipblas_ready && n_tok > 1) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        hipblasStatus_t st = hipblasSgemm(g_hipblas,
                                        HIPBLAS_OP_T,
                                        HIPBLAS_OP_N,
                                        (int)out_dim,
                                        (int)n_tok,
                                        (int)in_dim,
                                        &alpha,
                                        w,
                                        (int)in_dim,
                                        (const float *)x->ptr,
                                        (int)in_dim,
                                        &beta,
                                        (float *)out->ptr,
                                        (int)out_dim);
        return hipblas_ok(st, "f32 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_f32_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return hip_ok(hipGetLastError(), "matmul_f32 launch");
}

/* F16-activation entry for callers that already hold narrowed rows; returns 0
 * when the route is unavailable so the caller falls back to the F32 form. */
extern "C" int ds4_gpu_matmul_f16_f16_input_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x_h, uint64_t n_tok) {
    if (!x_h) return 0;
    return hip_matmul_f16_f16_input_tensor(out, model_map, model_size,
                                           weight_offset, in_dim, out_dim,
                                           (const __half *)x_h->ptr,
                                           x_h->bytes, n_tok);
}
