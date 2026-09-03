// DS4 attention output projection GEMM geometries on gfx1151.
//
// The two attention output projections cost about 1.1 s of a 4,096-token
// DeepSeek V4 Flash prefill, both at roughly 21 TFLOP/s against a measured
// ~59 TFLOP/s F16 WMMA ceiling, so the question this answers is which macro
// tile, wave split, and block swizzle move them and whether the limiter is DRAM
// traffic on operand re-reads or LDS issue inside the K loop.
//
//   out B: out[n][m] = Wt[k][m]^T . low[n][k]   m=4096 n=chunk k=8192
//   out A: per group g of 8, low[n][g*rank..] = Wa[g][k][rank]^T . heads[n][k]
//
// Operands rotate over three copies so the 32 MB MALL cannot hide a re-read the
// production run pays cold; see the note in benchmarks/deepseek-v4-flash.
//
//   nix develop -c tools/bench/build.sh dsv4_attn_out_gemm_bench
//   /tmp/dsv4_attn_out_gemm_bench [n_tokens]
//
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    hipError_t err_ = (expr);                                           \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

// ---------------------------------------------------------------------------
// Production kernel, verbatim from ds4_rocm_matmul.hip.hpp.
// ---------------------------------------------------------------------------
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
    const __half *__restrict__ weight_t = weight + (uint64_t)blockIdx.z * stride_a;
    const __half *__restrict__ low_h = act + (uint64_t)blockIdx.z * stride_b;
    out += (uint64_t)blockIdx.z * stride_c;
    constexpr uint32_t WAVES_M = BM / 16u / WMF;
    constexpr uint32_t WAVES_N = BN / 16u / WNF;
    constexpr uint32_t THREADS = WAVES_M * WAVES_N * 32u;
    constexpr uint32_t A_PITCH = A_ROWMAJOR ? BK : BM;
    constexpr uint32_t A_LINES = A_ROWMAJOR ? BM : BK;
    __shared__ __half shA[A_LINES * A_PITCH];
    __shared__ __half shB[BN * BK];

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

    for (uint32_t i = 0; i < WMF; i++) {
        for (uint32_t j = 0; j < WNF; j++) {
            const uint32_t mo = m0 + (wm * WMF + i) * 16u;
            const uint32_t no = n0 + (wn * WNF + j) * 16u;
            rocwmma::store_matrix_sync((float *)out + mo + (uint64_t)no * ldc,
                                       acc[i][j], ldc, rocwmma::mem_col_major);
        }
    }
}

// ---------------------------------------------------------------------------
// Double-buffered variant: the next K stage's operands are fetched into
// registers while the current stage's fragments feed the matrix ops, so one
// barrier per K step suffices instead of two.
// ---------------------------------------------------------------------------
template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t WMF, uint32_t WNF,
          uint32_t SWZ>
__global__ __launch_bounds__((BM / 16u / WMF) * (BN / 16u / WNF) * 32u) void
ds4_gemm_f16_wmma_pipe_kernel(float *__restrict__ out,
                              const __half *__restrict__ weight,
                              const __half *__restrict__ act,
                              uint32_t m,
                              uint32_t n,
                              uint32_t k,
                              uint32_t ldc,
                              uint64_t stride_a,
                              uint64_t stride_b,
                              uint64_t stride_c) {
    const __half *__restrict__ weight_t = weight + (uint64_t)blockIdx.z * stride_a;
    const __half *__restrict__ low_h = act + (uint64_t)blockIdx.z * stride_b;
    out += (uint64_t)blockIdx.z * stride_c;
    constexpr uint32_t WAVES_M = BM / 16u / WMF;
    constexpr uint32_t WAVES_N = BN / 16u / WNF;
    constexpr uint32_t THREADS = WAVES_M * WAVES_N * 32u;
    constexpr uint32_t A_PITCH = BM;
    __shared__ __half shA[2u * BK * A_PITCH];
    __shared__ __half shB[2u * BN * BK];

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

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half,
                                     rocwmma::col_major>;
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
    constexpr uint32_t A_PER = (A_GROUPS + THREADS - 1u) / THREADS;
    constexpr uint32_t B_PER = (B_GROUPS + THREADS - 1u) / THREADS;

    uint4 ra[A_PER];
    uint4 rb[B_PER];
    const auto fetch = [&](uint32_t k0) {
        for (uint32_t u = 0; u < A_PER; u++) {
            const uint32_t g = tid + u * THREADS;
            if (g >= A_GROUPS) break;
            const uint32_t run = g % A_RUN;
            const uint32_t line = g / A_RUN;
            ra[u] = *reinterpret_cast<const uint4 *>(
                    &weight_t[(uint64_t)(k0 + line) * m + m0 + run * 8u]);
        }
        for (uint32_t u = 0; u < B_PER; u++) {
            const uint32_t g = tid + u * THREADS;
            if (g >= B_GROUPS) break;
            const uint32_t run = g % B_RUN;
            const uint32_t line = g / B_RUN;
            rb[u] = *reinterpret_cast<const uint4 *>(
                    &low_h[(uint64_t)(n0 + line) * k + k0 + run * 8u]);
        }
    };
    const auto store = [&](uint32_t buf) {
        for (uint32_t u = 0; u < A_PER; u++) {
            const uint32_t g = tid + u * THREADS;
            if (g >= A_GROUPS) break;
            const uint32_t run = g % A_RUN;
            const uint32_t line = g / A_RUN;
            *reinterpret_cast<uint4 *>(
                    &shA[buf * BK * A_PITCH + line * A_PITCH + run * 8u]) = ra[u];
        }
        for (uint32_t u = 0; u < B_PER; u++) {
            const uint32_t g = tid + u * THREADS;
            if (g >= B_GROUPS) break;
            const uint32_t run = g % B_RUN;
            const uint32_t line = g / B_RUN;
            *reinterpret_cast<uint4 *>(
                    &shB[buf * BN * BK + line * BK + run * 8u]) = rb[u];
        }
    };

    fetch(0);
    store(0);
    uint32_t buf = 0;
    for (uint32_t k0 = 0; k0 < k; k0 += BK) {
        if (k0 + BK < k) fetch(k0 + BK);
        __syncthreads();
        for (uint32_t kk = 0; kk < BK; kk += 16u) {
            frag_a fa[WMF];
            frag_b fb[WNF];
            for (uint32_t i = 0; i < WMF; i++) {
                const uint32_t mm = (wm * WMF + i) * 16u;
                rocwmma::load_matrix_sync(
                        fa[i], &shA[buf * BK * A_PITCH + kk * A_PITCH + mm],
                        A_PITCH);
            }
            for (uint32_t j = 0; j < WNF; j++) {
                rocwmma::load_matrix_sync(
                        fb[j], &shB[buf * BN * BK + (wn * WNF + j) * 16u * BK + kk],
                        BK);
            }
            for (uint32_t i = 0; i < WMF; i++)
                for (uint32_t j = 0; j < WNF; j++)
                    rocwmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        if (k0 + BK < k) {
            __syncthreads();
            buf ^= 1u;
            store(buf);
        }
    }

    for (uint32_t i = 0; i < WMF; i++) {
        for (uint32_t j = 0; j < WNF; j++) {
            const uint32_t mo = m0 + (wm * WMF + i) * 16u;
            const uint32_t no = n0 + (wn * WNF + j) * 16u;
            rocwmma::store_matrix_sync(out + mo + (uint64_t)no * ldc, acc[i][j],
                                       ldc, rocwmma::mem_col_major);
        }
    }
}

// ---------------------------------------------------------------------------

static constexpr int kCopies = 3;

struct Problem {
    const char *name;
    uint32_t m, n, k, batch;
    uint32_t ldc;
};

struct Buffers {
    __half *a[kCopies];
    __half *b[kCopies];
    float *c;
    std::vector<float> ref;
};

static double bench(const Problem &p, Buffers &buf,
                    void (*launch)(const Problem &, Buffers &, int),
                    int iters) {
    for (int i = 0; i < 2; i++) launch(p, buf, i % kCopies);
    HIP_OK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    HIP_OK(hipEventCreate(&e0));
    HIP_OK(hipEventCreate(&e1));
    HIP_OK(hipEventRecord(e0));
    for (int i = 0; i < iters; i++) launch(p, buf, i % kCopies);
    HIP_OK(hipEventRecord(e1));
    HIP_OK(hipDeviceSynchronize());
    float ms = 0.f;
    HIP_OK(hipEventElapsedTime(&ms, e0, e1));
    HIP_OK(hipEventDestroy(e0));
    HIP_OK(hipEventDestroy(e1));
    return ms / iters;
}

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t WMF, uint32_t WNF,
          uint32_t SWZ, bool PIPE>
static void launch_geom(const Problem &p, Buffers &buf, int copy) {
    const dim3 grid(p.m / BM, p.n / BN, p.batch);
    const dim3 block((BM / 16u / WMF) * (BN / 16u / WNF) * 32u);
    const uint64_t sa = (uint64_t)p.k * p.m;
    const uint64_t sb = (uint64_t)p.n * p.k;
    const uint64_t sc = (uint64_t)p.m;
    if constexpr (PIPE) {
        ds4_gemm_f16_wmma_pipe_kernel<BM, BN, BK, WMF, WNF, SWZ>
                <<<grid, block>>>(buf.c, buf.a[copy], buf.b[copy], p.m, p.n, p.k,
                                  p.ldc, sa, sb, sc);
    } else {
        ds4_gemm_f16_wmma_kernel<BM, BN, BK, WMF, WNF, SWZ, false, float>
                <<<grid, block>>>(buf.c, buf.a[copy], buf.b[copy], p.m, p.n, p.k,
                                  p.ldc, sa, sb, sc);
    }
}

struct Result {
    std::string label;
    double ms;
    double tflops;
    double maxdiff;
};

static std::vector<Result> g_results;

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t WMF, uint32_t WNF,
          uint32_t SWZ, bool PIPE>
static void run_geom(const Problem &p, Buffers &buf, int iters) {
    constexpr uint32_t threads = (BM / 16u / WMF) * (BN / 16u / WNF) * 32u;
    if (threads > 1024u || threads == 0u) return;
    if (p.m % BM || p.n % BN || p.k % BK) return;
    if (BM % (16u * WMF) || BN % (16u * WNF)) return;

    const size_t cn = (size_t)p.batch * p.n * p.ldc;
    HIP_OK(hipMemset(buf.c, 0, cn * sizeof(float)));
    launch_geom<BM, BN, BK, WMF, WNF, SWZ, PIPE>(p, buf, 0);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::printf("  %-42s LAUNCH FAIL %s\n", "geom", hipGetErrorString(err));
        (void)hipGetLastError();
        return;
    }
    std::vector<float> host(cn);
    HIP_OK(hipMemcpy(host.data(), buf.c, cn * sizeof(float), hipMemcpyDeviceToHost));
    double maxdiff = 0.0;
    if (buf.ref.empty()) {
        buf.ref = host;
    } else {
        for (size_t i = 0; i < cn; i++) {
            const double d = std::fabs((double)host[i] - (double)buf.ref[i]);
            if (d > maxdiff) maxdiff = d;
        }
    }

    const double ms = bench(p, buf, launch_geom<BM, BN, BK, WMF, WNF, SWZ, PIPE>,
                            iters);
    const double flops = 2.0 * p.m * p.n * p.k * p.batch;
    char label[128];
    std::snprintf(label, sizeof(label), "%s BM%u BN%u BK%u W%ux%u SWZ%u%s",
                  PIPE ? "pipe" : "base", BM, BN, BK, WMF, WNF, SWZ,
                  PIPE ? "" : "");
    g_results.push_back({label, ms, flops / (ms * 1e-3) / 1e12, maxdiff});
    std::printf("  %-40s %8.3f ms  %6.2f TFLOP/s  maxdiff %.3g\n", label, ms,
                flops / (ms * 1e-3) / 1e12, maxdiff);
    std::fflush(stdout);
}

int main(int argc, char **argv) {
    const uint32_t tokens = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 4096u;

    const Problem problems[] = {
        /* attention output B: out[n][m] = Wt[k][m]^T . low[n][k] */
        {"outB m4096 k8192", 4096u, tokens, 8192u, 1u, 4096u},
        /* attention output A, one group: low[n][rank] = Wa[k][rank]^T . heads[n][k] */
        {"outA g8 m1024 k4096", 1024u, tokens, 4096u, 8u, 8192u},
    };

    for (const Problem &p : problems) {
        std::printf("\n=== %s  (m=%u n=%u k=%u batch=%u ldc=%u) ===\n", p.name,
                    p.m, p.n, p.k, p.batch, p.ldc);
        Buffers buf{};
        const size_t na = (size_t)p.batch * p.k * p.m;
        const size_t nb = (size_t)p.batch * p.n * p.k;
        const size_t nc = (size_t)p.batch * p.n * p.ldc;
        std::vector<__half> ha(na), hb(nb);
        for (size_t i = 0; i < na; i++)
            ha[i] = __float2half(((int)(i * 1103515245u >> 16) % 17 - 8) * 0.05f);
        for (size_t i = 0; i < nb; i++)
            hb[i] = __float2half(((int)(i * 22695477u >> 16) % 13 - 6) * 0.07f);
        for (int c = 0; c < kCopies; c++) {
            HIP_OK(hipMalloc(&buf.a[c], na * sizeof(__half)));
            HIP_OK(hipMalloc(&buf.b[c], nb * sizeof(__half)));
            HIP_OK(hipMemcpy(buf.a[c], ha.data(), na * sizeof(__half),
                             hipMemcpyHostToDevice));
            HIP_OK(hipMemcpy(buf.b[c], hb.data(), nb * sizeof(__half),
                             hipMemcpyHostToDevice));
        }
        HIP_OK(hipMalloc(&buf.c, nc * sizeof(float)));

        const int iters = 20;
        /* Production geometry first, so it defines the reference output. */
        run_geom<256, 128, 32, 4, 2, 2, false>(p, buf, iters);
        run_geom<256, 128, 32, 4, 2, 1, false>(p, buf, iters);
        run_geom<256, 128, 32, 4, 2, 4, false>(p, buf, iters);
        run_geom<256, 128, 32, 4, 2, 8, false>(p, buf, iters);
        run_geom<256, 128, 32, 4, 2, 16, false>(p, buf, iters);
        run_geom<256, 128, 64, 4, 2, 4, false>(p, buf, iters);
        run_geom<256, 128, 32, 2, 2, 4, false>(p, buf, iters);
        run_geom<256, 128, 32, 4, 4, 4, false>(p, buf, iters);
        run_geom<256, 128, 32, 8, 2, 4, false>(p, buf, iters);
        run_geom<128, 128, 32, 4, 2, 4, false>(p, buf, iters);
        run_geom<128, 256, 32, 2, 4, 4, false>(p, buf, iters);
        run_geom<256, 256, 32, 4, 4, 4, false>(p, buf, iters);
        run_geom<256, 256, 32, 4, 2, 4, false>(p, buf, iters);
        run_geom<512, 128, 32, 8, 2, 4, false>(p, buf, iters);
        run_geom<256, 64, 32, 4, 2, 8, false>(p, buf, iters);
        run_geom<256, 128, 32, 4, 2, 4, true>(p, buf, iters);
        run_geom<256, 256, 32, 4, 4, 4, true>(p, buf, iters);

        for (int c = 0; c < kCopies; c++) {
            HIP_OK(hipFree(buf.a[c]));
            HIP_OK(hipFree(buf.b[c]));
        }
        HIP_OK(hipFree(buf.c));
    }
    return 0;
}
