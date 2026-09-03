// DS4 mixed-window / indexed attention producer, phase ablation on gfx1151.
//
// attention_mixed_heads32_wmma_kernel is the largest single pool of unused
// throughput in a DeepSeek V4 Flash prefill: about 1.17 s of a 4,096-token
// chunk at 8 to 15% of the measured 55 TFLOP/s WMMA ceiling, and FETCH_SIZE
// counters price it at 8 GB/s against a 240 GB/s DRAM ceiling, so it is neither
// arithmetic nor bandwidth bound. One workgroup per CU (58 KiB of LDS, 1,024
// threads) and five barriers per 16-row KV block mean every phase's latency is
// exposed, which is a hypothesis this measures rather than assumes.
//
// Each variant removes exactly one phase, keeping the rest and the barriers, so
// the delta is that phase's contribution to the critical path. Results are only
// meaningful together: the ablations are not correct kernels.
//
//   nix develop -c tools/bench/build.sh dsv4_attn_mixed_bench
//   /tmp/dsv4_attn_mixed_bench [n_tokens]
//
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

/* Production top_k is 512; the kernel's 1024 cap only costs LDS here. */
#define TOPK_CAP 512u

enum Ablate : uint32_t {
    ABL_NONE = 0u,
    ABL_NO_STAGE = 1u,   // KV tile filled from registers, no global read
    ABL_NO_QK = 2u,      // scores written without the matrix multiply
    ABL_NO_SOFTMAX = 4u, // statistics skipped, rescale forced to one
    ABL_NO_PV = 8u,      // value matrix multiply skipped
    ABL_NO_BARRIER = 16u,// barriers removed (races; timing only)
};

/* QK_ACC independent accumulators break the K reduction's dependency chain:
 * the score pass issues DIM/BK back-to-back matrix ops into one accumulator, so
 * every one waits on its predecessor, and only two of the block's 32 waves are
 * on that chain. QK_WG additionally splits K across wave groups and reduces the
 * partials through LDS. Both reassociate the score sum. */
/* QT stages q transposed once, so the score pass reads both operands
 * row-major. The current form loads K as a col_major matrix_b, which rocwmma
 * has to gather; the score pass and the value pass issue the same number of
 * matrix ops per row block yet the score pass costs about eight times as much,
 * and neither spreading it over more waves nor halving its LDS reads moves it,
 * which points at the operand layout rather than parallelism or bandwidth. */
template <bool INDEXED, uint32_t ABL, uint32_t QK_ACC = 1u, uint32_t QK_WG = 1u,
          bool QHOIST = false, bool QT = false>
__global__ __launch_bounds__(1024, 1) static void attn_mixed_bench_kernel(
        float *heads, const float *sinks, const float *q, const float *raw_kv,
        const void *comp_kv, const int32_t *topk, uint32_t n_tokens,
        uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t top_k, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    constexpr uint32_t BM = 16u, BN = 16u, BK = 16u;
    constexpr uint32_t HEADS = 32u, ROWS = 16u, DIM = 512u;
    constexpr uint32_t LDS_DIM = DIM + 4u;
    constexpr bool BAR = (ABL & ABL_NO_BARRIER) == 0u;

    const uint32_t t = (uint32_t)blockIdx.x;
    const uint32_t head0 = (uint32_t)blockIdx.y * HEADS;
    if (t >= n_tokens || head_dim != DIM || head0 + HEADS > n_head) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[INDEXED ? TOPK_CAP : 1u];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t comp_count_s;
    constexpr uint32_t QT_PITCH = HEADS + 2u;
    __shared__ half q_half[QT ? 1u : HEADS * LDS_DIM];
    __shared__ half qt_half[QT ? DIM * QT_PITCH : 1u];
    __shared__ half kv_half[ROWS * LDS_DIM];
    __shared__ float scores[HEADS * ROWS];
    __shared__ half probs[HEADS * ROWS];
    __shared__ float softmax_max[HEADS];
    __shared__ float softmax_den[HEADS];
    __shared__ float softmax_rescale[HEADS];
    __shared__ float qk_part[QK_WG > 1u ? 2u * (QK_WG - 1u) * BM * ROWS : 1u];

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0u) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    if (tid == 0u) {
        uint32_t raw_count = 0u, raw_first_idx = 0u;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_count_s = raw_count;
        if constexpr (INDEXED) {
            uint32_t comp_count = 0u;
            for (uint32_t i = 0u; i < top_k && comp_count < TOPK_CAP; i++) {
                const int32_t ci = topk[(uint64_t)t * top_k + i];
                if (ci >= 0 && (uint32_t)ci < n_comp &&
                    (uint32_t)ci < visible_comp) {
                    comp_rows[comp_count++] = (uint32_t)ci;
                }
            }
            comp_count_s = comp_count;
        } else {
            comp_count_s = visible_comp;
        }
        for (uint32_t r = 0u; r < raw_count; r++) {
            raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
        }
    }
    __syncthreads();

    for (uint32_t j = tid; j < HEADS * DIM; j += blockDim.x) {
        const uint32_t h = j / DIM;
        const uint32_t d = j - h * DIM;
        const half v =
            __float2half(q[((uint64_t)t * n_head + head0 + h) * DIM + d]);
        if constexpr (QT) {
            qt_half[d * QT_PITCH + h] = v;
        } else {
            q_half[h * LDS_DIM + d] = v;
        }
    }
    if (tid < HEADS) {
        softmax_max[tid] = sinks[head0 + tid];
        softmax_den[tid] = 1.0f;
        softmax_rescale[tid] = 1.0f;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half,
                                     rocwmma::row_major>;
    using frag_b_col = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half,
                                         rocwmma::col_major>;
    using frag_b_row = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half,
                                         rocwmma::row_major>;
    using frag_c =
        rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;

    frag_c out0, out1;
    rocwmma::fill_fragment(out0, 0.0f);
    rocwmma::fill_fragment(out1, 0.0f);
    const uint32_t raw_count = raw_count_s;
    const uint32_t n_score = raw_count + comp_count_s;
    const float score_scale = rsqrtf((float)DIM);
    const uint32_t lane = tid & 31u;
    const uint32_t acc_row_base = lane >> 4u;

    /* q_half never changes across the row-block loop, so the score pass reloads
     * the same DIM/BK matrix_a fragments once per block. Holding them costs
     * (DIM/BK/QK_WG) * 8 halves per lane. */
    constexpr uint32_t QK_WAVES_C = 2u * QK_WG;
    constexpr uint32_t K_STEPS_C = DIM / BK / QK_WG;
    frag_a qa_hold[QHOIST ? K_STEPS_C : 1u];
    if (QHOIST && wave < QK_WAVES_C) {
        const uint32_t hb0 = wave / QK_WG;
        const uint32_t kg0 = wave % QK_WG;
#pragma unroll
        for (uint32_t i = 0u; i < K_STEPS_C; i++) {
            rocwmma::load_matrix_sync(
                qa_hold[i],
                q_half + hb0 * BM * LDS_DIM + kg0 * (DIM / QK_WG) + i * BK,
                LDS_DIM);
        }
    }

    for (uint32_t row0 = 0u; row0 < n_score; row0 += ROWS) {
        const uint32_t nr = n_score - row0 < ROWS ? n_score - row0 : ROWS;
        if constexpr ((ABL & ABL_NO_STAGE) == 0u) {
            for (uint32_t j = tid; j < ROWS * DIM; j += blockDim.x) {
                const uint32_t rr = j / DIM;
                const uint32_t d = j - rr * DIM;
                half v = __float2half(0.0f);
                if (rr < nr) {
                    const uint32_t sr = row0 + rr;
                    if (sr < raw_count) {
                        v = __float2half(raw_kv[(uint64_t)raw_rows[sr] * DIM + d]);
                    } else {
                        uint32_t comp_row = sr - raw_count;
                        if constexpr (INDEXED) comp_row = comp_rows[comp_row];
                        v = __float2half(
                            ((const float *)comp_kv)[(uint64_t)comp_row * DIM + d]);
                    }
                }
                kv_half[rr * LDS_DIM + d] = v;
            }
        } else {
            for (uint32_t j = tid; j < ROWS * DIM; j += blockDim.x) {
                const uint32_t rr = j / DIM;
                const uint32_t d = j - rr * DIM;
                kv_half[rr * LDS_DIM + d] = __float2half((float)(d & 7u) * 0.01f);
            }
        }
        if (BAR) __syncthreads();

        constexpr uint32_t QK_WAVES = 2u * QK_WG;
        constexpr uint32_t K_PER_WG = DIM / QK_WG;
        frag_c score_acc[QK_ACC];
        const uint32_t hb = wave / QK_WG;
        const uint32_t kg = wave % QK_WG;
        if (wave < QK_WAVES) {
#pragma unroll
            for (uint32_t a = 0u; a < QK_ACC; a++)
                rocwmma::fill_fragment(score_acc[a], 0.0f);
        }
        if constexpr ((ABL & ABL_NO_QK) == 0u) {
            if (wave < QK_WAVES) {
                uint32_t idx = 0u;
                for (uint32_t k0 = kg * K_PER_WG;
                     k0 < (kg + 1u) * K_PER_WG; k0 += BK, idx++) {
                    const uint32_t a = QK_ACC == 1u ? 0u : idx % QK_ACC;
                    if constexpr (QT) {
                        frag_a ka;
                        frag_b_row qb;
                        rocwmma::load_matrix_sync(ka, kv_half + k0, LDS_DIM);
                        rocwmma::load_matrix_sync(
                            qb, qt_half + k0 * QT_PITCH + hb * BM, QT_PITCH);
                        rocwmma::mma_sync(score_acc[a], ka, qb, score_acc[a]);
                    } else {
                        frag_b_col kb;
                        rocwmma::load_matrix_sync(kb, kv_half + k0, LDS_DIM);
                        if constexpr (QHOIST) {
                            rocwmma::mma_sync(score_acc[a], qa_hold[idx], kb,
                                              score_acc[a]);
                        } else {
                            frag_a qa;
                            rocwmma::load_matrix_sync(
                                qa, q_half + hb * BM * LDS_DIM + k0, LDS_DIM);
                            rocwmma::mma_sync(score_acc[a], qa, kb, score_acc[a]);
                        }
                    }
                }
            }
        }
        if (wave < QK_WAVES) {
#pragma unroll
            for (uint32_t a = 1u; a < QK_ACC; a++) {
#pragma unroll
                for (uint32_t e = 0u; e < score_acc[0].num_elements; e++)
                    score_acc[0].x[e] += score_acc[a].x[e];
            }
        }
        if constexpr (QK_WG == 1u) {
            if (wave < 2u) {
                /* QT's accumulator is [kv row][local head], so it lands as
                 * scoresT[row][head] with the head block choosing the column
                 * offset. The readers below index accordingly. */
                if constexpr (QT) {
                    rocwmma::store_matrix_sync(scores + hb * BM, score_acc[0],
                                               HEADS, rocwmma::mem_row_major);
                } else {
                    rocwmma::store_matrix_sync(scores + hb * BM * ROWS,
                                               score_acc[0], ROWS,
                                               rocwmma::mem_row_major);
                }
            }
            if (BAR) __syncthreads();
        } else {
            /* Group 0 stores into the final slot and the rest into scratch, so
             * the reduction is an accumulate and the scratch is one group
             * smaller -- the difference between fitting 64 KiB and not. */
            if (wave < QK_WAVES) {
                if (kg == 0u) {
                    if constexpr (QT) {
                        rocwmma::store_matrix_sync(scores + hb * BM,
                                                   score_acc[0], HEADS,
                                                   rocwmma::mem_row_major);
                    } else {
                        rocwmma::store_matrix_sync(scores + hb * BM * ROWS,
                                                   score_acc[0], ROWS,
                                                   rocwmma::mem_row_major);
                    }
                } else {
                    rocwmma::store_matrix_sync(
                        qk_part + (hb * (QK_WG - 1u) + kg - 1u) * BM * ROWS,
                        score_acc[0], ROWS, rocwmma::mem_row_major);
                }
            }
            if (BAR) __syncthreads();
            for (uint32_t j = tid; j < 2u * BM * ROWS; j += blockDim.x) {
                const uint32_t b = j / (BM * ROWS);
                const uint32_t o = j - b * (BM * ROWS);
                float sum = 0.0f;
#pragma unroll
                for (uint32_t gq = 1u; gq < QK_WG; gq++)
                    sum += qk_part[(b * (QK_WG - 1u) + gq - 1u) * BM * ROWS + o];
                if constexpr (QT) {
                    const uint32_t r = o / BM;
                    const uint32_t hl = o - r * BM;
                    scores[r * HEADS + b * BM + hl] += sum;
                } else {
                    scores[b * BM * ROWS + o] += sum;
                }
            }
            if (BAR) __syncthreads();
        }

        if constexpr ((ABL & ABL_NO_SOFTMAX) == 0u) {
            if (wave < HEADS) {
                const float s = lane < nr
                        ? (QT ? scores[lane * HEADS + wave]
                              : scores[wave * ROWS + lane]) * score_scale
                        : -INFINITY;
                float m_new = s;
                for (int off = 16; off > 0; off >>= 1) {
                    m_new = fmaxf(m_new, __shfl_xor(m_new, off, 32));
                }
                const float m_old = softmax_max[wave];
                m_new = fmaxf(m_old, m_new);
                float block_sum = lane < nr ? expf(s - m_new) : 0.0f;
                for (int off = 16; off > 0; off >>= 1) {
                    block_sum += __shfl_xor(block_sum, off, 32);
                }
                if (lane == 0u) {
                    const float factor = expf(m_old - m_new);
                    softmax_den[wave] = softmax_den[wave] * factor + block_sum;
                    softmax_max[wave] = m_new;
                    softmax_rescale[wave] = factor;
                }
            }
            if (BAR) __syncthreads();
        }

        if (row0 != 0u) {
#pragma unroll
            for (uint32_t e = 0u; e < out0.num_elements; e++) {
                const uint32_t r = 2u * e + acc_row_base;
                out0.x[e] *= softmax_rescale[r];
                out1.x[e] *= softmax_rescale[BM + r];
            }
        }

        for (uint32_t j = tid; j < HEADS * ROWS; j += blockDim.x) {
            const uint32_t h = j / ROWS;
            const uint32_t r = j - h * ROWS;
            float p = 0.0f;
            if (r < nr) {
                const float sc = QT ? scores[r * HEADS + h] : scores[h * ROWS + r];
                p = expf(sc * score_scale - softmax_max[h]);
            }
            probs[j] = __float2half(p);
        }
        if (BAR) __syncthreads();

        if constexpr ((ABL & ABL_NO_PV) == 0u) {
            frag_a p0, p1;
            frag_b_row vb;
            rocwmma::load_matrix_sync(p0, probs, ROWS);
            rocwmma::load_matrix_sync(p1, probs + BM * ROWS, ROWS);
            rocwmma::load_matrix_sync(vb, kv_half + wave * BN, LDS_DIM);
            rocwmma::mma_sync(out0, p0, vb, out0);
            rocwmma::mma_sync(out1, p1, vb, out1);
        }
        if (BAR) __syncthreads();
    }

#pragma unroll
    for (uint32_t e = 0u; e < out0.num_elements; e++) {
        const uint32_t r = 2u * e + acc_row_base;
        const float d0 = softmax_den[r];
        const float d1 = softmax_den[BM + r];
        if (d0 != 0.0f) out0.x[e] /= d0;
        if (d1 != 0.0f) out1.x[e] /= d1;
    }
    float *out = heads + ((uint64_t)t * n_head + head0) * DIM;
    rocwmma::store_matrix_sync(out + wave * BN, out0, DIM,
                               rocwmma::mem_row_major);
    rocwmma::store_matrix_sync(out + BM * DIM + wave * BN, out1, DIM,
                               rocwmma::mem_row_major);
}

struct Ctx {
    uint32_t n_tokens, n_head, head_dim, n_comp, top_k, window, ratio;
    uint32_t n_raw, raw_cap, raw_start, pos0;
    float *heads, *sinks, *q, *raw_kv, *comp_kv;
    int32_t *topk;
};

/* Max absolute deviation of a variant's heads output from the baseline. The
 * ablations are expected to differ; the QK reassociation variants are not. */
static std::vector<float> g_ref;

template <bool INDEXED, uint32_t ABL, uint32_t QK_ACC = 1u, uint32_t QK_WG = 1u,
          bool QHOIST = false, bool QT = false>
static double run(const Ctx &c, int iters) {
    const dim3 grid(c.n_tokens, c.n_head / 32u, 1u);
    const auto launch = [&] {
        attn_mixed_bench_kernel<INDEXED, ABL, QK_ACC, QK_WG, QHOIST, QT><<<grid, 1024>>>(
            c.heads, c.sinks, c.q, c.raw_kv, c.comp_kv, c.topk, c.n_tokens,
            c.pos0, c.n_raw, c.raw_cap, c.raw_start, c.n_comp, c.top_k,
            c.window, c.ratio, c.n_head, c.head_dim);
    };
    launch();
    HIP_OK(hipDeviceSynchronize());
    if constexpr (ABL == ABL_NONE) {
        const size_t n = (size_t)c.n_tokens * c.n_head * c.head_dim;
        std::vector<float> host(n);
        HIP_OK(hipMemcpy(host.data(), c.heads, n * sizeof(float),
                         hipMemcpyDeviceToHost));
        if (g_ref.empty()) {
            g_ref = host;
        } else {
            double md = 0.0;
            for (size_t i = 0; i < n; i++) {
                const double d = std::fabs((double)host[i] - (double)g_ref[i]);
                if (d > md) md = d;
            }
            std::printf("      (maxdiff vs baseline %.3g)\n", md);
        }
    }
    hipEvent_t e0, e1;
    HIP_OK(hipEventCreate(&e0));
    HIP_OK(hipEventCreate(&e1));
    HIP_OK(hipEventRecord(e0));
    for (int i = 0; i < iters; i++) launch();
    HIP_OK(hipEventRecord(e1));
    HIP_OK(hipDeviceSynchronize());
    float ms = 0.f;
    HIP_OK(hipEventElapsedTime(&ms, e0, e1));
    HIP_OK(hipEventDestroy(e0));
    HIP_OK(hipEventDestroy(e1));
    return ms / iters;
}

template <bool INDEXED>
static void sweep(const Ctx &c, const char *label, int iters) {
    std::printf("\n=== %s  tokens=%u n_comp=%u top_k=%u window=%u ratio=%u ===\n",
                label, c.n_tokens, c.n_comp, c.top_k, c.window, c.ratio);
    g_ref.clear();
    const double base = run<INDEXED, ABL_NONE>(c, iters);
    std::printf("  %-22s %9.3f ms\n", "full", base);
    struct Case { const char *name; double ms; };
    Case cases[] = {
        {"no KV stage", run<INDEXED, ABL_NO_STAGE>(c, iters)},
        {"no QK mma", run<INDEXED, ABL_NO_QK>(c, iters)},
        {"no softmax", run<INDEXED, ABL_NO_SOFTMAX>(c, iters)},
        {"no PV mma", run<INDEXED, ABL_NO_PV>(c, iters)},
        {"no barriers", run<INDEXED, ABL_NO_BARRIER>(c, iters)},
        {"QK+PV only", run<INDEXED, ABL_NO_STAGE | ABL_NO_SOFTMAX>(c, iters)},
        {"stage only", run<INDEXED, ABL_NO_QK | ABL_NO_PV | ABL_NO_SOFTMAX>(c, iters)},
    };
    for (const Case &cs : cases) {
        std::printf("  %-22s %9.3f ms   delta %+8.3f (%+5.1f%%)\n", cs.name,
                    cs.ms, cs.ms - base, 100.0 * (cs.ms - base) / base);
    }
    Case variants[] = {
        {"QK_ACC 2", run<INDEXED, ABL_NONE, 2u>(c, iters)},
        {"QK_ACC 4", run<INDEXED, ABL_NONE, 4u>(c, iters)},
        {"QK_ACC 8", run<INDEXED, ABL_NONE, 8u>(c, iters)},
        {"QK_WG 2", run<INDEXED, ABL_NONE, 1u, 2u>(c, iters)},
        {"QK_WG 4", run<INDEXED, ABL_NONE, 1u, 4u>(c, iters)},
        {"QK_WG 4 ACC 2", run<INDEXED, ABL_NONE, 2u, 4u>(c, iters)},
        {"QHOIST WG1", run<INDEXED, ABL_NONE, 1u, 1u, true>(c, iters)},
        {"QHOIST WG2", run<INDEXED, ABL_NONE, 1u, 2u, true>(c, iters)},
        {"QHOIST WG4", run<INDEXED, ABL_NONE, 1u, 4u, true>(c, iters)},
        {"QHOIST WG4 ACC2", run<INDEXED, ABL_NONE, 2u, 4u, true>(c, iters)},
        {"QT WG1", run<INDEXED, ABL_NONE, 1u, 1u, false, true>(c, iters)},
        {"QT WG2", run<INDEXED, ABL_NONE, 1u, 2u, false, true>(c, iters)},
        {"QT WG4", run<INDEXED, ABL_NONE, 1u, 4u, false, true>(c, iters)},
    };
    for (const Case &cs : variants) {
        std::printf("  %-22s %9.3f ms   delta %+8.3f (%+5.1f%%)\n", cs.name,
                    cs.ms, cs.ms - base, 100.0 * (cs.ms - base) / base);
    }
    std::fflush(stdout);
}

int main(int argc, char **argv) {
    const uint32_t tokens = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 1024u;
    Ctx c{};
    c.n_tokens = tokens;
    c.n_head = 64u;
    c.head_dim = 512u;
    c.n_comp = 4096u;
    c.top_k = 512u;
    c.window = 128u;
    c.ratio = 4u;
    c.n_raw = tokens;
    c.raw_cap = tokens;
    c.raw_start = 0u;
    c.pos0 = 0u;

    const size_t nq = (size_t)tokens * c.n_head * c.head_dim;
    const size_t nraw = (size_t)c.raw_cap * c.head_dim;
    const size_t ncomp = (size_t)c.n_comp * c.head_dim;
    const size_t ntopk = (size_t)tokens * c.top_k;

    std::vector<float> hq(nq), hraw(nraw), hcomp(ncomp), hsinks(c.n_head);
    std::vector<int32_t> htopk(ntopk);
    for (size_t i = 0; i < nq; i++) hq[i] = ((int)(i % 31) - 15) * 0.02f;
    for (size_t i = 0; i < nraw; i++) hraw[i] = ((int)(i % 29) - 14) * 0.03f;
    for (size_t i = 0; i < ncomp; i++) hcomp[i] = ((int)(i % 23) - 11) * 0.04f;
    for (uint32_t h = 0; h < c.n_head; h++) hsinks[h] = 0.1f;
    /* Distinct rows per token, as a real top-k is. */
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t i = 0; i < c.top_k; i++) {
            htopk[(size_t)t * c.top_k + i] =
                (int32_t)((t * 7u + i * 13u) % c.n_comp);
        }
    }

    HIP_OK(hipMalloc(&c.heads, nq * sizeof(float)));
    HIP_OK(hipMalloc(&c.q, nq * sizeof(float)));
    HIP_OK(hipMalloc(&c.raw_kv, nraw * sizeof(float)));
    HIP_OK(hipMalloc(&c.comp_kv, ncomp * sizeof(float)));
    HIP_OK(hipMalloc(&c.sinks, c.n_head * sizeof(float)));
    HIP_OK(hipMalloc(&c.topk, ntopk * sizeof(int32_t)));
    HIP_OK(hipMemcpy(c.q, hq.data(), nq * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(c.raw_kv, hraw.data(), nraw * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(c.comp_kv, hcomp.data(), ncomp * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(c.sinks, hsinks.data(), c.n_head * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(c.topk, htopk.data(), ntopk * sizeof(int32_t), hipMemcpyHostToDevice));

    const int iters = 5;
    sweep<true>(c, "indexed (ratio 4)", iters);

    Ctx w = c;
    w.ratio = 128u;
    w.n_comp = tokens / 128u ? tokens / 128u : 1u;
    sweep<false>(w, "mixed window (ratio 128)", iters);
    return 0;
}
