// Multi-token BF16 GEMM engines at the Qwen3-ASR-1.7B prefill and audio-tower
// shapes.
//
// These projections multiply a transposed BF16 weight by BF16 activations into a
// float32 result (`OP_T/OP_N`, `HIP_R_32F` output). rocBLAS has no WMMA kernel
// for that combination and resolves it to a scalar Tensile kernel, so the
// runtime routes them through hipBLASLt instead. hipBLASLt then has to pick an
// algorithm, and the production picker is a rank heuristic calibrated on
// Qwen3.8-27B's shapes at batch 2048 -- this harness measures what that pick
// costs at ASR's batch 196-211 by timing every algorithm the heuristic offers.
//
// It answers one question: is the remaining prefill gap an algorithm-selection
// problem, or does it need a hand-written WMMA kernel?
//
//   nix develop -c tools/bench/build.sh asr_prefill_gemm_bench
//   /tmp/asr_prefill_gemm_bench

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
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

#define BLAS_OK(expr)                                                      \
  do {                                                                     \
    hipblasStatus_t st_ = (expr);                                           \
    if (st_ != HIPBLAS_STATUS_SUCCESS) {                                    \
      std::fprintf(stderr, "%s:%d hipblas error %d\n", __FILE__, __LINE__,  \
                   static_cast<int>(st_));                                  \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

namespace {

// Measured WMMA BF16 ceiling on gfx1151 (tools/bench/gfx1151_peak.hip).
constexpr double kWmmaPeakTflops = 55.05;
constexpr double kDramReadPeakGBs = 240.24;
constexpr int kMaxAlgorithms = 64;

// ---------------------------------------------------------------------------
// The repository's blocked BF16 WMMA GEMM, copied verbatim from
// tools/bench/bf16_gemm_bench.hip so it can be scored against the libraries on
// cold weights at these shapes. If a hand-written kernel is going to replace the
// library here, this is the starting point it has to beat.
// ---------------------------------------------------------------------------
// Operands are staged in WMMA fragment order, [k step][tile][16 rows][16 k], so
// a fragment read is 16 contiguous bf16 = one 32-byte load per lane.
// ---------------------------------------------------------------------------

using v16bf = __attribute__((__vector_size__(16 * sizeof(short)))) short;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f WmmaBf16(v16bf a, v16bf b, v8f c) {
#if defined(__has_builtin) && \
    __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
  return c;
#endif
}

__device__ __forceinline__ v16bf LoadBfFrag(const short* p) {
  union {
    v16bf f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

// KS is the number of bf16 per staged row: 16 payload plus PAD of bank padding.
template<int BM, int BN, int BK, int WM, int WN, int PAD>
__launch_bounds__(WM* WN * 32) __global__ void BlockedBf16Wmma(
    const short* __restrict__ w, const short* __restrict__ x,
    float* __restrict__ y, int batch, int m, int k) {
  constexpr int kThreads = WM * WN * 32;
  constexpr int RS = BM / 16;
  constexpr int TS = BN / 16;
  constexpr int WRS = RS / WM;
  constexpr int WTS = TS / WN;
  constexpr int KS = 16 + PAD;
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert((BM * BK) % kThreads == 0 && (BN * BK) % kThreads == 0);
  constexpr int PFA = (BM * BK) / kThreads;
  constexpr int PFB = (BN * BK) / kThreads;

  __shared__ short s_a[BK][RS][16][KS];
  __shared__ short s_b[BK][TS][16][KS];
  static_assert(sizeof(s_a) >= static_cast<std::size_t>(WM) * WN * 256 * 4,
                "the weight stage doubles as the store transpose scratch");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sl = lane & 15;
  const int wr = wave / WN;
  const int wt = wave % WN;

  const int r_block = static_cast<int>(blockIdx.y) * BM;
  const int t_block = static_cast<int>(blockIdx.x) * BN;
  const int ksteps = k / 16;

  v8f acc[WRS][WTS] = {};

  // Hoisted 64-bit row bases, clamped so an out-of-range row reads row m-1 and
  // is masked out at the store instead of branching in the K loop.
  const short* w_row[PFA];
#pragma unroll
  for (int p = 0; p < PFA; ++p) {
    const int r = r_block + (((p * kThreads) + tid) / BK);
    w_row[p] = w + (static_cast<std::size_t>((r < m) ? r : (m - 1)) * k);
  }
  const short* x_row[PFB];
#pragma unroll
  for (int p = 0; p < PFB; ++p) {
    const int t = t_block + (((p * kThreads) + tid) / BK);
    x_row[p] = x + (static_cast<std::size_t>((t < batch) ? t : (batch - 1)) * k);
  }

  for (int ks0 = 0; ks0 < ksteps; ks0 += BK) {
    __syncthreads();
    // Stage weights: each thread owns one (row, k step) 32-byte chunk.
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = (p * kThreads) + tid;
      const int row = idx / BK;
      const int ks = idx % BK;
      const int kk = ks0 + ks;
      const short* src = w_row[p] + (static_cast<std::size_t>(kk) * 16);
      short* dst = &s_a[ks][row / 16][row % 16][0];
      if (kk < ksteps) {
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
        *reinterpret_cast<uint4*>(dst + 8) =
            *reinterpret_cast<const uint4*>(src + 8);
      } else {
        *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = (p * kThreads) + tid;
      const int tok = idx / BK;
      const int ks = idx % BK;
      const int kk = ks0 + ks;
      const short* src = x_row[p] + (static_cast<std::size_t>(kk) * 16);
      short* dst = &s_b[ks][tok / 16][tok % 16][0];
      if (kk < ksteps) {
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
        *reinterpret_cast<uint4*>(dst + 8) =
            *reinterpret_cast<const uint4*>(src + 8);
      } else {
        *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < BK; ++ks) {
      v16bf a[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        a[i] = LoadBfFrag(&s_a[ks][(wr * WRS) + i][sl][0]);
      }
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const v16bf b = LoadBfFrag(&s_b[ks][(wt * WTS) + j][sl][0]);
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
          acc[i][j] = WmmaBf16(a[i], b, acc[i][j]);
        }
      }
    }
  }

  // C element l of a wave32 tile is row 2l + lane/16, column lane%16. The output
  // is [token][row], so transpose through the (now dead) weight stage.
  __syncthreads();
  const int hi = lane >> 4;
  float* scratch = reinterpret_cast<float*>(&s_a[0][0][0][0]) + (wave * 256);
#pragma unroll
  for (int i = 0; i < WRS; ++i) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[(sl * 16) + (2 * l) + hi] = acc[i][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int r0 = r_block + (((wr * WRS) + i) * 16);
      const int t0 = t_block + (((wt * WTS) + j) * 16);
      if (t0 + 16 <= batch && r0 + 16 <= m) {
        const std::size_t base =
            (static_cast<std::size_t>(t0) * m) + static_cast<std::size_t>(r0);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane;
          y[base + (static_cast<std::size_t>(flat >> 4) * m) + (flat & 15)] =
              scratch[flat];
        }
      } else {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane;
          const int tok = t0 + (flat >> 4);
          const int r = r0 + (flat & 15);
          if (tok < batch && r < m) {
            y[(static_cast<std::size_t>(tok) * m) + r] = scratch[flat];
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

template<int BMv, int BNv, int BKv, int WMv, int WNv, int PADv>
struct Cfg {
  static constexpr int bm = BMv;
  static constexpr int bn = BNv;
  static constexpr int bk = BKv;
  static constexpr int wm = WMv;
  static constexpr int wn = WNv;
  static constexpr int pad = PADv;
};


struct Case {
  const char* name;
  std::size_t m;      // output columns, i.e. weight rows
  std::size_t k;      // reduction
  std::size_t batch;  // tokens
};

// Qwen3-ASR-1.7B. Text decoder: 28 layers, hidden 2048, FFN 6144, 8 KV heads x
// 128, 211 prompt tokens. Audio tower: 24 layers, hidden 1024, FFN 4096, 196
// tokens, plus the flattened convolution projection and the 1024->2048 output.
const Case kCases[] = {
    {"text q/o", 2048, 2048, 211},   {"text k/v", 1024, 2048, 211},
    {"text gate/up", 6144, 2048, 211}, {"text down", 2048, 6144, 211},
    {"audio qkv/o", 1024, 1024, 196}, {"audio fc1", 4096, 1024, 196},
    {"audio fc2", 1024, 4096, 196},   {"audio conv proj", 1024, 7680, 196},
    {"audio out proj", 2048, 1024, 196},
};

// Mirrors SelectHipblasLtHeuristicRank in
// src/families/qwen/kernels/hipblaslt_gemm.hip, so the row labelled
// `production rank` is the algorithm the runtime actually installs.
constexpr std::size_t kLargeBatch = 512;
constexpr std::size_t kLargeWideRank = 3;
constexpr std::size_t kTallRank = 4;
constexpr std::size_t kWideFfnThreshold = 2;

std::size_t ProductionRank(std::size_t batch, std::size_t m, std::size_t k,
                           std::size_t count) {
  if (batch >= kLargeBatch) {
    if (m <= 1024) {
      return kLargeWideRank;
    }
    return k >= kWideFfnThreshold * m ? kTallRank : kLargeWideRank;
  }
  const std::size_t tall_m_threshold = batch >= 128 ? k : 2 * k;
  if (m > tall_m_threshold || k >= kWideFfnThreshold * m) {
    return kTallRank;
  }
  return count - 1;
}

// Iteration index is passed so a caller can rotate weight replicas and keep the
// stream out of the 32 MB MALL, which a repeat loop over one 25 MB projection
// would otherwise sit entirely inside.
float TimeLoop(const std::function<void(int)>& launch, int iterations) {
  launch(0);
  HIP_OK(hipDeviceSynchronize());
  hipEvent_t start;
  hipEvent_t stop;
  HIP_OK(hipEventCreate(&start));
  HIP_OK(hipEventCreate(&stop));
  HIP_OK(hipEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch(i);
  }
  HIP_OK(hipEventRecord(stop));
  HIP_OK(hipEventSynchronize(stop));
  float ms = 0.0F;
  HIP_OK(hipEventElapsedTime(&ms, start, stop));
  HIP_OK(hipEventDestroy(start));
  HIP_OK(hipEventDestroy(stop));
  return ms / static_cast<float>(iterations);
}

}  // namespace

int main(int argc, char** argv) {
  int iterations = 20;
  bool dump_algorithms = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-i" && i + 1 < argc) {
      iterations = std::atoi(argv[++i]);
    } else if (std::string(argv[i]) == "--algos") {
      dump_algorithms = true;
    }
  }

  hipblasHandle_t blas = nullptr;
  BLAS_OK(hipblasCreate(&blas));
  hipblasLtHandle_t lt = nullptr;
  if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "hipblasLtCreate failed\n");
    return 1;
  }
  void* workspace = nullptr;
  const std::size_t workspace_bytes = 64ULL << 20;
  HIP_OK(hipMalloc(&workspace, workspace_bytes));

  std::printf("WMMA bf16 peak %.2f TFLOPS, iterations %d\n\n", kWmmaPeakTflops,
              iterations);
  std::printf("DRAM read peak 240.24 GB/s; weights are streamed cold\n\n");
  std::printf("%-16s %6s %6s %6s | %-18s %8s %8s %6s %9s %6s\n", "case", "m",
              "k", "batch", "engine", "ms", "TFLOPS", "%peak", "GB/s",
              "%bw");

  double total_hipblas = 0.0;
  double total_production = 0.0;
  double total_best = 0.0;
  double total_blocked = 0.0;
  std::size_t best_cases = 0;
  const std::size_t case_count = sizeof(kCases) / sizeof(kCases[0]);

  for (const Case& c : kCases) {
    const std::size_t weights = c.m * c.k;
    const std::size_t inputs = c.batch * c.k;
    const std::size_t outputs = c.batch * c.m;
    const double flops =
        2.0 * static_cast<double>(c.batch) * static_cast<double>(c.m) *
        static_cast<double>(c.k);

    // Enough weight replicas that one pass over them exceeds the MALL, so the
    // measured rate is against DRAM the way prefill actually sees it: each
    // layer's weights are cold.
    constexpr std::size_t kColdWorkingSetBytes = 256ULL << 20;
    const std::size_t weight_bytes = weights * sizeof(hip_bfloat16);
    const std::size_t replicas = std::max<std::size_t>(
        4, (kColdWorkingSetBytes + weight_bytes - 1) / weight_bytes);

    hip_bfloat16* d_w = nullptr;
    hip_bfloat16* d_x = nullptr;
    float* d_y = nullptr;
    float* d_reference = nullptr;
    HIP_OK(hipMalloc(&d_w, replicas * weight_bytes));
    HIP_OK(hipMalloc(&d_x, inputs * sizeof(hip_bfloat16)));
    HIP_OK(hipMalloc(&d_y, outputs * sizeof(float)));
    HIP_OK(hipMalloc(&d_reference, outputs * sizeof(float)));
    {
      std::vector<hip_bfloat16> host(std::max(weights, inputs));
      for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<hip_bfloat16>(
            0.7F * std::sin(0.0131F * static_cast<float>(i % 811)));
      }
      for (std::size_t replica = 0; replica < replicas; ++replica) {
        HIP_OK(hipMemcpy(d_w + (replica * weights), host.data(), weight_bytes,
                         hipMemcpyHostToDevice));
      }
      for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<hip_bfloat16>(
            0.9F * std::cos(0.0177F * static_cast<float>(i % 643)));
      }
      HIP_OK(hipMemcpy(d_x, host.data(), inputs * sizeof(hip_bfloat16),
                       hipMemcpyHostToDevice));
    }

    const float alpha = 1.0F;
    const float beta = 0.0F;

    // hipBLAS, exactly as LaunchHipblasGEMMBF16 issues it.
    const auto hipblas_launch = [&](int iteration) {
      const hip_bfloat16* w =
          d_w + ((static_cast<std::size_t>(iteration) % replicas) * weights);
      BLAS_OK(hipblasGemmEx(
          blas, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(c.m),
          static_cast<int>(c.batch), static_cast<int>(c.k), &alpha, w,
          HIP_R_16BF, static_cast<int>(c.k), d_x, HIP_R_16BF,
          static_cast<int>(c.k), &beta, d_y, HIP_R_32F, static_cast<int>(c.m),
          HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    };
    const float hipblas_ms = TimeLoop(hipblas_launch, iterations);
    HIP_OK(hipMemcpy(d_reference, d_y, outputs * sizeof(float),
                     hipMemcpyDeviceToDevice));
    std::vector<float> reference(outputs);
    HIP_OK(hipMemcpy(reference.data(), d_reference, outputs * sizeof(float),
                     hipMemcpyDeviceToHost));
    double reference_magnitude = 1e-6;
    for (const float value : reference) {
      reference_magnitude =
          std::max(reference_magnitude, std::abs(static_cast<double>(value)));
    }

    const auto report = [&](const char* label, double ms, const char* suffix) {
      const double tflops = flops / (ms * 1e-3) / 1e12;
      const double gbs =
          static_cast<double>(weight_bytes) / (ms * 1e-3) / 1e9;
      std::printf("%-16s %6s %6s %6s | %-18s %8.3f %8.2f %5.0f%% %9.1f %5.0f%%%s\n",
                  "", "", "", "", label, ms, tflops,
                  100.0 * tflops / kWmmaPeakTflops, gbs,
                  100.0 * gbs / kDramReadPeakGBs, suffix);
    };
    std::printf("%-16s %6zu %6zu %6zu |\n", c.name, c.m, c.k, c.batch);
    report("hipblas", hipblas_ms, "");

    // hipBLASLt: time every heuristic algorithm.
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t la = nullptr;
    hipblasLtMatrixLayout_t lb = nullptr;
    hipblasLtMatrixLayout_t lc = nullptr;
    hipblasLtMatmulPreference_t preference = nullptr;
    const hipblasOperation_t transpose_a = HIPBLAS_OP_T;
    const hipblasOperation_t transpose_b = HIPBLAS_OP_N;
    if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) ==
            HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, c.k, c.m, c.k) ==
            HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, c.k, c.batch, c.k) ==
            HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F, c.m, c.batch, c.m) ==
            HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatmulPreferenceCreate(&preference) ==
            HIPBLAS_STATUS_SUCCESS) {
      (void)hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA,
                                            &transpose_a, sizeof(transpose_a));
      (void)hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB,
                                            &transpose_b, sizeof(transpose_b));
      (void)hipblasLtMatmulPreferenceSetAttribute(
          preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_bytes, sizeof(workspace_bytes));

      std::vector<hipblasLtMatmulHeuristicResult_t> results(kMaxAlgorithms);
      int found = 0;
      if (hipblasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, preference,
                                          kMaxAlgorithms, results.data(),
                                          &found) == HIPBLAS_STATUS_SUCCESS &&
          found > 0) {
        const std::size_t rank = ProductionRank(
            c.batch, c.m, c.k, static_cast<std::size_t>(found));
        double best_ms = 0.0;
        int best_index = -1;
        double production_ms = 0.0;
        for (int i = 0; i < found; ++i) {
          if (results[i].state != HIPBLAS_STATUS_SUCCESS ||
              results[i].workspaceSize > workspace_bytes) {
            continue;
          }
          const auto launch = [&](int iteration) {
            const hip_bfloat16* w =
                d_w +
                ((static_cast<std::size_t>(iteration) % replicas) * weights);
            (void)hipblasLtMatmul(lt, op, &alpha, w, la, d_x, lb, &beta, d_y,
                                  lc, d_y, lc, &results[i].algo, workspace,
                                  results[i].workspaceSize, nullptr);
          };
          HIP_OK(hipMemset(d_y, 0, outputs * sizeof(float)));
          const double ms = TimeLoop(launch, iterations);

          std::vector<float> got(outputs);
          HIP_OK(hipMemcpy(got.data(), d_y, outputs * sizeof(float),
                           hipMemcpyDeviceToHost));
          // Normalize by the reference's largest magnitude, not per element:
          // with BF16 operands and k up to 7680 individual outputs sit near
          // zero, and a per-element ratio there rejects correct kernels.
          double worst_absolute = 0.0;
          for (std::size_t e = 0; e < outputs; ++e) {
            worst_absolute =
                std::max(worst_absolute,
                         std::abs(static_cast<double>(got[e]) - reference[e]));
          }
          const double worst = worst_absolute / reference_magnitude;
          const bool correct = worst < 2e-2;
          if (correct && (best_index < 0 || ms < best_ms)) {
            best_ms = ms;
            best_index = i;
          }
          if (static_cast<std::size_t>(i) == rank) {
            production_ms = ms;
          }
          if (dump_algorithms) {
            std::printf(
                "%-16s %6s %6s %6s | %-18s %8.3f %8.2f %5s %9.1f %5s   %s%s\n",
                "", "", "", "", (std::string("  algo #") +
                                 std::to_string(i)).c_str(),
                ms, flops / (ms * 1e-3) / 1e12, "",
                static_cast<double>(weight_bytes) / (ms * 1e-3) / 1e9, "",
                correct ? "ok" : "INCORRECT",
                static_cast<std::size_t>(i) == rank ? "  <- production rank"
                                                    : "");
          }
        }
        char suffix[96];
        if (production_ms > 0.0) {
          std::snprintf(suffix, sizeof(suffix), "  (algo #%zu of %d)", rank,
                        found);
          report("hipblaslt prod rank", production_ms, suffix);
        }
        if (best_index >= 0) {
          std::snprintf(suffix, sizeof(suffix),
                        "  (algo #%d of %d) x%.2f over prod", best_index, found,
                        production_ms > 0.0 ? production_ms / best_ms : 0.0);
          report("hipblaslt best", best_ms, suffix);
          total_best += best_ms;
          ++best_cases;
        }
        total_production += production_ms > 0.0 ? production_ms : hipblas_ms;
      }
      hipblasLtMatmulPreferenceDestroy(preference);
      hipblasLtMatrixLayoutDestroy(lc);
      hipblasLtMatrixLayoutDestroy(lb);
      hipblasLtMatrixLayoutDestroy(la);
      hipblasLtMatmulDescDestroy(op);
    }
    // Hand-written blocked WMMA, same cold weight stream.
    double blocked_best = 0.0;
    const char* blocked_best_name = nullptr;
    const auto run_blocked = [&](const char* label, auto tag) {
      constexpr int BM = decltype(tag)::bm;
      constexpr int BN = decltype(tag)::bn;
      constexpr int BK = decltype(tag)::bk;
      constexpr int WM = decltype(tag)::wm;
      constexpr int WN = decltype(tag)::wn;
      constexpr int PAD = decltype(tag)::pad;
      if (c.m % BM != 0) {
        return;
      }
      const dim3 grid((static_cast<int>(c.batch) + BN - 1) / BN,
                      (static_cast<int>(c.m) + BM - 1) / BM);
      const auto launch = [&](int iteration) {
        const hip_bfloat16* w =
            d_w + ((static_cast<std::size_t>(iteration) % replicas) * weights);
        BlockedBf16Wmma<BM, BN, BK, WM, WN, PAD>
            <<<grid, WM * WN * 32>>>(reinterpret_cast<const short*>(w),
                                     reinterpret_cast<const short*>(d_x), d_y,
                                     static_cast<int>(c.batch),
                                     static_cast<int>(c.m),
                                     static_cast<int>(c.k));
      };
      HIP_OK(hipMemset(d_y, 0, outputs * sizeof(float)));
      launch(0);
      if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
        return;
      }
      std::vector<float> got(outputs);
      HIP_OK(hipMemcpy(got.data(), d_y, outputs * sizeof(float),
                       hipMemcpyDeviceToHost));
      double worst_absolute = 0.0;
      for (std::size_t e = 0; e < outputs; ++e) {
        worst_absolute = std::max(
            worst_absolute, std::abs(static_cast<double>(got[e]) - reference[e]));
      }
      if (worst_absolute / reference_magnitude >= 2e-2) {
        std::printf("%-16s %6s %6s %6s | %-18s   INCORRECT maxabs/|ref|=%.2e\n",
                    "", "", "", "", label, worst_absolute / reference_magnitude);
        return;
      }
      const double ms = TimeLoop(launch, iterations);
      if (blocked_best == 0.0 || ms < blocked_best) {
        blocked_best = ms;
        blocked_best_name = label;
      }
    };
    run_blocked("wmma 128x128x2", Cfg<128, 128, 2, 2, 2, 0>{});
    run_blocked("wmma 128x128x4", Cfg<128, 128, 4, 2, 2, 8>{});
    run_blocked("wmma 256x128x4", Cfg<256, 128, 4, 2, 2, 0>{});
    run_blocked("wmma 128x256x4", Cfg<128, 256, 4, 2, 2, 0>{});
    if (blocked_best > 0.0) {
      char suffix[96];
      std::snprintf(suffix, sizeof(suffix), "  (%s)", blocked_best_name);
      report("wmma blocked best", blocked_best, suffix);
      total_blocked += blocked_best;
    }

    total_hipblas += hipblas_ms;

    HIP_OK(hipFree(d_w));
    HIP_OK(hipFree(d_x));
    HIP_OK(hipFree(d_y));
    HIP_OK(hipFree(d_reference));
    std::printf("\n");
  }

  std::printf("-- summed over one instance of every shape --\n");
  std::printf("hipblas             %8.3f ms\n", total_hipblas);
  std::printf("hipblaslt prod rank %8.3f ms\n", total_production);
  if (best_cases == case_count) {
    std::printf("hipblaslt best algo %8.3f ms   x%.2f over prod rank\n",
                total_best, total_production / total_best);
  } else {
    std::printf(
        "hipblaslt best algo   incomplete: only %zu of %zu cases had a "
        "correct algorithm\n",
        best_cases, case_count);
  }

  std::printf("wmma blocked best    %8.3f ms\n", total_blocked);

  hipFree(workspace);
  hipblasLtDestroy(lt);
  hipblasDestroy(blas);
  return 0;
}
