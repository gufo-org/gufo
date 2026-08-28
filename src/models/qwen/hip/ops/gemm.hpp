#ifndef GUFO_MODELS_QWEN_HIP_OPS_GEMM_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_GEMM_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/gemm_route.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace gufo::hip {

struct HipblasLtDispatchInfo {
  int algorithm_id{-1};
  std::string solution_name;
  std::string kernel_name;
  std::string plan_source{"heuristic"};
  std::string persistent_cache_status{"disabled"};
  std::size_t workspace_bytes{0};
  double plan_resolution_us{0.0};
  bool plan_cache_hit{false};
};

struct HipblasLtGemmOptions {
  std::string plan_database_path;
  std::size_t tuning_workspace_bytes{0};
  bool ignore_environment{false};
};

struct HipblasLtTuningOptions {
  std::uint32_t warmup{2};
  std::uint32_t repetitions{5};
  std::size_t max_algorithms{64};
};

struct HipblasLtTuningResult {
  int algorithm_id{-1};
  int heuristic_algorithm_id{-1};
  std::string solution_name;
  std::string kernel_name;
  std::size_t workspace_bytes{0};
  double median_us{0.0};
  double heuristic_median_us{0.0};
  double verified_speedup{1.0};
  std::size_t supported_algorithms{0};
  std::size_t measured_algorithms{0};
  bool retained_tuned_algorithm{false};
};

/// Cached hipBLASLt BF16 GEMM plans for prompt-processing projections.
class HipblasLtGemm {
public:
  explicit HipblasLtGemm(HipblasLtGemmOptions options = {});
  ~HipblasLtGemm();

  HipblasLtGemm(const HipblasLtGemm&) = delete;
  HipblasLtGemm& operator=(const HipblasLtGemm&) = delete;
  HipblasLtGemm(HipblasLtGemm&&) noexcept;
  HipblasLtGemm& operator=(HipblasLtGemm&&) noexcept;

  /// Computes Y[B, M] = X_bf16[B, K] * A_bf16[M, K]^T.
  /// Returns false when hipBLASLt cannot provide a supported plan.
  [[nodiscard]] bool RunBf16(const void* a_bf16, const void* x_bf16, float* y,
                             std::size_t batch_size, std::size_t m,
                             std::size_t k, hipStream_t stream = nullptr,
                             HipblasLtDispatchInfo* dispatch_info = nullptr);

  /// Benchmarks supported algorithms, installs the fastest plan, and retains
  /// it for SavePlans(). Buffers must use the same layout as RunBf16().
  [[nodiscard]] bool TuneBf16(const void* a_bf16, const void* x_bf16, float* y,
                              std::size_t batch_size, std::size_t m,
                              std::size_t k,
                              const HipblasLtTuningOptions& options,
                              HipblasLtTuningResult* result,
                              hipStream_t stream = nullptr);

  /// Atomically writes all resolved plans using the current hardware/ROCm key.
  [[nodiscard]] bool SavePlans(const std::string& path,
                               std::string* error = nullptr) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Computes Matrix-Vector Multiplication: y = A * x
/// Supports both F32 and BF16 weights A
void LaunchGEMV(
    const void* A, core::GgmlType a_type, const float* x, float* y,
    std::size_t M, std::size_t K, hipStream_t stream = nullptr,
    models::qwen::QwenGemmMode mode = models::qwen::QwenGemmMode::kHipDecode);

/// Computes Matrix-Vector Multiplication with a residual-add epilogue:
/// y = A*x + residual (opt-c010-ssm-gate-residual). The residual is read
/// before y is written, so residual may alias y (in-place accumulate).
void LaunchGEMVResidual(const void* A, core::GgmlType a_type, const float* x,
                        float* y, const float* residual, std::size_t M,
                        std::size_t K, hipStream_t stream = nullptr);

/// opt-c014-layer-prefetch: asynchronous page-touch of a weight region on the
/// given stream. Reads one 16B chunk per 4KiB page; never writes.
void LaunchLayerWeightPrefetch(const void* data, std::size_t bytes,
                               hipStream_t stream = nullptr);

/// opt-c1xx-q8k-gemv: computes y = A * x where A is stored as block_q8_K
/// ({ float d; int8_t qs[256]; int16_t bsums[16]; }, QK_K=256). The dot runs
/// at Q8 (activation quantized to int8, integer MAC, single fp scale at block
/// end); weights are not dequantized to fp16. Requires K % 256 == 0.
void LaunchQ8KBlockGEMV(const void* A, core::GgmlType type, const float* x,
                        float* y, std::size_t M, std::size_t K,
                        hipStream_t stream = nullptr);

/// Dequantizes a block_q8_K weight region ({ float d; int8_t qs[256]; int16_t
/// bsums[16]; }, QK=256) into a BF16 scratch buffer for the prefill hipblas
/// GEMM path (opt-c162-q8k-prefill-dequant). hipBLAS cannot consume Q8_K
/// blocks, so prefill dequantizes to BF16 and runs the GF16 GEMM on the
/// scratch. out[b*256 + j] = (hip_bfloat16)(d * qs[j]) per block, matching the
/// CPU DequantizeQ8_K oracle. Requires n_elems % 256 == 0.
void LaunchDequantizeQ8KToBf16(const void* w, hip_bfloat16* out,
                               std::size_t n_elems,
                               hipStream_t stream = nullptr);

/// Type-dispatched dequantize-to-BF16 for the prefill hipblas GEMM path
/// (opt-c162-q8k-prefill-dequant). Converts a quantized weight region
/// (block_q8_0, block_q5_K, block_q6_K, block_q8_K) into a BF16 scratch
/// buffer. out[b*qk + j] = (hip_bfloat16)(per_value_dequant(block[b], j)) per
/// block, matching the CPU DequantizeQ8_0/Q5_K/Q6_K/Q8_K oracles. Q8_0 uses
/// QK=32, Q5_K/Q6_K/Q8_K use QK=256; requires n_elems to be a whole number of
/// blocks. Unsupported types are a no-op.
void LaunchDequantizeToBf16(core::GgmlType type, const void* w,
                            hip_bfloat16* out, std::size_t n_elems,
                            hipStream_t stream = nullptr);

/// Directly computes Y[B, M] = X_bf16[B, K] * W_quant[M, K]^T on quantized
/// weights (block_q8_0 / block_q8_K / block_q5_K / block_q6_K) with NO
/// dequantize-to-BF16 and NO BF16 GEMM (opt-c162-prefill-quant-direct). Each
/// output row is computed by one warp reading the quant blocks directly.
void LaunchBatchedQuantGEMM(core::GgmlType type, const void* w,
                            const void* bf16_x, float* y, std::size_t batch,
                            std::size_t m, std::size_t k,
                            hipStream_t stream = nullptr);

/// Directly computes Y[B, M] = X_bf16[B, K] * W_quant[M, K]^T without
/// quantizing the activation. The Q8_0 small-batch route shares each weight
/// load across up to eight verification rows.
void LaunchBatchedQuantGEMMBf16(core::GgmlType type, const void* w,
                                const void* bf16_x, float* y, std::size_t batch,
                                std::size_t m, std::size_t k,
                                hipStream_t stream = nullptr);

/// Directly computes Y[B, M] = X_fp32[B, K] * W_quant[M, K]^T using the same
/// quant-block dot product and FP32 activation precision as decode. Intended
/// for short verification batches where exact target-path numerics matter.
/// Q8_0 and Q8_K batches of two through eight rows share weight loads while
/// preserving the isolated production GEMV arithmetic order.
void LaunchBatchedQuantGEMMFp32(core::GgmlType type, const void* w,
                                const float* fp32_x, float* y,
                                std::size_t batch, std::size_t m, std::size_t k,
                                hipStream_t stream = nullptr);

/// Quantizes BF16 activation tensor X[B, K] to block_q8_1 activation blocks
void LaunchQuantizeActivationQ8_1(const void* bf16_x, void* q8_1_out,
                                  std::size_t batch, std::size_t k,
                                  hipStream_t stream = nullptr);

/// Quantizes FP32 activation tensor X[B, K] to block_q8_1 activation blocks
/// True when the fused SSM post-norm/gate + Q8_1 quantize kernel can take this
/// head shape.
[[nodiscard]] bool IsFusedSSMEpilogueQuantizeQ8_1Supported(
    std::uint32_t val_dim, std::size_t inner_size) noexcept;

/// SSM per-head post-RMSNorm + SiLU gate writing the tiled Q8_1 activation
/// directly (opt-c174-ssm-epilogue-quant). Use when `ssm_out` is Q8_0, so the
/// FP32 gated row has no other consumer. Bit-identical to
/// BatchedSSMPostNormGateKernel followed by an FP32 quantize.
void LaunchBatchedFusedSSMPostNormGateQuantizeQ8_1(
    const float* raw_out, const float* ssm_norm, const float* gate,
    void* q8_1_out, std::size_t batch_size, std::size_t inner_size,
    std::uint32_t num_heads, std::uint32_t val_dim,
    hipStream_t stream = nullptr);

/// True when the fused RMSNorm + Q8_1 quantize kernel can take this row length.
[[nodiscard]] bool IsFusedRMSNormQuantizeQ8_1Supported(
    std::size_t dim) noexcept;

/// Optional residual add + RMSNorm + tiled Q8_1 quantize in one pass
/// (opt-c173-norm-quant). Use when every projection reading the norm is Q8_0,
/// so neither the FP32 normed row nor the BF16 staging copy is needed. With
/// `residual` non-null the sum `x + residual` is normalized and also written to
/// `sum_out` for the next residual link. The normed values are bit-identical to
/// LaunchBatchedResidualAdd followed by LaunchBatchedRMSNorm.
void LaunchBatchedFusedRMSNormQuantizeQ8_1(
    const float* x, const float* residual, const float* weight, float* sum_out,
    void* q8_1_out, std::size_t batch_size, std::size_t dim, float eps,
    hipStream_t stream = nullptr);

void LaunchQuantizeActivationQ8_1FromFp32(const float* fp32_x, void* q8_1_out,
                                          std::size_t batch, std::size_t k,
                                          hipStream_t stream = nullptr);

/// Fuses SwiGLU activation (SiLU(gate) * up) with Q8_1 quantization into one
/// kernel
void LaunchBatchedFusedSwiGLUQuantizeQ8_1(const float* gate, const float* up,
                                          void* q8_1_out, std::size_t batch,
                                          std::size_t k,
                                          hipStream_t stream = nullptr);

/// Directly computes Y[B, M] = X_q8_1[B, K] * W_q8_0[M, K]^T using LDS-staged
/// Matrix Core (WMMA) operations with pre-quantized activation blocks.
void LaunchBatchedQuantGEMMPreQuantized(core::GgmlType type, const void* w,
                                        const void* q8_1_x, float* y,
                                        std::size_t batch, std::size_t m,
                                        std::size_t k,
                                        hipStream_t stream = nullptr);

/// Directly computes dual GEMM:
///   Y_gate[B, M] = X_q8_1[B, K] * W_gate[M, K]^T
///   Y_up[B, M]   = X_q8_1[B, K] * W_up[M, K]^T
/// in a single fused kernel pass sharing activation memory reads and registers.
void LaunchBatchedDualQuantGEMMPreQuantized(
    core::GgmlType type, const void* w_gate, const void* w_up,
    const void* q8_1_x, float* y_gate, float* y_up, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream = nullptr);

/// Batched GEMM: Y[B, M] = X[B, K] * A[M, K]^T
void LaunchBatchedGEMM(const void* A, bool is_bf16, const float* X, float* Y,
                       std::size_t batch_size, std::size_t M, std::size_t K,
                       hipStream_t stream = nullptr);

/// Exact small-batch BF16-weight GEMM with FP32 activations. The per-output
/// accumulation order matches the decode GEMV while activation tiles are
/// shared across the batch.
void LaunchExactBf16GEMMFp32Batch8(const void* A, const float* X, float* Y,
                                   std::size_t M, std::size_t K,
                                   hipStream_t stream = nullptr);

/// Converts float buffer to bfloat16 buffer on GPU
void LaunchFloatToBfloat16(const float* in, void* out, std::size_t num_elements,
                           hipStream_t stream = nullptr);

/// Hardware-accelerated Batched GEMM: Y[B, M] = X[B, K] * A[M, K]^T using
/// hipBLAS
void LaunchHipblasGEMM(hipblasHandle_t handle, const void* A, bool is_bf16,
                       const float* X, float* Y, std::size_t batch_size,
                       std::size_t M, std::size_t K, void* d_x_bf16_buf,
                       hipStream_t stream = nullptr);

/// Direct BF16 GEMM without input conversion: Y[B, M] = X_bf16[B, K] *
/// A_bf16[M, K]^T
void LaunchHipblasGEMMBF16(hipblasHandle_t handle, const void* A_bf16,
                           const void* d_x_bf16, float* Y,
                           std::size_t batch_size, std::size_t M, std::size_t K,
                           hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_GEMM_HPP_
