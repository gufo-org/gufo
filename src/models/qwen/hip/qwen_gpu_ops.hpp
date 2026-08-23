#ifndef STRIX_MODELS_QWEN_HIP_QWEN_GPU_OPS_HPP_
#define STRIX_MODELS_QWEN_HIP_QWEN_GPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace strix::hip {

inline constexpr std::size_t kSsmReplayCapacity = 16;

struct SsmReplayCapture {
  float* qkv{nullptr};
  float* alpha{nullptr};
  float* beta{nullptr};
  const std::uint32_t* position{nullptr};
  const std::uint32_t* enabled{nullptr};
};

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

/// Asynchronously copies embedding row for token_id into out_hidden
void LaunchEmbeddingLookup(const void* table, core::GgmlType type,
                           std::uint32_t token_id, float* out_hidden,
                           std::size_t hidden_size,
                           hipStream_t stream = nullptr);

/// Asynchronously copies embedding row for *d_token_id into out_hidden
void LaunchEmbeddingLookup(const void* table, core::GgmlType type,
                           const std::uint32_t* d_token_id, float* out_hidden,
                           std::size_t hidden_size,
                           hipStream_t stream = nullptr);

/// Computes RMSNorm on GPU: out = (x / sqrt(mean(x^2) + eps)) * weight
void LaunchRMSNorm(const float* x, const float* weight, float* out,
                   std::size_t dim, float eps = 1e-6F,
                   hipStream_t stream = nullptr);

/// Computes per-head RMSNorm across num_heads
void LaunchPerHeadRMSNorm(const float* x, const float* weight, float* out,
                          std::uint32_t num_heads, std::uint32_t head_dim,
                          float eps = 1e-6F, hipStream_t stream = nullptr);

/// Computes residual add: out = a + b
void LaunchResidualAdd(const float* a, const float* b, float* out,
                       std::size_t dim, hipStream_t stream = nullptr);

/// Fuses the residual add with the subsequent RMSNorm into one launch
/// (opt-c010-residual-rmsnorm): writes out_sum = a + b in place and
/// out = (out_sum / sqrt(mean(out_sum^2) + eps)) * weight. Unfused reference:
/// LaunchResidualAdd followed by LaunchRMSNorm.
void LaunchFusedResidualAddRMSNorm(const float* a, const float* b,
                                   float* out_sum, const float* weight,
                                   float* out, std::size_t dim,
                                   float eps = 1e-6F,
                                   hipStream_t stream = nullptr);

/// De-interleaves [Q0 (head_dim), Gate0 (head_dim), Q1, Gate1, ...] into
/// separate Q and Gate buffers
void LaunchUnpackQG(const float* qg_interleaved, float* q_out, float* gate_out,
                    std::uint32_t num_heads, std::uint32_t head_dim,
                    hipStream_t stream = nullptr);

/// Computes Rotary Position Embedding (RoPE) on Q and K heads
void LaunchRoPE(float* q, float* k, std::uint32_t num_heads,
                std::uint32_t num_kv_heads, std::uint32_t head_dim,
                std::uint32_t rotary_dim, std::uint32_t pos, float rope_theta,
                hipStream_t stream = nullptr);

/// Computes Rotary Position Embedding (RoPE) reading position from device
/// memory
void LaunchRoPE(float* q, float* k, std::uint32_t num_heads,
                std::uint32_t num_kv_heads, std::uint32_t head_dim,
                std::uint32_t rotary_dim, const std::uint32_t* d_pos,
                float rope_theta, hipStream_t stream = nullptr);

/// Computes SwiGLU: out = SiLU(gate) * up
void LaunchSwiGLU(const float* gate, const float* up, float* out,
                  std::size_t intermediate_size, hipStream_t stream = nullptr);

/// Computes Matrix-Vector Multiplication: y = A * x
/// Supports both F32 and BF16 weights A
void LaunchGEMV(const void* A, core::GgmlType a_type, const float* x, float* y,
                std::size_t M, std::size_t K, hipStream_t stream = nullptr);

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

/// Computes Fused SSM Input Projections (QKV, Gate, Alpha, Beta) in a single
/// kernel
void LaunchFusedSSMInputProjections(
    const void* qkv_w, core::GgmlType qkv_type, const void* gate_w,
    core::GgmlType gate_type, const void* alpha_w, core::GgmlType alpha_type,
    const void* beta_w, core::GgmlType beta_type, const float* x,
    float* qkv_out, float* gate_out, float* alpha_out, float* beta_out,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank, hipStream_t stream = nullptr);

/// Computes Fused QKV Projections for Full Attention layers in a single kernel
void LaunchFusedQKVProjections(const void* q_w, core::GgmlType q_type,
                               const void* k_w, core::GgmlType k_type,
                               const void* v_w, core::GgmlType v_type,
                               const float* x, float* q_out, float* k_out,
                               float* v_out, std::size_t q_dim,
                               std::size_t kv_dim, std::size_t hidden_size,
                               hipStream_t stream = nullptr);

/// Computes Fused SwiGLU GEMV: out = SiLU(W_gate * x) * (W_up * x)
void LaunchFusedSwiGLUGEMV(const void* gate_w, core::GgmlType gate_type,
                           const void* up_w, core::GgmlType up_type,
                           const float* x, float* out,
                           std::size_t intermediate_size,
                           std::size_t hidden_size, hipStream_t stream = nullptr);

/// Fused layer pre-RMSNorm + QKV projections (opt-c010-rmsnorm-projection).
/// Computes the norm over x and feeds the projections, matching the unfused
/// RMSNormKernel + LaunchFusedQKVProjections chain bit-for-bit.
void LaunchFusedRMSNormQKVProjections(
    const float* x, const float* norm_w, float eps, const void* q_w,
    bool q_is_bf16, const void* k_w, bool k_is_bf16, const void* v_w,
    bool v_is_bf16, float* q_out, float* k_out, float* v_out, std::size_t q_dim,
    std::size_t kv_dim, std::size_t hidden_size, hipStream_t stream = nullptr);

/// Fused layer pre-RMSNorm + SSM input projections (QKV, Gate, Alpha, Beta).
void LaunchFusedRMSNormSSMInputProjections(
    const float* x, const float* norm_w, float eps, const void* qkv_w,
    bool qkv_is_bf16, const void* gate_w, bool gate_is_bf16,
    const void* alpha_w, bool alpha_is_bf16, const void* beta_w,
    bool beta_is_bf16, float* qkv_out, float* gate_out, float* alpha_out,
    float* beta_out, std::size_t hidden_size, std::size_t qkv_size,
    std::size_t inner_size, std::size_t time_step_rank,
    hipStream_t stream = nullptr);

/// Fused layer pre-RMSNorm + FFN SwiGLU gate/up GEMV (BF16 weights).
void LaunchFusedRMSNormSwiGLUGEMV(const float* x, const float* norm_w,
                                  float eps, const void* gate_w,
                                  const void* up_w, float* out,
                                  std::size_t intermediate_size,
                                  std::size_t hidden_size,
                                  hipStream_t stream = nullptr);

/// Computes Grouped-Query Softmax Attention with KV-cache and optional gating
/// on GPU (maintaining both FP32 and FP16 cache representations). When
/// skip_kv_write is true the KV cache is assumed already written (e.g. by the
/// fused QK norm+RoPE kernel, opt-c010-qk-rope-kv).
void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, std::uint32_t pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream = nullptr,
                     float* split_k_scratch = nullptr,
                     bool skip_kv_write = false);

/// Computes Grouped-Query Softmax Attention reading position from device memory
void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, const std::uint32_t* d_pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream = nullptr, bool skip_kv_write = false);

/// Fuses per-head Q/K RMSNorm, RoPE, and the KV-cache write for a single decode
/// token into one launch (opt-c010-qk-rope-kv). Writes the normed+roped Q into
/// q_out and the normed+roped K into k_out, K/V into the FP32+FP16 caches.
void LaunchFusedQKNormRoPEKvWrite(
    const float* q, const float* k, const float* v, const float* q_weight,
    const float* k_weight, float* q_out, float* k_out, float* k_cache,
    float* v_cache, void* k_cache_f16, void* v_cache_f16,
    std::uint32_t layer_idx, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps = 1e-6F,
    hipStream_t stream = nullptr);

void LaunchSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    hipStream_t stream = nullptr, SsmReplayCapture replay_capture = {});

/// Computes parallel GPU argmax reduction over logits
void LaunchGPUArgmax(const float* logits, std::uint32_t* out_token,
                     std::size_t vocab_size, hipStream_t stream = nullptr);

// =========================================================================
// Batched GPU Operators for High-Throughput Prompt Processing (Prefill)
// =========================================================================

/// Batched Embedding lookup for B tokens
void LaunchBatchedEmbeddingLookup(const void* table, core::GgmlType type,
                                  const std::uint32_t* token_ids,
                                  float* out_hidden, std::size_t batch_size,
                                  std::size_t hidden_size,
                                  hipStream_t stream = nullptr);

/// Batched RMSNorm across B tokens (optional BF16 output in single pass)
void LaunchBatchedRMSNorm(const float* x, const float* weight, float* out,
                          void* out_bf16, std::size_t batch_size,
                          std::size_t dim, float eps = 1e-6F,
                          hipStream_t stream = nullptr);

/// Batched Per-Head RMSNorm across B tokens
void LaunchBatchedPerHeadRMSNorm(const float* x, const float* weight,
                                 float* out, std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t head_dim, float eps = 1e-6F,
                                 hipStream_t stream = nullptr);

/// Batched Residual Add across B tokens
void LaunchBatchedResidualAdd(const float* a, const float* b, float* out,
                              std::size_t batch_size, std::size_t dim,
                              hipStream_t stream = nullptr);

/// Batched fuse of residual add and RMSNorm across B tokens into one launch
/// (opt-c010-residual-rmsnorm), with optional BF16 normed output. Unfused
/// reference: LaunchBatchedResidualAdd followed by LaunchBatchedRMSNorm.
void LaunchBatchedFusedResidualAddRMSNorm(const float* a, const float* b,
                                          float* out_sum, const float* weight,
                                          float* out, void* out_bf16,
                                          std::size_t batch_size,
                                          std::size_t dim, float eps = 1e-6F,
                                          hipStream_t stream = nullptr);

/// Batched Unpack Q and Gate across B tokens
void LaunchBatchedUnpackQG(const float* qg_interleaved, float* q_out,
                           float* gate_out, std::size_t batch_size,
                           std::uint32_t num_heads, std::uint32_t head_dim,
                           hipStream_t stream = nullptr);

/// Batched RoPE across B tokens starting at pos
void LaunchBatchedRoPE(float* q, float* k, std::size_t batch_size,
                       std::uint32_t num_heads, std::uint32_t num_kv_heads,
                       std::uint32_t head_dim, std::uint32_t rotary_dim,
                       std::uint32_t start_pos, float rope_theta,
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

/// Batched GEMM: Y[B, M] = X[B, K] * A[M, K]^T
void LaunchBatchedGEMM(const void* A, bool is_bf16, const float* X, float* Y,
                       std::size_t batch_size, std::size_t M, std::size_t K,
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

/// Batched SwiGLU activation: out = (gate * sigmoid(gate)) * up (optional BF16
/// output)
void LaunchBatchedSwiGLUActivation(const float* gate, const float* up,
                                   float* out, void* out_bf16,
                                   std::size_t num_elements,
                                   hipStream_t stream = nullptr);

/// Batched Fused SSM Input Projections across B tokens
void LaunchBatchedFusedSSMInputProjections(
    const void* qkv_w, bool qkv_is_bf16, const void* gate_w, bool gate_is_bf16,
    const void* alpha_w, bool alpha_is_bf16, const void* beta_w,
    bool beta_is_bf16, const float* X, float* qkv_out, float* gate_out,
    float* alpha_out, float* beta_out, std::size_t batch_size,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank, hipStream_t stream = nullptr);

/// Batched Fused QKV Projections across B tokens
void LaunchBatchedFusedQKVProjections(
    const void* q_w, bool q_is_bf16, const void* k_w, bool k_is_bf16,
    const void* v_w, bool v_is_bf16, const float* X, float* q_out, float* k_out,
    float* v_out, std::size_t batch_size, std::size_t q_dim, std::size_t kv_dim,
    std::size_t hidden_size, hipStream_t stream = nullptr);

/// Batched Fused SwiGLU GEMM: Out[B, intermediate] = SiLU(X[B, K] * W_gate^T) *
/// (X[B, K] * W_up^T), with optional BF16 output (opt-c010-ffn-swiglu)
void LaunchBatchedFusedSwiGLUGEMM(const void* gate_w, bool gate_is_bf16,
                                  const void* up_w, bool up_is_bf16,
                                  const float* X, float* out, void* out_bf16,
                                  std::size_t batch_size,
                                  std::size_t intermediate_size,
                                  std::size_t hidden_size,
                                  hipStream_t stream = nullptr);

/// Batched Causal Attention for B tokens with KV Cache. When skip_kv_write is
/// true the KV cache is assumed already written by the fused prefill kernel
/// (opt-c010-qk-rope-kv).
void LaunchBatchedAttention(const float* q, const float* k, const float* v,
                            const float* gate, float* k_cache, float* v_cache,
                            void* k_cache_f16, void* v_cache_f16,
                            float* out_context, std::uint32_t layer_idx,
                            std::uint32_t start_pos, std::size_t batch_size,
                            std::uint32_t max_context, std::uint32_t num_heads,
                            std::uint32_t num_kv_heads, std::uint32_t head_dim,
                            hipStream_t stream = nullptr,
                            bool skip_kv_write = false);

/// Batched fuse of per-head Q/K RMSNorm, RoPE, and the KV-cache write across B
/// tokens into one launch (opt-c010-qk-rope-kv). Writes the normed+roped Q into
/// q_out and the normed+roped K into k_out, K/V into the FP32+FP16 caches.
void LaunchBatchedFusedQKNormRoPEKvWrite(
    const float* q, const float* k, const float* v, const float* q_weight,
    const float* k_weight, float* q_out, float* k_out, float* k_cache,
    float* v_cache, void* k_cache_f16, void* v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps = 1e-6F,
    hipStream_t stream = nullptr);

/// Qwen3.8-specific causal GQA tile for gfx1151. The kernel processes 16 query
/// positions and two query heads per block while reusing one FP16 K/V tile.
/// Returns false for unsupported model shapes.
[[nodiscard]] bool LaunchBatchedAttentionTile(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr);

/// Causal GQA through ROCm Composable Kernel. Inputs and outputs remain FP32
/// at the executor boundary; the fused attention operator uses FP16 tiles with
/// FP32 accumulation and online softmax. Returns false for unsupported shapes.
[[nodiscard]] bool LaunchBatchedAttentionCk(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    void* scratch_f16, float* out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr);

/// Large-batch causal attention using float32 QK/PV GEMMs and one reusable
/// [batch, context] score buffer.
void LaunchBatchedAttentionGemm(
    hipblasHandle_t handle, const float* q, const float* k, const float* v,
    const float* gate, float* k_cache, float* v_cache, void* k_cache_f16,
    void* v_cache_f16, float* scores, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr);

/// Batched Causal SSM Conv1D + DeltaNet Recurrence for B tokens
void LaunchBatchedSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t batch_size, std::size_t qkv_size,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim, hipStream_t stream = nullptr);

/// Batched Causal SSM Conv1D + DeltaNet recurrence with the per-head
/// post-RMSNorm + SiLU gate folded into the recurrence epilogue
/// (opt-c010-ssm-gate-residual). Writes the final gated output into out_buf.
void LaunchBatchedSSMConvRecurrenceNormGate(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t batch_size, std::size_t qkv_size,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim, hipStream_t stream = nullptr);

}  // namespace strix::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_QWEN_GPU_OPS_HPP_
