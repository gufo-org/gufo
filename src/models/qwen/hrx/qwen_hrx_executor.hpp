#ifndef GUFO_MODELS_QWEN_HRX_QWEN_HRX_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN_HRX_QWEN_HRX_EXECUTOR_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/hrx/hrx_backend.hpp"
#include "src/core/hrx/hrx_graph_executor.hpp"
#include "src/core/hrx/hrx_module_loader.hpp"
#include "src/core/hrx/hrx_utils.hpp"
#include "src/models/qwen/forward.hpp"

namespace gufo::hrx {

class QwenHrxExecutor {
public:
  explicit QwenHrxExecutor(int device_index = 0);
  ~QwenHrxExecutor();

  bool Initialize(
      const std::string& loom_artifact_path = "/tmp/qwen_swiglu.fb");
  bool InitializeAllKernels(
      const std::string& kernels_dir = "share/gufo/kernels");

  [[nodiscard]] bool IsReady() const noexcept { return is_ready_; }
  [[nodiscard]] HrxBackend& Backend() noexcept { return backend_; }
  [[nodiscard]] HrxModuleLoader& Loader() noexcept { return loader_; }
  [[nodiscard]] HrxGraphDecodeExecutor& GraphExecutor() noexcept {
    return graph_executor_;
  }

  /// Dispatches the fused Loom SwiGLU block through native HRX queue
  bool DispatchSwiGLU(hrx_buffer_t input_buf, hrx_buffer_t gate_buf,
                      hrx_buffer_t up_buf, hrx_buffer_t out_buf,
                      uint32_t num_rows, uint32_t hidden_dim = 5120);

  /// Dispatches Pre-RMSNorm + QKV Projection
  bool DispatchRMSNormQKV(hrx_buffer_t input_buf, hrx_buffer_t gamma_buf,
                          hrx_buffer_t w_qkv_buf, hrx_buffer_t out_buf,
                          uint32_t num_rows = 10240,
                          uint32_t hidden_dim = 5120);

  /// Dispatches RoPE + KV Cache store
  bool DispatchRoPEKVCache(hrx_buffer_t q_buf, hrx_buffer_t k_buf,
                           hrx_buffer_t v_buf, hrx_buffer_t cos_buf,
                           hrx_buffer_t sin_buf, hrx_buffer_t k_cache_buf,
                           hrx_buffer_t v_cache_buf, uint32_t num_heads = 64,
                           uint32_t head_dim = 128);

  /// Dispatches DeltaNet SSM linear recurrence
  bool DispatchDeltaNetRecurrence(hrx_buffer_t q_buf, hrx_buffer_t k_buf,
                                  hrx_buffer_t v_buf, hrx_buffer_t state_buf,
                                  hrx_buffer_t out_buf, uint32_t num_heads = 32,
                                  uint32_t head_dim = 128);

  /// Dispatches Down GEMV + Hidden Residual Add
  bool DispatchDownResidual(hrx_buffer_t input_buf, hrx_buffer_t w_down_buf,
                            hrx_buffer_t residual_buf, hrx_buffer_t out_buf,
                            uint32_t hidden_dim = 5120,
                            uint32_t intermediate_dim = 17408);

  /// Dispatches Final RMSNorm + LM-Head Projection
  bool DispatchFinalNormHead(hrx_buffer_t input_buf, hrx_buffer_t gamma_buf,
                             hrx_buffer_t lm_head_buf, hrx_buffer_t logits_buf,
                             uint32_t vocab_size = 152064,
                             uint32_t hidden_dim = 5120);

  /// Dispatches Macro-Fused Full Attention Layer (RMSNorm + QKV + RoPE + KV
  /// Store)
  bool DispatchLayerAttention(hrx_buffer_t input_buf, hrx_buffer_t gamma_buf,
                              hrx_buffer_t w_qkv_buf, hrx_buffer_t cos_buf,
                              hrx_buffer_t sin_buf, hrx_buffer_t q_out_buf,
                              hrx_buffer_t k_cache_buf,
                              hrx_buffer_t v_cache_buf, uint32_t qkv_dim = 8192,
                              uint32_t hidden_dim = 5120);

  /// Dispatches Macro-Fused Full FFN Layer (Norm + SwiGLU + Down + Residual)
  bool DispatchLayerFFN(hrx_buffer_t input_buf, hrx_buffer_t gamma_buf,
                        hrx_buffer_t w_down_buf, hrx_buffer_t residual_buf,
                        hrx_buffer_t out_buf, uint32_t hidden_dim = 5120,
                        uint32_t intermediate_dim = 17408);

  /// Builds a graph-executable decode step and instantiates it
  bool BuildAndInstantiateDecodeGraph(
      hrx_buffer_t hidden_buf, hrx_buffer_t attn_gamma, hrx_buffer_t w_qkv,
      hrx_buffer_t cos_buf, hrx_buffer_t sin_buf, hrx_buffer_t q_out,
      hrx_buffer_t k_cache, hrx_buffer_t v_cache, hrx_buffer_t ffn_gamma,
      hrx_buffer_t w_down, hrx_buffer_t out_buf);

  /// Executes the instantiated decode graph DAG
  bool ExecuteDecodeGraph();

private:
  HrxBackend& backend_;
  HrxModuleLoader loader_;
  HrxGraphDecodeExecutor graph_executor_;

  hrx_executable_t swiglu_executable_{nullptr};
  hrx_executable_t rmsnorm_qkv_executable_{nullptr};
  hrx_executable_t rope_kv_executable_{nullptr};
  hrx_executable_t deltanet_recurrence_executable_{nullptr};
  hrx_executable_t down_residual_executable_{nullptr};
  hrx_executable_t final_norm_head_executable_{nullptr};
  hrx_executable_t layer_attn_executable_{nullptr};
  hrx_executable_t layer_ffn_executable_{nullptr};

  bool is_ready_{false};
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_EXECUTOR_HPP_
