#include "src/models/qwen/hrx/qwen_hrx_executor.hpp"

#include <filesystem>
#include <iostream>

namespace gufo::hrx {

QwenHrxExecutor::QwenHrxExecutor(int device_index)
    : backend_(HrxBackend::Instance()) {
  (void)backend_.Initialize(device_index);
}

QwenHrxExecutor::~QwenHrxExecutor() {
  loader_.UnloadAll();
}

bool QwenHrxExecutor::Initialize(const std::string& loom_artifact_path) {
  if (!backend_.IsInitialized()) {
    return false;
  }

  hrx_status_t status =
      loader_.LoadFromFile(backend_.Device(), "qwen_swiglu", loom_artifact_path,
                           "amdgpu", "gfx1151");
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }

  swiglu_executable_ = loader_.GetExecutable("qwen_swiglu");
  if (swiglu_executable_ == nullptr) {
    return false;
  }

  is_ready_ = true;
  return true;
}

bool QwenHrxExecutor::InitializeAllKernels(const std::string& kernels_dir) {
  if (!backend_.IsInitialized()) {
    return false;
  }

  const struct {
    const char* name;
    const char* filename;
    hrx_executable_t* target_ptr;
  } kernel_specs[] = {
      {"qwen_swiglu", "qwen_fused_swiglu_bf16.fb", &swiglu_executable_},
      {"qwen_rmsnorm_qkv", "qwen_fused_rmsnorm_qkv_bf16.fb",
       &rmsnorm_qkv_executable_},
      {"qwen_rope_kv", "qwen_fused_rope_kv_cache_bf16.fb",
       &rope_kv_executable_},
      {"qwen_deltanet_recurrence", "qwen_fused_deltanet_recurrence_bf16.fb",
       &deltanet_recurrence_executable_},
      {"qwen_down_residual", "qwen_fused_down_residual_bf16.fb",
       &down_residual_executable_},
      {"qwen_final_norm_head", "qwen_fused_final_norm_head_bf16.fb",
       &final_norm_head_executable_},
      {"qwen_layer_attn", "qwen_fused_layer_attn_bf16.fb",
       &layer_attn_executable_},
      {"qwen_layer_ffn", "qwen_fused_layer_ffn_bf16.fb",
       &layer_ffn_executable_},
  };

  for (const auto& spec : kernel_specs) {
    const std::string found_path =
        (std::filesystem::path(kernels_dir) / spec.filename).string();
    if (std::filesystem::exists(found_path)) {
      hrx_status_t status = loader_.LoadFromFile(
          backend_.Device(), spec.name, found_path, "amdgpu", "gfx1151");
      if (hrx_status_is_ok(status)) {
        *spec.target_ptr = loader_.GetExecutable(spec.name);
      } else {
        hrx_status_ignore(status);
      }
    }
  }

  is_ready_ =
      (swiglu_executable_ != nullptr || layer_attn_executable_ != nullptr);
  return is_ready_;
}

bool QwenHrxExecutor::DispatchSwiGLU(hrx_buffer_t input_buf,
                                     hrx_buffer_t gate_buf, hrx_buffer_t up_buf,
                                     hrx_buffer_t out_buf, uint32_t num_rows,
                                     uint32_t hidden_dim) {
  if (!is_ready_ || swiglu_executable_ == nullptr) {
    return false;
  }

  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = (num_rows + 1) / 2;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 160;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  hrx_buffer_ref_t bindings[4];
  bindings[0] = {input_buf, 0, hidden_dim * sizeof(float)};
  bindings[1] = {gate_buf, 0, num_rows * hidden_dim * sizeof(uint16_t)};
  bindings[2] = {up_buf, 0, num_rows * hidden_dim * sizeof(uint16_t)};
  bindings[3] = {out_buf, 0, num_rows * sizeof(float)};

  uint32_t rows_param = num_rows;

  hrx_status_t status =
      hrx_stream_dispatch(backend_.Stream(), swiglu_executable_, 0, &config,
                          &rows_param, sizeof(rows_param), bindings, 4, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchRMSNormQKV(
    hrx_buffer_t input_buf, hrx_buffer_t gamma_buf, hrx_buffer_t w_qkv_buf,
    hrx_buffer_t out_buf, uint32_t num_rows, uint32_t hidden_dim) {
  if (rmsnorm_qkv_executable_ == nullptr) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = (num_rows + 1) / 2;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 160;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  hrx_buffer_ref_t bindings[4];
  bindings[0] = {input_buf, 0, hidden_dim * sizeof(float)};
  bindings[1] = {gamma_buf, 0, hidden_dim * sizeof(float)};
  bindings[2] = {w_qkv_buf, 0, num_rows * hidden_dim * sizeof(uint16_t)};
  bindings[3] = {out_buf, 0, num_rows * sizeof(float)};

  uint32_t rows_param = num_rows;
  hrx_status_t status = hrx_stream_dispatch(
      backend_.Stream(), rmsnorm_qkv_executable_, 0, &config, &rows_param,
      sizeof(rows_param), bindings, 4, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchRoPEKVCache(
    hrx_buffer_t q_buf, hrx_buffer_t k_buf, hrx_buffer_t v_buf,
    hrx_buffer_t cos_buf, hrx_buffer_t sin_buf, hrx_buffer_t k_cache_buf,
    hrx_buffer_t v_cache_buf, uint32_t num_heads, uint32_t head_dim) {
  if (rope_kv_executable_ == nullptr) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = num_heads;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 64;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  hrx_buffer_ref_t bindings[7];
  bindings[0] = {q_buf, 0, num_heads * head_dim * sizeof(float)};
  bindings[1] = {k_buf, 0, num_heads * head_dim * sizeof(float)};
  bindings[2] = {v_buf, 0, num_heads * head_dim * sizeof(float)};
  bindings[3] = {cos_buf, 0, (head_dim / 2) * sizeof(float)};
  bindings[4] = {sin_buf, 0, (head_dim / 2) * sizeof(float)};
  bindings[5] = {k_cache_buf, 0, num_heads * head_dim * sizeof(float)};
  bindings[6] = {v_cache_buf, 0, num_heads * head_dim * sizeof(float)};

  uint32_t heads_param = num_heads;
  hrx_status_t status =
      hrx_stream_dispatch(backend_.Stream(), rope_kv_executable_, 0, &config,
                          &heads_param, sizeof(heads_param), bindings, 7, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchDeltaNetRecurrence(
    hrx_buffer_t q_buf, hrx_buffer_t k_buf, hrx_buffer_t v_buf,
    hrx_buffer_t state_buf, hrx_buffer_t out_buf, uint32_t num_heads,
    uint32_t head_dim) {
  if (deltanet_recurrence_executable_ == nullptr) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = num_heads;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 64;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  hrx_buffer_ref_t bindings[5];
  bindings[0] = {q_buf, 0, num_heads * head_dim * sizeof(float)};
  bindings[1] = {k_buf, 0, num_heads * head_dim * sizeof(float)};
  bindings[2] = {v_buf, 0, num_heads * head_dim * sizeof(float)};
  bindings[3] = {state_buf, 0, num_heads * head_dim * head_dim * sizeof(float)};
  bindings[4] = {out_buf, 0, num_heads * head_dim * sizeof(float)};

  uint32_t heads_param = num_heads;
  hrx_status_t status = hrx_stream_dispatch(
      backend_.Stream(), deltanet_recurrence_executable_, 0, &config,
      &heads_param, sizeof(heads_param), bindings, 5, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchDownResidual(
    hrx_buffer_t input_buf, hrx_buffer_t w_down_buf, hrx_buffer_t residual_buf,
    hrx_buffer_t out_buf, uint32_t hidden_dim, uint32_t intermediate_dim) {
  if (down_residual_executable_ == nullptr) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = (hidden_dim + 1) / 2;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 544;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  hrx_buffer_ref_t bindings[4];
  bindings[0] = {input_buf, 0, intermediate_dim * sizeof(float)};
  bindings[1] = {w_down_buf, 0,
                 hidden_dim * intermediate_dim * sizeof(uint16_t)};
  bindings[2] = {residual_buf, 0, hidden_dim * sizeof(float)};
  bindings[3] = {out_buf, 0, hidden_dim * sizeof(float)};

  uint32_t hidden_param = hidden_dim;
  hrx_status_t status = hrx_stream_dispatch(
      backend_.Stream(), down_residual_executable_, 0, &config, &hidden_param,
      sizeof(hidden_param), bindings, 4, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchFinalNormHead(
    hrx_buffer_t input_buf, hrx_buffer_t gamma_buf, hrx_buffer_t lm_head_buf,
    hrx_buffer_t logits_buf, uint32_t vocab_size, uint32_t hidden_dim) {
  if (final_norm_head_executable_ == nullptr) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = (vocab_size + 1) / 2;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 160;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  hrx_buffer_ref_t bindings[4];
  bindings[0] = {input_buf, 0, hidden_dim * sizeof(float)};
  bindings[1] = {gamma_buf, 0, hidden_dim * sizeof(float)};
  bindings[2] = {lm_head_buf, 0, vocab_size * hidden_dim * sizeof(uint16_t)};
  bindings[3] = {logits_buf, 0, vocab_size * sizeof(float)};

  uint32_t vocab_param = vocab_size;
  hrx_status_t status = hrx_stream_dispatch(
      backend_.Stream(), final_norm_head_executable_, 0, &config, &vocab_param,
      sizeof(vocab_param), bindings, 4, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchLayerAttention(
    hrx_buffer_t input_buf, hrx_buffer_t gamma_buf, hrx_buffer_t w_qkv_buf,
    hrx_buffer_t cos_buf, hrx_buffer_t sin_buf, hrx_buffer_t q_out_buf,
    hrx_buffer_t k_cache_buf, hrx_buffer_t v_cache_buf, uint32_t qkv_dim,
    uint32_t hidden_dim) {
  if (layer_attn_executable_ == nullptr) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = (qkv_dim + 1) / 2;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 160;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  hrx_buffer_ref_t bindings[8];
  bindings[0] = {input_buf, 0, hidden_dim * sizeof(float)};
  bindings[1] = {gamma_buf, 0, hidden_dim * sizeof(float)};
  bindings[2] = {w_qkv_buf, 0, qkv_dim * hidden_dim * sizeof(uint16_t)};
  bindings[3] = {cos_buf, 0, 64 * sizeof(float)};
  bindings[4] = {sin_buf, 0, 64 * sizeof(float)};
  bindings[5] = {q_out_buf, 0, qkv_dim * sizeof(float)};
  bindings[6] = {k_cache_buf, 0, qkv_dim * sizeof(float)};
  bindings[7] = {v_cache_buf, 0, qkv_dim * sizeof(float)};

  uint32_t qkv_param = qkv_dim;
  hrx_status_t status =
      hrx_stream_dispatch(backend_.Stream(), layer_attn_executable_, 0, &config,
                          &qkv_param, sizeof(qkv_param), bindings, 8, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchLayerFFN(
    hrx_buffer_t input_buf, hrx_buffer_t gamma_buf, hrx_buffer_t w_down_buf,
    hrx_buffer_t residual_buf, hrx_buffer_t out_buf, uint32_t hidden_dim,
    uint32_t intermediate_dim) {
  if (layer_ffn_executable_ == nullptr) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = (hidden_dim + 1) / 2;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 544;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  hrx_buffer_ref_t bindings[5];
  bindings[0] = {input_buf, 0, intermediate_dim * sizeof(float)};
  bindings[1] = {gamma_buf, 0, intermediate_dim * sizeof(float)};
  bindings[2] = {w_down_buf, 0,
                 hidden_dim * intermediate_dim * sizeof(uint16_t)};
  bindings[3] = {residual_buf, 0, hidden_dim * sizeof(float)};
  bindings[4] = {out_buf, 0, hidden_dim * sizeof(float)};

  uint32_t hidden_param = hidden_dim;
  hrx_status_t status =
      hrx_stream_dispatch(backend_.Stream(), layer_ffn_executable_, 0, &config,
                          &hidden_param, sizeof(hidden_param), bindings, 5, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::BuildAndInstantiateDecodeGraph(
    hrx_buffer_t hidden_buf, hrx_buffer_t attn_gamma, hrx_buffer_t w_qkv,
    hrx_buffer_t cos_buf, hrx_buffer_t sin_buf, hrx_buffer_t q_out,
    hrx_buffer_t k_cache, hrx_buffer_t v_cache, hrx_buffer_t ffn_gamma,
    hrx_buffer_t w_down, hrx_buffer_t out_buf) {
  if (layer_attn_executable_ == nullptr || layer_ffn_executable_ == nullptr) {
    return false;
  }

  HrxGraphCaptureKey key{1, 1};
  if (!graph_executor_.InitializeGraph(backend_.Device(), key)) {
    return false;
  }

  // 1. Attention Layer Node
  hrx_dispatch_config_t attn_config{};
  attn_config.workgroup_count[0] = (8192 + 1) / 2;
  attn_config.workgroup_count[1] = 1;
  attn_config.workgroup_count[2] = 1;
  attn_config.workgroup_size[0] = 160;
  attn_config.workgroup_size[1] = 1;
  attn_config.workgroup_size[2] = 1;
  attn_config.subgroup_size = 32;

  hrx_buffer_ref_t attn_bindings[8];
  attn_bindings[0] = {hidden_buf, 0, 5120 * sizeof(float)};
  attn_bindings[1] = {attn_gamma, 0, 5120 * sizeof(float)};
  attn_bindings[2] = {w_qkv, 0, 8192 * 5120 * sizeof(uint16_t)};
  attn_bindings[3] = {cos_buf, 0, 64 * sizeof(float)};
  attn_bindings[4] = {sin_buf, 0, 64 * sizeof(float)};
  attn_bindings[5] = {q_out, 0, 8192 * sizeof(float)};
  attn_bindings[6] = {k_cache, 0, 8192 * sizeof(float)};
  attn_bindings[7] = {v_cache, 0, 8192 * sizeof(float)};

  uint32_t qkv_rows = 8192;
  hrx_graph_node_t attn_node = nullptr;
  if (!graph_executor_.AddKernelNode(
          layer_attn_executable_, 0, attn_config, attn_bindings, 8, &qkv_rows,
          sizeof(qkv_rows), nullptr, 0, &attn_node)) {
    return false;
  }

  // 2. FFN Layer Node (dependent on Attention Node)
  hrx_dispatch_config_t ffn_config{};
  ffn_config.workgroup_count[0] = (5120 + 1) / 2;
  ffn_config.workgroup_count[1] = 1;
  ffn_config.workgroup_count[2] = 1;
  ffn_config.workgroup_size[0] = 544;
  ffn_config.workgroup_size[1] = 1;
  ffn_config.workgroup_size[2] = 1;
  ffn_config.subgroup_size = 32;

  hrx_buffer_ref_t ffn_bindings[5];
  ffn_bindings[0] = {q_out, 0, 17408 * sizeof(float)};
  ffn_bindings[1] = {ffn_gamma, 0, 17408 * sizeof(float)};
  ffn_bindings[2] = {w_down, 0, 5120 * 17408 * sizeof(uint16_t)};
  ffn_bindings[3] = {hidden_buf, 0, 5120 * sizeof(float)};
  ffn_bindings[4] = {out_buf, 0, 5120 * sizeof(float)};

  uint32_t ffn_rows = 5120;
  hrx_graph_node_t ffn_node = nullptr;
  if (!graph_executor_.AddKernelNode(
          layer_ffn_executable_, 0, ffn_config, ffn_bindings, 5, &ffn_rows,
          sizeof(ffn_rows), &attn_node, 1, &ffn_node)) {
    return false;
  }

  return graph_executor_.Instantiate();
}

bool QwenHrxExecutor::ExecuteDecodeGraph() {
  if (!graph_executor_.IsInstantiated()) {
    return false;
  }
  return graph_executor_.Launch(backend_.Stream());
}

}  // namespace gufo::hrx
