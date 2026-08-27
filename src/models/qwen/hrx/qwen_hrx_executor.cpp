#include "src/models/qwen/hrx/qwen_hrx_executor.hpp"

#include <array>
#include <cstddef>
#include <filesystem>

namespace gufo::hrx {

QwenHrxExecutor::QwenHrxExecutor(QwenHrxArtifactContract contract,
                                 int device_index)
    : contract_(contract), backend_(HrxBackend::Instance()) {
  (void)backend_.Initialize(device_index);
}

QwenHrxExecutor::~QwenHrxExecutor() {
  loader_.UnloadAll();
}

void QwenHrxExecutor::ResetKernelState() {
  swiglu_executable_ = nullptr;
  rmsnorm_qkv_executable_ = nullptr;
  rope_kv_executable_ = nullptr;
  deltanet_recurrence_executable_ = nullptr;
  down_residual_executable_ = nullptr;
  final_norm_head_executable_ = nullptr;
  layer_attn_executable_ = nullptr;
  layer_ffn_executable_ = nullptr;
  missing_kernel_artifacts_.clear();
  is_ready_ = false;
}

bool QwenHrxExecutor::Initialize(const std::string& loom_artifact_path) {
  loader_.UnloadAll();
  ResetKernelState();
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

  missing_kernel_artifacts_ = {
      "qwen_fused_rmsnorm_qkv_bf16.fb",
      "qwen_fused_rope_kv_cache_bf16.fb",
      "qwen_fused_deltanet_recurrence_bf16.fb",
      "qwen_fused_down_residual_bf16.fb",
      "qwen_fused_final_norm_head_bf16.fb",
      "qwen_fused_layer_attn_bf16.fb",
      "qwen_fused_layer_ffn_bf16.fb",
  };
  return true;
}

bool QwenHrxExecutor::InitializeAllKernels(const std::string& kernels_dir) {
  loader_.UnloadAll();
  ResetKernelState();
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
    if (!std::filesystem::exists(found_path)) {
      missing_kernel_artifacts_.emplace_back(spec.filename);
      continue;
    }

    hrx_status_t status = loader_.LoadFromFile(
        backend_.Device(), spec.name, found_path, "amdgpu", "gfx1151");
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      missing_kernel_artifacts_.emplace_back(spec.filename);
      continue;
    }

    *spec.target_ptr = loader_.GetExecutable(spec.name);
    if (*spec.target_ptr == nullptr) {
      missing_kernel_artifacts_.emplace_back(spec.filename);
    }
  }

  is_ready_ = missing_kernel_artifacts_.empty();
  return is_ready_;
}

bool QwenHrxExecutor::DispatchSwiGLU(hrx_buffer_t input_buf,
                                     hrx_buffer_t gate_buf, hrx_buffer_t up_buf,
                                     hrx_buffer_t out_buf, uint32_t num_rows) {
  if (swiglu_executable_ == nullptr || num_rows < 2 ||
      num_rows > contract_.FfnSize() || (num_rows % 2) != 0) {
    return false;
  }

  const std::uint32_t hidden_dim = contract_.HiddenSize();
  const std::size_t input_bytes =
      static_cast<std::size_t>(hidden_dim) * sizeof(float);
  const std::size_t weight_bytes = static_cast<std::size_t>(num_rows) *
                                   hidden_dim * sizeof(std::uint16_t);
  const std::size_t output_bytes =
      static_cast<std::size_t>(num_rows) * sizeof(float);
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = num_rows / 2;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 160;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  hrx_buffer_ref_t bindings[4];
  bindings[0] = {input_buf, 0, input_bytes};
  bindings[1] = {gate_buf, 0, weight_bytes};
  bindings[2] = {up_buf, 0, weight_bytes};
  bindings[3] = {out_buf, 0, output_bytes};

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
    hrx_buffer_t out_buf, uint32_t num_rows) {
  if (rmsnorm_qkv_executable_ == nullptr || num_rows < 2 ||
      num_rows > contract_.SsmQkvWidth() || (num_rows % 2) != 0) {
    return false;
  }
  const std::uint32_t hidden_dim = contract_.HiddenSize();
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
    hrx_buffer_t v_cache_buf) {
  if (rope_kv_executable_ == nullptr) {
    return false;
  }
  const std::uint32_t q_heads = contract_.QHeadCount();
  const std::uint32_t kv_heads = contract_.KvHeadCount();
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = q_heads;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = contract_.RotaryDim() / 2;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  hrx_buffer_ref_t bindings[7];
  bindings[0] = {q_buf, 0, contract_.QWidth() * sizeof(float)};
  bindings[1] = {k_buf, 0, contract_.KWidth() * sizeof(float)};
  bindings[2] = {v_buf, 0, contract_.VWidth() * sizeof(float)};
  bindings[3] = {cos_buf, 0, (contract_.RotaryDim() / 2) * sizeof(float)};
  bindings[4] = {sin_buf, 0, (contract_.RotaryDim() / 2) * sizeof(float)};
  bindings[5] = {k_cache_buf, 0, contract_.KWidth() * sizeof(float)};
  bindings[6] = {v_cache_buf, 0, contract_.VWidth() * sizeof(float)};

  const std::array<std::uint32_t, 2> head_params{q_heads, kv_heads};
  hrx_status_t status =
      hrx_stream_dispatch(backend_.Stream(), rope_kv_executable_, 0, &config,
                          head_params.data(), sizeof(head_params), bindings, 7,
                          0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchDeltaNetRecurrence(
    hrx_buffer_t state_buf, hrx_buffer_t q_buf, hrx_buffer_t v_buf,
    hrx_buffer_t beta_buf, hrx_buffer_t out_buf) {
  if (deltanet_recurrence_executable_ == nullptr) {
    return false;
  }
  const std::uint32_t num_heads = contract_.SsmHeadCount();
  const std::uint32_t state_size = contract_.SsmStateSize();
  const std::uint32_t value_size = contract_.SsmValueSize();
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = num_heads;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  hrx_buffer_ref_t bindings[5];
  bindings[0] = {state_buf, 0,
                 num_heads * value_size * state_size * sizeof(float)};
  bindings[1] = {q_buf, 0,
                 num_heads * state_size * sizeof(std::uint16_t)};
  bindings[2] = {v_buf, 0, num_heads * value_size * sizeof(float)};
  bindings[3] = {beta_buf, 0, num_heads * sizeof(float)};
  bindings[4] = {out_buf, 0, num_heads * value_size * sizeof(float)};

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
    hrx_buffer_t out_buf) {
  if (down_residual_executable_ == nullptr) {
    return false;
  }
  const std::uint32_t hidden_dim = contract_.HiddenSize();
  const std::uint32_t intermediate_dim = contract_.FfnSize();
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
    hrx_buffer_t logits_buf) {
  if (final_norm_head_executable_ == nullptr) {
    return false;
  }
  const std::uint32_t vocab_size = contract_.VocabSize();
  const std::uint32_t hidden_dim = contract_.HiddenSize();
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
    hrx_buffer_t k_cache_buf, hrx_buffer_t v_cache_buf) {
  if (layer_attn_executable_ == nullptr) {
    return false;
  }
  const std::uint32_t qkv_dim = contract_.QkvWidth();
  const std::uint32_t hidden_dim = contract_.HiddenSize();
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
  bindings[3] = {cos_buf, 0,
                 (contract_.RotaryDim() / 2) * sizeof(float)};
  bindings[4] = {sin_buf, 0,
                 (contract_.RotaryDim() / 2) * sizeof(float)};
  bindings[5] = {q_out_buf, 0, contract_.QWidth() * sizeof(float)};
  bindings[6] = {k_cache_buf, 0, contract_.KWidth() * sizeof(float)};
  bindings[7] = {v_cache_buf, 0, contract_.VWidth() * sizeof(float)};

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
    hrx_buffer_t residual_buf, hrx_buffer_t out_buf) {
  if (layer_ffn_executable_ == nullptr) {
    return false;
  }
  const std::uint32_t hidden_dim = contract_.HiddenSize();
  const std::uint32_t intermediate_dim = contract_.FfnSize();
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

  const std::uint32_t hidden_dim = contract_.HiddenSize();
  const std::uint32_t ffn_dim = contract_.FfnSize();
  const std::uint32_t qkv_dim = contract_.QkvWidth();

  // 1. Attention Layer Node
  hrx_dispatch_config_t attn_config{};
  attn_config.workgroup_count[0] = (qkv_dim + 1) / 2;
  attn_config.workgroup_count[1] = 1;
  attn_config.workgroup_count[2] = 1;
  attn_config.workgroup_size[0] = 160;
  attn_config.workgroup_size[1] = 1;
  attn_config.workgroup_size[2] = 1;
  attn_config.subgroup_size = 32;

  hrx_buffer_ref_t attn_bindings[8];
  attn_bindings[0] = {hidden_buf, 0, hidden_dim * sizeof(float)};
  attn_bindings[1] = {attn_gamma, 0, hidden_dim * sizeof(float)};
  attn_bindings[2] = {w_qkv, 0,
                      qkv_dim * hidden_dim * sizeof(std::uint16_t)};
  attn_bindings[3] = {cos_buf, 0,
                      (contract_.RotaryDim() / 2) * sizeof(float)};
  attn_bindings[4] = {sin_buf, 0,
                      (contract_.RotaryDim() / 2) * sizeof(float)};
  attn_bindings[5] = {q_out, 0, contract_.QWidth() * sizeof(float)};
  attn_bindings[6] = {k_cache, 0, contract_.KWidth() * sizeof(float)};
  attn_bindings[7] = {v_cache, 0, contract_.VWidth() * sizeof(float)};

  std::uint32_t qkv_rows = qkv_dim;
  hrx_graph_node_t attn_node = nullptr;
  if (!graph_executor_.AddKernelNode(
          layer_attn_executable_, 0, attn_config, attn_bindings, 8, &qkv_rows,
          sizeof(qkv_rows), nullptr, 0, &attn_node)) {
    return false;
  }

  // 2. FFN Layer Node (dependent on Attention Node)
  hrx_dispatch_config_t ffn_config{};
  ffn_config.workgroup_count[0] = (hidden_dim + 1) / 2;
  ffn_config.workgroup_count[1] = 1;
  ffn_config.workgroup_count[2] = 1;
  ffn_config.workgroup_size[0] = 544;
  ffn_config.workgroup_size[1] = 1;
  ffn_config.workgroup_size[2] = 1;
  ffn_config.subgroup_size = 32;

  hrx_buffer_ref_t ffn_bindings[5];
  ffn_bindings[0] = {q_out, 0, ffn_dim * sizeof(float)};
  ffn_bindings[1] = {ffn_gamma, 0, ffn_dim * sizeof(float)};
  ffn_bindings[2] = {w_down, 0,
                     hidden_dim * ffn_dim * sizeof(std::uint16_t)};
  ffn_bindings[3] = {hidden_buf, 0, hidden_dim * sizeof(float)};
  ffn_bindings[4] = {out_buf, 0, hidden_dim * sizeof(float)};

  std::uint32_t ffn_rows = hidden_dim;
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
