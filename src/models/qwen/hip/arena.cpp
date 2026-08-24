#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace strix::hip {
namespace {

constexpr std::uint32_t kMaxPromptBatch = 4096;

}  // namespace

QwenGpuArena::QwenGpuArena(const core::ModelConfig& config,
                           std::uint32_t max_context)
    : config_(config),
      max_context_(std::max(max_context, 1U)),
      max_batch_(std::min(max_context_, kMaxPromptBatch)) {
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamCreate(&prefetch_stream));
  HIP_CHECK(hipEventCreateWithFlags(&prefetch_event, hipEventDisableTiming));
  HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
  HIPBLAS_CHECK(hipblasSetStream(hipblas_handle, stream));
  hipblaslt_gemm = std::make_unique<HipblasLtGemm>();

  const std::size_t hidden_size = config_.hidden_size;
  const std::size_t intermediate_size = config_.intermediate_size;
  const std::size_t vocab_size = config_.vocab_size;
  const std::size_t num_layers = config_.num_layers;
  const std::size_t num_kv_heads = config_.num_key_value_heads;
  const std::size_t head_dim = config_.head_dim;
  const std::size_t batch = max_batch_;
  const std::size_t attention_size = config_.AttentionSize();
  const std::size_t kv_size = num_kv_heads * head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config_.SsmQkvSize();
  const std::size_t ssm_inner_size = config_.ssm_inner_size;
  const std::size_t recurrent_width = std::max(attention_size, ssm_inner_size);
  const std::size_t projection_width =
      std::max(q_projection_size, ssm_qkv_size);
  const std::size_t time_step_rank = config_.ssm_time_step_rank;

  HIP_CHECK(hipMalloc(&d_hidden, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q, batch * attention_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, batch * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, batch * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_attn_out, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_gate, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_up, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_act, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_out, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_qkv, batch * projection_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out, batch * ssm_qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_gate, batch * recurrent_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_out, batch * recurrent_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_buf, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_buf, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_logits, vocab_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_prompt_tokens,
                      std::max<std::size_t>(batch, 2) * sizeof(std::uint32_t)));

  const std::size_t scratch_elements =
      batch * std::max<std::size_t>(
                  {intermediate_size, hidden_size, projection_width,
                   ssm_qkv_size + ssm_inner_size + (2 * time_step_rank)});
  HIP_CHECK(
      hipMalloc(&d_scratch_bf16, scratch_elements * sizeof(hip_bfloat16)));

  // Weight BF16 scratch: largest per-layer matmul weight in bf16 elements,
  // used by the prefill dequant-to-BF16 then BF16 GEMM path.
  const std::size_t max_weight_elems =
      hidden_size *
      std::max<std::size_t>({q_projection_size, kv_size, attention_size,
                             intermediate_size, ssm_qkv_size, ssm_inner_size,
                             time_step_rank});
  HIP_CHECK(
      hipMalloc(&d_weights_bf16, max_weight_elems * sizeof(hip_bfloat16)));

  const std::size_t total_kv = config_.FullAttentionLayerCount() *
                               num_kv_heads * max_context_ * head_dim;
  HIP_CHECK(hipMalloc(&d_kv_cache, total_kv * sizeof(float) * 2));
  HIP_CHECK(
      hipMalloc(&d_attention_kv_f16, total_kv * sizeof(std::uint16_t) * 2));

  const std::size_t total_conv =
      num_layers * ssm_qkv_size * config_.ssm_conv_kernel;
  HIP_CHECK(hipMalloc(&d_ssm_conv_state, total_conv * sizeof(float)));

  const std::size_t total_deltanet = num_layers * time_step_rank *
                                     config_.ssm_state_size *
                                     config_.SsmValueSize();
  HIP_CHECK(hipMalloc(&d_ssm_deltanet_state, total_deltanet * sizeof(float)));

  Reset();
}

QwenGpuArena::~QwenGpuArena() {
  FreeAll();
}

QwenGpuScratchView QwenGpuArena::GetScratchView(
    std::size_t batch_size) noexcept {
  const std::size_t batch = std::min<std::size_t>(batch_size, max_batch_);
  const std::size_t hidden = config_.hidden_size;
  const std::size_t attention = config_.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config_.num_key_value_heads) * config_.head_dim;
  const std::size_t q_projection = 2 * attention;
  const std::size_t ssm_qkv = config_.SsmQkvSize();
  const std::size_t recurrent = std::max<std::size_t>(
      attention, config_.ssm_inner_size);
  const std::size_t projection =
      std::max<std::size_t>(q_projection, ssm_qkv);
  const std::size_t bf16_scratch =
      batch * std::max<std::size_t>(
                  {config_.intermediate_size, hidden, projection,
                   ssm_qkv + config_.ssm_inner_size +
                       (2 * config_.ssm_time_step_rank)});
  const std::size_t weight_bf16 =
      hidden * std::max<std::size_t>(
                   {q_projection, kv, attention, config_.intermediate_size,
                    ssm_qkv, config_.ssm_inner_size,
                    config_.ssm_time_step_rank});
  return {
      .hidden = {d_hidden, batch * hidden},
      .normed = {d_normed, batch * hidden},
      .q = {d_q, batch * attention},
      .k = {d_k, batch * kv},
      .v = {d_v, batch * kv},
      .attention_out = {d_attn_out, batch * hidden},
      .ffn_gate = {d_ffn_gate, batch * config_.intermediate_size},
      .ffn_up = {d_ffn_up, batch * config_.intermediate_size},
      .ffn_activation = {d_ffn_act, batch * config_.intermediate_size},
      .ffn_out = {d_ffn_out, batch * hidden},
      .ssm_qkv = {d_ssm_qkv, batch * projection},
      .conv_out = {d_conv_out, batch * ssm_qkv},
      .ssm_gate = {d_ssm_gate, batch * recurrent},
      .ssm_out = {d_ssm_out, batch * recurrent},
      .alpha = {d_alpha_buf, batch * config_.ssm_time_step_rank},
      .beta = {d_beta_buf, batch * config_.ssm_time_step_rank},
      .logits = {d_logits, config_.vocab_size},
      .bf16 = {static_cast<hip_bfloat16*>(d_scratch_bf16), bf16_scratch},
      .weight_bf16 = {d_weights_bf16, weight_bf16},
      .prompt_tokens = {d_prompt_tokens,
                        std::max<std::size_t>(batch, 2)},
  };
}

QwenGpuArena::QwenGpuArena(QwenGpuArena&& other) noexcept
    : config_(other.config_),
      max_context_(other.max_context_),
      max_batch_(other.max_batch_) {
  d_hidden = other.d_hidden;
  d_normed = other.d_normed;
  d_q = other.d_q;
  d_k = other.d_k;
  d_v = other.d_v;
  d_attn_out = other.d_attn_out;
  d_ffn_gate = other.d_ffn_gate;
  d_ffn_up = other.d_ffn_up;
  d_ffn_act = other.d_ffn_act;
  d_ffn_out = other.d_ffn_out;
  d_ssm_qkv = other.d_ssm_qkv;
  d_conv_out = other.d_conv_out;
  d_ssm_gate = other.d_ssm_gate;
  d_ssm_out = other.d_ssm_out;
  d_alpha_buf = other.d_alpha_buf;
  d_beta_buf = other.d_beta_buf;
  d_logits = other.d_logits;
  d_attention_kv_f16 = other.d_attention_kv_f16;
  d_kv_cache = other.d_kv_cache;
  d_ssm_conv_state = other.d_ssm_conv_state;
  d_ssm_deltanet_state = other.d_ssm_deltanet_state;
  d_prompt_tokens = other.d_prompt_tokens;
  stream = other.stream;
  hipblas_handle = other.hipblas_handle;
  hipblaslt_gemm = std::move(other.hipblaslt_gemm);
  d_scratch_bf16 = other.d_scratch_bf16;
  d_weights_bf16 = other.d_weights_bf16;
  d_saved_ssm_conv_state_ = other.d_saved_ssm_conv_state_;
  d_saved_ssm_deltanet_state_ = other.d_saved_ssm_deltanet_state_;
  d_ssm_replay_qkv_ = other.d_ssm_replay_qkv_;
  d_ssm_replay_alpha_ = other.d_ssm_replay_alpha_;
  d_ssm_replay_beta_ = other.d_ssm_replay_beta_;
  d_ssm_replay_enabled_ = other.d_ssm_replay_enabled_;
  saved_context_ = other.saved_context_;
  replay_last_position_ = other.replay_last_position_;
  replay_captured_positions_ = other.replay_captured_positions_;
  has_saved_state_ = other.has_saved_state_;
  replay_capture_active_ = other.replay_capture_active_;

  other.d_hidden = nullptr;
  other.d_normed = nullptr;
  other.d_q = nullptr;
  other.d_k = nullptr;
  other.d_v = nullptr;
  other.d_attn_out = nullptr;
  other.d_ffn_gate = nullptr;
  other.d_ffn_up = nullptr;
  other.d_ffn_act = nullptr;
  other.d_ffn_out = nullptr;
  other.d_ssm_qkv = nullptr;
  other.d_conv_out = nullptr;
  other.d_ssm_gate = nullptr;
  other.d_ssm_out = nullptr;
  other.d_alpha_buf = nullptr;
  other.d_beta_buf = nullptr;
  other.d_logits = nullptr;
  other.d_attention_kv_f16 = nullptr;
  other.d_kv_cache = nullptr;
  other.d_ssm_conv_state = nullptr;
  other.d_ssm_deltanet_state = nullptr;
  other.d_prompt_tokens = nullptr;
  other.stream = nullptr;
  other.hipblas_handle = nullptr;
  other.d_scratch_bf16 = nullptr;
  other.d_weights_bf16 = nullptr;
  other.d_saved_ssm_conv_state_ = nullptr;
  other.d_saved_ssm_deltanet_state_ = nullptr;
  other.d_ssm_replay_qkv_ = nullptr;
  other.d_ssm_replay_alpha_ = nullptr;
  other.d_ssm_replay_beta_ = nullptr;
  other.d_ssm_replay_enabled_ = nullptr;
  other.saved_context_ = 0;
  other.replay_last_position_ = 0;
  other.replay_captured_positions_ = 0;
  other.has_saved_state_ = false;
  other.replay_capture_active_ = false;
}

QwenGpuArena& QwenGpuArena::operator=(QwenGpuArena&& other) noexcept {
  if (this != &other) {
    FreeAll();
    config_ = other.config_;
    max_context_ = other.max_context_;
    max_batch_ = other.max_batch_;
    d_hidden = other.d_hidden;
    d_normed = other.d_normed;
    d_q = other.d_q;
    d_k = other.d_k;
    d_v = other.d_v;
    d_attn_out = other.d_attn_out;
    d_ffn_gate = other.d_ffn_gate;
    d_ffn_up = other.d_ffn_up;
    d_ffn_act = other.d_ffn_act;
    d_ffn_out = other.d_ffn_out;
    d_ssm_qkv = other.d_ssm_qkv;
    d_conv_out = other.d_conv_out;
    d_ssm_gate = other.d_ssm_gate;
    d_ssm_out = other.d_ssm_out;
    d_alpha_buf = other.d_alpha_buf;
    d_beta_buf = other.d_beta_buf;
    d_logits = other.d_logits;
    d_attention_kv_f16 = other.d_attention_kv_f16;
    d_kv_cache = other.d_kv_cache;
    d_ssm_conv_state = other.d_ssm_conv_state;
    d_ssm_deltanet_state = other.d_ssm_deltanet_state;
    d_prompt_tokens = other.d_prompt_tokens;
    stream = other.stream;
    hipblas_handle = other.hipblas_handle;
    hipblaslt_gemm = std::move(other.hipblaslt_gemm);
    d_scratch_bf16 = other.d_scratch_bf16;
    d_weights_bf16 = other.d_weights_bf16;
    d_saved_ssm_conv_state_ = other.d_saved_ssm_conv_state_;
    d_saved_ssm_deltanet_state_ = other.d_saved_ssm_deltanet_state_;
    d_ssm_replay_qkv_ = other.d_ssm_replay_qkv_;
    d_ssm_replay_alpha_ = other.d_ssm_replay_alpha_;
    d_ssm_replay_beta_ = other.d_ssm_replay_beta_;
    d_ssm_replay_enabled_ = other.d_ssm_replay_enabled_;
    saved_context_ = other.saved_context_;
    replay_last_position_ = other.replay_last_position_;
    replay_captured_positions_ = other.replay_captured_positions_;
    has_saved_state_ = other.has_saved_state_;
    replay_capture_active_ = other.replay_capture_active_;

    other.d_hidden = nullptr;
    other.d_normed = nullptr;
    other.d_q = nullptr;
    other.d_k = nullptr;
    other.d_v = nullptr;
    other.d_attn_out = nullptr;
    other.d_ffn_gate = nullptr;
    other.d_ffn_up = nullptr;
    other.d_ffn_act = nullptr;
    other.d_ffn_out = nullptr;
    other.d_ssm_qkv = nullptr;
    other.d_conv_out = nullptr;
    other.d_ssm_gate = nullptr;
    other.d_ssm_out = nullptr;
    other.d_alpha_buf = nullptr;
    other.d_beta_buf = nullptr;
    other.d_logits = nullptr;
    other.d_attention_kv_f16 = nullptr;
    other.d_kv_cache = nullptr;
    other.d_ssm_conv_state = nullptr;
    other.d_ssm_deltanet_state = nullptr;
    other.d_prompt_tokens = nullptr;
    other.stream = nullptr;
    other.hipblas_handle = nullptr;
    other.d_scratch_bf16 = nullptr;
    other.d_weights_bf16 = nullptr;
    other.d_saved_ssm_conv_state_ = nullptr;
    other.d_saved_ssm_deltanet_state_ = nullptr;
    other.d_ssm_replay_qkv_ = nullptr;
    other.d_ssm_replay_alpha_ = nullptr;
    other.d_ssm_replay_beta_ = nullptr;
    other.d_ssm_replay_enabled_ = nullptr;
    other.saved_context_ = 0;
    other.replay_last_position_ = 0;
    other.replay_captured_positions_ = 0;
    other.has_saved_state_ = false;
    other.replay_capture_active_ = false;
  }
  return *this;
}

void QwenGpuArena::Reset() noexcept {
  const std::size_t num_layers = config_.num_layers;
  const std::size_t num_kv_heads = config_.num_key_value_heads;
  const std::size_t head_dim = config_.head_dim;
  const std::size_t total_kv = config_.FullAttentionLayerCount() *
                               num_kv_heads * max_context_ * head_dim * 2;
  const std::size_t total_conv =
      num_layers * config_.SsmQkvSize() * config_.ssm_conv_kernel;
  const std::size_t total_deltanet = num_layers * config_.ssm_time_step_rank *
                                     config_.ssm_state_size *
                                     config_.SsmValueSize();

  if (d_kv_cache != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_kv_cache, 0, total_kv * sizeof(float), stream));
  }
  if (d_ssm_conv_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_ssm_conv_state, 0, total_conv * sizeof(float),
                             stream));
  }
  if (d_ssm_deltanet_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_ssm_deltanet_state, 0,
                             total_deltanet * sizeof(float), stream));
  }
  DisableSsmReplayCapture();
  saved_context_ = 0;
  replay_last_position_ = 0;
  replay_captured_positions_ = 0;
  has_saved_state_ = false;
}

void QwenGpuArena::AllocateRecurrentSnapshot() {
  if (d_saved_ssm_conv_state_ != nullptr) {
    return;
  }

  const std::size_t total_conv =
      config_.num_layers * config_.SsmQkvSize() * config_.ssm_conv_kernel;
  const std::size_t total_deltanet =
      config_.num_layers * config_.ssm_time_step_rank * config_.ssm_state_size *
      config_.SsmValueSize();

  HIP_CHECK(hipMalloc(&d_saved_ssm_conv_state_, total_conv * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_saved_ssm_deltanet_state_, total_deltanet * sizeof(float)));
}

void QwenGpuArena::SaveState(std::uint32_t valid_context) {
  if (valid_context > max_context_) {
    throw std::length_error("saved GPU state exceeds the context length");
  }
  AllocateRecurrentSnapshot();

  const std::size_t total_conv =
      config_.num_layers * config_.SsmQkvSize() * config_.ssm_conv_kernel;
  const std::size_t total_deltanet =
      config_.num_layers * config_.ssm_time_step_rank * config_.ssm_state_size *
      config_.SsmValueSize();

  // KV entries are append-only and every attention launch is bounded by its
  // explicit position. Draft entries beyond valid_context can remain in place:
  // accepted positions reuse them and rejected positions are overwritten.
  HIP_CHECK(hipMemcpyAsync(d_saved_ssm_conv_state_, d_ssm_conv_state,
                           total_conv * sizeof(float), hipMemcpyDeviceToDevice,
                           stream));
  HIP_CHECK(hipMemcpyAsync(d_saved_ssm_deltanet_state_, d_ssm_deltanet_state,
                           total_deltanet * sizeof(float),
                           hipMemcpyDeviceToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  saved_context_ = valid_context;
  replay_last_position_ = valid_context;
  replay_captured_positions_ = 0;
  has_saved_state_ = true;
}

void QwenGpuArena::RestoreState() {
  if (!has_saved_state_) {
    throw std::logic_error("GPU state has not been saved");
  }

  const std::size_t total_conv =
      config_.num_layers * config_.SsmQkvSize() * config_.ssm_conv_kernel;
  const std::size_t total_deltanet =
      config_.num_layers * config_.ssm_time_step_rank * config_.ssm_state_size *
      config_.SsmValueSize();

  HIP_CHECK(hipMemcpyAsync(d_ssm_conv_state, d_saved_ssm_conv_state_,
                           total_conv * sizeof(float), hipMemcpyDeviceToDevice,
                           stream));
  HIP_CHECK(hipMemcpyAsync(d_ssm_deltanet_state, d_saved_ssm_deltanet_state_,
                           total_deltanet * sizeof(float),
                           hipMemcpyDeviceToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
}

bool QwenGpuArena::AllocateSsmReplayLog() {
  if (d_ssm_replay_qkv_ != nullptr) {
    return false;
  }

  const std::size_t layer_slots =
      static_cast<std::size_t>(config_.num_layers) * kSsmReplayCapacity;
  HIP_CHECK(hipMalloc(&d_ssm_replay_qkv_,
                      layer_slots * config_.SsmQkvSize() * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_ssm_replay_alpha_,
                layer_slots * config_.ssm_time_step_rank * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_ssm_replay_beta_,
                layer_slots * config_.ssm_time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_replay_enabled_, sizeof(std::uint32_t)));
  return true;
}

bool QwenGpuArena::BeginSsmReplayCapture() {
  const bool allocated = AllocateSsmReplayLog();
  HIP_CHECK(
      hipMemsetAsync(d_ssm_replay_enabled_, 1, sizeof(std::uint32_t), stream));
  replay_capture_active_ = true;
  replay_last_position_ = saved_context_;
  replay_captured_positions_ = 0;
  return allocated;
}

void QwenGpuArena::DisableSsmReplayCapture() noexcept {
  if (d_ssm_replay_enabled_ != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_ssm_replay_enabled_, 0, sizeof(std::uint32_t),
                             stream));
  }
  replay_capture_active_ = false;
}

void QwenGpuArena::MarkSsmReplayPosition(std::uint32_t position) noexcept {
  if (replay_capture_active_ && position >= saved_context_) {
    replay_last_position_ = std::max(replay_last_position_, position);
    replay_captured_positions_ =
        std::max(replay_captured_positions_,
                 static_cast<std::size_t>(position - saved_context_) + 1);
  }
}

bool QwenGpuArena::CanReplaySsmPosition(std::uint32_t position) const noexcept {
  if (!has_saved_state_ || d_ssm_replay_qkv_ == nullptr ||
      replay_captured_positions_ == 0) {
    return false;
  }
  return replay_captured_positions_ <= kSsmReplayCapacity &&
         position >= saved_context_ && position <= replay_last_position_;
}

SsmReplayCapture QwenGpuArena::GetSsmReplayCapture() const noexcept {
  return {
      .qkv = d_ssm_replay_qkv_,
      .alpha = d_ssm_replay_alpha_,
      .beta = d_ssm_replay_beta_,
      .position = d_prompt_tokens + 1,
      .enabled = d_ssm_replay_enabled_,
  };
}

const float* QwenGpuArena::GetReplayQkv(std::uint32_t layer,
                                        std::uint32_t position) const {
  if (!CanReplaySsmPosition(position)) {
    throw std::out_of_range("SSM replay position is outside the capture");
  }
  const std::size_t slot =
      static_cast<std::size_t>(position) % kSsmReplayCapacity;
  const std::size_t offset =
      (static_cast<std::size_t>(layer) * kSsmReplayCapacity + slot) *
      config_.SsmQkvSize();
  return d_ssm_replay_qkv_ + offset;
}

const float* QwenGpuArena::GetReplayAlpha(std::uint32_t layer,
                                          std::uint32_t position) const {
  if (!CanReplaySsmPosition(position)) {
    throw std::out_of_range("SSM replay position is outside the capture");
  }
  const std::size_t slot =
      static_cast<std::size_t>(position) % kSsmReplayCapacity;
  const std::size_t offset =
      (static_cast<std::size_t>(layer) * kSsmReplayCapacity + slot) *
      config_.ssm_time_step_rank;
  return d_ssm_replay_alpha_ + offset;
}

const float* QwenGpuArena::GetReplayBeta(std::uint32_t layer,
                                         std::uint32_t position) const {
  if (!CanReplaySsmPosition(position)) {
    throw std::out_of_range("SSM replay position is outside the capture");
  }
  const std::size_t slot =
      static_cast<std::size_t>(position) % kSsmReplayCapacity;
  const std::size_t offset =
      (static_cast<std::size_t>(layer) * kSsmReplayCapacity + slot) *
      config_.ssm_time_step_rank;
  return d_ssm_replay_beta_ + offset;
}

void QwenGpuArena::FreeAll() noexcept {
  if (d_hidden != nullptr)
    HIP_CHECK(hipFree(d_hidden));
  if (d_normed != nullptr)
    HIP_CHECK(hipFree(d_normed));
  if (d_q != nullptr)
    HIP_CHECK(hipFree(d_q));
  if (d_k != nullptr)
    HIP_CHECK(hipFree(d_k));
  if (d_v != nullptr)
    HIP_CHECK(hipFree(d_v));
  if (d_attn_out != nullptr)
    HIP_CHECK(hipFree(d_attn_out));
  if (d_ffn_gate != nullptr)
    HIP_CHECK(hipFree(d_ffn_gate));
  if (d_ffn_up != nullptr)
    HIP_CHECK(hipFree(d_ffn_up));
  if (d_ffn_act != nullptr)
    HIP_CHECK(hipFree(d_ffn_act));
  if (d_ffn_out != nullptr)
    HIP_CHECK(hipFree(d_ffn_out));
  if (d_ssm_qkv != nullptr)
    HIP_CHECK(hipFree(d_ssm_qkv));
  if (d_conv_out != nullptr)
    HIP_CHECK(hipFree(d_conv_out));
  if (d_ssm_gate != nullptr)
    HIP_CHECK(hipFree(d_ssm_gate));
  if (d_ssm_out != nullptr)
    HIP_CHECK(hipFree(d_ssm_out));
  if (d_alpha_buf != nullptr)
    HIP_CHECK(hipFree(d_alpha_buf));
  if (d_beta_buf != nullptr)
    HIP_CHECK(hipFree(d_beta_buf));
  if (d_logits != nullptr)
    HIP_CHECK(hipFree(d_logits));
  if (d_attention_kv_f16 != nullptr)
    HIP_CHECK(hipFree(d_attention_kv_f16));
  if (d_kv_cache != nullptr)
    HIP_CHECK(hipFree(d_kv_cache));
  if (d_ssm_conv_state != nullptr)
    HIP_CHECK(hipFree(d_ssm_conv_state));
  if (d_ssm_deltanet_state != nullptr)
    HIP_CHECK(hipFree(d_ssm_deltanet_state));
  if (d_prompt_tokens != nullptr)
    HIP_CHECK(hipFree(d_prompt_tokens));
  if (d_scratch_bf16 != nullptr)
    HIP_CHECK(hipFree(d_scratch_bf16));
  if (d_weights_bf16 != nullptr)
    HIP_CHECK(hipFree(d_weights_bf16));
  if (d_saved_ssm_conv_state_ != nullptr)
    HIP_CHECK(hipFree(d_saved_ssm_conv_state_));
  if (d_saved_ssm_deltanet_state_ != nullptr)
    HIP_CHECK(hipFree(d_saved_ssm_deltanet_state_));
  if (d_ssm_replay_qkv_ != nullptr)
    HIP_CHECK(hipFree(d_ssm_replay_qkv_));
  if (d_ssm_replay_alpha_ != nullptr)
    HIP_CHECK(hipFree(d_ssm_replay_alpha_));
  if (d_ssm_replay_beta_ != nullptr)
    HIP_CHECK(hipFree(d_ssm_replay_beta_));
  if (d_ssm_replay_enabled_ != nullptr)
    HIP_CHECK(hipFree(d_ssm_replay_enabled_));
  if (hipblas_handle != nullptr)
    HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));
  hipblaslt_gemm.reset();
  if (stream != nullptr)
    HIP_CHECK(hipStreamDestroy(stream));
  if (prefetch_stream != nullptr)
    HIP_CHECK(hipStreamDestroy(prefetch_stream));
  if (prefetch_event != nullptr)
    HIP_CHECK(hipEventDestroy(prefetch_event));

  d_hidden = nullptr;
  d_normed = nullptr;
  d_q = nullptr;
  d_k = nullptr;
  d_v = nullptr;
  d_attn_out = nullptr;
  d_ffn_gate = nullptr;
  d_ffn_up = nullptr;
  d_ffn_act = nullptr;
  d_ffn_out = nullptr;
  d_ssm_qkv = nullptr;
  d_conv_out = nullptr;
  d_ssm_gate = nullptr;
  d_ssm_out = nullptr;
  d_alpha_buf = nullptr;
  d_beta_buf = nullptr;
  d_logits = nullptr;
  d_attention_kv_f16 = nullptr;
  d_kv_cache = nullptr;
  d_ssm_conv_state = nullptr;
  d_ssm_deltanet_state = nullptr;
  d_prompt_tokens = nullptr;
  d_scratch_bf16 = nullptr;
  d_weights_bf16 = nullptr;
  d_saved_ssm_conv_state_ = nullptr;
  d_saved_ssm_deltanet_state_ = nullptr;
  d_ssm_replay_qkv_ = nullptr;
  d_ssm_replay_alpha_ = nullptr;
  d_ssm_replay_beta_ = nullptr;
  d_ssm_replay_enabled_ = nullptr;
  saved_context_ = 0;
  replay_last_position_ = 0;
  replay_captured_positions_ = 0;
  has_saved_state_ = false;
  replay_capture_active_ = false;
  hipblas_handle = nullptr;
  stream = nullptr;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
