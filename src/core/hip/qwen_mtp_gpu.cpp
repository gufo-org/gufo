#if defined(ENGINE_ENABLE_HIP)
#include "src/core/hip/qwen_mtp_gpu.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/hip/detail/qwen_attention_policy.hpp"
#include "src/core/hip/qwen_gpu_ops.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#if defined(ENGINE_ENABLE_XRT)
#include "src/core/diagnostics/system_inventory.h"
#include "src/core/xdna2/device.h"
#endif

namespace strix::hip {
namespace {

constexpr std::size_t kPackChunkRows = 32;

[[nodiscard]] std::uint16_t FloatToBfloat16Bits(float value) noexcept {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounding_bias = 0x7FFFU + ((bits >> 16U) & 1U);
  bits += rounding_bias;
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] const void* QuantizedRow(const models::QwenTensorRef& tensor,
                                       std::size_t row,
                                       std::size_t columns) noexcept {
  const std::size_t row_bytes = quant::QuantizedRowBytes(tensor.type, columns);
  if (row_bytes == 0) {
    return nullptr;
  }
  return static_cast<const std::uint8_t*>(tensor.data) + (row * row_bytes);
}

void ReadMatrixRow(const models::QwenTensorRef& tensor, std::size_t row,
                   std::size_t columns, float* output) {
  if (tensor.type == core::GgmlType::kF32) {
    const auto* source =
        static_cast<const float*>(tensor.data) + (row * columns);
    std::copy_n(source, columns, output);
    return;
  }
  if (tensor.type == core::GgmlType::kBF16) {
    const std::size_t offset = row * columns;
    for (std::size_t column = 0; column < columns; ++column) {
      const auto bits =
          static_cast<const std::uint16_t*>(tensor.data)[offset + column];
      output[column] =
          std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    }
    return;
  }

  const void* source = QuantizedRow(tensor, row, columns);
  if (source == nullptr) {
    throw std::runtime_error("unsupported MTP matrix quantization");
  }
  switch (tensor.type) {
    case core::GgmlType::kQ3_K:
      quant::DequantizeQ3_K(source, output, columns);
      return;
    case core::GgmlType::kQ4_K:
      quant::DequantizeQ4_K(source, output, columns);
      return;
    case core::GgmlType::kQ6_K:
      quant::DequantizeQ6_K(source, output, columns);
      return;
    default:
      throw std::runtime_error("unsupported MTP matrix type");
  }
}

[[nodiscard]] void* AllocateDevice(std::size_t bytes) {
  void* pointer = nullptr;
  const auto error = hipMalloc(&pointer, bytes);
  if (error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hipMalloc failed: ") +
                             hipGetErrorString(error));
  }
  return pointer;
}

void CopyToDevice(void* destination, const void* source, std::size_t bytes) {
  const auto error =
      hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice);
  if (error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hipMemcpy failed: ") +
                             hipGetErrorString(error));
  }
}

models::QwenTensorRef CopyVectorF32(const models::QwenTensorRef& source,
                                    std::size_t elements,
                                    std::vector<void*>& allocations,
                                    std::size_t& packed_bytes) {
  std::vector<float> host(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    host[index] = source.Get(index);
  }
  const std::size_t bytes = elements * sizeof(float);
  void* device = AllocateDevice(bytes);
  allocations.push_back(device);
  CopyToDevice(device, host.data(), bytes);
  packed_bytes += bytes;
  return {
      .data = device,
      .type = core::GgmlType::kF32,
      .num_elements = elements,
  };
}

models::QwenTensorRef PackMatrixBf16(const models::QwenTensorRef& source,
                                     std::size_t rows, std::size_t columns,
                                     std::vector<void*>& allocations,
                                     std::size_t& packed_bytes) {
  const std::size_t total_elements = rows * columns;
  const std::size_t total_bytes = total_elements * sizeof(std::uint16_t);
  void* device = AllocateDevice(total_bytes);
  allocations.push_back(device);

  const std::size_t chunk_rows = std::min(rows, kPackChunkRows);
  std::vector<float> float_rows(chunk_rows * columns);
  std::vector<std::uint16_t> bf16_rows(chunk_rows * columns);
  for (std::size_t start = 0; start < rows; start += chunk_rows) {
    const std::size_t count = std::min(chunk_rows, rows - start);
    for (std::size_t local_row = 0; local_row < count; ++local_row) {
      ReadMatrixRow(source, start + local_row, columns,
                    float_rows.data() + (local_row * columns));
    }
    const std::size_t chunk_elements = count * columns;
    for (std::size_t index = 0; index < chunk_elements; ++index) {
      bf16_rows[index] = FloatToBfloat16Bits(float_rows[index]);
    }
    CopyToDevice(static_cast<std::uint8_t*>(device) +
                     (start * columns * sizeof(std::uint16_t)),
                 bf16_rows.data(), chunk_elements * sizeof(std::uint16_t));
  }

  packed_bytes += total_bytes;
  return {
      .data = device,
      .type = core::GgmlType::kBF16,
      .num_elements = total_elements,
  };
}

void ReleaseAllocations(std::vector<void*>& allocations) noexcept {
  for (void* allocation : allocations) {
    if (allocation != nullptr) {
      (void)hipFree(allocation);
    }
  }
  allocations.clear();
}

void PackPrivateWeights(speculative::QwenMtpWeights& weights,
                        std::vector<void*>& allocations,
                        std::size_t& packed_bytes) {
  const std::size_t hidden = weights.config.hidden_size;
  const std::size_t intermediate = weights.config.intermediate_size;
  const std::size_t attention = weights.config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(weights.config.num_key_value_heads) *
      weights.config.head_dim;

  weights.embedding_norm =
      CopyVectorF32(weights.embedding_norm, hidden, allocations, packed_bytes);
  weights.hidden_norm =
      CopyVectorF32(weights.hidden_norm, hidden, allocations, packed_bytes);
  weights.shared_head_norm = CopyVectorF32(weights.shared_head_norm, hidden,
                                           allocations, packed_bytes);
  weights.fusion_projection = PackMatrixBf16(
      weights.fusion_projection, hidden, 2 * hidden, allocations, packed_bytes);

  auto& layer = weights.layer;
  layer.attn_norm =
      CopyVectorF32(layer.attn_norm, hidden, allocations, packed_bytes);
  layer.attn_q = PackMatrixBf16(layer.attn_q, 2 * attention, hidden,
                                allocations, packed_bytes);
  layer.attn_k =
      PackMatrixBf16(layer.attn_k, kv, hidden, allocations, packed_bytes);
  layer.attn_v =
      PackMatrixBf16(layer.attn_v, kv, hidden, allocations, packed_bytes);
  layer.attn_output = PackMatrixBf16(layer.attn_output, hidden, attention,
                                     allocations, packed_bytes);
  layer.attn_q_norm = CopyVectorF32(layer.attn_q_norm, weights.config.head_dim,
                                    allocations, packed_bytes);
  layer.attn_k_norm = CopyVectorF32(layer.attn_k_norm, weights.config.head_dim,
                                    allocations, packed_bytes);
  layer.ffn_norm =
      CopyVectorF32(layer.ffn_norm, hidden, allocations, packed_bytes);
  layer.ffn_gate = PackMatrixBf16(layer.ffn_gate, intermediate, hidden,
                                  allocations, packed_bytes);
  layer.ffn_up = PackMatrixBf16(layer.ffn_up, intermediate, hidden, allocations,
                                packed_bytes);
  layer.ffn_down = PackMatrixBf16(layer.ffn_down, hidden, intermediate,
                                  allocations, packed_bytes);
}

template<typename T>
void AllocateBuffer(T*& pointer, std::size_t elements) {
  pointer = static_cast<T*>(AllocateDevice(elements * sizeof(T)));
}

}  // namespace

QwenMtpGpuModel::QwenMtpGpuModel(
    std::shared_ptr<const core::GgufReader> mtp_reader,
    std::shared_ptr<const QwenGpuModel> target_model,
    speculative::QwenMtpWeights weights,
    models::QwenTensorRef raw_fusion_projection, std::vector<void*> allocations,
    std::size_t packed_weight_bytes, double pack_time_seconds)
    : mtp_reader_(std::move(mtp_reader)),
      target_model_(std::move(target_model)),
      weights_(std::move(weights)),
      raw_fusion_projection_(raw_fusion_projection),
      allocations_(std::move(allocations)),
      packed_weight_bytes_(packed_weight_bytes),
      pack_time_seconds_(pack_time_seconds) {}

QwenMtpGpuModel::~QwenMtpGpuModel() {
  ReleaseAllocations(allocations_);
}

std::shared_ptr<const QwenMtpGpuModel> QwenMtpGpuModel::Create(
    std::shared_ptr<const core::GgufReader> mtp_reader,
    std::shared_ptr<const QwenGpuModel> target_model, std::string* error_msg) {
  if (mtp_reader == nullptr || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "MTP and target GPU models must not be null";
    }
    return nullptr;
  }
  auto weights =
      speculative::QwenMtpWeights::LoadFromGguf(*mtp_reader, error_msg);
  if (!weights.has_value()) {
    return nullptr;
  }
  const auto& target_weights = target_model->GetWeights();
  if (target_weights.config.hidden_size != weights->config.hidden_size ||
      target_weights.config.vocab_size != weights->config.vocab_size ||
      target_weights.token_embd.empty() || target_weights.output.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "MTP and target GPU model dimensions are incompatible";
    }
    return nullptr;
  }

  std::vector<void*> allocations;
  std::size_t packed_bytes = 0;
  const auto raw_fusion_projection = weights->fusion_projection;
  const auto start = std::chrono::steady_clock::now();
  try {
    PackPrivateWeights(*weights, allocations, packed_bytes);
  } catch (const std::exception& exception) {
    ReleaseAllocations(allocations);
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }
  weights->token_embedding = target_weights.token_embd;
  weights->output = target_weights.output;
  const double pack_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();

  return std::shared_ptr<const QwenMtpGpuModel>(
      new QwenMtpGpuModel(std::move(mtp_reader), std::move(target_model),
                          std::move(*weights), raw_fusion_projection,
                          std::move(allocations), packed_bytes, pack_seconds));
}

QwenMtpGpuExecutor::QwenMtpGpuExecutor(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
    QwenMtpExecutionMode execution_mode)
    : model_(std::move(model)),
      max_context_(max_context),
      execution_mode_(execution_mode),
      h_last_hidden_(model_->GetConfig().hidden_size),
      h_logits_(model_->GetConfig().vocab_size)
#if defined(ENGINE_ENABLE_XRT)
      ,
      h_npu_input_(execution_mode == QwenMtpExecutionMode::kHybridNpuEhProj
                       ? 2 * model_->GetConfig().hidden_size
                       : 0),
      h_npu_output_(execution_mode == QwenMtpExecutionMode::kHybridNpuEhProj
                        ? model_->GetConfig().hidden_size
                        : 0)
#endif
{
  try {
    Allocate();
#if defined(ENGINE_ENABLE_XRT)
    if (execution_mode_ == QwenMtpExecutionMode::kHybridNpuEhProj) {
      const auto inventory = diagnostics::CollectSystemInventory();
      const auto device = xdna2::DiscoverXrtDevice(0, inventory);
      xdna2::QwenMtpEhProjFailure failure;
      npu_eh_proj_ = xdna2::QwenMtpEhProjSession::Create(
          {}, device, model_->GetRawFusionProjection(), &failure);
      if (npu_eh_proj_ == nullptr) {
        throw std::runtime_error("MTP NPU eh_proj initialization failed: " +
                                 failure.category + ": " + failure.message);
      }
    }
#else
    if (execution_mode_ == QwenMtpExecutionMode::kHybridNpuEhProj) {
      throw std::runtime_error(
          "MTP NPU execution requested without ENGINE_ENABLE_XRT");
    }
#endif
    Reset();
  } catch (...) {
    Free();
    throw;
  }
}

QwenMtpGpuExecutor::~QwenMtpGpuExecutor() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  Free();
}

std::unique_ptr<QwenMtpGpuExecutor> QwenMtpGpuExecutor::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
    std::string* error_msg, QwenMtpExecutionMode execution_mode) {
  if (model == nullptr || max_context == 0 ||
      max_context > model->GetConfig().context_length) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU executor context is invalid";
    }
    return nullptr;
  }
  try {
    return std::unique_ptr<QwenMtpGpuExecutor>(
        new QwenMtpGpuExecutor(std::move(model), max_context, execution_mode));
  } catch (const std::exception& exception) {
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }
}

void QwenMtpGpuExecutor::Allocate() {
  const auto& config = model_->GetConfig();
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;

  const auto stream_error = hipStreamCreate(&stream_);
  if (stream_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP stream creation failed: ") +
                             hipGetErrorString(stream_error));
  }
  AllocateBuffer(d_target_hidden_, hidden);
  AllocateBuffer(d_embedding_, hidden);
  AllocateBuffer(d_fusion_, 2 * hidden);
  AllocateBuffer(d_hidden_, hidden);
  AllocateBuffer(d_normed_, hidden);
  AllocateBuffer(d_qg_, 2 * attention);
  AllocateBuffer(d_q_, attention);
  AllocateBuffer(d_k_, kv);
  AllocateBuffer(d_v_, kv);
  AllocateBuffer(d_gate_, attention);
  AllocateBuffer(d_context_, attention);
  AllocateBuffer(d_attn_out_, hidden);
  AllocateBuffer(d_ffn_act_, intermediate);
  AllocateBuffer(d_ffn_out_, hidden);
  AllocateBuffer(d_feedback_hidden_, hidden);
  AllocateBuffer(d_logits_, config.vocab_size);
  AllocateBuffer(d_kv_cache_, 2 * total_kv);
  d_kv_cache_f16_ = AllocateDevice(2 * total_kv * sizeof(std::uint16_t));
  AllocateBuffer(d_split_k_scratch_,
                 detail::DecodeAttentionScratchElements(
                     config.num_attention_heads, config.head_dim));
  AllocateBuffer(d_out_token_, 1);
}

void QwenMtpGpuExecutor::Free() noexcept {
  const auto free_buffer = [](auto*& pointer) {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
      pointer = nullptr;
    }
  };
  free_buffer(d_target_hidden_);
  free_buffer(d_embedding_);
  free_buffer(d_fusion_);
  free_buffer(d_hidden_);
  free_buffer(d_normed_);
  free_buffer(d_qg_);
  free_buffer(d_q_);
  free_buffer(d_k_);
  free_buffer(d_v_);
  free_buffer(d_gate_);
  free_buffer(d_context_);
  free_buffer(d_attn_out_);
  free_buffer(d_ffn_act_);
  free_buffer(d_ffn_out_);
  free_buffer(d_feedback_hidden_);
  free_buffer(d_logits_);
  free_buffer(d_kv_cache_);
  free_buffer(d_kv_cache_f16_);
  free_buffer(d_split_k_scratch_);
  free_buffer(d_out_token_);
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void QwenMtpGpuExecutor::Reset() noexcept {
  const auto& config = model_->GetConfig();
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;
  (void)hipMemsetAsync(d_kv_cache_, 0, 2 * total_kv * sizeof(float), stream_);
  (void)hipMemsetAsync(d_kv_cache_f16_, 0, 2 * total_kv * sizeof(std::uint16_t),
                       stream_);
  next_position_ = 0;
  hybrid_metrics_ = {};
}

void QwenMtpGpuExecutor::Rewind(std::uint32_t position) {
  if (position > next_position_) {
    throw std::out_of_range("MTP rewind position exceeds executed context");
  }
  next_position_ = position;
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardTargetHidden(
    tokenization::TokenId input_token, std::span<const float> target_hidden,
    std::uint32_t position, bool compute_logits) {
  if (target_hidden.size() != model_->GetConfig().hidden_size) {
    throw std::invalid_argument("MTP target hidden width is invalid");
  }
  const auto copy_error = hipMemcpyAsync(d_target_hidden_, target_hidden.data(),
                                         target_hidden.size_bytes(),
                                         hipMemcpyHostToDevice, stream_);
  if (copy_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hidden copy failed: ") +
                             hipGetErrorString(copy_error));
  }
  return Run(input_token, d_target_hidden_, position, compute_logits);
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardFeedback(
    tokenization::TokenId input_token, std::uint32_t position,
    bool compute_logits) {
  return Run(input_token, d_feedback_hidden_, position, compute_logits);
}

tokenization::TokenId QwenMtpGpuExecutor::Run(tokenization::TokenId input_token,
                                              const float* hidden_input,
                                              std::uint32_t position,
                                              bool compute_logits) {
  if (position != next_position_ || position >= max_context_) {
    throw std::out_of_range("MTP GPU position is not sequential");
  }
  const auto& weights = model_->GetWeights();
  const auto& config = weights.config;
  const auto& layer = weights.layer;
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;

  LaunchEmbeddingLookup(weights.token_embedding.data,
                        weights.token_embedding.type, input_token, d_embedding_,
                        hidden, stream_);
  LaunchRMSNorm(d_embedding_,
                static_cast<const float*>(weights.embedding_norm.data),
                d_fusion_, hidden, 1.0e-6F, stream_);
  LaunchRMSNorm(hidden_input,
                static_cast<const float*>(weights.hidden_norm.data),
                d_fusion_ + hidden, hidden, 1.0e-6F, stream_);
  if (execution_mode_ == QwenMtpExecutionMode::kHybridNpuEhProj) {
#if defined(ENGINE_ENABLE_XRT)
    const auto gpu_to_host_start = std::chrono::steady_clock::now();
    const auto download_error = hipMemcpyAsync(
        h_npu_input_.data(), d_fusion_, h_npu_input_.size() * sizeof(float),
        hipMemcpyDeviceToHost, stream_);
    const auto download_sync_error = hipStreamSynchronize(stream_);
    if (download_error != hipSuccess || download_sync_error != hipSuccess) {
      throw std::runtime_error("MTP GPU-to-NPU input copy failed");
    }
    const auto gpu_to_host_end = std::chrono::steady_clock::now();
    xdna2::QwenMtpEhProjRunMetrics metrics;
    xdna2::QwenMtpEhProjFailure failure;
    if (!npu_eh_proj_->Run(h_npu_input_, h_npu_output_, &metrics, &failure)) {
      throw std::runtime_error("MTP NPU eh_proj failed: " + failure.category +
                               ": " + failure.message);
    }
    const auto host_to_gpu_start = std::chrono::steady_clock::now();
    const auto upload_error = hipMemcpyAsync(
        d_hidden_, h_npu_output_.data(), h_npu_output_.size() * sizeof(float),
        hipMemcpyHostToDevice, stream_);
    const auto upload_sync_error = hipStreamSynchronize(stream_);
    if (upload_error != hipSuccess || upload_sync_error != hipSuccess) {
      throw std::runtime_error("MTP NPU-to-GPU output copy failed");
    }
    const auto host_to_gpu_end = std::chrono::steady_clock::now();
    ++hybrid_metrics_.projection_count;
    hybrid_metrics_.gpu_to_host_us += std::chrono::duration<double, std::micro>(
                                          gpu_to_host_end - gpu_to_host_start)
                                          .count();
    hybrid_metrics_.activation_pack_us += metrics.activation_pack_us;
    hybrid_metrics_.npu_command_us += metrics.command_us;
    hybrid_metrics_.npu_end_to_end_us += metrics.end_to_end_us;
    hybrid_metrics_.host_to_gpu_us += std::chrono::duration<double, std::micro>(
                                          host_to_gpu_end - host_to_gpu_start)
                                          .count();
#else
    throw std::runtime_error(
        "MTP NPU execution requested without ENGINE_ENABLE_XRT");
#endif
  } else {
    LaunchGEMV(weights.fusion_projection.data, core::GgmlType::kBF16, d_fusion_, d_hidden_,
               hidden, 2 * hidden, stream_);
  }

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.attn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  LaunchFusedQKVProjections(layer.attn_q.data, core::GgmlType::kBF16,
                            layer.attn_k.data, core::GgmlType::kBF16,
                            layer.attn_v.data, core::GgmlType::kBF16, d_normed_,
                            d_qg_, d_k_, d_v_, 2 * attention, kv, hidden,
                            stream_);
  LaunchUnpackQG(d_qg_, d_q_, d_gate_, config.num_attention_heads,
                 config.head_dim, stream_);
  LaunchPerHeadRMSNorm(d_q_, static_cast<const float*>(layer.attn_q_norm.data),
                       d_q_, config.num_attention_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchPerHeadRMSNorm(d_k_, static_cast<const float*>(layer.attn_k_norm.data),
                       d_k_, config.num_key_value_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchRoPE(d_q_, d_k_, config.num_attention_heads, config.num_key_value_heads,
             config.head_dim, config.rotary_dim, position, config.rope_theta,
             stream_);
  LaunchAttention(
      d_q_, d_k_, d_v_, d_gate_, d_kv_cache_, d_kv_cache_ + total_kv,
      d_kv_cache_f16_, static_cast<std::uint16_t*>(d_kv_cache_f16_) + total_kv,
      d_context_, 0, position, max_context_, config.num_attention_heads,
      config.num_key_value_heads, config.head_dim, stream_, d_split_k_scratch_);
  LaunchGEMV(layer.attn_output.data, core::GgmlType::kBF16, d_context_, d_attn_out_, hidden,
             attention, stream_);
  LaunchResidualAdd(d_hidden_, d_attn_out_, d_hidden_, hidden, stream_);

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.ffn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, core::GgmlType::kBF16,
                        layer.ffn_up.data, core::GgmlType::kBF16, d_normed_,
                        d_ffn_act_, config.intermediate_size, hidden, stream_);
  LaunchGEMV(layer.ffn_down.data, core::GgmlType::kBF16, d_ffn_act_, d_ffn_out_, hidden,
             config.intermediate_size, stream_);
  LaunchResidualAdd(d_hidden_, d_ffn_out_, d_hidden_, hidden, stream_);
  LaunchRMSNorm(d_hidden_,
                static_cast<const float*>(weights.shared_head_norm.data),
                d_feedback_hidden_, hidden, 1.0e-6F, stream_);
  ++next_position_;

  if (!compute_logits) {
    return 0;
  }
  LaunchGEMV(weights.output.data, weights.output.type,
             d_feedback_hidden_, d_logits_, config.vocab_size, hidden, stream_);
  LaunchGPUArgmax(d_logits_, d_out_token_, config.vocab_size, stream_);
  tokenization::TokenId result = 0;
  const auto copy_error = hipMemcpyAsync(&result, d_out_token_, sizeof(result),
                                         hipMemcpyDeviceToHost, stream_);
  if (copy_error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP GPU result synchronization failed");
  }
  return result;
}

std::span<const float> QwenMtpGpuExecutor::CopyLastHidden() {
  const auto error = hipMemcpyAsync(h_last_hidden_.data(), d_feedback_hidden_,
                                    h_last_hidden_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP hidden synchronization failed");
  }
  return h_last_hidden_;
}

std::span<const float> QwenMtpGpuExecutor::CopyLastLogits() {
  const auto error = hipMemcpyAsync(h_logits_.data(), d_logits_,
                                    h_logits_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP logit synchronization failed");
  }
  return h_logits_;
}

QwenMtpGpuDraftBackend::QwenMtpGpuDraftBackend(
    std::unique_ptr<QwenMtpGpuExecutor> executor, QwenMtpGpuDraftConfig config)
    : executor_(std::move(executor)),
      config_(config),
      target_hidden_(executor_->GetHiddenSize()) {}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, QwenMtpGpuDraftConfig config,
    std::string* error_msg) {
  if (model == nullptr || config.max_draft_tokens == 0) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU draft configuration is invalid";
    }
    return nullptr;
  }
  auto executor = QwenMtpGpuExecutor::Create(
      std::move(model), config.max_context, error_msg, config.execution_mode);
  if (executor == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<QwenMtpGpuDraftBackend>(
      new QwenMtpGpuDraftBackend(std::move(executor), config));
}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::CreateFromGguf(
    std::string_view model_path,
    std::shared_ptr<const QwenGpuModel> target_model,
    QwenMtpGpuDraftConfig config, std::string* error_msg) {
  if (model_path.empty() || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GGUF path and target GPU model are required";
    }
    return nullptr;
  }
  auto reader_owner =
      core::GgufReader::OpenFile(std::string(model_path), error_msg);
  if (reader_owner == nullptr) {
    return nullptr;
  }
  std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  auto model = QwenMtpGpuModel::Create(std::move(reader),
                                       std::move(target_model), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), config, error_msg);
}

bool QwenMtpGpuDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  Reset();
  if (context.prompt_tokens.empty() || context.hidden_size == 0 ||
      context.hidden_size != target_hidden_.size() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    last_error_ = "MTP prompt hidden-state shape is invalid";
    return false;
  }

  try {
    for (std::size_t index = 0; index + 1 < context.prompt_tokens.size();
         ++index) {
      const auto hidden = context.prompt_hidden_states.subspan(
          index * context.hidden_size, context.hidden_size);
      (void)executor_->ForwardTargetHidden(
          context.prompt_tokens[index + 1], hidden,
          static_cast<std::uint32_t>(index), false);
    }
    const std::size_t final_offset =
        (context.prompt_tokens.size() - 1) * context.hidden_size;
    const auto final_hidden =
        context.prompt_hidden_states.subspan(final_offset, context.hidden_size);
    std::ranges::copy(final_hidden, target_hidden_.begin());
    executor_->ResetHybridMetrics();
    primed_ = true;
    return true;
  } catch (const std::exception& exception) {
    last_error_ = exception.what();
    Reset();
    return false;
  }
}

speculative::DraftProposal QwenMtpGpuDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  if (!primed_ || prompt_tokens.empty() || current_pos == 0) {
    throw std::logic_error("MTP GPU draft backend is not primed");
  }
  if (proposal_active_) {
    throw std::logic_error("MTP GPU proposal feedback is pending");
  }
  if (executor_->GetNextPosition() + 1 != current_pos) {
    throw std::logic_error("MTP GPU draft position is inconsistent");
  }

  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  const std::uint32_t count = std::min(max_tokens, config_.max_draft_tokens);
  if (count == 0) {
    return proposal;
  }

  proposal_checkpoint_ = executor_->GetNextPosition();
  proposal_input_ = prompt_tokens.back();
  proposed_tokens_.clear();
  proposed_tokens_.reserve(count);

  auto token = executor_->ForwardTargetHidden(proposal_input_, target_hidden_,
                                              proposal_checkpoint_, true);
  proposed_tokens_.push_back(token);
  for (std::uint32_t index = 1; index < count; ++index) {
    token =
        executor_->ForwardFeedback(token, proposal_checkpoint_ + index, true);
    proposed_tokens_.push_back(token);
  }
  proposal.tokens = proposed_tokens_;
  proposal_active_ = true;
  return proposal;
}

void QwenMtpGpuDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  if (!proposal_active_ || accepted.size() > proposed_tokens_.size()) {
    throw std::logic_error("MTP GPU proposal feedback is invalid");
  }

  executor_->Rewind(proposal_checkpoint_);
  (void)executor_->ForwardTargetHidden(proposal_input_, target_hidden_,
                                       proposal_checkpoint_, false);
  for (std::size_t index = 0; index < accepted.size(); ++index) {
    (void)executor_->ForwardFeedback(
        accepted[index],
        proposal_checkpoint_ + static_cast<std::uint32_t>(index) + 1, false);
  }
  proposal_active_ = false;
  proposed_tokens_.clear();
}

void QwenMtpGpuDraftBackend::UpdateTargetHidden(std::span<const float> hidden) {
  if (hidden.size() != target_hidden_.size()) {
    throw std::invalid_argument("MTP target hidden-state shape is invalid");
  }
  std::ranges::copy(hidden, target_hidden_.begin());
}

void QwenMtpGpuDraftBackend::Reset() noexcept {
  executor_->Reset();
  std::ranges::fill(target_hidden_, 0.0F);
  proposed_tokens_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = false;
  proposal_active_ = false;
  last_error_.clear();
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
