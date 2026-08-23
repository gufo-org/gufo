#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/qwen_mtp_gpu.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen/hip/mtp/detail/qwen_mtp_gpu_allocation.hpp"
#include "src/core/quant/ggml_dequant.hpp"

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
  void* device = detail::AllocateDevice(bytes);
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
  void* device = detail::AllocateDevice(total_bytes);
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

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
