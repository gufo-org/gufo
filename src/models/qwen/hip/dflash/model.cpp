#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/dflash.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"

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
    throw std::runtime_error("unsupported DFlash matrix quantization");
  }
  switch (tensor.type) {
    case core::GgmlType::kQ8_0:
      quant::DequantizeQ8_0(source, output, columns);
      return;
    case core::GgmlType::kQ4_K:
      quant::DequantizeQ4_K(source, output, columns);
      return;
    case core::GgmlType::kQ6_K:
      quant::DequantizeQ6_K(source, output, columns);
      return;
    default:
      throw std::runtime_error("unsupported DFlash tensor format for packing");
  }
}

void CopyToDevice(void* destination, const void* source, std::size_t bytes) {
  const auto error =
      hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice);
  if (error != hipSuccess) {
    throw std::runtime_error(std::string("DFlash hipMemcpy failed: ") +
                             hipGetErrorString(error));
  }
}

models::QwenTensorRef CopyVectorF32(const models::QwenTensorRef& source,
                                    std::size_t elements,
                                    std::vector<void*>& allocations,
                                    std::size_t& packed_bytes) {
  if (source.empty() || elements == 0) {
    return {};
  }
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
  if (source.empty() || rows == 0 || columns == 0) {
    return {};
  }
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

void PackDFlashWeights(speculative::QwenDFlashWeights& weights,
                      std::vector<void*>& allocations,
                      std::size_t& packed_bytes) {
  const auto& cfg = weights.config;
  const auto& df_cfg = weights.dflash_config;
  const std::size_t hidden = cfg.hidden_size;
  const std::size_t intermediate = cfg.intermediate_size;
  const std::size_t attention = cfg.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(cfg.num_key_value_heads) * cfg.head_dim;
  const std::size_t enc_in_dim = df_cfg.target_layer_ids.size() * hidden;

  // Feature Fusion Encoder
  weights.fc_norm =
      CopyVectorF32(weights.fc_norm, hidden, allocations, packed_bytes);
  weights.fc_projection =
      PackMatrixBf16(weights.fc_projection, hidden, enc_in_dim, allocations,
                     packed_bytes);

  // Dynamic Convolutions (DFlash-2)
  if (!weights.in_conv_weight.empty()) {
    weights.in_conv_weight = CopyVectorF32(weights.in_conv_weight, hidden * 2,
                                           allocations, packed_bytes);
    weights.in_conv_bias = CopyVectorF32(weights.in_conv_bias, hidden,
                                         allocations, packed_bytes);
  }
  if (!weights.out_conv_weight.empty()) {
    weights.out_conv_weight = CopyVectorF32(weights.out_conv_weight, hidden * 2,
                                            allocations, packed_bytes);
    weights.out_conv_bias = CopyVectorF32(weights.out_conv_bias, hidden,
                                          allocations, packed_bytes);
  }

  // Draft Output Norm if present
  if (!weights.output_norm.empty()) {
    weights.output_norm = CopyVectorF32(weights.output_norm, hidden,
                                        allocations, packed_bytes);
  }

  // Draft Transformer Layers
  for (std::size_t i = 0; i < weights.layers.size(); ++i) {
    auto& layer = weights.layers[i];
    layer.attn_norm =
        CopyVectorF32(layer.attn_norm, hidden, allocations, packed_bytes);
    layer.attn_q = PackMatrixBf16(layer.attn_q, attention, hidden,
                                  allocations, packed_bytes);
    layer.attn_k =
        PackMatrixBf16(layer.attn_k, kv, hidden, allocations, packed_bytes);
    layer.attn_v =
        PackMatrixBf16(layer.attn_v, kv, hidden, allocations, packed_bytes);
    layer.attn_output = PackMatrixBf16(layer.attn_output, hidden, attention,
                                       allocations, packed_bytes);
    layer.attn_q_norm = CopyVectorF32(layer.attn_q_norm, cfg.head_dim,
                                      allocations, packed_bytes);
    layer.attn_k_norm = CopyVectorF32(layer.attn_k_norm, cfg.head_dim,
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
}

}  // namespace

QwenDFlashGpuModel::QwenDFlashGpuModel(
    std::shared_ptr<const core::GgufReader> dflash_reader,
    std::shared_ptr<const QwenGpuModel> target_model,
    speculative::QwenDFlashWeights weights,
    std::vector<void*> allocations,
    std::size_t packed_weight_bytes)
    : dflash_reader_(std::move(dflash_reader)),
      target_model_(std::move(target_model)),
      weights_(std::move(weights)),
      allocations_(std::move(allocations)),
      packed_weight_bytes_(packed_weight_bytes) {}

QwenDFlashGpuModel::~QwenDFlashGpuModel() {
  for (void* pointer : allocations_) {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
    }
  }
}

std::shared_ptr<const QwenDFlashGpuModel> QwenDFlashGpuModel::Create(
    std::shared_ptr<const core::GgufReader> dflash_reader,
    std::shared_ptr<const QwenGpuModel> target_model,
    std::string* error_msg) {
  if (dflash_reader == nullptr || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash reader and target GPU model are required";
    }
    return nullptr;
  }

  auto raw_weights =
      speculative::QwenDFlashWeights::LoadFromGguf(*dflash_reader, error_msg);
  if (!raw_weights.has_value()) {
    return nullptr;
  }

  std::vector<void*> allocations;
  std::size_t packed_bytes = 0;
  try {
    speculative::QwenDFlashWeights weights = *raw_weights;

    // Pack all private DFlash matrices and vectors into GPU device memory
    PackDFlashWeights(weights, allocations, packed_bytes);

    // Unconditionally bind GPU device pointers from the target model for tied weights
    weights.token_embedding = target_model->GetWeights().token_embd;
    weights.output = target_model->GetWeights().output;
    if (weights.output_norm.empty()) {
      weights.output_norm = target_model->GetWeights().output_norm;
    }

    return std::shared_ptr<const QwenDFlashGpuModel>(
        new QwenDFlashGpuModel(std::move(dflash_reader), std::move(target_model),
                               std::move(weights), std::move(allocations),
                               packed_bytes));
  } catch (const std::exception& ex) {
    for (void* ptr : allocations) {
      if (ptr != nullptr) (void)hipFree(ptr);
    }
    if (error_msg != nullptr) {
      *error_msg = ex.what();
    }
    return nullptr;
  }
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
