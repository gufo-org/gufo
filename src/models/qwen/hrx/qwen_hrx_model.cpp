#include "src/models/qwen/hrx/qwen_hrx_model.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <utility>

#include "src/core/hrx/hrx_utils.hpp"

namespace gufo::hrx {
namespace {

enum class WeightPlacement { kMapped, kDeviceLocal };

constexpr std::size_t kWeightTransferChunkBytes = 64ULL * 1024ULL * 1024ULL;

std::optional<WeightPlacement> ParseWeightPlacement(std::string* error_msg) {
  const auto* value = std::getenv("GUFO_HRX_WEIGHT_MODE");
  if (value == nullptr || std::string_view(value) == "device-local" ||
      std::string_view(value) == "copy") {
    return WeightPlacement::kDeviceLocal;
  }
  if (std::string_view(value) == "mapped") {
    return WeightPlacement::kMapped;
  }
  if (error_msg != nullptr) {
    *error_msg = "GUFO_HRX_WEIGHT_MODE must be device-local, copy, or mapped";
  }
  return std::nullopt;
}

void SetHrxError(hrx_status_t status, std::string_view operation,
                 std::string* error_msg) {
  char* message = nullptr;
  std::size_t message_length = 0;
  const auto format_status =
      hrx_status_to_string(status, &message, &message_length);
  if (error_msg != nullptr) {
    *error_msg = std::string(operation) + ": " +
                 (message != nullptr ? message : "unknown HRX error");
  }
  hrx_status_free_message(message);
  hrx_status_ignore(format_status);
  hrx_status_ignore(status);
}

/// Returns a single binding covering two weight tensors when they occupy one
/// HRX buffer back to back. Fused GEMV routes require that exact layout.
std::optional<HrxBufferBinding> MergeAdjacent(
    const HrxBufferBinding& first, const HrxBufferBinding& second) noexcept {
  if (!first.IsValid() || !second.IsValid() ||
      first.buffer != second.buffer ||
      first.offset > std::numeric_limits<std::size_t>::max() - first.length ||
      first.offset + first.length != second.offset ||
      first.length > std::numeric_limits<std::size_t>::max() - second.length) {
    return std::nullopt;
  }
  return HrxBufferBinding{.buffer = first.buffer,
                          .offset = first.offset,
                          .length = first.length + second.length};
}

}  // namespace

QwenHrxModel::ImportedRegion::ImportedRegion(const void* data,
                                             std::size_t byte_size,
                                             hrx_buffer_t owned_buffer) noexcept
    : host_data(data), size(byte_size), buffer(owned_buffer) {}

QwenHrxModel::ImportedRegion::~ImportedRegion() {
  if (buffer != nullptr) {
    hrx_buffer_release(buffer);
  }
}

QwenHrxModel::ImportedRegion::ImportedRegion(ImportedRegion&& other) noexcept
    : host_data(std::exchange(other.host_data, nullptr)),
      size(std::exchange(other.size, 0)),
      buffer(std::exchange(other.buffer, nullptr)) {}

QwenHrxModel::ImportedRegion& QwenHrxModel::ImportedRegion::operator=(
    ImportedRegion&& other) noexcept {
  if (this != &other) {
    if (buffer != nullptr) {
      hrx_buffer_release(buffer);
    }
    host_data = std::exchange(other.host_data, nullptr);
    size = std::exchange(other.size, 0);
    buffer = std::exchange(other.buffer, nullptr);
  }
  return *this;
}

QwenHrxModel::QwenHrxModel(
    core::ModelConfig config,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
    QwenHrxArtifactContract contract, std::vector<ImportedRegion> regions,
    bool uses_device_local_weights)
    : config_(std::move(config)),
      contract_(contract),
      tokenizer_(std::move(tokenizer)),
      regions_(std::move(regions)),
      uses_device_local_weights_(uses_device_local_weights) {}

QwenHrxModel::~QwenHrxModel() = default;

std::unique_ptr<QwenHrxModel> QwenHrxModel::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, hrx_device_t device,
    std::string* error_msg) {
  if (reader == nullptr || device == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "HRX model creation requires a GGUF reader and device";
    }
    return nullptr;
  }

  auto weights = models::QwenModelWeights::LoadFromGguf(*reader, error_msg);
  if (!weights.has_value()) {
    return nullptr;
  }
  const auto contract =
      QwenHrxArtifactContract::FromConfig(weights->config, error_msg);
  if (!contract.has_value()) {
    return nullptr;
  }
  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
  if (tokenizer == nullptr) {
    return nullptr;
  }
  const auto placement = ParseWeightPlacement(error_msg);
  if (!placement.has_value()) {
    return nullptr;
  }
  const bool device_local = *placement == WeightPlacement::kDeviceLocal;

  const auto mapped_regions = reader->GetMappedRegions();
  if (mapped_regions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader has no mapped weight regions";
    }
    return nullptr;
  }

  std::vector<ImportedRegion> regions;
  regions.reserve(mapped_regions.size());
  const hrx_buffer_params_t params{
      .type = device_local ? HRX_MEMORY_TYPE_DEVICE_LOCAL
                           : HRX_MEMORY_TYPE_HOST_VISIBLE |
                                 HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      .access = device_local ? HRX_MEMORY_ACCESS_ALL : HRX_MEMORY_ACCESS_READ,
      .usage =
          device_local ? HRX_BUFFER_USAGE_DEFAULT : HRX_BUFFER_USAGE_STORAGE,
      .queue_affinity = 0,
  };
  for (const auto& mapped_region : mapped_regions) {
    hrx_buffer_t buffer{nullptr};
    std::string_view operation = device_local
                                     ? "failed to allocate HRX device-local "
                                       "GGUF shard storage"
                                     : "failed to import a mapped GGUF shard "
                                       "into HRX";
    auto status =
        device_local
            ? hrx_allocator_allocate_buffer(hrx_device_allocator(device),
                                            params, mapped_region.size, &buffer)
            : hrx_allocator_import_buffer(hrx_device_allocator(device), params,
                                          const_cast<void*>(mapped_region.data),
                                          mapped_region.size, &buffer);
    if (hrx_status_is_ok(status) && device_local) {
      operation = "failed to copy a GGUF shard into HRX device-local storage";
      const auto* source = static_cast<const std::byte*>(mapped_region.data);
      for (std::size_t offset = 0; offset < mapped_region.size;
           offset += kWeightTransferChunkBytes) {
        const std::size_t chunk_size =
            std::min(kWeightTransferChunkBytes, mapped_region.size - offset);
        status = hrx_synchronous_h2d(device, source + offset, buffer, offset,
                                     chunk_size);
        if (!hrx_status_is_ok(status)) {
          break;
        }
      }
    }
    if (!hrx_status_is_ok(status)) {
      SetHrxError(status, operation, error_msg);
      if (buffer != nullptr) {
        hrx_buffer_release(buffer);
      }
      return nullptr;
    }
    regions.emplace_back(mapped_region.data, mapped_region.size, buffer);
  }

  auto model = std::unique_ptr<QwenHrxModel>(
      new QwenHrxModel(weights->config, std::move(tokenizer), *contract,
                       std::move(regions), device_local));
  if (!model->BuildNativeQ8Bindings(*weights, error_msg)) {
    return nullptr;
  }
  return model;
}

std::optional<HrxBufferBinding> QwenHrxModel::BindTensor(
    const models::QwenTensorRef& tensor, core::GgmlType expected_type,
    std::size_t expected_elements, std::string* error_msg) const noexcept {
  if (!ValidateHrxTensorPayload(tensor, expected_type, expected_elements,
                                error_msg)) {
    return std::nullopt;
  }
  auto binding = BindStorage(tensor);
  if (!binding.has_value()) {
    if (error_msg != nullptr) {
      *error_msg = "validated GGUF tensor is outside the imported HRX shards";
    }
    return std::nullopt;
  }
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return binding;
}

std::optional<HrxBufferBinding> QwenHrxModel::BindBf16Matrix(
    const models::QwenTensorRef& tensor, std::size_t rows, std::size_t columns,
    std::string* error_msg) const noexcept {
  const auto elements = HrxMatrixElementCount(rows, columns);
  if (!elements.has_value()) {
    if (error_msg != nullptr) {
      *error_msg = "HRX BF16 matrix dimensions are zero or overflow size_t";
    }
    return std::nullopt;
  }
  return BindTensor(tensor, core::GgmlType::kBF16, *elements, error_msg);
}

std::optional<HrxBufferBinding> QwenHrxModel::BindQ8_0Matrix(
    const models::QwenTensorRef& tensor, std::size_t rows, std::size_t columns,
    std::string* error_msg) const noexcept {
  const auto elements = HrxMatrixElementCount(rows, columns);
  if (!elements.has_value() || (columns % 32) != 0) {
    if (error_msg != nullptr) {
      *error_msg =
          "native HRX Q8_0 matrix dimensions are zero, overflow, or have a "
          "column count not divisible by 32";
    }
    return std::nullopt;
  }
  return BindTensor(tensor, core::GgmlType::kQ8_0, *elements, error_msg);
}

std::optional<HrxBufferBinding> QwenHrxModel::BindF32Vector(
    const models::QwenTensorRef& tensor, std::size_t elements,
    std::string* error_msg) const noexcept {
  return BindTensor(tensor, core::GgmlType::kF32, elements, error_msg);
}

bool QwenHrxModel::BuildNativeQ8Bindings(
    const models::QwenModelWeights& weights, std::string* error_msg) {
  const auto& config = weights.config;
  const std::size_t hidden = config.hidden_size;
  const std::size_t ffn = config.intermediate_size;
  const std::size_t q = config.AttentionSize();
  const std::size_t q_gate = 2 * q;
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t ssm_qkv = config.SsmQkvSize();
  const std::size_t ssm_inner = config.ssm_inner_size;
  const std::size_t rank = config.ssm_time_step_rank;

  const auto reject_binding = [error_msg](const std::string& name,
                                          const std::string& detail) {
    if (error_msg != nullptr) {
      *error_msg = "native HRX Q8_0 binding rejected '" + name + "': " + detail;
    }
    return false;
  };
  const auto bind_q8 = [this, &reject_binding](
                           const models::QwenTensorRef& tensor,
                           std::size_t rows, std::size_t columns,
                           const std::string& name,
                           HrxBufferBinding* destination) {
    std::string detail;
    const auto binding = BindQ8_0Matrix(tensor, rows, columns, &detail);
    if (!binding.has_value()) {
      return reject_binding(name, detail);
    }
    *destination = *binding;
    return true;
  };
  const auto bind_f32 = [this, &reject_binding](
                            const models::QwenTensorRef& tensor,
                            std::size_t elements, const std::string& name,
                            HrxBufferBinding* destination) {
    std::string detail;
    const auto binding = BindF32Vector(tensor, elements, &detail);
    if (!binding.has_value()) {
      return reject_binding(name, detail);
    }
    *destination = *binding;
    return true;
  };

  QwenHrxWeightBindings bindings;
  if (!bind_q8(weights.token_embd, config.vocab_size, hidden,
               "token_embd.weight", &bindings.token_embedding) ||
      !bind_f32(weights.output_norm, hidden, "output_norm.weight",
                &bindings.output_norm) ||
      !bind_q8(weights.output, config.vocab_size, hidden, "output.weight",
               &bindings.output)) {
    return false;
  }

  bindings.layers.resize(weights.layers.size());
  for (std::size_t index = 0; index < weights.layers.size(); ++index) {
    const auto& layer = weights.layers[index];
    auto& native = bindings.layers[index];
    native.is_full_attention = layer.is_full_attention;
    const std::string prefix = "blk." + std::to_string(index) + ".";
    if (!bind_f32(layer.attn_norm, hidden, prefix + "attn_norm.weight",
                  &native.attn_norm) ||
        !bind_f32(layer.ffn_norm, hidden, prefix + "ffn_norm.weight",
                  &native.ffn_norm) ||
        !bind_q8(layer.ffn_gate, ffn, hidden, prefix + "ffn_gate.weight",
                 &native.ffn_gate) ||
        !bind_q8(layer.ffn_up, ffn, hidden, prefix + "ffn_up.weight",
                 &native.ffn_up) ||
        !bind_q8(layer.ffn_down, hidden, ffn, prefix + "ffn_down.weight",
                 &native.ffn_down)) {
      return false;
    }
    native.ffn_gate_up =
        MergeAdjacent(native.ffn_gate, native.ffn_up).value_or(HrxBufferBinding{});

    if (layer.is_full_attention) {
      if (!bind_q8(layer.attn_q, q_gate, hidden, prefix + "attn_q.weight",
                   &native.attn_q) ||
          !bind_q8(layer.attn_k, kv, hidden, prefix + "attn_k.weight",
                   &native.attn_k) ||
          !bind_q8(layer.attn_v, kv, hidden, prefix + "attn_v.weight",
                   &native.attn_v) ||
          !bind_q8(layer.attn_output, hidden, q, prefix + "attn_output.weight",
                   &native.attn_output) ||
          !bind_f32(layer.attn_q_norm, config.head_dim,
                    prefix + "attn_q_norm.weight", &native.attn_q_norm) ||
          !bind_f32(layer.attn_k_norm, config.head_dim,
                    prefix + "attn_k_norm.weight", &native.attn_k_norm)) {
        return false;
      }
      continue;
    }

    if (!bind_q8(layer.attn_qkv, ssm_qkv, hidden, prefix + "attn_qkv.weight",
                 &native.attn_qkv) ||
        !bind_q8(layer.attn_gate, ssm_inner, hidden,
                 prefix + "attn_gate.weight", &native.attn_gate) ||
        !bind_f32(layer.ssm_a, rank, prefix + "ssm_a", &native.ssm_a) ||
        !bind_f32(layer.ssm_conv1d, ssm_qkv * config.ssm_conv_kernel,
                  prefix + "ssm_conv1d.weight", &native.ssm_conv1d) ||
        !bind_f32(layer.ssm_dt, rank, prefix + "ssm_dt.bias", &native.ssm_dt) ||
        !bind_q8(layer.ssm_alpha, rank, hidden, prefix + "ssm_alpha.weight",
                 &native.ssm_alpha) ||
        !bind_q8(layer.ssm_beta, rank, hidden, prefix + "ssm_beta.weight",
                 &native.ssm_beta) ||
        !bind_f32(layer.ssm_norm, config.SsmValueSize(),
                  prefix + "ssm_norm.weight", &native.ssm_norm) ||
        !bind_q8(layer.ssm_out, hidden, ssm_inner, prefix + "ssm_out.weight",
                 &native.ssm_out)) {
      return false;
    }
    native.ssm_alpha_beta =
        MergeAdjacent(native.ssm_alpha, native.ssm_beta).value_or(HrxBufferBinding{});
  }

  native_bindings_ = std::move(bindings);
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return true;
}

std::optional<HrxBufferBinding> QwenHrxModel::BindStorage(
    const models::QwenTensorRef& tensor) const noexcept {
  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor.data);
  const std::size_t tensor_size = tensor.EncodedSizeBytes();
  for (const auto& region : regions_) {
    const auto region_address =
        reinterpret_cast<std::uintptr_t>(region.host_data);
    if (tensor_address < region_address) {
      continue;
    }
    const std::size_t offset = tensor_address - region_address;
    if (offset < region.size && tensor_size != 0 &&
        tensor_size <= region.size - offset) {
      const HrxBufferBinding binding{
          .buffer = region.buffer, .offset = offset, .length = tensor_size};
      if (binding.IsValid()) {
        return binding;
      }
      return std::nullopt;
    }
  }
  return std::nullopt;
}

}  // namespace gufo::hrx
