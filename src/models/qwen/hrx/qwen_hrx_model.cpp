#include "src/models/qwen/hrx/qwen_hrx_model.hpp"

#include <cstdint>
#include <utility>

#include "src/core/hrx/hrx_utils.hpp"

namespace gufo::hrx {

QwenHrxModel::QwenHrxModel(
    std::shared_ptr<const core::GgufReader> reader,
    models::QwenModelWeights weights,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
    QwenHrxArtifactContract contract, std::vector<ImportedRegion> regions)
    : reader_(std::move(reader)),
      weights_(std::move(weights)),
      contract_(contract),
      tokenizer_(std::move(tokenizer)),
      regions_(std::move(regions)) {}

QwenHrxModel::~QwenHrxModel() {
  for (auto& region : regions_) {
    if (region.buffer != nullptr) {
      hrx_buffer_release(region.buffer);
    }
  }
}

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
      .type = HRX_MEMORY_TYPE_HOST_VISIBLE | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      .access = HRX_MEMORY_ACCESS_READ,
      .usage = HRX_BUFFER_USAGE_STORAGE,
      .queue_affinity = 0,
  };
  for (const auto& mapped_region : mapped_regions) {
    hrx_buffer_t buffer{nullptr};
    const auto status = hrx_allocator_import_buffer(
        hrx_device_allocator(device), params,
        const_cast<void*>(mapped_region.data), mapped_region.size, &buffer);
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      if (buffer != nullptr) {
        hrx_buffer_release(buffer);
      }
      for (auto& region : regions) {
        if (region.buffer != nullptr) {
          hrx_buffer_release(region.buffer);
        }
      }
      if (error_msg != nullptr) {
        *error_msg = "failed to import a mapped GGUF shard into HRX";
      }
      return nullptr;
    }
    regions.push_back({.host_data = mapped_region.data,
                       .size = mapped_region.size,
                       .buffer = buffer});
  }

  auto model = std::unique_ptr<QwenHrxModel>(
      new QwenHrxModel(std::move(reader), std::move(*weights),
                       std::move(tokenizer), *contract, std::move(regions)));
  if (!model->BuildNativeQ8Bindings(error_msg)) {
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

bool QwenHrxModel::BuildNativeQ8Bindings(std::string* error_msg) {
  const auto& config = weights_.config;
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
  if (!bind_q8(weights_.token_embd, config.vocab_size, hidden,
               "token_embd.weight", &bindings.token_embedding) ||
      !bind_f32(weights_.output_norm, hidden, "output_norm.weight",
                &bindings.output_norm) ||
      !bind_q8(weights_.output, config.vocab_size, hidden, "output.weight",
               &bindings.output)) {
    return false;
  }

  bindings.layers.resize(weights_.layers.size());
  for (std::size_t index = 0; index < weights_.layers.size(); ++index) {
    const auto& layer = weights_.layers[index];
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
