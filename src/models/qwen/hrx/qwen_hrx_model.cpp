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
      .usage = HRX_BUFFER_USAGE_STORAGE_READ,
      .queue_affinity = 0,
  };
  for (const auto& mapped_region : mapped_regions) {
    hrx_buffer_t buffer{nullptr};
    const auto status = hrx_allocator_import_buffer(
        hrx_device_allocator(device), params,
        const_cast<void*>(mapped_region.data), mapped_region.size, &buffer);
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
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

  return std::unique_ptr<QwenHrxModel>(
      new QwenHrxModel(std::move(reader), std::move(*weights),
                       std::move(tokenizer), *contract, std::move(regions)));
}

std::optional<HrxBufferBinding> QwenHrxModel::Bind(
    const models::QwenTensorRef& tensor) const noexcept {
  if (tensor.empty()) {
    return std::nullopt;
  }
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
      return HrxBufferBinding{
          .buffer = region.buffer, .offset = offset, .length = tensor_size};
    }
  }
  return std::nullopt;
}

}  // namespace gufo::hrx
