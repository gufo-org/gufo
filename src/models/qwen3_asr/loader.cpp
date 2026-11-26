#include "src/models/qwen3_asr/loader.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gufo/ext/platform/mapped_file.hpp>
#include <gufo/ext/platform/prefault.hpp>
#include <gufo/ext/safetensors.hpp>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace gufo::models::qwen3_asr {
namespace {

class MappedFile {
public:
  MappedFile() = default;
  ~MappedFile() { Reset(); }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&&) = delete;
  MappedFile& operator=(MappedFile&&) = delete;

  [[nodiscard]] bool Open(const std::filesystem::path& path) {
    mapped_ = platform::ReadOnlyMappedFile::Open(path, &open_error_);
    if (!mapped_.valid())
      return false;
    data_ = const_cast<std::uint8_t*>(mapped_.data());
    size_ = mapped_.size();
    return true;
  }

  [[nodiscard]] const std::string& open_error() const noexcept {
    return open_error_;
  }

  [[nodiscard]] const std::byte* data() const noexcept {
    return static_cast<const std::byte*>(data_);
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  platform::PlatformFile file_descriptor() const {
    return mapped_.PlatformHandle();
  }

private:
  void Reset() noexcept {
    mapped_.Close();
    data_ = nullptr;
    size_ = 0;
  }

  platform::ReadOnlyMappedFile mapped_;
  std::string open_error_;
  void* data_{nullptr};
  std::size_t size_{0};
};

DType ParseDType(safetensors::Dtype dtype) noexcept {
  switch (dtype) {
    case safetensors::Dtype::kF32:
      return DType::kF32;
    case safetensors::Dtype::kBF16:
      return DType::kBF16;
    case safetensors::Dtype::kF16:
      return DType::kF16;
    case safetensors::Dtype::kI64:
      return DType::kI64;
    case safetensors::Dtype::kI32:
      return DType::kI32;
    case safetensors::Dtype::kU8:
      return DType::kU8;
    default:
      return DType::kUnknown;
  }
}

bool LoadSafetensors(const std::filesystem::path& path, TensorStore* store,
                     std::vector<std::shared_ptr<void>>* mappings,
                     std::vector<MappedRegion>* regions, std::string* error) {
  auto mapping = std::make_shared<MappedFile>();
  if (!mapping->Open(path)) {
    if (error != nullptr && !mapping->open_error().empty())
      *error = "cannot open " + path.string() + ": " + mapping->open_error();
    return false;
  }
  safetensors::Header header;
  try {
    header = safetensors::Parse(std::span(mapping->data(), mapping->size()));
  } catch (const std::exception& failure) {
    if (error != nullptr)
      *error = failure.what();
    return false;
  }
  const std::size_t before = store->size();
  for (auto& entry : header.tensors) {
    Tensor tensor;
    tensor.name = std::move(entry.name);
    tensor.dtype = ParseDType(entry.dtype);
    // Types this model has no use for are skipped, as are tensors without
    // elements.
    if (DTypeSize(tensor.dtype) == 0 || entry.begin == entry.end)
      continue;
    tensor.shape = std::move(entry.shape);
    tensor.data = mapping->data() + header.payload_offset + entry.begin;
    tensor.byte_count = entry.end - entry.begin;
    if (!store->Add(std::move(tensor))) {
      return false;
    }
  }
  if (store->size() == before) {
    return false;
  }
  regions->push_back({
      .data = mapping->data(),
      .size = mapping->size(),
      .payload_offset = static_cast<std::size_t>(header.payload_offset),
      .file_descriptor = mapping->file_descriptor(),
  });
  mappings->push_back(std::move(mapping));
  return true;
}

}  // namespace

void LoadResult::ReleaseHostPages() const noexcept {
  for (const auto& region : mapped_regions)
    platform::ReleaseMappedPages(region.data, region.size);
}

std::vector<core::GgufMappedRegion> LoadResult::WeightUploadRegions() const {
  std::vector<core::GgufMappedRegion> result;
  for (const auto& region : mapped_regions)
    result.push_back({region.data, region.size, region.file_descriptor});
  return result;
}

std::vector<MappedRegion> LoadResult::RegionsFor(
    std::initializer_list<std::string_view> prefixes) const {
  std::vector<MappedRegion> result;
  if (!store)
    return result;
  const auto system_page_bytes = platform::SystemPageSize();
  if (system_page_bytes % 16 != 0)
    throw std::runtime_error("ASR cannot determine mapped weight alignment");
  const auto page_bytes = static_cast<std::size_t>(system_page_bytes);
  for (const auto& region : mapped_regions) {
    std::size_t begin = region.size, end = 0;
    const auto base = reinterpret_cast<std::uintptr_t>(region.data);
    for (const auto& [name, tensor] : store->tensors_) {
      if (!std::ranges::any_of(
              prefixes, [&](auto prefix) { return name.starts_with(prefix); }))
        continue;
      const auto address = reinterpret_cast<std::uintptr_t>(tensor.data);
      if (address < base || address - base >= region.size)
        continue;
      const auto offset = address - base;
      if (tensor.byte_count > region.size - offset)
        throw std::runtime_error("ASR tensor exceeds its mapped shard");
      begin = std::min(begin, offset);
      end = std::max(end, offset + tensor.byte_count);
    }
    if (end == 0)
      continue;
    // Keep mapping fallback page aligned; all tensor offsets retain the
    // original shard's alignment modulo 16 for the copied representation.
    begin -= begin % page_bytes;
    result.push_back(
        {region.data + begin, end - begin, region.payload_offset % 16U});
  }
  return result;
}

LoadResult LoadModelDirectory(const std::string& model_dir) {
  LoadResult result;
  result.store = std::make_unique<TensorStore>();
  const auto config = LoadModelConfigFromPath(model_dir);
  if (!config.has_value()) {
    result.error = "qwen3_asr: cannot parse " + model_dir + "/config.json";
    return result;
  }
  result.config = *config;
  if (!IsSupportedModelConfig(result.config)) {
    result.error = "qwen3_asr: expected the Qwen3-ASR-1.7B BF16 configuration";
    return result;
  }

  const std::filesystem::path root(model_dir);
  for (const std::string_view shard : {"model-00001-of-00002.safetensors",
                                       "model-00002-of-00002.safetensors"}) {
    const std::filesystem::path path = root / shard;
    std::string load_error;
    if (!LoadSafetensors(path, result.store.get(), &result.mappings,
                         &result.mapped_regions, &load_error)) {
      result.error = "qwen3_asr: cannot load " + path.string();
      if (!load_error.empty())
        result.error += ": " + load_error;
      return result;
    }
  }
  result.ok = true;
  return result;
}

bool LooksLikeQwen3Asr(const std::string& model_dir) {
  const auto config = LoadModelConfigFromPath(model_dir);
  return config.has_value() && IsSupportedModelConfig(*config);
}

}  // namespace gufo::models::qwen3_asr
