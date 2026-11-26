#include "src/models/qwen3_tts/loader.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gufo/ext/platform/mapped_file.hpp>
#include <gufo/ext/platform/prefault.hpp>
#include <gufo/ext/safetensors.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace gufo::models::qwen3_tts {
namespace {

/// RAII mmap holder; moved into LoadResult::mappings.
class MappedFile {
public:
  MappedFile() = default;
  ~MappedFile() { Unmap(); }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  MappedFile(MappedFile&&) = delete;
  MappedFile& operator=(MappedFile&&) = delete;

  [[nodiscard]] const std::string& open_error() const noexcept {
    return open_error_;
  }

  void* base() const { return base_; }
  std::size_t size() const { return size_; }
  platform::PlatformFile file_descriptor() const {
    return mapped_.PlatformHandle();
  }

  /// Maps the file read-only. Returns false on failure; the reason is
  /// then available through open_error().
  bool Open(const std::filesystem::path& path) {
    mapped_ = platform::ReadOnlyMappedFile::Open(path, &open_error_);
    if (!mapped_.valid())
      return false;
    base_ = const_cast<std::uint8_t*>(mapped_.data());
    size_ = mapped_.size();
    return true;
  }

private:
  void Unmap() {
    mapped_.Close();
    base_ = nullptr;
    size_ = 0;
  }

  platform::ReadOnlyMappedFile mapped_;
  std::string open_error_;
  void* base_ = nullptr;
  std::size_t size_ = 0;
};

DType DTypeFromSafetensors(safetensors::Dtype dtype) {
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

/// Parses one safetensors file, adding tensors to `store`.
/// Adds a ownership mapping to `mappings` if the file is usable.
bool LoadSafetensorsFile(const std::filesystem::path& path, TensorStore* store,
                         std::vector<std::shared_ptr<void>>* mappings,
                         std::vector<MappedRegion>* mapped_regions,
                         std::string* error) {
  auto mapped = std::make_shared<MappedFile>();
  if (!mapped->Open(path)) {
    if (error != nullptr && !mapped->open_error().empty())
      *error = "cannot open " + path.string() + ": " + mapped->open_error();
    return false;
  }
  const auto* bytes = static_cast<const std::byte*>(mapped->base());
  safetensors::Header header;
  try {
    header = safetensors::Parse(std::span(bytes, mapped->size()));
  } catch (const std::exception& failure) {
    if (error != nullptr)
      *error = failure.what();
    return false;
  }
  const std::size_t count_before = store->size();
  for (auto& entry : header.tensors) {
    Tensor tensor;
    tensor.name = std::move(entry.name);
    tensor.dtype = DTypeFromSafetensors(entry.dtype);
    // Types this model has no use for are skipped, as are tensors without
    // elements.
    if (DTypeSize(tensor.dtype) == 0 || entry.begin == entry.end) {
      continue;
    }
    tensor.shape = std::move(entry.shape);
    tensor.data = bytes + static_cast<std::ptrdiff_t>(header.payload_offset +
                                                      entry.begin);
    tensor.byte_count = entry.end - entry.begin;
    store->Add(std::move(tensor));
  }
  if (store->size() == count_before) {
    return false;
  }
  mapped_regions->push_back(
      {.data = bytes,
       .size = static_cast<std::size_t>(mapped->size()),
       .payload_offset = static_cast<std::size_t>(header.payload_offset),
       .file_descriptor = mapped->file_descriptor()});
  mappings->push_back(std::move(mapped));
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
    std::string_view prefix) const {
  std::vector<MappedRegion> result;
  if (!store)
    return result;
  const auto page_size = platform::SystemPageSize();
  if (page_size % 16 != 0)
    throw std::runtime_error("TTS cannot determine mapped weight alignment");
  for (const auto& region : mapped_regions) {
    std::size_t begin = region.size, end = 0;
    const auto base = reinterpret_cast<std::uintptr_t>(region.data);
    for (const auto& tensor : store->tensors_) {
      if (!tensor.name.starts_with(prefix))
        continue;
      const auto address = reinterpret_cast<std::uintptr_t>(tensor.data);
      if (address < base || address - base >= region.size)
        continue;
      const auto offset = address - base;
      if (tensor.byte_count > region.size - offset)
        throw std::runtime_error("TTS tensor exceeds its mapped shard");
      begin = std::min(begin, offset);
      end = std::max(end, offset + tensor.byte_count);
    }
    if (end != 0) {
      begin -= begin % static_cast<std::size_t>(page_size);
      result.push_back(
          {region.data + begin, end - begin, region.payload_offset % 16U});
    }
  }
  return result;
}

LoadResult LoadModelDirectory(const std::string& model_dir) {
  LoadResult result;
  result.store = std::make_unique<TensorStore>();

  const auto config = LoadModelConfigFromPath(model_dir);
  if (!config.has_value()) {
    result.error = "qwen3_tts: cannot parse " + model_dir + "/config.json";
    return result;
  }
  result.config = *config;
  if (!IsSupportedModelConfig(result.config)) {
    result.error =
        "qwen3_tts: expected a supported 12Hz 1.7B CustomVoice, "
        "VoiceDesign, or Base model configuration";
    return result;
  }

  const std::filesystem::path root(model_dir);
  std::string load_error;
  if (!LoadSafetensorsFile(root / "model.safetensors", result.store.get(),
                           &result.mappings, &result.mapped_regions,
                           &load_error)) {
    result.error =
        "qwen3_tts: cannot load " + (root / "model.safetensors").string();
    if (!load_error.empty())
      result.error += ": " + load_error;
    return result;
  }
  const std::filesystem::path speech_weights =
      root / "speech_tokenizer" / "model.safetensors";
  load_error.clear();
  if (!LoadSafetensorsFile(speech_weights, result.store.get(), &result.mappings,
                           &result.mapped_regions, &load_error)) {
    result.error = "qwen3_tts: cannot load " + speech_weights.string();
    if (!load_error.empty())
      result.error += ": " + load_error;
    return result;
  }
  result.ok = true;
  return result;
}

bool LooksLikeQwen3Tts(const std::string& model_dir) {
  const auto config = LoadModelConfigFromPath(model_dir);
  return config.has_value() && IsSupportedModelConfig(*config);
}

}  // namespace gufo::models::qwen3_tts
