#include "src/models/qwen3_tts/loader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/json.hpp"

namespace gufo::models::qwen3_tts {
namespace {

namespace json = gufo::json;

/// RAII mmap holder; moved into LoadResult::mappings.
class MappedFile {
public:
  MappedFile() = default;
  ~MappedFile() { Unmap(); }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  MappedFile(MappedFile&& other) noexcept
      : base_(std::exchange(other.base_, nullptr)), size_(other.size_) {}

  MappedFile& operator=(MappedFile&& other) noexcept {
    if (this != &other) {
      Unmap();
      base_ = std::exchange(other.base_, nullptr);
      size_ = other.size_;
    }
    return *this;
  }

  void* base() const { return base_; }
  std::size_t size() const { return size_; }

  /// Maps the file read-only. Returns false on failure.
  bool Open(const std::string& path) {
    struct stat status{};
    if (stat(path.c_str(), &status) != 0 || status.st_size <= 0) {
      return false;
    }
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      return false;
    }
    void* mapped = mmap(nullptr, static_cast<std::size_t>(status.st_size),
                        PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
      return false;
    }
    base_ = mapped;
    size_ = static_cast<std::size_t>(status.st_size);
    return true;
  }

private:
  void Unmap() {
    if (base_ != nullptr) {
      munmap(base_, size_);
      base_ = nullptr;
    }
  }

  void* base_ = nullptr;
  std::size_t size_ = 0;
};

DType DTypeFromString(std::string_view name) {
  if (name == "F32" || name == "f32") {
    return DType::kF32;
  }
  if (name == "BF16" || name == "bf16") {
    return DType::kBF16;
  }
  if (name == "F16" || name == "f16") {
    return DType::kF16;
  }
  if (name == "I64" || name == "i64") {
    return DType::kI64;
  }
  if (name == "I32" || name == "i32") {
    return DType::kI32;
  }
  if (name == "U8" || name == "u8") {
    return DType::kU8;
  }
  return DType::kUnknown;
}

std::uint64_t ReadLE64(const std::byte* p) {
  std::uint64_t value = 0;
  for (int index = 0; index < 8; ++index) {
    value |=
        static_cast<std::uint64_t>(std::to_integer<unsigned char>(p[index]))
        << (index * 8U);
  }
  return value;
}

/// Parses one safetensors file, adding tensors to `store`.
/// Adds a ownership mapping to `mappings` if the file is usable.
bool LoadSafetensorsFile(const std::string& path, TensorStore* store,
                         std::vector<std::shared_ptr<void>>* mappings,
                         std::vector<MappedRegion>* mapped_regions) {
  auto mapped = std::make_shared<MappedFile>();
  if (!mapped->Open(path) || mapped->size() < 8U) {
    return false;
  }
  const auto* bytes = static_cast<const std::byte*>(mapped->base());
  const std::uint64_t header_bytes = ReadLE64(bytes);
  if (header_bytes == 0 || header_bytes > mapped->size() - 8U) {
    return false;
  }
  json::Value header;
  try {
    header =
        json::parse(std::string_view(reinterpret_cast<const char*>(bytes + 8),
                                     static_cast<std::size_t>(header_bytes)));
  } catch (const std::exception&) {
    return false;
  }
  if (!header.is_object()) {
    return false;
  }

  const std::uint64_t payload_offset = 8U + header_bytes;
  const std::size_t count_before = store->size();
  for (const auto& [name, value] : header.members()) {
    if (name == "__metadata__" || !value.is_object()) {
      continue;
    }
    Tensor tensor;
    tensor.name = name;
    tensor.dtype = DTypeFromString(value.member_str("dtype"));
    const json::Value* shape_array = value.find("shape");
    if (shape_array != nullptr && shape_array->is_array()) {
      for (const auto& dim : shape_array->items()) {
        tensor.shape.push_back(dim.as_size());
      }
    }
    const auto* offsets = value.find("data_offsets");
    if (offsets == nullptr || !offsets->is_array() || offsets->size() != 2) {
      continue;
    }
    const std::uint64_t begin = offsets->items()[0].as_size();
    const std::uint64_t end = offsets->items()[1].as_size();
    const auto elements = tensor.NumElements();
    const std::size_t element_bytes = DTypeSize(tensor.dtype);
    if (elements == std::nullopt || element_bytes == 0 || begin >= end ||
        end > static_cast<std::uint64_t>(mapped->size()) - payload_offset) {
      continue;
    }
    if (*elements > std::numeric_limits<std::uint64_t>::max() / element_bytes) {
      continue;
    }
    const std::uint64_t bytes_count = end - begin;
    const std::uint64_t expected_bytes =
        static_cast<std::uint64_t>(*elements) * element_bytes;
    if (expected_bytes != bytes_count) {
      continue;
    }
    tensor.data = bytes + static_cast<std::ptrdiff_t>(payload_offset + begin);
    tensor.byte_count = bytes_count;
    store->Add(std::move(tensor));
  }
  if (store->size() == count_before) {
    return false;
  }
  mapped_regions->push_back(
      {.data = bytes,
       .size = static_cast<std::size_t>(mapped->size()),
       .payload_offset = static_cast<std::size_t>(payload_offset)});
  mappings->push_back(std::move(mapped));
  return true;
}

}  // namespace

std::vector<MappedRegion> LoadResult::RegionsFor(
    std::string_view prefix) const {
  std::vector<MappedRegion> result;
  if (!store)
    return result;
  const auto page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || page_size % 16 != 0)
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
  if (!LoadSafetensorsFile((root / "model.safetensors").string(),
                           result.store.get(), &result.mappings,
                           &result.mapped_regions)) {
    result.error =
        "qwen3_tts: cannot load " + (root / "model.safetensors").string();
    return result;
  }
  const std::filesystem::path speech_weights =
      root / "speech_tokenizer" / "model.safetensors";
  if (!LoadSafetensorsFile(speech_weights.string(), result.store.get(),
                           &result.mappings, &result.mapped_regions)) {
    result.error = "qwen3_tts: cannot load " + speech_weights.string();
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
