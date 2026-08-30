#include "src/models/qwen3_asr/loader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "src/models/qwen3_tts/json.hpp"

namespace gufo::models::qwen3_asr {
namespace {

namespace json = gufo::models::qwen3_tts::json;

class MappedFile {
public:
  MappedFile() = default;
  ~MappedFile() { Reset(); }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&&) = delete;
  MappedFile& operator=(MappedFile&&) = delete;

  [[nodiscard]] bool Open(const std::filesystem::path& path) {
    struct stat status{};
    if (stat(path.c_str(), &status) != 0 || status.st_size <= 0) {
      return false;
    }
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
      return false;
    }
    void* mapping = mmap(nullptr, static_cast<std::size_t>(status.st_size),
                         PROT_READ, MAP_PRIVATE, descriptor, 0);
    close(descriptor);
    if (mapping == MAP_FAILED) {
      return false;
    }
    data_ = mapping;
    size_ = static_cast<std::size_t>(status.st_size);
    (void)madvise(data_, size_, MADV_SEQUENTIAL);
    return true;
  }

  [[nodiscard]] const std::byte* data() const noexcept {
    return static_cast<const std::byte*>(data_);
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
  void Reset() noexcept {
    if (data_ != nullptr) {
      (void)munmap(data_, size_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  void* data_{nullptr};
  std::size_t size_{0};
};

DType ParseDType(std::string_view name) noexcept {
  if (name == "F32") {
    return DType::kF32;
  }
  if (name == "BF16") {
    return DType::kBF16;
  }
  if (name == "F16") {
    return DType::kF16;
  }
  if (name == "I64") {
    return DType::kI64;
  }
  if (name == "I32") {
    return DType::kI32;
  }
  if (name == "U8") {
    return DType::kU8;
  }
  return DType::kUnknown;
}

std::uint64_t ReadLittleEndian64(const std::byte* bytes) noexcept {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index) {
    value |=
        static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[index]))
        << (index * 8U);
  }
  return value;
}

bool LoadSafetensors(const std::filesystem::path& path, TensorStore* store,
                     std::vector<std::shared_ptr<void>>* mappings,
                     std::vector<MappedRegion>* regions) {
  auto mapping = std::make_shared<MappedFile>();
  if (!mapping->Open(path) || mapping->size() < 8U) {
    return false;
  }
  const std::uint64_t header_size = ReadLittleEndian64(mapping->data());
  if (header_size == 0 || header_size > mapping->size() - 8U) {
    return false;
  }
  json::Value header;
  try {
    header = json::parse(
        std::string_view(reinterpret_cast<const char*>(mapping->data() + 8),
                         static_cast<std::size_t>(header_size)));
  } catch (const std::exception&) {
    return false;
  }
  if (!header.is_object()) {
    return false;
  }

  const std::uint64_t payload_offset = 8U + header_size;
  const std::size_t before = store->size();
  for (const auto& [name, specification] : header.members()) {
    if (name == "__metadata__" || !specification.is_object()) {
      continue;
    }
    Tensor tensor;
    tensor.name = name;
    tensor.dtype = ParseDType(specification.member_str("dtype"));
    if (const json::Value* shape = specification.find("shape");
        shape != nullptr && shape->is_array()) {
      for (const auto& dimension : shape->items()) {
        tensor.shape.push_back(dimension.as_size());
      }
    }
    const json::Value* offsets = specification.find("data_offsets");
    if (offsets == nullptr || !offsets->is_array() || offsets->size() != 2) {
      continue;
    }
    const std::uint64_t begin = offsets->items()[0].as_size();
    const std::uint64_t end = offsets->items()[1].as_size();
    const auto elements = tensor.NumElements();
    const std::size_t element_size = DTypeSize(tensor.dtype);
    if (!elements.has_value() || element_size == 0 || begin >= end ||
        end > mapping->size() - payload_offset ||
        *elements > std::numeric_limits<std::uint64_t>::max() / element_size ||
        end - begin != static_cast<std::uint64_t>(*elements) * element_size) {
      continue;
    }
    tensor.data = mapping->data() + payload_offset + begin;
    tensor.byte_count = end - begin;
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
      .payload_offset = static_cast<std::size_t>(payload_offset),
  });
  mappings->push_back(std::move(mapping));
  return true;
}

}  // namespace

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
    if (!LoadSafetensors(path, result.store.get(), &result.mappings,
                         &result.mapped_regions)) {
      result.error = "qwen3_asr: cannot load " + path.string();
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
