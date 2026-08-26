#include "src/core/gguf_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <optional>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace strix::core {

namespace {

constexpr std::array<char, 4> kGgufMagic = {'G', 'G', 'U', 'F'};

template<typename T>
bool ReadPod(const std::uint8_t* data, std::size_t size, std::size_t& offset,
             T& out) {
  if (offset + sizeof(T) > size) {
    return false;
  }
  std::memcpy(&out, data + offset, sizeof(T));
  offset += sizeof(T);
  return true;
}

bool ReadString(const std::uint8_t* data, std::size_t size, std::size_t& offset,
                std::string_view& out) {
  std::uint64_t len = 0;
  if (!ReadPod(data, size, offset, len)) {
    return false;
  }
  if (offset + len > size) {
    return false;
  }
  out = std::string_view(reinterpret_cast<const char*>(data + offset), len);
  offset += len;
  return true;
}

}  // namespace

GgufReader::~GgufReader() {
  if (owns_mmap_ && mmap_addr_ != nullptr && size_ > 0) {
    munmap(mmap_addr_, size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

GgufReader::GgufReader(GgufReader&& other) noexcept
    : data_(other.data_),
      mmap_addr_(other.mmap_addr_),
      size_(other.size_),
      fd_(other.fd_),
      owns_mmap_(other.owns_mmap_),
      version_(other.version_),
      alignment_(other.alignment_),
      metadata_(std::move(other.metadata_)),
      tensors_(std::move(other.tensors_)),
      tensor_index_(std::move(other.tensor_index_)),
      mapped_regions_(std::move(other.mapped_regions_)),
      shards_(std::move(other.shards_)) {
  other.data_ = nullptr;
  other.mmap_addr_ = nullptr;
  other.size_ = 0;
  other.fd_ = -1;
  other.owns_mmap_ = false;
}

GgufReader& GgufReader::operator=(GgufReader&& other) noexcept {
  if (this != &other) {
    if (owns_mmap_ && mmap_addr_ != nullptr && size_ > 0) {
      munmap(mmap_addr_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }

    data_ = other.data_;
    mmap_addr_ = other.mmap_addr_;
    size_ = other.size_;
    fd_ = other.fd_;
    owns_mmap_ = other.owns_mmap_;
    version_ = other.version_;
    alignment_ = other.alignment_;
    metadata_ = std::move(other.metadata_);
    tensors_ = std::move(other.tensors_);
    tensor_index_ = std::move(other.tensor_index_);
    mapped_regions_ = std::move(other.mapped_regions_);
    shards_ = std::move(other.shards_);

    other.data_ = nullptr;
    other.mmap_addr_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
    other.owns_mmap_ = false;
  }
  return *this;
}

std::unique_ptr<GgufReader> GgufReader::OpenFile(
    const std::filesystem::path& path, std::string* error_msg) {
  auto first = OpenSingleFile(path, error_msg);
  if (!first) {
    return nullptr;
  }

  const auto split_count = first->GetMetadataUint32("split.count").value_or(1U);
  if (split_count <= 1) {
    return first;
  }
  return OpenSplitFileSet(path, std::move(first), error_msg);
}

std::unique_ptr<GgufReader> GgufReader::OpenSingleFile(
    const std::filesystem::path& path, std::string* error_msg) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to open file: " + path.string();
    }
    return nullptr;
  }

  struct stat sb{};
  if (fstat(fd, &sb) != 0 || sb.st_size <= 0) {
    close(fd);
    if (error_msg != nullptr) {
      *error_msg = "Invalid or empty file: " + path.string();
    }
    return nullptr;
  }

  const auto size = static_cast<std::size_t>(sb.st_size);
  void* const addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    close(fd);
    if (error_msg != nullptr) {
      *error_msg = "Failed to mmap file: " + path.string();
    }
    return nullptr;
  }
  (void)madvise(addr, size, MADV_SEQUENTIAL);

  auto reader = std::unique_ptr<GgufReader>(new GgufReader());
  reader->mmap_addr_ = addr;
  reader->data_ = static_cast<const std::uint8_t*>(addr);
  reader->size_ = size;
  reader->fd_ = fd;
  reader->owns_mmap_ = true;

  if (!reader->ParseHeaders(error_msg)) {
    return nullptr;
  }
  reader->mapped_regions_.push_back({reader->data_, reader->size_});
  return reader;
}

std::unique_ptr<GgufReader> GgufReader::OpenSplitFileSet(
    const std::filesystem::path& path, std::unique_ptr<GgufReader> first,
    std::string* error_msg) {
  const auto split_count = first->GetMetadataUint32("split.count").value_or(1U);
  const auto split_no = first->GetMetadataUint32("split.no");
  const auto total_tensor_count =
      first->GetMetadataUint64("split.tensors.count");
  if (!split_no.has_value() || *split_no >= split_count ||
      !total_tensor_count.has_value()) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid split GGUF metadata in " + path.string();
    }
    return nullptr;
  }

  const std::string filename = path.filename().string();
  const std::regex split_pattern{R"(^(.*)-([0-9]+)-of-([0-9]+)\.gguf$)"};
  std::smatch match;
  if (!std::regex_match(filename, match, split_pattern)) {
    if (error_msg != nullptr) {
      *error_msg =
          "Split GGUF filename does not match "
          "<name>-00001-of-00002.gguf: " +
          filename;
    }
    return nullptr;
  }

  const std::string prefix = match[1].str();
  const std::size_t index_width = match[2].str().size();
  const std::size_t count_width = match[3].str().size();
  const auto build_path = [&](std::uint32_t index) {
    std::ostringstream name;
    name << prefix << '-' << std::setfill('0')
         << std::setw(static_cast<int>(index_width)) << (index + 1) << "-of-"
         << std::setw(static_cast<int>(count_width)) << split_count << ".gguf";
    return path.parent_path() / name.str();
  };

  std::vector<std::unique_ptr<GgufReader>> shards(split_count);
  shards[*split_no] = std::move(first);
  for (std::uint32_t i = 0; i < split_count; ++i) {
    if (shards[i]) {
      continue;
    }
    const auto shard_path = build_path(i);
    shards[i] = OpenSingleFile(shard_path, error_msg);
    if (!shards[i]) {
      if (error_msg != nullptr && error_msg->empty()) {
        *error_msg = "Failed to open split GGUF shard: " + shard_path.string();
      }
      return nullptr;
    }
  }

  const auto architecture =
      shards[0]->GetMetadataString("general.architecture");
  for (std::uint32_t i = 0; i < split_count; ++i) {
    const auto shard_no = shards[i]->GetMetadataUint32("split.no");
    const auto shard_count = shards[i]->GetMetadataUint32("split.count");
    const auto shard_architecture =
        shards[i]->GetMetadataString("general.architecture");
    if (!shard_no.has_value() || *shard_no != i || !shard_count.has_value() ||
        *shard_count != split_count ||
        shards[i]->GetVersion() != shards[0]->GetVersion() ||
        (shard_architecture.has_value() &&
         shard_architecture != architecture)) {
      if (error_msg != nullptr) {
        *error_msg =
            "Inconsistent split GGUF shard metadata: " + build_path(i).string();
      }
      return nullptr;
    }
  }

  auto combined = std::unique_ptr<GgufReader>(new GgufReader());
  combined->version_ = shards[0]->version_;
  combined->alignment_ = shards[0]->alignment_;
  combined->metadata_ = shards[0]->metadata_;
  combined->tensors_.reserve(static_cast<std::size_t>(*total_tensor_count));
  combined->tensor_index_.reserve(
      static_cast<std::size_t>(*total_tensor_count));
  combined->mapped_regions_.reserve(split_count);

  for (const auto& shard : shards) {
    combined->size_ += shard->size_;
    combined->mapped_regions_.push_back({shard->data_, shard->size_});
    for (const auto& tensor : shard->tensors_) {
      if (combined->tensor_index_.contains(tensor.name)) {
        if (error_msg != nullptr) {
          *error_msg = "Duplicate tensor across split GGUF shards: " +
                       std::string(tensor.name);
        }
        return nullptr;
      }
      combined->tensor_index_[tensor.name] = combined->tensors_.size();
      combined->tensors_.push_back(tensor);
    }
  }

  if (combined->tensors_.size() != *total_tensor_count) {
    if (error_msg != nullptr) {
      *error_msg = "Split GGUF tensor count mismatch: expected " +
                   std::to_string(*total_tensor_count) + ", found " +
                   std::to_string(combined->tensors_.size());
    }
    return nullptr;
  }

  combined->shards_ = std::move(shards);
  return combined;
}

std::unique_ptr<GgufReader> GgufReader::OpenMemory(const void* data,
                                                   std::size_t size,
                                                   std::string* error_msg) {
  if (data == nullptr || size < 24) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid memory buffer for GGUF";
    }
    return nullptr;
  }

  auto reader = std::unique_ptr<GgufReader>(new GgufReader());
  reader->mmap_addr_ = nullptr;
  reader->data_ = static_cast<const std::uint8_t*>(data);
  reader->size_ = size;
  reader->fd_ = -1;
  reader->owns_mmap_ = false;

  if (!reader->ParseHeaders(error_msg)) {
    return nullptr;
  }
  reader->mapped_regions_.push_back({reader->data_, reader->size_});
  return reader;
}

bool GgufReader::ParseHeaders(std::string* error_msg) {
  if (size_ < 24) {
    if (error_msg != nullptr) {
      *error_msg = "File too small for GGUF header";
    }
    return false;
  }

  if (std::memcmp(data_, kGgufMagic.data(), 4) != 0) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid GGUF magic header";
    }
    return false;
  }

  std::size_t offset = 4;
  if (!ReadPod(data_, size_, offset, version_)) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to read GGUF version";
    }
    return false;
  }

  if (version_ != 2 && version_ != 3) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported GGUF version: " + std::to_string(version_);
    }
    return false;
  }

  std::uint64_t tensor_count = 0;
  std::uint64_t metadata_count = 0;
  if (!ReadPod(data_, size_, offset, tensor_count) ||
      !ReadPod(data_, size_, offset, metadata_count)) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to read tensor/metadata counts";
    }
    return false;
  }

  metadata_.reserve(metadata_count);

  // Parse metadata key-value pairs
  for (std::uint64_t i = 0; i < metadata_count; ++i) {
    std::string_view key;
    if (!ReadString(data_, size_, offset, key)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read metadata key";
      }
      return false;
    }

    std::uint32_t val_type_raw = 0;
    if (!ReadPod(data_, size_, offset, val_type_raw)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read metadata type for key " + std::string(key);
      }
      return false;
    }

    const auto val_type = static_cast<GgufValueType>(val_type_raw);
    GgufMetadataValue meta_val;
    meta_val.type = val_type;

    switch (val_type) {
      case GgufValueType::kUint8: {
        std::uint8_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt8: {
        std::int8_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kUint16: {
        std::uint16_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt16: {
        std::int16_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kUint32: {
        std::uint32_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt32: {
        std::int32_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kFloat32: {
        float v = 0.0F;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<double>(v);
        break;
      }
      case GgufValueType::kBool: {
        std::uint8_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = (v != 0);
        break;
      }
      case GgufValueType::kString: {
        std::string_view s;
        if (!ReadString(data_, size_, offset, s)) {
          return false;
        }
        meta_val.value = s;
        break;
      }
      case GgufValueType::kUint64: {
        std::uint64_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kInt64: {
        std::int64_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kFloat64: {
        double v = 0.0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kArray: {
        std::uint32_t item_type_raw = 0;
        std::uint64_t array_len = 0;
        if (!ReadPod(data_, size_, offset, item_type_raw) ||
            !ReadPod(data_, size_, offset, array_len)) {
          return false;
        }
        const auto item_type = static_cast<GgufValueType>(item_type_raw);
        if (item_type == GgufValueType::kString) {
          std::vector<std::string_view> str_arr;
          str_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::string_view s;
            if (!ReadString(data_, size_, offset, s)) {
              return false;
            }
            str_arr.push_back(s);
          }
          meta_val.value = str_arr;
        } else if (item_type == GgufValueType::kUint32 ||
                   item_type == GgufValueType::kUint64) {
          std::vector<std::uint64_t> u_arr;
          u_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::uint64_t val = 0;
            if (item_type == GgufValueType::kUint32) {
              std::uint32_t val32 = 0;
              if (!ReadPod(data_, size_, offset, val32)) {
                return false;
              }
              val = val32;
            } else {
              if (!ReadPod(data_, size_, offset, val)) {
                return false;
              }
            }
            u_arr.push_back(val);
          }
          meta_val.value = u_arr;
        } else if (item_type == GgufValueType::kInt32 ||
                   item_type == GgufValueType::kInt64) {
          std::vector<std::int64_t> i_arr;
          i_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::int64_t val = 0;
            if (item_type == GgufValueType::kInt32) {
              std::int32_t val32 = 0;
              if (!ReadPod(data_, size_, offset, val32)) {
                return false;
              }
              val = val32;
            } else if (!ReadPod(data_, size_, offset, val)) {
              return false;
            }
            i_arr.push_back(val);
          }
          meta_val.value = i_arr;
        } else {
          // Skip other array types cleanly
          std::size_t item_size = 4;
          if (item_type == GgufValueType::kUint8 ||
              item_type == GgufValueType::kInt8 ||
              item_type == GgufValueType::kBool) {
            item_size = 1;
          } else if (item_type == GgufValueType::kUint16 ||
                     item_type == GgufValueType::kInt16) {
            item_size = 2;
          } else if (item_type == GgufValueType::kUint64 ||
                     item_type == GgufValueType::kInt64 ||
                     item_type == GgufValueType::kFloat64) {
            item_size = 8;
          }
          if (offset + (array_len * item_size) > size_) {
            return false;
          }
          offset += (array_len * item_size);
        }
        break;
      }
    }

    if (key == "general.alignment") {
      if (std::holds_alternative<std::uint64_t>(meta_val.value)) {
        alignment_ = std::get<std::uint64_t>(meta_val.value);
      }
    }
    metadata_.emplace(key, std::move(meta_val));
  }

  // Parse tensor infos
  tensors_.reserve(tensor_count);
  tensor_index_.reserve(tensor_count);

  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    GgufTensorInfo info;
    if (!ReadString(data_, size_, offset, info.name)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read tensor name";
      }
      return false;
    }

    std::uint32_t n_dims = 0;
    if (!ReadPod(data_, size_, offset, n_dims) || n_dims > 8) {
      if (error_msg != nullptr) {
        *error_msg = "Invalid tensor dimensions count";
      }
      return false;
    }

    info.dimensions.resize(n_dims);
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      if (!ReadPod(data_, size_, offset, info.dimensions[d])) {
        return false;
      }
    }

    std::uint32_t type_raw = 0;
    if (!ReadPod(data_, size_, offset, type_raw)) {
      return false;
    }
    info.type = static_cast<GgmlType>(type_raw);

    if (!ReadPod(data_, size_, offset, info.offset)) {
      return false;
    }

    tensor_index_[info.name] = tensors_.size();
    tensors_.push_back(std::move(info));
  }

  // Align data payload base to alignment boundary
  const std::size_t data_base = (offset + alignment_ - 1) & ~(alignment_ - 1);
  for (auto& tensor : tensors_) {
    if (data_base + tensor.offset > size_) {
      if (error_msg != nullptr) {
        *error_msg =
            "Tensor offset exceeds file size: " + std::string(tensor.name);
      }
      return false;
    }
    tensor.data = data_ + data_base + tensor.offset;
  }

  return true;
}

const GgufMetadataValue* GgufReader::FindMetadata(
    std::string_view key) const noexcept {
  auto it = metadata_.find(key);
  if (it != metadata_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::optional<std::string_view> GgufReader::GetMetadataString(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr &&
      std::holds_alternative<std::string_view>(meta->value)) {
    return std::get<std::string_view>(meta->value);
  }
  return std::nullopt;
}

std::optional<std::uint32_t> GgufReader::GetMetadataUint32(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta == nullptr) {
    return std::nullopt;
  }
  if (const auto* value = std::get_if<std::uint64_t>(&meta->value)) {
    if (std::in_range<std::uint32_t>(*value)) {
      return static_cast<std::uint32_t>(*value);
    }
  } else if (const auto* value = std::get_if<std::int64_t>(&meta->value)) {
    if (std::in_range<std::uint32_t>(*value)) {
      return static_cast<std::uint32_t>(*value);
    }
  }
  return std::nullopt;
}

std::optional<std::uint64_t> GgufReader::GetMetadataUint64(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta == nullptr) {
    return std::nullopt;
  }
  if (const auto* value = std::get_if<std::uint64_t>(&meta->value)) {
    return *value;
  }
  if (const auto* value = std::get_if<std::int64_t>(&meta->value)) {
    if (std::in_range<std::uint64_t>(*value)) {
      return static_cast<std::uint64_t>(*value);
    }
  }
  return std::nullopt;
}

std::optional<float> GgufReader::GetMetadataFloat32(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<double>(meta->value)) {
    return static_cast<float>(std::get<double>(meta->value));
  }
  return std::nullopt;
}

std::optional<bool> GgufReader::GetMetadataBool(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<bool>(meta->value)) {
    return std::get<bool>(meta->value);
  }
  return std::nullopt;
}

std::vector<std::string_view> GgufReader::GetMetadataStringArray(
    std::string_view key) const {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr &&
      std::holds_alternative<std::vector<std::string_view>>(meta->value)) {
    return std::get<std::vector<std::string_view>>(meta->value);
  }
  return {};
}

const GgufTensorInfo* GgufReader::FindTensor(
    std::string_view name) const noexcept {
  auto it = tensor_index_.find(name);
  if (it != tensor_index_.end()) {
    return &tensors_[it->second];
  }
  return nullptr;
}

bool GgufReader::HasTensor(std::string_view name) const noexcept {
  return tensor_index_.contains(name);
}

bool GgufReader::HasMtpTensors() const noexcept {
  return std::ranges::any_of(tensors_, [](const auto& t) {
    return t.name.starts_with("mtp.") ||
           t.name.find(".mtp.") != std::string_view::npos ||
           t.name.find(".nextn.") != std::string_view::npos;
  });
}

bool GgufReader::HasVisionTensors() const noexcept {
  return std::ranges::any_of(tensors_, [](const auto& t) {
    return t.name.starts_with("model.visual") ||
           t.name.starts_with("visual.") || t.name.starts_with("v.");
  });
}

std::optional<ModelConfig> GgufReader::ExtractModelConfig(
    std::string* error_msg) const {
  ModelConfig config;

  // 1. Check architecture prefix (e.g. "qwen2", "qwen3", "qwen35", "llama")
  auto arch_str = GetMetadataString("general.architecture").value_or("qwen");
  config.architecture = std::string(arch_str);
  const std::string prefix = config.architecture + ".";

  // 2. Read standard architectural hyperparameters
  const auto total_layers =
      GetMetadataUint32(prefix + "block_count").value_or(config.num_layers);
  if (auto val = GetMetadataUint32(prefix + "embedding_length")) {
    config.hidden_size = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "feed_forward_length")) {
    config.intermediate_size = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "attention.head_count")) {
    config.num_attention_heads = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "attention.head_count_kv")) {
    config.num_key_value_heads = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "attention.key_length")) {
    config.head_dim = *val;
  } else if (config.num_attention_heads > 0) {
    config.head_dim = config.hidden_size / config.num_attention_heads;
  }
  if (auto val = GetMetadataUint32(prefix + "context_length")) {
    config.context_length = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "full_attention_interval")) {
    config.full_attention_interval = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "ssm.conv_kernel")) {
    config.ssm_conv_kernel = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "ssm.state_size")) {
    config.ssm_state_size = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "ssm.group_count")) {
    config.ssm_group_count = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "ssm.time_step_rank")) {
    config.ssm_time_step_rank = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "ssm.inner_size")) {
    config.ssm_inner_size = *val;
  }
  if (auto val = GetMetadataFloat32(prefix + "rope.freq_base")) {
    config.rope_theta = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "rope.dimension_count")) {
    config.rotary_dim = *val;
  } else {
    config.rotary_dim = config.head_dim;
  }

  // 3. MTP speculative layers are included in qwen35.block_count but are not
  // part of the main autoregressive transformer stack.
  const auto embedded_mtp_layers =
      GetMetadataUint32(prefix + "nextn_predict_layers");
  config.mtp_num_layers =
      embedded_mtp_layers.value_or(HasMtpTensors() ? 1U : 0U);
  const std::uint32_t layers_in_block_count = embedded_mtp_layers.value_or(0U);
  if (layers_in_block_count > total_layers) {
    if (error_msg != nullptr) {
      *error_msg = "MTP layer count exceeds qwen block count";
    }
    return std::nullopt;
  }
  config.num_layers = total_layers - layers_in_block_count;

  // 4. Assert text-only contract
  if (HasVisionTensors()) {
    config.is_text_only = false;
    if (error_msg != nullptr) {
      *error_msg =
          "Model contains vision encoder tensors; text-only contract violation";
    }
    return std::nullopt;
  }
  config.is_text_only = true;

  // 5. Model name
  if (auto name = GetMetadataString("general.name")) {
    config.model_name = std::string(*name);
  }
  if (const auto* token_embd = FindTensor("token_embd.weight");
      token_embd != nullptr && config.hidden_size > 0) {
    const auto elements = token_embd->ElementCount();
    if ((elements % config.hidden_size) == 0) {
      config.vocab_size =
          static_cast<std::uint32_t>(elements / config.hidden_size);
    }
  }

  // 6. Validate configuration against Qwen3.5/3.8 structural rules
  if (!config.IsValidQwen()) {
    if (error_msg != nullptr) {
      *error_msg = "Model config failed Qwen structural validation (head_dim=" +
                   std::to_string(config.head_dim) +
                   ", layers=" + std::to_string(config.num_layers) + ")";
    }
    return std::nullopt;
  }

  return config;
}

}  // namespace strix::core
