// A frozen copy of GgufReader's header parsing in 0.35.0
// (src/core/gguf_reader.cpp), which gguf_header_test compares with the Rust
// parser. ParseHeaders and its helpers are unchanged; the reader members it
// uses are fields of Reader, and OpenMemory and OpenSingleFile's handling
// around it returns a Result (marked "Test hook"). Do not change them to
// match new behavior.
#include "tests/core/gguf_header_reference.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::gguf_header_reference {
namespace {

using core::GgmlType;
using core::GgufMetadataValue;
using core::GgufOpenOptions;
using core::GgufTensorInfo;
using core::GgufValueType;
namespace quant = gufo::quant;

constexpr std::array<char, 4> kGgufMagic = {'G', 'G', 'U', 'F'};

template<typename T>
bool ReadPod(const std::uint8_t* data, std::size_t size, std::size_t& offset,
             T& out) {
  if (offset > size || sizeof(T) > size - offset) {
    return false;
  }
  std::memcpy(&out, data + offset, sizeof(T));
  offset += sizeof(T);
  return true;
}

std::string UnsizedError(std::string_view name) {
  return "Invalid GGUF tensor shape or unsupported storage type: " +
         std::string(name);
}

bool ReadString(const std::uint8_t* data, std::size_t size, std::size_t& offset,
                std::string_view& out) {
  std::uint64_t len = 0;
  if (!ReadPod(data, size, offset, len)) {
    return false;
  }
  if (offset > size || len > size - offset) {
    return false;
  }
  out = std::string_view(reinterpret_cast<const char*>(data + offset), len);
  offset += len;
  return true;
}

// Test hook: the GgufReader members that ParseHeaders reads and writes.
struct Reader {
  const std::uint8_t* data_{nullptr};
  std::size_t size_{0};
  std::uint32_t version_{0};
  std::uint64_t alignment_{32};
  std::unordered_map<std::string_view, GgufMetadataValue> metadata_;
  std::vector<GgufTensorInfo> tensors_;
  std::unordered_map<std::string_view, std::size_t> tensor_index_;

  [[nodiscard]] const GgufTensorInfo* FirstUnsizedTensor() const noexcept {
    const auto found =
        std::ranges::find(tensors_, false, &GgufTensorInfo::sized);
    return found == tensors_.end() ? nullptr : &*found;
  }

  bool ParseHeaders(std::string* error_msg, GgufOpenOptions options = {});
};

bool Reader::ParseHeaders(std::string* error_msg, GgufOpenOptions options) {
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

  // Bound host-side descriptor allocations independently of mapped weight size.
  std::size_t budget = 128 * 1024 * 1024;
  const auto charge = [&](std::uint64_t count, std::size_t item_bytes) {
    if (count > budget / item_bytes)
      return false;
    budget -= static_cast<std::size_t>(count) * item_bytes;
    return true;
  };
  if (metadata_count > (size_ - offset) / 13 ||
      tensor_count > (size_ - offset) / 32 ||
      !charge(metadata_count, sizeof(GgufMetadataValue) + 64) ||
      !charge(tensor_count, sizeof(GgufTensorInfo) + 128)) {
    if (error_msg)
      *error_msg = "GGUF descriptor counts exceed file or memory limits";
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

    if (key.empty() || key.size() > 65535 || !charge(key.size(), 1)) {
      if (error_msg)
        *error_msg = "GGUF metadata key exceeds limits";
      return false;
    }
    std::uint32_t val_type_raw = 0;
    if (!ReadPod(data_, size_, offset, val_type_raw)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read metadata type for key " + std::string(key);
      }
      return false;
    }

    if (val_type_raw > static_cast<std::uint32_t>(GgufValueType::kFloat64))
      return false;
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
        if (!ReadPod(data_, size_, offset, v) || v > 1) {
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
        if (!charge(s.size(), 1))
          return false;
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
        if (item_type_raw >
                static_cast<std::uint32_t>(GgufValueType::kFloat64) ||
            item_type_raw == static_cast<std::uint32_t>(GgufValueType::kArray))
          return false;
        constexpr std::size_t item_bytes[] = {1, 1, 2, 2, 4, 4, 4,
                                              1, 8, 0, 8, 8, 8};
        if (offset > size_ ||
            array_len > (size_ - offset) / item_bytes[item_type_raw] ||
            !charge(array_len,
                    item_type_raw == 8 ? sizeof(std::string_view) : 8))
          return false;
        const auto item_type = static_cast<GgufValueType>(item_type_raw);
        if (item_type == GgufValueType::kString) {
          std::vector<std::string_view> str_arr;
          str_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::string_view s;
            if (!ReadString(data_, size_, offset, s)) {
              return false;
            }
            if (!charge(s.size(), 1))
              return false;
            str_arr.push_back(s);
          }
          meta_val.value = std::move(str_arr);
        } else if (item_type == GgufValueType::kFloat32 ||
                   item_type == GgufValueType::kFloat64) {
          const auto bytes = item_type == GgufValueType::kFloat32 ? 4U : 8U;
          if (offset > size_ || array_len > (size_ - offset) / bytes)
            return false;
          std::vector<double> values;
          values.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            double value = 0;
            if (item_type == GgufValueType::kFloat32) {
              float scalar = 0;
              if (!ReadPod(data_, size_, offset, scalar))
                return false;
              value = scalar;
            } else if (!ReadPod(data_, size_, offset, value)) {
              return false;
            }
            values.push_back(value);
          }
          meta_val.value = std::move(values);
        } else if (item_type == GgufValueType::kUint8 ||
                   item_type == GgufValueType::kUint16 ||
                   item_type == GgufValueType::kUint32 ||
                   item_type == GgufValueType::kBool ||
                   item_type == GgufValueType::kUint64) {
          // Writers pick the narrowest integer type; widen every width.
          std::vector<std::uint64_t> u_arr;
          u_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::uint64_t val = 0;
            if (item_type == GgufValueType::kUint8) {
              std::uint8_t val8 = 0;
              if (!ReadPod(data_, size_, offset, val8))
                return false;
              val = val8;
            } else if (item_type == GgufValueType::kUint16) {
              std::uint16_t val16 = 0;
              if (!ReadPod(data_, size_, offset, val16))
                return false;
              val = val16;
            } else if (item_type == GgufValueType::kUint32) {
              std::uint32_t val32 = 0;
              if (!ReadPod(data_, size_, offset, val32)) {
                return false;
              }
              val = val32;
            } else if (item_type == GgufValueType::kBool) {
              std::uint8_t boolean = 0;
              if (!ReadPod(data_, size_, offset, boolean) || boolean > 1)
                return false;
              val = boolean;
            } else {
              if (!ReadPod(data_, size_, offset, val)) {
                return false;
              }
            }
            u_arr.push_back(val);
          }
          meta_val.value = std::move(u_arr);
        } else if (item_type == GgufValueType::kInt8 ||
                   item_type == GgufValueType::kInt16 ||
                   item_type == GgufValueType::kInt32 ||
                   item_type == GgufValueType::kInt64) {
          std::vector<std::int64_t> i_arr;
          i_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::int64_t val = 0;
            if (item_type == GgufValueType::kInt8) {
              std::int8_t val8 = 0;
              if (!ReadPod(data_, size_, offset, val8))
                return false;
              val = val8;
            } else if (item_type == GgufValueType::kInt16) {
              std::int16_t val16 = 0;
              if (!ReadPod(data_, size_, offset, val16))
                return false;
              val = val16;
            } else if (item_type == GgufValueType::kInt32) {
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
          meta_val.value = std::move(i_arr);
        } else {
          // Every scalar item type is decoded above; never store an array
          // entry without its values.
          return false;
        }
        break;
      }
    }

    if (key == "general.alignment") {
      const auto* value = std::get_if<std::uint64_t>(&meta_val.value);
      if (!value || !std::has_single_bit(*value) || *value > UINT32_MAX) {
        if (error_msg)
          *error_msg = "Invalid GGUF alignment";
        return false;
      }
      alignment_ = *value;
    }
    if (!metadata_.emplace(key, std::move(meta_val)).second) {
      if (error_msg)
        *error_msg = "Duplicate GGUF metadata key";
      return false;
    }
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

    if (info.name.empty() || info.name.size() > 65535 ||
        !charge(info.name.size(), 1))
      return false;
    std::uint32_t n_dims = 0;
    if (!ReadPod(data_, size_, offset, n_dims) || n_dims == 0 || n_dims > 8) {
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
    if (type_raw > UINT16_MAX)
      return false;
    info.type = static_cast<GgmlType>(type_raw);
    const auto elements = info.ElementCount();
    const auto block = quant::QuantizedBlockElements(info.type);
    info.size_bytes = quant::EncodedSizeBytes(info.type, elements);
    if (!info.size_bytes || (block && info.dimensions.front() % block != 0)) {
      // An extension may own this layout; its extent follows from the next
      // tensor's offset once every offset is known.
      if (options.allow_unsized_tensors && elements != 0) {
        info.sized = false;
        info.size_bytes = 0;
      } else {
        if (error_msg)
          *error_msg = UnsizedError(info.name);
        return false;
      }
    }

    if (!ReadPod(data_, size_, offset, info.offset)) {
      return false;
    }

    if (!tensor_index_.emplace(info.name, tensors_.size()).second) {
      if (error_msg)
        *error_msg = "Duplicate GGUF tensor name";
      return false;
    }
    tensors_.push_back(std::move(info));
  }

  // Align data payload base to alignment boundary. A tensorless file, such as
  // a metadata-only first split shard, may end unpadded after its header.
  const auto padding = (alignment_ - offset % alignment_) % alignment_;
  if (tensors_.empty())
    return true;
  if (offset > size_ || padding > size_ - offset) {
    if (error_msg)
      *error_msg = "GGUF data section is truncated before its alignment";
    return false;
  }
  const std::size_t data_base = offset + padding;
  const std::size_t data_size = size_ - data_base;
  std::vector<std::uint64_t> offsets;
  if (FirstUnsizedTensor() != nullptr) {
    offsets.reserve(tensors_.size());
    for (const auto& tensor : tensors_)
      offsets.push_back(tensor.offset);
    std::ranges::sort(offsets);
  }
  for (auto& tensor : tensors_) {
    if (!tensor.sized && tensor.offset < data_size) {
      const auto next = std::ranges::upper_bound(offsets, tensor.offset);
      tensor.size_bytes = static_cast<std::size_t>(
          (next == offsets.end() ? data_size : *next) - tensor.offset);
    }
    if (tensor.offset % alignment_ != 0 || tensor.offset > data_size ||
        (!tensor.sized && tensor.size_bytes == 0) ||
        tensor.size_bytes > data_size - tensor.offset) {
      if (error_msg != nullptr) {
        *error_msg = "Tensor extent or alignment is invalid: " +
                     std::string(tensor.name);
      }
      return false;
    }
    tensor.data = data_ + data_base + tensor.offset;
  }

  return true;
}

}  // namespace

// Test hook: OpenSingleFile's handling around ParseHeaders.
Result Parse(const std::uint8_t* data, std::size_t size,
             core::GgufOpenOptions options) {
  Result result;
  std::string* error_msg = &result.error;
  Reader reader;
  reader.data_ = data;
  reader.size_ = size;
  try {
    if (!reader.ParseHeaders(error_msg, options)) {
      if (error_msg && error_msg->empty())
        *error_msg = "Malformed or truncated GGUF";
      return result;
    }
  } catch (const std::bad_alloc&) {
    if (error_msg)
      *error_msg = "Insufficient memory for GGUF descriptors";
    return result;
  }
  result.opened = true;
  result.version = reader.version_;
  result.alignment = reader.alignment_;
  result.metadata = std::move(reader.metadata_);
  result.tensors = std::move(reader.tensors_);
  return result;
}

// Test hook: OpenMemory's checks before ParseHeaders.
Result OpenMemory(const void* data, std::size_t size,
                  core::GgufOpenOptions options) {
  if (data == nullptr || size < 24) {
    Result result;
    result.error = "Invalid memory buffer for GGUF";
    return result;
  }
  return Parse(static_cast<const std::uint8_t*>(data), size, options);
}

}  // namespace gufo::gguf_header_reference
