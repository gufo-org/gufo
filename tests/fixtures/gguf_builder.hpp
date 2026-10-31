#ifndef GUFO_TESTS_FIXTURES_GGUF_BUILDER_HPP_
#define GUFO_TESTS_FIXTURES_GGUF_BUILDER_HPP_

// Synthetic GGUF builder shared by the core reader tests and the Qwen family
// configuration test, so both exercise the same byte layout.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::test_fixture {

// Helper to construct a synthetic valid GGUF binary in memory
class GgufBuilder {
public:
  GgufBuilder(std::uint32_t version = 3) {
    // Magic "GGUF"
    const char magic[4] = {'G', 'G', 'U', 'F'};
    AppendBytes(magic, 4);
    AppendPod(version);
    // Reserve space for tensor_count and metadata_count (filled on Build)
    tensor_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
    metadata_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
  }

  void AddMetadataString(std::string_view key, std::string_view val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendString(val);
    metadata_count_++;
  }

  void AddMetadataUint32(std::string_view key, std::uint32_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint32));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataUint64(std::string_view key, std::uint64_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint64));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataInt32(std::string_view key, std::int32_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kInt32));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataFloat32(std::string_view key, float val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kFloat32));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataBool(std::string_view key, bool val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kBool));
    std::uint8_t b = val ? 1 : 0;
    AppendPod(b);
    metadata_count_++;
  }

  void AddMetadataInt32Array(std::string_view key,
                             const std::vector<std::int32_t>& values) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kInt32));
    AppendPod(static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) {
      AppendPod(value);
    }
    metadata_count_++;
  }

  template<class T>
  void AddMetadataArray(std::string_view key, gufo::core::GgufValueType type,
                        const std::vector<T>& values) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(type));
    AppendPod(static_cast<std::uint64_t>(values.size()));
    for (const auto value : values)
      AppendPod(value);
    ++metadata_count_;
  }

  void AddTensor(std::string_view name, const std::vector<std::uint64_t>& dims,
                 gufo::core::GgmlType type, std::uint64_t offset) {
    tensors_to_write_.push_back({std::string(name), dims, type, offset});
  }

  std::vector<std::uint8_t> Build(std::size_t payload_bytes = 1024) {
    // Patch metadata count
    std::memcpy(buffer_.data() + metadata_count_pos_, &metadata_count_,
                sizeof(metadata_count_));

    // Write tensors
    std::uint64_t tensor_count = tensors_to_write_.size();
    std::memcpy(buffer_.data() + tensor_count_pos_, &tensor_count,
                sizeof(tensor_count));

    for (const auto& t : tensors_to_write_) {
      AppendString(t.name);
      AppendPod(static_cast<std::uint32_t>(t.dims.size()));
      for (auto d : t.dims) {
        AppendPod(d);
      }
      AppendPod(static_cast<std::uint32_t>(t.type));
      AppendPod(t.offset);
    }

    // Align to 32 bytes
    std::size_t rem = buffer_.size() % 32;
    if (rem != 0) {
      buffer_.resize(buffer_.size() + (32 - rem), 0);
    }

    // Fixtures carry the complete declared payload, just like real files.
    for (const auto& tensor : tensors_to_write_) {
      std::size_t elements = 1;
      for (auto dimension : tensor.dims)
        elements *= dimension;
      payload_bytes =
          std::max(payload_bytes,
                   static_cast<std::size_t>(tensor.offset) +
                       gufo::quant::EncodedSizeBytes(tensor.type, elements));
    }
    // Append payload dummy data
    buffer_.resize(buffer_.size() + payload_bytes, 0xAB);
    return buffer_;
  }

private:
  struct TensorRecord {
    std::string name;
    std::vector<std::uint64_t> dims;
    gufo::core::GgmlType type;
    std::uint64_t offset;
  };

  void AppendBytes(const void* data, std::size_t len) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), p, p + len);
  }

  template<typename T>
  void AppendPod(T val) {
    AppendBytes(&val, sizeof(T));
  }

  void AppendString(std::string_view s) {
    std::uint64_t len = s.size();
    AppendPod(len);
    AppendBytes(s.data(), len);
  }

  std::vector<std::uint8_t> buffer_;
  std::size_t tensor_count_pos_{0};
  std::size_t metadata_count_pos_{0};
  std::uint64_t metadata_count_{0};
  std::vector<TensorRecord> tensors_to_write_;
};


}  // namespace gufo::test_fixture

#endif  // GUFO_TESTS_FIXTURES_GGUF_BUILDER_HPP_
