#ifndef GUFO_CORE_GGUF_READER_HPP_
#define GUFO_CORE_GGUF_READER_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "src/core/model_config.hpp"

namespace gufo::core {

/// GGML tensor types including standard GGML and custom Strix Halo layouts.
enum class GgmlType : std::uint16_t {
  kF32 = 0,
  kF16 = 1,
  kQ4_0 = 2,
  kQ4_1 = 3,
  kQ5_0 = 6,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ8_1 = 9,
  kQ2_K = 10,
  kQ3_K = 11,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kQ8_K = 15,
  kIQ2_XXS = 16,
  kIQ4_NL = 20,
  kIQ3_S = 21,
  kIQ4_XS = 23,
  kBF16 = 30,

  // Strix Halo specialized hardware quantization types (Wave32 & XDNA2 NPU
  // aligned)
  kStrixSHQ4_T16 = 1000,
  kStrixSHQ6_T16 = 1001,
  kStrixSHQ8_T16 = 1002,
};

[[nodiscard]] constexpr std::string_view ToString(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return "F32";
    case GgmlType::kF16:
      return "F16";
    case GgmlType::kQ4_0:
      return "Q4_0";
    case GgmlType::kQ4_1:
      return "Q4_1";
    case GgmlType::kQ5_0:
      return "Q5_0";
    case GgmlType::kQ5_1:
      return "Q5_1";
    case GgmlType::kQ8_0:
      return "Q8_0";
    case GgmlType::kQ8_1:
      return "Q8_1";
    case GgmlType::kQ2_K:
      return "Q2_K";
    case GgmlType::kQ3_K:
      return "Q3_K";
    case GgmlType::kQ4_K:
      return "Q4_K";
    case GgmlType::kQ5_K:
      return "Q5_K";
    case GgmlType::kQ6_K:
      return "Q6_K";
    case GgmlType::kQ8_K:
      return "Q8_K";
    case GgmlType::kIQ2_XXS:
      return "IQ2_XXS";
    case GgmlType::kIQ4_NL:
      return "IQ4_NL";
    case GgmlType::kIQ3_S:
      return "IQ3_S";
    case GgmlType::kIQ4_XS:
      return "IQ4_XS";
    case GgmlType::kBF16:
      return "BF16";
    case GgmlType::kStrixSHQ4_T16:
      return "GUFO_SHQ4_T16";
    case GgmlType::kStrixSHQ6_T16:
      return "GUFO_SHQ6_T16";
    case GgmlType::kStrixSHQ8_T16:
      return "GUFO_SHQ8_T16";
  }
  return "UNKNOWN";
}

/// GGUF metadata value types as specified by the GGUF binary format.
enum class GgufValueType : std::uint8_t {
  kUint8 = 0,
  kInt8 = 1,
  kUint16 = 2,
  kInt16 = 3,
  kUint32 = 4,
  kInt32 = 5,
  kFloat32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kUint64 = 10,
  kInt64 = 11,
  kFloat64 = 12,
};

/// Metadata entry holding typed scalar, string, or array data.
struct GgufMetadataValue {
  GgufValueType type{GgufValueType::kUint32};
  std::variant<std::uint64_t, std::int64_t, double, bool, std::string_view,
               std::vector<std::string_view>, std::vector<std::uint64_t>,
               std::vector<std::int64_t>, std::vector<double>>
      value;
};

/// Metadata and memory pointer for a single tensor inside the GGUF container.
struct GgufTensorInfo {
  std::string_view name;
  std::vector<std::uint64_t> dimensions;
  GgmlType type{GgmlType::kF16};
  std::uint64_t offset{0};
  const void* data{nullptr};
  std::size_t size_bytes{0};

  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    if (dimensions.empty()) {
      return 0;
    }
    std::uint64_t count = 1;
    for (const auto dimension : dimensions) {
      if (dimension == 0 ||
          count > std::numeric_limits<std::uint64_t>::max() / dimension) {
        return 0;
      }
      count *= dimension;
    }
    return count;
  }
};

/// One contiguous mapped GGUF file region. Split models expose one region per
/// shard while single-file and in-memory readers expose exactly one.
struct GgufMappedRegion {
  const void* data{nullptr};
  std::size_t size{0};
  /// Borrowed descriptor, valid for the reader lifetime; -1 for memory images.
  int file_descriptor{-1};
};

/// Zero-copy, lightweight GGUF binary reader and tensor indexer.
class GgufReader {
public:
  ~GgufReader();

  GgufReader(const GgufReader&) = delete;
  GgufReader& operator=(const GgufReader&) = delete;
  GgufReader(GgufReader&& other) noexcept;
  GgufReader& operator=(GgufReader&& other) noexcept;

  /// Memory-maps a .gguf file from disk and parses metadata and tensor headers.
  [[nodiscard]] static std::unique_ptr<GgufReader> OpenFile(
      const std::filesystem::path& path, std::string* error_msg = nullptr);

  /// Parses a GGUF file from an existing in-memory buffer without file mapping.
  [[nodiscard]] static std::unique_ptr<GgufReader> OpenMemory(
      const void* data, std::size_t size, std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t GetVersion() const noexcept { return version_; }
  [[nodiscard]] std::uint64_t GetTensorCount() const noexcept {
    return tensors_.size();
  }
  [[nodiscard]] std::uint64_t GetMetadataCount() const noexcept {
    return metadata_.size();
  }
  [[nodiscard]] std::uint64_t GetAlignment() const noexcept {
    return alignment_;
  }
  /// Returns the contiguous backing pointer for single-file readers only.
  [[nodiscard]] const void* GetData() const noexcept { return data_; }
  [[nodiscard]] std::size_t GetSize() const noexcept { return size_; }
  [[nodiscard]] std::span<const GgufMappedRegion> GetMappedRegions()
      const noexcept {
    return mapped_regions_;
  }

  /// Metadata lookup helpers
  [[nodiscard]] const GgufMetadataValue* FindMetadata(
      std::string_view key) const noexcept;
  [[nodiscard]] std::optional<std::string_view> GetMetadataString(
      std::string_view key) const noexcept;
  [[nodiscard]] std::optional<std::uint32_t> GetMetadataUint32(
      std::string_view key) const noexcept;
  [[nodiscard]] std::optional<std::uint64_t> GetMetadataUint64(
      std::string_view key) const noexcept;
  [[nodiscard]] std::optional<float> GetMetadataFloat32(
      std::string_view key) const noexcept;
  [[nodiscard]] std::optional<bool> GetMetadataBool(
      std::string_view key) const noexcept;
  [[nodiscard]] std::vector<std::string_view> GetMetadataStringArray(
      std::string_view key) const;
  /// Tensor lookup helpers
  [[nodiscard]] const GgufTensorInfo* FindTensor(
      std::string_view name) const noexcept;
  [[nodiscard]] bool HasTensor(std::string_view name) const noexcept;
  [[nodiscard]] std::span<const GgufTensorInfo> GetTensors() const noexcept {
    return tensors_;
  }

  /// Extracts and validates model configuration from embedded config or
  /// standard keys.
  [[nodiscard]] std::optional<ModelConfig> ExtractModelConfig(
      std::string* error_msg = nullptr) const;

  /// Names the weight format this file was produced in, for benchmark and
  /// diagnostic output. Prefers the `general.file_type` the quantizer recorded
  /// (llama.cpp's ftype, so the label matches `llama-bench` on the same
  /// artifact) and otherwise falls back to whichever tensor type holds the most
  /// bytes. Never empty.
  [[nodiscard]] std::string GetQuantizationLabel() const;

  /// Inspects presence of MTP speculative tensors
  [[nodiscard]] bool HasMtpTensors() const noexcept;

  /// Inspects presence of vision encoder tensors
  [[nodiscard]] bool HasVisionTensors() const noexcept;

private:
  GgufReader() = default;

  [[nodiscard]] static std::unique_ptr<GgufReader> OpenSingleFile(
      const std::filesystem::path& path, std::string* error_msg);
  [[nodiscard]] static std::unique_ptr<GgufReader> OpenSplitFileSet(
      const std::filesystem::path& path, std::unique_ptr<GgufReader> first,
      std::string* error_msg);

  bool ParseHeaders(std::string* error_msg);

  const std::uint8_t* data_{nullptr};
  void* mmap_addr_{nullptr};
  std::size_t size_{0};
  int fd_{-1};
  bool owns_mmap_{false};

  std::uint32_t version_{0};
  std::uint64_t alignment_{32};
  std::unordered_map<std::string_view, GgufMetadataValue> metadata_;
  std::vector<GgufTensorInfo> tensors_;
  std::unordered_map<std::string_view, std::size_t> tensor_index_;
  std::vector<GgufMappedRegion> mapped_regions_;
  std::vector<std::unique_ptr<GgufReader>> shards_;
};

}  // namespace gufo::core

#endif  // GUFO_CORE_GGUF_READER_HPP_
