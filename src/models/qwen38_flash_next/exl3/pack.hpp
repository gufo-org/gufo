// Read-only view of an EXL3 checkpoint directory: every `*.safetensors`
// file mapped, its tensors indexed by name, and the JSON configuration.
#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_PACK_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_PACK_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/platform/mapped_file.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::models::qwen38_flash_next::exl3 {

enum class DType : std::uint8_t { kF16, kBF16, kF32, kI16, kI32, kI64 };

struct PackTensor {
  std::string name;
  DType dtype{DType::kF16};
  std::vector<std::uint64_t> shape;
  const void* data{nullptr};  ///< Mapped payload.
  std::uint64_t size_bytes{0};
  std::uint32_t file{0};         ///< Index into Pack::Regions().
  std::uint64_t file_offset{0};  ///< Payload offset inside that file.

  [[nodiscard]] std::uint64_t Elements() const noexcept {
    std::uint64_t n = 1;
    for (const auto d : shape)
      n *= d;
    return n;
  }
};

class Pack {
public:
  /// Maps the checkpoint under `directory`. Every safetensors file found
  /// there is indexed; `config.json` must exist.
  [[nodiscard]] static std::unique_ptr<Pack> Open(
      const std::filesystem::path& directory, std::string* error);

  [[nodiscard]] const PackTensor* Find(std::string_view name) const noexcept;
  [[nodiscard]] const json::Value& Config() const noexcept { return config_; }
  /// The `text_config` object, or the whole config when it has none.
  [[nodiscard]] const json::Value& TextConfig() const noexcept;
  [[nodiscard]] const std::filesystem::path& Directory() const noexcept {
    return directory_;
  }
  /// One region per safetensors file, in Regions() index order, with a
  /// borrowed handle the weight uploader reads with direct I/O.
  [[nodiscard]] std::span<const core::GgufMappedRegion> Regions() const {
    return regions_;
  }
  [[nodiscard]] std::size_t TensorCount() const noexcept {
    return tensors_.size();
  }
  /// sha256 over config.json and every safetensors header, in file order:
  /// the checkpoint's identity for cache keys.
  [[nodiscard]] const std::string& Identity() const noexcept {
    return identity_;
  }
  /// Reads a whole text file next to the checkpoint ("" when missing).
  [[nodiscard]] std::string ReadText(std::string_view file) const;

private:
  Pack() = default;
  bool AddFile(const std::filesystem::path& path, std::string* error);

  std::filesystem::path directory_;
  json::Value config_;
  std::vector<platform::ReadOnlyMappedFile> files_;
  std::vector<core::GgufMappedRegion> regions_;
  std::vector<PackTensor> tensors_;
  std::unordered_map<std::string_view, std::size_t> index_;
  std::string identity_;
};

/// True when `directory` holds an EXL3 checkpoint of Qwen3.8-Flash-Next:
/// config.json with model_type qwen4_exp and quant_method exl3.
[[nodiscard]] bool IsFlashNextPack(const std::filesystem::path& directory);

/// The checkpoint's `quantization_config.bits` (4.05, ...), or 0.
[[nodiscard]] double PackBits(const std::filesystem::path& directory);

/// The text configuration's `max_position_embeddings`, or 0.
[[nodiscard]] std::uint32_t PackContextLength(
    const std::filesystem::path& directory);

}  // namespace gufo::models::qwen38_flash_next::exl3

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_PACK_HPP_
