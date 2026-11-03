#ifndef GUFO_MODELS_QWEN3_ASR_LOADER_HPP_
#define GUFO_MODELS_QWEN3_ASR_LOADER_HPP_

#include <cstddef>
#include <gufo/ext/gguf_reader.hpp>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_asr/config.hpp"
#include "src/models/qwen3_asr/tensor.hpp"

namespace gufo::models::qwen3_asr {

struct MappedRegion {
  const std::byte* data{nullptr};
  std::size_t size{0};
  std::size_t payload_offset{0};
  platform::PlatformFile file_descriptor{platform::kInvalidFile};
};

struct LoadResult {
  bool ok{false};
  std::string error;
  ModelConfig config;
  std::unique_ptr<TensorStore> store;
  std::vector<std::shared_ptr<void>> mappings;
  std::vector<MappedRegion> mapped_regions;

  /// Restrict each mapped shard to tensors used by this component, preserving
  /// the original payload alignment for GPU upload.
  [[nodiscard]] std::vector<MappedRegion> RegionsFor(
      std::initializer_list<std::string_view> prefixes) const;

  /// Drops the resident pages of every mapped shard once the caller's weights
  /// live in device memory. Pointers in `store` stay valid (pages refault).
  void ReleaseHostPages() const noexcept;
  [[nodiscard]] std::vector<core::GgufMappedRegion> WeightUploadRegions() const;
};

[[nodiscard]] LoadResult LoadModelDirectory(const std::string& model_dir);
[[nodiscard]] bool LooksLikeQwen3Asr(const std::string& model_dir);

}  // namespace gufo::models::qwen3_asr

#endif  // GUFO_MODELS_QWEN3_ASR_LOADER_HPP_
