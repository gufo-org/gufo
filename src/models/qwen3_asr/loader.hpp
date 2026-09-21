#ifndef GUFO_MODELS_QWEN3_ASR_LOADER_HPP_
#define GUFO_MODELS_QWEN3_ASR_LOADER_HPP_

#include <cstddef>
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
};

[[nodiscard]] LoadResult LoadModelDirectory(const std::string& model_dir);
[[nodiscard]] bool LooksLikeQwen3Asr(const std::string& model_dir);

}  // namespace gufo::models::qwen3_asr

#endif  // GUFO_MODELS_QWEN3_ASR_LOADER_HPP_
