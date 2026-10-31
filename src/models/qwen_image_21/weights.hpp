#ifndef GUFO_MODELS_QWEN_IMAGE_21_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_WEIGHTS_HPP_

#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"

namespace gufo::models::qwen_image_21 {

struct Weight {
  std::vector<int> shape;
  const void* data{nullptr};
  std::size_t bytes{0};
  bool bf16{true};
  /// Index into Weights::Files() and byte offset of `data` in that file.
  std::uint32_t file{0};
  std::uint64_t offset{0};
};

json::Value ReadJson(const std::filesystem::path& path);

/// One safetensors file and the prefix its tensor names receive.
struct SafetensorsSource {
  std::filesystem::path path;
  std::string prefix;
};

class Weights {
public:
  explicit Weights(const std::filesystem::path& root);
  /// Maps BF16/F32 safetensors files for another image model that reuses this
  /// runtime. Only the payload layout is validated, not a model configuration.
  static Weights FromFiles(std::span<const SafetensorsSource> files,
                           json::Value vae_config = {});
  const Weight& Get(std::string_view name) const;
  bool Contains(std::string_view name) const;
  void Prefetch(std::string_view prefix) const;
  /// Every tensor whose name starts with `prefix`, in name order.
  std::vector<std::pair<std::string_view, const Weight*>> Group(
      std::string_view prefix) const;
  /// The mapped safetensors files with readable handles, for WeightUpload.
  std::span<const core::GgufMappedRegion> Files() const { return files_; }
  const json::Value& VaeConfig() const { return vae_config_; }

private:
  Weights() = default;
  void Load(const std::filesystem::path& file, const std::string& prefix);
  std::vector<std::shared_ptr<void>> mappings_;
  std::vector<std::span<const std::byte>> regions_;
  std::vector<core::GgufMappedRegion> files_;
  std::map<std::string, Weight, std::less<>> weights_;
  json::Value vae_config_;
};

}  // namespace gufo::models::qwen_image_21
#endif
