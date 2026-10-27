#ifndef GUFO_MODELS_FISH_DRAMA3_LOADER_HPP_
#define GUFO_MODELS_FISH_DRAMA3_LOADER_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/platform/mapped_file.hpp"
#include "src/models/fish_drama3/tensor.hpp"

namespace gufo::models::fish_drama3 {

class ModelSafetensors {
public:
  [[nodiscard]] const TensorView* Find(const std::string& name) const;
  [[nodiscard]] const TensorView& Require(const std::string& name) const;
  [[nodiscard]] const std::unordered_map<std::string, TensorView>& Tensors()
      const {
    return tensors_;
  }
  [[nodiscard]] std::size_t NumShards() const { return shards_.size(); }

private:
  friend std::unique_ptr<ModelSafetensors> LoadSafetensors(
      const std::string& model_dir);
  friend std::unique_ptr<ModelSafetensors> LoadSingleSafetensorsFile(
      const std::string& file_path);
  friend std::unique_ptr<ModelSafetensors> LoadCodecSafetensorsOrPth(
      const std::string& model_dir);

  std::vector<gufo::platform::ReadOnlyMappedFile> shards_;
  std::vector<std::vector<float>> owned_f32_buffers_;
  std::unordered_map<std::string, TensorView> tensors_;
};

[[nodiscard]] std::unique_ptr<ModelSafetensors> LoadSafetensors(
    const std::string& model_dir);
[[nodiscard]] std::unique_ptr<ModelSafetensors> LoadSingleSafetensorsFile(
    const std::string& file_path);
[[nodiscard]] std::unique_ptr<ModelSafetensors> LoadCodecSafetensorsOrPth(
    const std::string& model_dir);

}  // namespace gufo::models::fish_drama3

#endif  // GUFO_MODELS_FISH_DRAMA3_LOADER_HPP_
