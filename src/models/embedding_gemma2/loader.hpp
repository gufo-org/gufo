#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_LOADER_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_LOADER_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "src/models/embedding_gemma2/config.hpp"
#include "src/models/embedding_gemma2/tensor.hpp"

namespace gufo::models::embedding_gemma2 {

/// Tensor name prefix of the text backbone in model.safetensors. The vision
/// and audio towers share the file and are never read.
inline constexpr const char* kTextTensorPrefix = "language_model.";

struct LoadResult {
  bool ok{false};
  std::string error;
  ModelConfig config;
  std::unique_ptr<TensorStore> store;
  /// Keeps the memory-mapped checkpoint alive while tensors point into it.
  std::vector<std::shared_ptr<void>> mappings;
};

/// Maps `<model_dir>/model.safetensors`, indexes the text-backbone tensors
/// and checks that each one exists with the shape and dtype the config
/// requires.
[[nodiscard]] LoadResult LoadModelDirectory(const std::string& model_dir);
[[nodiscard]] bool LooksLikeEmbeddingGemma2(const std::string& model_dir);

struct ExpectedTensor {
  std::string name;
  std::vector<std::uint64_t> shape;
};

/// Names every tensor the text backbone needs, with its expected shape.
[[nodiscard]] std::vector<ExpectedTensor> ExpectedTextTensors(
    const TextConfig& config);

}  // namespace gufo::models::embedding_gemma2

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_LOADER_HPP_
