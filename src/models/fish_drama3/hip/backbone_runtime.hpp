#ifndef GUFO_MODELS_FISH_DRAMA3_HIP_BACKBONE_RUNTIME_HPP_
#define GUFO_MODELS_FISH_DRAMA3_HIP_BACKBONE_RUNTIME_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/fish_drama3/config.hpp"
#include "src/models/fish_drama3/loader.hpp"

namespace gufo::models::fish_drama3::hip {

class BackboneHipRuntime {
public:
  ~BackboneHipRuntime();

  BackboneHipRuntime(const BackboneHipRuntime&) = delete;
  BackboneHipRuntime& operator=(const BackboneHipRuntime&) = delete;

  [[nodiscard]] static std::unique_ptr<BackboneHipRuntime> Create(
      const ModelSafetensors& store, const ModelConfig& model_config,
      std::size_t max_cache_steps = 4096, std::string* error = nullptr);

  void Reset();

  [[nodiscard]] std::vector<float> TokenEmbedding(
      std::span<const std::int32_t> token_ids) const;

  // Runs `steps` frames starting at current `valid_steps()`, advancing
  // `valid_steps()` by `steps`. Writes the final normed hidden states [steps *
  // hidden_size] to `out_hidden`.
  [[nodiscard]] bool ForwardEmbeddings(
      std::span<const float> embeddings, std::size_t steps,
      std::vector<float>* out_hidden, std::string* error = nullptr,
      std::vector<float>* out_last_logits = nullptr);

  // Computes compact [1 + codebook_size] semantic logits ([EOS, semantic
  // 0..4095]) from a single frame's final normed hidden state [hidden_size].
  [[nodiscard]] bool ComputeLmHeadLogits(std::span<const float> last_hidden,
                                         std::vector<float>* out_logits,
                                         std::string* error = nullptr);

  [[nodiscard]] std::size_t valid_steps() const noexcept;
  [[nodiscard]] const BackboneConfig& config() const noexcept;

private:
  struct Impl;
  explicit BackboneHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::fish_drama3::hip

#endif  // GUFO_MODELS_FISH_DRAMA3_HIP_BACKBONE_RUNTIME_HPP_
