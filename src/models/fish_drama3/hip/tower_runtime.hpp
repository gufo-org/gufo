#ifndef GUFO_MODELS_FISH_DRAMA3_HIP_TOWER_RUNTIME_HPP_
#define GUFO_MODELS_FISH_DRAMA3_HIP_TOWER_RUNTIME_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/fish_drama3/config.hpp"
#include "src/models/fish_drama3/loader.hpp"
#include "src/models/fish_drama3/prompt.hpp"

namespace gufo::models::fish_drama3::hip {

class TowerHipRuntime {
public:
  ~TowerHipRuntime();

  TowerHipRuntime(const TowerHipRuntime&) = delete;
  TowerHipRuntime& operator=(const TowerHipRuntime&) = delete;

  [[nodiscard]] static std::unique_ptr<TowerHipRuntime> Create(
      const ModelSafetensors& store, const TowerConfig& config,
      std::size_t max_cache_steps = 4096, std::string* error = nullptr);

  void Reset();

  void MixBackboneEmbeddings(std::vector<float>* embeddings,
                             std::span<const Drama3Frame> frames) const;

  [[nodiscard]] std::vector<float> SharedInput(
      std::span<const Drama3Frame> frames) const;

  [[nodiscard]] bool Prefill(std::span<const float> backbone_hidden,
                             std::span<const Drama3Frame> frames,
                             std::string* error = nullptr);

  [[nodiscard]] bool RunFrame(std::span<const float> backbone_hidden,
                              const Drama3Frame& current_frame,
                              std::int32_t semantic_code, std::size_t position,
                              float temperature, int top_k, float top_p,
                              const std::array<float, 9>& uniforms,
                              Drama3Frame* out_frame,
                              std::string* error = nullptr);

  [[nodiscard]] const TowerConfig& config() const noexcept;

private:
  struct Impl;
  explicit TowerHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::fish_drama3::hip

#endif  // GUFO_MODELS_FISH_DRAMA3_HIP_TOWER_RUNTIME_HPP_
