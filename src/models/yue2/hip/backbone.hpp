#ifndef GUFO_MODELS_YUE2_HIP_BACKBONE_HPP_
#define GUFO_MODELS_YUE2_HIP_BACKBONE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/yue2/config.hpp"
#include "src/models/yue2/sampler.hpp"

namespace gufo::models::yue2::hip {

/// BF16 key/value rows of one AR token stream, head-major per layer.
class KvCache {
public:
  ~KvCache();
  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;

  [[nodiscard]] std::size_t length() const { return length_; }
  [[nodiscard]] std::size_t capacity() const { return capacity_; }

private:
  friend class Backbone;
  KvCache() = default;
  std::uint16_t* keys_{nullptr};
  std::uint16_t* values_{nullptr};
  std::size_t capacity_{0};
  std::size_t length_{0};
};

/// Called after each midpoint step; returning false cancels.
using StepProgress = std::function<bool(int completed, int total)>;

/// The YuE2 Mixture-of-Transformers backbone on gfx1151: the AR path for
/// planning and semantic tokens, and the NAR path for acoustic flow matching.
/// Activations keep PyTorch's eager BF16 boundaries; the KV caches are BF16.
/// One request at a time.
class Backbone {
public:
  ~Backbone();
  Backbone(const Backbone&) = delete;
  Backbone& operator=(const Backbone&) = delete;

  [[nodiscard]] static std::unique_ptr<Backbone> Create(
      const std::filesystem::path& model_root, const BackboneConfig& config,
      std::string* error);

  [[nodiscard]] std::unique_ptr<KvCache> NewCache(std::size_t capacity);

  /// Appends `tokens` to the cache. When `window` is not empty, writes the
  /// BF16-valued logits of the last token's window rows to `logits`.
  void Append(KvCache& cache, std::span<const std::uint32_t> tokens,
              VocabularyWindow window, std::vector<float>* logits);

  /// Causal forward over `tokens` in a scratch cache; returns the window
  /// logits of each row in `rows` (teacher forcing, for quality checks).
  void TeacherForce(std::span<const std::uint32_t> tokens,
                    std::span<const std::uint32_t> rows,
                    VocabularyWindow window, std::vector<float>* logits);

  /// The acoustic flow-matching velocity for `state` (`[frames][64]`) at raw
  /// time `raw_t`, attending to every AR row of `prefix`.
  void Velocity(const KvCache& prefix, std::span<const float> state,
                double raw_t, std::vector<float>* velocity);

  /// Solves the 32-step (by default) midpoint ODE from `noise`
  /// (`[frames][64]`) to the FP32 latents. Fails when cancelled.
  [[nodiscard]] bool Synthesize(const KvCache& prefix,
                                std::span<const float> noise, int steps,
                                const StepProgress& progress,
                                std::vector<float>* latents,
                                std::string* error);

  [[nodiscard]] const BackboneConfig& config() const;

private:
  struct Impl;
  explicit Backbone(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::yue2::hip

#endif  // GUFO_MODELS_YUE2_HIP_BACKBONE_HPP_
