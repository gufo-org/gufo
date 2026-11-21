#ifndef GUFO_MODELS_YUE2_SAMPLER_HPP_
#define GUFO_MODELS_YUE2_SAMPLER_HPP_

#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "src/models/yue2/protocol.hpp"

namespace gufo::models::yue2 {

/// The contiguous vocabulary rows a phase can select, end token included.
/// The runtime computes logits only for these rows.
struct VocabularyWindow {
  std::uint32_t begin{0};
  std::uint32_t end{0};
  [[nodiscard]] std::uint32_t size() const { return end - begin; }
};

[[nodiscard]] VocabularyWindow PhaseWindow(Phase phase);
[[nodiscard]] std::uint32_t PhaseEndToken(Phase phase);

/// YuE2's request-local token sampler (upstream `sampling.distribution`):
/// phase range mask, minimum-length end ban, windowed multiplicative
/// repetition penalty, temperature, top-k, then top-p. Guidance mixes the two
/// branches in BF16 first. `legacy_bf16` keeps the planning-off historical
/// arithmetic in BF16 and always retains three top-p candidates.
///
/// Draws use a request-local generator: the same seed gives the same tokens
/// for the same logits, but not torch's `multinomial` stream.
class TokenSampler {
public:
  TokenSampler(Phase phase, const Sampling& sampling, std::uint64_t seed,
               bool legacy_bf16);

  /// Returns the next token id. Rows are BF16-valued logits of
  /// `PhaseWindow(phase)`; `unconditional` is empty without guidance.
  [[nodiscard]] std::uint32_t Next(std::span<const float> conditional,
                                   std::span<const float> unconditional,
                                   float guidance);

  [[nodiscard]] std::size_t steps() const { return step_; }

private:
  struct Candidate {
    std::uint32_t index;
    float score;
  };

  Phase phase_;
  Sampling sampling_;
  bool legacy_bf16_;
  VocabularyWindow window_;
  std::uint32_t end_index_;
  std::size_t step_{0};
  std::vector<std::uint32_t> history_;
  std::vector<float> scores_;
  std::vector<Candidate> candidates_;
  std::vector<float> top_;
  std::mt19937_64 generator_;
};

}  // namespace gufo::models::yue2

#endif  // GUFO_MODELS_YUE2_SAMPLER_HPP_
