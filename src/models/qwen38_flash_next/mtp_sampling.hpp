#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "src/core/sampling.hpp"

namespace gufo::models::qwen38_flash_next {

enum class DraftDecision { kAccept, kReject, kStop };

inline constexpr std::size_t kMtpCandidates = 64;

struct MtpCandidateLogits {
  std::array<sampling::TokenId, kMtpCandidates> ids{};
  std::array<float, kMtpCandidates> logits{};
  std::size_t size{0};
};

struct MtpProposal {
  std::array<sampling::TokenId, kMtpCandidates> ids{};
  std::array<float, kMtpCandidates> probabilities{};
  std::size_t size{0};
  sampling::TokenId token{0};
  float probability{0.0F};
};

/// Top-1 mass inside the proposal's own candidate set, used as a per-position
/// confidence signal. A wide verify batch is not free on a model with hundreds
/// of experts: verifying n tokens touches far more distinct experts than one
/// token does, so slots the predictor is unlikely to win are better left
/// unspent. Renormalizing over the returned candidates rather than the full
/// vocabulary is deliberate — the head already reduced to them, and the tail
/// only shifts the scale.
[[nodiscard]] inline float DraftConfidence(const MtpCandidateLogits& c) {
  if (c.size == 0) {
    return 0.0F;
  }
  float largest = c.logits[0];
  for (std::size_t i = 1; i < c.size; ++i) {
    largest = std::max(largest, c.logits[i]);
  }
  float sum = 0.0F;
  for (std::size_t i = 0; i < c.size; ++i) {
    sum += std::exp(c.logits[i] - largest);
  }
  return sum > 0.0F ? 1.0F / sum : 0.0F;
}

/// The proposal has bounded support; the target distribution stays complete.
/// Integer probability masses sum to 2^24, so the exported F32 q sums to
/// exactly one and is exactly the distribution used to draw the proposal.
inline MtpProposal SampleMtpProposal(const MtpCandidateLogits& candidates,
                                     const sampling::SamplerState& sampler,
                                     std::uint64_t* rng) {
  if (candidates.size == 0 || candidates.size > kMtpCandidates) {
    throw std::invalid_argument("invalid MTP candidate count");
  }
  const auto distribution =
      sampler.Distribution(std::span(candidates.logits).first(candidates.size),
                           std::span(candidates.ids).first(candidates.size));
  constexpr std::uint32_t units = 1U << 24;
  std::array<std::uint32_t, kMtpCandidates> mass{};
  MtpProposal proposal;
  proposal.size = distribution.entries().size();
  std::uint32_t total = 0;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    const auto& entry = distribution.entries()[i];
    proposal.ids[i] = candidates.ids[entry.token];
    mass[i] = static_cast<std::uint32_t>(std::floor(entry.value * units));
    total += mass[i];
  }
  mass[0] += units - total;
  const auto draw = static_cast<std::uint32_t>(sampling::Uniform(rng) * units);
  std::uint32_t cumulative = 0;
  bool selected = false;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    proposal.probabilities[i] = static_cast<float>(mass[i]) / units;
    cumulative += mass[i];
    if (!selected && draw < cumulative) {
      proposal.token = proposal.ids[i];
      proposal.probability = proposal.probabilities[i];
      selected = true;
    }
  }
  return proposal;
}

struct MtpVerification {
  sampling::TokenId token;
  bool accepted;
};

/// The same FP64 target distribution drives AR draws, support, acceptance
/// and residual correction. Only proposal q uses the compact F32 masses.
inline MtpVerification VerifyMtpProposal(std::span<const float> logits,
                                         const MtpProposal& proposal,
                                         sampling::SamplerState& sampler) {
  if (proposal.size == 0 || proposal.size > kMtpCandidates ||
      proposal.token >= logits.size() || !std::isfinite(proposal.probability) ||
      proposal.probability <= 0 || proposal.probability > 1)
    throw std::invalid_argument("invalid MTP proposal");
  double total = 0;
  double selected = 0;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    if (proposal.ids[i] >= logits.size() ||
        !std::isfinite(proposal.probabilities[i]) ||
        proposal.probabilities[i] < 0)
      throw std::invalid_argument("invalid MTP proposal mass");
    total += proposal.probabilities[i];
    if (proposal.ids[i] == proposal.token)
      selected += proposal.probabilities[i];
  }
  if (total != 1 || selected != proposal.probability)
    throw std::invalid_argument("MTP proposal masses do not match the draw");
  const auto target = sampler.Distribution(logits);
  if (sampler.Uniform() * proposal.probability <
      target.probability(proposal.token))
    return {proposal.token, true};
  return {target.SampleResidual(
              std::span(proposal.ids).first(proposal.size),
              std::span(proposal.probabilities).first(proposal.size),
              sampler.mutable_rng_state()),
          false};
}

/// The draft is deterministic, so an exact target sample decides acceptance.
/// A rejected draw is deferred to the next cycle: restore its RNG so that
/// stopping at this prefix consumes exactly the draws made by ordinary decode.
template<typename IsStop>
DraftDecision VerifyDraft(std::span<const float> logits, std::int32_t draft,
                          sampling::SamplerState& sampler, IsStop is_stop) {
  const auto rng = sampler.rng_state();
  const auto token = sampler.Sample(logits);
  if (is_stop(static_cast<std::int32_t>(token))) {
    return DraftDecision::kStop;
  }
  if (token != static_cast<sampling::TokenId>(draft)) {
    sampler.SetRngState(rng);
    return DraftDecision::kReject;
  }
  sampler.Accept(token);
  return DraftDecision::kAccept;
}

}  // namespace gufo::models::qwen38_flash_next

#endif
