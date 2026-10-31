#ifndef GUFO_MODELS_GEMMA4_SIBLING_VERIFICATION_HPP_
#define GUFO_MODELS_GEMMA4_SIBLING_VERIFICATION_HPP_

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::gemma4 {

struct SiblingVerification {
  sampling::TokenId token;
  bool accepted;  ///< the chain's draft
  bool sibling;   ///< the sibling, after the draft's rejection
};

/// p/q verification of a draft with a sibling drawn from q without it:
/// recursive rejection sampling. The draft is accepted with min(1, p/q);
/// else the sibling, against the residual r = norm(max(p - q, 0)) and its
/// own proposal q' = q without the draft, with min(1, r/q'); else the token
/// comes from norm(max(r - q', 0)). The emitted token follows p exactly.
inline SiblingVerification VerifyWithSibling(
    const sampling::SamplingDistribution& target,
    const qwen38_flash_next::MtpProposal& q, sampling::TokenId sibling,
    sampling::SamplerState& sampler) {
  if (sampler.Uniform() * q.probability < target.probability(q.token)) {
    return {q.token, true, false};
  }
  const auto draft = [&q](sampling::TokenId token) {
    for (std::size_t i = 0; i < q.size; ++i) {
      if (q.ids[i] == token) {
        return static_cast<double>(q.probabilities[i]);
      }
    }
    return 0.0;
  };
  std::vector<sampling::Probability> residual;
  double total = 0.0;
  for (const auto& entry : target.entries()) {
    const double value = std::max(entry.value - draft(entry.token), 0.0);
    if (value > 0.0) {
      residual.push_back({.token = entry.token, .value = value});
      total += value;
    }
  }
  if (!(total > 0.0) || !std::isfinite(total)) {
    return {target.Sample(sampler.mutable_rng_state()), false, false};
  }
  const double rest = 1.0 - static_cast<double>(q.probability);
  const double own = draft(sibling) / rest;
  const double r =
      std::max(target.probability(sibling) - draft(sibling), 0.0) / total;
  if (sampler.Uniform() * own < r) {
    return {sibling, false, true};
  }
  std::vector<sampling::Probability> second;
  for (const auto& entry : residual) {
    const double other =
        entry.token == q.token ? 0.0 : draft(entry.token) / rest;
    const double value = std::max(entry.value / total - other, 0.0);
    if (value > 0.0) {
      second.push_back({.token = entry.token, .value = value});
    }
  }
  if (second.empty()) {
    return {sampling::SamplingDistribution(std::move(residual))
                .Sample(sampler.mutable_rng_state()),
            false, false};
  }
  return {sampling::SamplingDistribution(std::move(second))
              .Sample(sampler.mutable_rng_state()),
          false, false};
}

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_SIBLING_VERIFICATION_HPP_
