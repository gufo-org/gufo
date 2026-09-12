#include "src/core/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace gufo::sampling {
namespace {

struct Candidate {
  TokenId token;
  double logit;
};

[[nodiscard]] bool IsBetterCandidate(const Candidate& left,
                                     const Candidate& right) noexcept {
  if (left.logit == right.logit) {
    return left.token < right.token;
  }
  return left.logit > right.logit;
}

[[nodiscard]] bool IsBetterProbability(const Probability& left,
                                       const Probability& right) noexcept {
  if (left.value == right.value) {
    return left.token < right.token;
  }
  return left.value > right.value;
}

[[nodiscard]] std::size_t MinimumKept(const SamplingConfig& config,
                                      std::size_t size) {
  return std::min(size, std::max<std::size_t>(config.min_keep, 1));
}

void SortCandidates(std::vector<Candidate>* candidates) {
  std::ranges::sort(*candidates, IsBetterCandidate);
}

[[nodiscard]] std::vector<double> SoftmaxWeights(
    std::span<const Candidate> candidates, double temperature) {
  if (candidates.empty()) {
    throw std::runtime_error("sampling filters removed every token");
  }
  const double maximum = candidates.front().logit;
  std::vector<double> weights;
  weights.reserve(candidates.size());
  double sum = 0.0;
  for (const auto& candidate : candidates) {
    const double weight = std::exp((candidate.logit - maximum) / temperature);
    weights.push_back(weight);
    sum += weight;
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    throw std::runtime_error("logit softmax normalization failed");
  }
  for (double& weight : weights) {
    weight /= sum;
  }
  return weights;
}

void ApplyPenalties(std::vector<Candidate>* candidates,
                    const SamplingConfig& config,
                    std::span<const TokenId> recent_tokens) {
  if (!config.penalties_enabled() || config.repeat_last_n == 0 ||
      recent_tokens.empty()) {
    return;
  }
  const std::size_t first = recent_tokens.size() > config.repeat_last_n
                                ? recent_tokens.size() - config.repeat_last_n
                                : 0;
  std::unordered_map<TokenId, std::size_t> counts;
  for (const TokenId token : recent_tokens.subspan(first)) {
    ++counts[token];
  }
  for (Candidate& candidate : *candidates) {
    const auto found = counts.find(candidate.token);
    if (found == counts.end()) {
      continue;
    }
    if (config.repeat_penalty != 1.0F) {
      candidate.logit =
          candidate.logit <= 0.0
              ? candidate.logit * static_cast<double>(config.repeat_penalty)
              : candidate.logit / static_cast<double>(config.repeat_penalty);
    }
    candidate.logit -= static_cast<double>(config.frequency_penalty) *
                       static_cast<double>(found->second);
    candidate.logit -= static_cast<double>(config.presence_penalty);
  }
}

void ApplyTopK(std::vector<Candidate>* candidates,
               const SamplingConfig& config) {
  if (config.top_k <= 0) {
    SortCandidates(candidates);
    return;
  }
  const auto configured = static_cast<std::size_t>(config.top_k);
  const std::size_t keep =
      std::min(candidates->size(),
               std::max(configured, MinimumKept(config, candidates->size())));
  if (keep < candidates->size()) {
    std::partial_sort(candidates->begin(),
                      candidates->begin() + static_cast<std::ptrdiff_t>(keep),
                      candidates->end(), IsBetterCandidate);
  } else {
    SortCandidates(candidates);
  }
  candidates->resize(keep);
}

void ApplyTopP(std::vector<Candidate>* candidates,
               const SamplingConfig& config) {
  if (config.top_p >= 1.0F || candidates->size() <= 1) {
    return;
  }
  const auto probabilities = SoftmaxWeights(*candidates, 1.0);
  const std::size_t minimum = MinimumKept(config, candidates->size());
  double cumulative = 0.0;
  std::size_t keep = 0;
  while (keep < candidates->size()) {
    cumulative += probabilities[keep];
    ++keep;
    if (keep >= minimum && cumulative >= static_cast<double>(config.top_p)) {
      break;
    }
  }
  candidates->resize(keep);
}

void ApplyMinP(std::vector<Candidate>* candidates,
               const SamplingConfig& config) {
  if (config.min_p <= 0.0F || candidates->size() <= 1) {
    return;
  }
  const double threshold = std::log(static_cast<double>(config.min_p));
  const double maximum = candidates->front().logit;
  std::size_t keep = 0;
  while (keep < candidates->size() &&
         candidates->at(keep).logit - maximum >= threshold) {
    ++keep;
  }
  keep = std::max(keep, MinimumKept(config, candidates->size()));
  candidates->resize(std::min(keep, candidates->size()));
}

[[nodiscard]] std::uint64_t InitialRngState(std::int64_t seed) {
  if (seed >= 0) {
    return static_cast<std::uint64_t>(seed);
  }
  std::random_device random_device;
  return (static_cast<std::uint64_t>(random_device()) << 32U) ^
         static_cast<std::uint64_t>(random_device());
}

}  // namespace

void SamplingConfig::Validate() const {
  if (!std::isfinite(temperature) || temperature < 0.0F) {
    throw std::invalid_argument(
        "sampling temperature must be finite and nonnegative");
  }
  if (top_k < 0) {
    throw std::invalid_argument("sampling top-k must be nonnegative");
  }
  if (!std::isfinite(top_p) || top_p <= 0.0F || top_p > 1.0F) {
    throw std::invalid_argument("sampling top-p must be in (0, 1]");
  }
  if (!std::isfinite(min_p) || min_p < 0.0F || min_p > 1.0F) {
    throw std::invalid_argument("sampling min-p must be in [0, 1]");
  }
  if (seed < -1) {
    throw std::invalid_argument("sampling seed must be -1 or nonnegative");
  }
  if (!std::isfinite(repeat_penalty) || repeat_penalty <= 0.0F) {
    throw std::invalid_argument(
        "sampling repeat penalty must be finite and positive");
  }
  if (!std::isfinite(frequency_penalty)) {
    throw std::invalid_argument("sampling frequency penalty must be finite");
  }
  if (!std::isfinite(presence_penalty)) {
    throw std::invalid_argument("sampling presence penalty must be finite");
  }
}

bool SamplingConfig::penalties_enabled() const noexcept {
  return repeat_penalty != 1.0F || frequency_penalty != 0.0F ||
         presence_penalty != 0.0F;
}

bool SamplingConfig::uses_random_sampling() const noexcept {
  return temperature > 0.0F && (top_k != 1 || min_keep > 1);
}

bool SamplingConfig::can_use_unmodified_argmax() const noexcept {
  return temperature == 0.0F && !penalties_enabled();
}

SamplingDistribution::SamplingDistribution(std::vector<Probability> entries)
    : entries_(std::move(entries)) {
  if (entries_.empty()) {
    throw std::invalid_argument("sampling distribution cannot be empty");
  }
  std::unordered_set<TokenId> tokens;
  double sum = 0.0;
  for (const auto& entry : entries_) {
    if (!std::isfinite(entry.value) || entry.value < 0.0 ||
        !tokens.insert(entry.token).second) {
      throw std::invalid_argument("sampling distribution is malformed");
    }
    sum += entry.value;
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    throw std::invalid_argument("sampling distribution has no probability");
  }
  for (auto& entry : entries_) {
    entry.value /= sum;
  }
  std::ranges::sort(entries_,
                    [](const Probability& left, const Probability& right) {
                      if (left.value == right.value) {
                        return left.token < right.token;
                      }
                      return left.value > right.value;
                    });
}

std::span<const Probability> SamplingDistribution::entries() const noexcept {
  return entries_;
}

TokenId SamplingDistribution::best_token() const {
  if (entries_.empty()) {
    throw std::logic_error("sampling distribution is empty");
  }
  return entries_.front().token;
}

double SamplingDistribution::probability(TokenId token) const noexcept {
  for (const auto& entry : entries_) {
    if (entry.token == token) {
      return entry.value;
    }
  }
  return 0.0;
}

TokenId SamplingDistribution::Sample(std::uint64_t* rng_state) const {
  if (entries_.empty()) {
    throw std::logic_error("sampling distribution is empty");
  }
  if (entries_.size() == 1) {
    return entries_.front().token;
  }
  double sample = gufo::sampling::Uniform(rng_state);
  for (const auto& entry : entries_) {
    sample -= entry.value;
    if (sample <= 0.0) {
      return entry.token;
    }
  }
  return entries_.back().token;
}

TokenId SamplingDistribution::SampleResidual(
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities,
    std::uint64_t* rng_state) const {
  if (candidate_ids.size() != candidate_probabilities.size()) {
    throw std::invalid_argument("sparse probability row is malformed");
  }
  std::unordered_map<TokenId, double> draft;
  for (std::size_t index = 0; index < candidate_ids.size(); ++index) {
    const float probability_value = candidate_probabilities[index];
    if (!std::isfinite(probability_value) || probability_value < 0.0F) {
      throw std::invalid_argument(
          "sparse probability row contains an invalid value");
    }
    draft[candidate_ids[index]] += static_cast<double>(probability_value);
  }

  std::vector<Probability> residual;
  residual.reserve(entries_.size());
  double sum = 0.0;
  for (const auto& entry : entries_) {
    const double value = std::max(entry.value - draft[entry.token], 0.0);
    if (value > 0.0) {
      residual.push_back({.token = entry.token, .value = value});
      sum += value;
    }
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    return Sample(rng_state);
  }
  for (auto& entry : residual) {
    entry.value /= sum;
  }
  return SamplingDistribution(std::move(residual)).Sample(rng_state);
}

SamplingDistribution BuildDistribution(std::span<const float> logits,
                                       const SamplingConfig& config,
                                       std::span<const TokenId> recent_tokens) {
  config.Validate();
  if (logits.empty()) {
    throw std::invalid_argument("cannot sample an empty logit distribution");
  }
  if (logits.size() >
      static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
    throw std::invalid_argument("logit distribution exceeds token ID range");
  }

  std::vector<Candidate> candidates;
  candidates.reserve(logits.size());
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (std::isfinite(logits[index])) {
      candidates.push_back({
          .token = static_cast<TokenId>(index),
          .logit = static_cast<double>(logits[index]),
      });
    }
  }
  if (candidates.empty()) {
    throw std::runtime_error("logit distribution contains no finite values");
  }

  ApplyPenalties(&candidates, config, recent_tokens);
  ApplyTopK(&candidates, config);
  ApplyTopP(&candidates, config);
  ApplyMinP(&candidates, config);

  if (config.temperature == 0.0F) {
    return SamplingDistribution(
        {{.token = candidates.front().token, .value = 1.0}});
  }
  const auto probabilities =
      SoftmaxWeights(candidates, static_cast<double>(config.temperature));
  std::vector<Probability> entries;
  entries.reserve(candidates.size());
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    entries.push_back(
        {.token = candidates[index].token, .value = probabilities[index]});
  }
  return SamplingDistribution(std::move(entries));
}

SamplerState::SamplerState(SamplingConfig config,
                           std::span<const TokenId> initial_history)
    : config_(config),
      history_(initial_history.begin(), initial_history.end()),
      rng_state_(InitialRngState(config.seed)) {
  config_.Validate();
  TrimHistory();
  RebuildPenaltyCounts();
}

SamplerState::SamplerState(const SamplerState& other)
    : config_(other.config_),
      history_(other.history_),
      penalty_counts_(other.penalty_counts_),
      rng_state_(other.rng_state_) {}

SamplerState& SamplerState::operator=(const SamplerState& other) {
  if (this == &other) {
    return *this;
  }
  config_ = other.config_;
  history_ = other.history_;
  penalty_counts_ = other.penalty_counts_;
  candidate_scratch_.clear();
  rng_state_ = other.rng_state_;
  return *this;
}

const SamplingConfig& SamplerState::config() const noexcept {
  return config_;
}

std::span<const TokenId> SamplerState::history() const noexcept {
  return history_;
}

std::uint64_t SamplerState::rng_state() const noexcept {
  return rng_state_;
}

std::uint64_t* SamplerState::mutable_rng_state() noexcept {
  return &rng_state_;
}

void SamplerState::SetRngState(std::uint64_t state) noexcept {
  rng_state_ = state;
}

void SamplerState::ResetHistory(std::span<const TokenId> tokens) {
  history_.assign(tokens.begin(), tokens.end());
  TrimHistory();
  RebuildPenaltyCounts();
}

void SamplerState::Accept(TokenId token) {
  if (config_.repeat_last_n == 0) {
    return;
  }
  history_.push_back(token);
  TrimHistory();
  RebuildPenaltyCounts();
}

void SamplerState::Accept(std::span<const TokenId> tokens) {
  if (config_.repeat_last_n == 0) {
    return;
  }
  history_.insert(history_.end(), tokens.begin(), tokens.end());
  TrimHistory();
  RebuildPenaltyCounts();
}

SamplingDistribution SamplerState::Distribution(
    std::span<const float> logits) const {
  return BuildDistribution(logits, config_, history_);
}

TokenId SamplerState::Sample(std::span<const float> logits) {
  config_.Validate();
  if (logits.empty()) {
    throw std::invalid_argument("cannot sample an empty logit distribution");
  }
  if (logits.size() >
      static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
    throw std::invalid_argument("logit distribution exceeds token ID range");
  }
  if (config_.temperature == 0.0F) {
    return SampleGreedy(logits);
  }
  const bool min_p_needs_floor = config_.min_p > 0.0F && config_.min_keep > 1;
  if (config_.top_k == 0 && config_.top_p == 1.0F && !min_p_needs_floor) {
    return SampleLinear(logits);
  }
  return SampleSelected(logits);
}

TokenId SamplerState::SampleResidual(
    std::span<const float> target_logits,
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities) {
  return Distribution(target_logits)
      .SampleResidual(candidate_ids, candidate_probabilities, &rng_state_);
}

double SamplerState::Uniform() {
  return gufo::sampling::Uniform(&rng_state_);
}

void SamplerState::TrimHistory() {
  if (config_.repeat_last_n == 0) {
    history_.clear();
    return;
  }
  if (history_.size() > config_.repeat_last_n) {
    history_.erase(
        history_.begin(),
        history_.end() - static_cast<std::ptrdiff_t>(config_.repeat_last_n));
  }
}

void SamplerState::RebuildPenaltyCounts() {
  penalty_counts_.clear();
  if (!config_.penalties_enabled() || config_.repeat_last_n == 0) {
    return;
  }
  penalty_counts_.reserve(history_.size());
  for (const TokenId token : history_) {
    const auto found = std::ranges::find_if(
        penalty_counts_,
        [token](const auto& entry) { return entry.first == token; });
    if (found == penalty_counts_.end()) {
      penalty_counts_.emplace_back(token, 1U);
    } else {
      ++found->second;
    }
  }
  std::ranges::sort(penalty_counts_, [](const auto& left, const auto& right) {
    return left.first < right.first;
  });
}

double SamplerState::AdjustedLogit(TokenId token, float logit) const noexcept {
  if (penalty_counts_.empty()) {
    return static_cast<double>(logit);
  }
  const auto found = std::lower_bound(
      penalty_counts_.begin(), penalty_counts_.end(), token,
      [](const auto& entry, TokenId value) { return entry.first < value; });
  if (found == penalty_counts_.end() || found->first != token) {
    return static_cast<double>(logit);
  }

  double adjusted = static_cast<double>(logit);
  if (config_.repeat_penalty != 1.0F) {
    adjusted = adjusted <= 0.0
                   ? adjusted * static_cast<double>(config_.repeat_penalty)
                   : adjusted / static_cast<double>(config_.repeat_penalty);
  }
  adjusted -= static_cast<double>(config_.frequency_penalty) *
              static_cast<double>(found->second);
  adjusted -= static_cast<double>(config_.presence_penalty);
  return adjusted;
}

TokenId SamplerState::SampleGreedy(std::span<const float> logits) const {
  if (penalty_counts_.empty()) {
    float best_logit = -std::numeric_limits<float>::infinity();
    TokenId best_token = 0;
    bool found = false;
    for (std::size_t index = 0; index < logits.size(); ++index) {
      const float logit = logits[index];
      if (logit > best_logit && std::isfinite(logit)) {
        best_logit = logit;
        best_token = static_cast<TokenId>(index);
        found = true;
      }
    }
    if (!found) {
      throw std::runtime_error("logit distribution contains no finite values");
    }
    return best_token;
  }

  double best_logit = -std::numeric_limits<double>::infinity();
  TokenId best_token = 0;
  bool found = false;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (!std::isfinite(logits[index])) {
      continue;
    }
    const double adjusted =
        AdjustedLogit(static_cast<TokenId>(index), logits[index]);
    if (!std::isfinite(adjusted)) {
      throw std::runtime_error(
          "sampling penalties produced a non-finite logit");
    }
    if (!found || adjusted > best_logit) {
      best_logit = adjusted;
      best_token = static_cast<TokenId>(index);
      found = true;
    }
  }
  if (!found) {
    throw std::runtime_error("logit distribution contains no finite values");
  }
  return best_token;
}

TokenId SamplerState::SampleLinear(std::span<const float> logits) {
  if (penalty_counts_.empty()) {
    float maximum = -std::numeric_limits<float>::infinity();
    bool found = false;
    for (const float logit : logits) {
      if (logit > maximum && std::isfinite(logit)) {
        maximum = logit;
        found = true;
      }
    }
    if (!found) {
      throw std::runtime_error("logit distribution contains no finite values");
    }

    const double min_p_threshold =
        config_.min_p > 0.0F ? static_cast<double>(maximum) +
                                   std::log(static_cast<double>(config_.min_p))
                             : -std::numeric_limits<double>::infinity();
    const double inverse_temperature =
        1.0 / static_cast<double>(config_.temperature);
    double sum = 0.0;
    for (const float logit : logits) {
      if (static_cast<double>(logit) >= min_p_threshold) {
        sum += std::exp(
            (static_cast<double>(logit) - static_cast<double>(maximum)) *
            inverse_temperature);
      }
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
      throw std::runtime_error("logit softmax normalization failed");
    }

    double threshold = Uniform() * sum;
    TokenId last_token = 0;
    bool has_candidate = false;
    for (std::size_t index = 0; index < logits.size(); ++index) {
      const double logit = static_cast<double>(logits[index]);
      if (logit < min_p_threshold) {
        continue;
      }
      last_token = static_cast<TokenId>(index);
      has_candidate = true;
      threshold -= std::exp((logit - static_cast<double>(maximum)) *
                            inverse_temperature);
      if (threshold <= 0.0) {
        return last_token;
      }
    }
    if (!has_candidate) {
      throw std::runtime_error("sampling filters removed every token");
    }
    return last_token;
  }

  double maximum = -std::numeric_limits<double>::infinity();
  bool found = false;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (!std::isfinite(logits[index])) {
      continue;
    }
    const double adjusted =
        AdjustedLogit(static_cast<TokenId>(index), logits[index]);
    if (!std::isfinite(adjusted)) {
      throw std::runtime_error(
          "sampling penalties produced a non-finite logit");
    }
    maximum = std::max(maximum, adjusted);
    found = true;
  }
  if (!found) {
    throw std::runtime_error("logit distribution contains no finite values");
  }

  const double min_p_threshold =
      config_.min_p > 0.0F
          ? maximum + std::log(static_cast<double>(config_.min_p))
          : -std::numeric_limits<double>::infinity();
  const double inverse_temperature =
      1.0 / static_cast<double>(config_.temperature);
  double sum = 0.0;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (!std::isfinite(logits[index])) {
      continue;
    }
    const double adjusted =
        AdjustedLogit(static_cast<TokenId>(index), logits[index]);
    if (adjusted >= min_p_threshold) {
      sum += std::exp((adjusted - maximum) * inverse_temperature);
    }
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    throw std::runtime_error("logit softmax normalization failed");
  }

  double threshold = Uniform() * sum;
  TokenId last_token = 0;
  bool has_candidate = false;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (!std::isfinite(logits[index])) {
      continue;
    }
    const double adjusted =
        AdjustedLogit(static_cast<TokenId>(index), logits[index]);
    if (adjusted < min_p_threshold) {
      continue;
    }
    last_token = static_cast<TokenId>(index);
    has_candidate = true;
    threshold -= std::exp((adjusted - maximum) * inverse_temperature);
    if (threshold <= 0.0) {
      return last_token;
    }
  }
  if (!has_candidate) {
    throw std::runtime_error("sampling filters removed every token");
  }
  return last_token;
}

TokenId SamplerState::SampleSelected(std::span<const float> logits) {
  candidate_scratch_.clear();
  const auto read_adjusted = [&](std::size_t index) {
    const double adjusted =
        AdjustedLogit(static_cast<TokenId>(index), logits[index]);
    if (!std::isfinite(adjusted)) {
      throw std::runtime_error(
          "sampling penalties produced a non-finite logit");
    }
    return adjusted;
  };
  const auto select_best = [&](std::size_t limit) {
    candidate_scratch_.clear();
    candidate_scratch_.reserve(limit);
    for (std::size_t index = 0; index < logits.size(); ++index) {
      if (!std::isfinite(logits[index])) {
        continue;
      }
      const Probability candidate{
          .token = static_cast<TokenId>(index),
          .value = read_adjusted(index),
      };
      if (candidate_scratch_.size() < limit) {
        candidate_scratch_.push_back(candidate);
        std::push_heap(candidate_scratch_.begin(), candidate_scratch_.end(),
                       IsBetterProbability);
      } else if (IsBetterProbability(candidate, candidate_scratch_.front())) {
        std::pop_heap(candidate_scratch_.begin(), candidate_scratch_.end(),
                      IsBetterProbability);
        candidate_scratch_.back() = candidate;
        std::push_heap(candidate_scratch_.begin(), candidate_scratch_.end(),
                       IsBetterProbability);
      }
    }
    std::ranges::sort(candidate_scratch_, IsBetterProbability);
  };
  const auto select_all = [&]() {
    candidate_scratch_.clear();
    candidate_scratch_.reserve(logits.size());
    for (std::size_t index = 0; index < logits.size(); ++index) {
      if (!std::isfinite(logits[index])) {
        continue;
      }
      candidate_scratch_.push_back({
          .token = static_cast<TokenId>(index),
          .value = read_adjusted(index),
      });
    }
    std::ranges::sort(candidate_scratch_, IsBetterProbability);
  };

  double full_softmax_sum = 0.0;
  bool has_full_softmax_sum = false;
  if (config_.top_k > 0) {
    const std::size_t keep = std::min(
        logits.size(),
        std::max<std::size_t>(static_cast<std::size_t>(config_.top_k),
                              std::max<std::size_t>(config_.min_keep, 1)));
    select_best(keep);
  } else if (config_.top_p < 1.0F && logits.size() > 1024) {
    double maximum = -std::numeric_limits<double>::infinity();
    bool found = false;
    for (std::size_t index = 0; index < logits.size(); ++index) {
      if (!std::isfinite(logits[index])) {
        continue;
      }
      maximum = std::max(maximum, read_adjusted(index));
      found = true;
    }
    if (!found) {
      throw std::runtime_error("logit distribution contains no finite values");
    }
    for (std::size_t index = 0; index < logits.size(); ++index) {
      if (std::isfinite(logits[index])) {
        full_softmax_sum += std::exp(read_adjusted(index) - maximum);
      }
    }
    if (!(full_softmax_sum > 0.0) || !std::isfinite(full_softmax_sum)) {
      throw std::runtime_error("logit softmax normalization failed");
    }
    has_full_softmax_sum = true;

    constexpr std::size_t initial_top_p_candidates = 256;
    const std::size_t initial_keep = std::min(
        logits.size(), std::max(initial_top_p_candidates,
                                std::max<std::size_t>(config_.min_keep, 1)));
    select_best(initial_keep);

    double selected_mass = 0.0;
    for (const auto& candidate : candidate_scratch_) {
      selected_mass += std::exp(candidate.value - maximum);
    }
    if (selected_mass < static_cast<double>(config_.top_p) * full_softmax_sum) {
      select_all();
    }
  } else {
    select_all();
  }
  if (candidate_scratch_.empty()) {
    throw std::runtime_error("logit distribution contains no finite values");
  }

  std::size_t keep = candidate_scratch_.size();
  const std::size_t minimum = MinimumKept(config_, candidate_scratch_.size());
  if (config_.top_p < 1.0F && keep > 1) {
    const double maximum = candidate_scratch_.front().value;
    double sum = full_softmax_sum;
    if (!has_full_softmax_sum) {
      sum = 0.0;
      for (const auto& candidate : candidate_scratch_) {
        sum += std::exp(candidate.value - maximum);
      }
    }
    const double target = static_cast<double>(config_.top_p) * sum;
    double cumulative = 0.0;
    std::size_t top_p_keep = 0;
    while (top_p_keep < keep) {
      cumulative += std::exp(candidate_scratch_[top_p_keep].value - maximum);
      ++top_p_keep;
      if (top_p_keep >= minimum && cumulative >= target) {
        break;
      }
    }
    keep = top_p_keep;
  }
  if (config_.min_p > 0.0F && keep > 1) {
    const double threshold = candidate_scratch_.front().value +
                             std::log(static_cast<double>(config_.min_p));
    std::size_t min_p_keep = 0;
    while (min_p_keep < keep &&
           candidate_scratch_[min_p_keep].value >= threshold) {
      ++min_p_keep;
    }
    keep = std::min(keep, std::max(min_p_keep, minimum));
  }
  candidate_scratch_.resize(keep);

  if (candidate_scratch_.size() == 1) {
    return candidate_scratch_.front().token;
  }
  const double maximum = candidate_scratch_.front().value;
  const double inverse_temperature =
      1.0 / static_cast<double>(config_.temperature);
  double sum = 0.0;
  for (const auto& candidate : candidate_scratch_) {
    sum += std::exp((candidate.value - maximum) * inverse_temperature);
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    throw std::runtime_error("logit softmax normalization failed");
  }

  double threshold = Uniform() * sum;
  for (const auto& candidate : candidate_scratch_) {
    threshold -= std::exp((candidate.value - maximum) * inverse_temperature);
    if (threshold <= 0.0) {
      return candidate.token;
    }
  }
  return candidate_scratch_.back().token;
}

std::uint64_t NextRandom(std::uint64_t* state) {
  if (state == nullptr) {
    throw std::invalid_argument("sampling requires RNG state");
  }
  std::uint64_t value = *state;
  if (value == 0) {
    value = UINT64_C(0x9e3779b97f4a7c15);
  }
  value ^= value >> 12U;
  value ^= value << 25U;
  value ^= value >> 27U;
  *state = value;
  return value * UINT64_C(0x2545f4914f6cdd1d);
}

double Uniform(std::uint64_t* state) {
  const std::uint64_t value = NextRandom(state);
  return static_cast<double>((value >> 40U) & UINT64_C(0xffffff)) / 16777216.0;
}

TokenId SampleLogits(std::span<const float> logits, float temperature,
                     std::uint64_t* rng_state) {
  if (temperature > 0.0F && rng_state == nullptr) {
    throw std::invalid_argument("sampling requires RNG state");
  }
  SamplingConfig config;
  config.temperature = temperature;
  config.seed = 0;
  SamplerState sampler(config);
  if (rng_state != nullptr) {
    sampler.SetRngState(*rng_state);
  }
  const TokenId token = sampler.Sample(logits);
  if (rng_state != nullptr) {
    *rng_state = sampler.rng_state();
  }
  return token;
}

double TokenProbability(std::span<const float> logits, float temperature,
                        TokenId token) {
  if (token >= logits.size()) {
    throw std::out_of_range("sampled token exceeds target vocabulary");
  }
  SamplingConfig config;
  config.temperature = temperature;
  return BuildDistribution(logits, config).probability(token);
}

TokenId SampleResidual(std::span<const float> target_logits, float temperature,
                       std::span<const TokenId> candidate_ids,
                       std::span<const float> candidate_probabilities,
                       std::uint64_t* rng_state) {
  SamplingConfig config;
  config.temperature = temperature;
  return BuildDistribution(target_logits, config)
      .SampleResidual(candidate_ids, candidate_probabilities, rng_state);
}

}  // namespace gufo::sampling
