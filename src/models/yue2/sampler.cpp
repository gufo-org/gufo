#include "src/models/yue2/sampler.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_map>

namespace gufo::models::yue2 {
namespace {

constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

float RoundBfloat16(float value) {
  if (std::isnan(value))
    return value;
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return std::bit_cast<float>(bits & 0xFFFF0000U);
}

}  // namespace

VocabularyWindow PhaseWindow(Phase phase) {
  // Planning selects ordinary text plus </abc>; the few control rows between
  // them are masked. The codec rows sit directly after </music>.
  if (phase == Phase::kAbc)
    return {0, kAbcEnd + 1};
  return {kMusicEnd, kCodecOffset + kCodecSize};
}

std::uint32_t PhaseEndToken(Phase phase) {
  return phase == Phase::kAbc ? kAbcEnd : kMusicEnd;
}

TokenSampler::TokenSampler(Phase phase, const Sampling& sampling,
                           std::uint64_t seed, bool legacy_bf16)
    : phase_(phase),
      sampling_(sampling),
      legacy_bf16_(legacy_bf16),
      window_(PhaseWindow(phase)),
      end_index_(PhaseEndToken(phase) - PhaseWindow(phase).begin),
      generator_(seed) {
  scores_.resize(window_.size());
  candidates_.reserve(window_.size());
  history_.reserve(sampling.max_tokens);
}

std::uint32_t TokenSampler::Next(std::span<const float> conditional,
                                 std::span<const float> unconditional,
                                 float guidance) {
  const std::size_t n = window_.size();
  const bool mixed = guidance != 1.0F && unconditional.size() == n;
  const auto narrow = [this](float value) {
    return legacy_bf16_ ? RoundBfloat16(value) : value;
  };
  for (std::size_t i = 0; i < n; ++i) {
    float value = conditional[i];
    if (mixed) {
      // Historical BF16 CFG: subtract, scale and add, each rounded.
      const float base = unconditional[i];
      value = RoundBfloat16(
          base + RoundBfloat16(guidance * RoundBfloat16(value - base)));
    }
    scores_[i] = value;
  }
  // Phase range mask.
  if (phase_ == Phase::kAbc) {
    for (std::size_t i = kEndOfDocument; i < n; ++i) {
      if (i != end_index_)
        scores_[i] = kNegativeInfinity;
    }
  }
  if (step_ < sampling_.min_tokens)
    scores_[end_index_] = kNegativeInfinity;

  // Windowed multiplicative repetition penalty over the last tokens.
  if (sampling_.repetition_penalty != 1.0F && !history_.empty()) {
    const std::size_t window =
        std::min<std::size_t>(sampling_.penalty_window, history_.size());
    std::unordered_map<std::uint32_t, std::uint32_t> counts;
    for (std::size_t i = history_.size() - window; i < history_.size(); ++i)
      ++counts[history_[i]];
    for (const auto& [index, count] : counts) {
      const float alpha = narrow(static_cast<float>(
          std::pow(static_cast<double>(sampling_.repetition_penalty),
                   static_cast<double>(count))));
      float& score = scores_[index];
      score = narrow(score < 0.0F ? score * alpha : score / alpha);
    }
  }

  std::uint32_t chosen = 0;
  if (sampling_.temperature == 0.0F) {
    chosen = static_cast<std::uint32_t>(
        std::max_element(scores_.begin(), scores_.end()) - scores_.begin());
  } else {
    if (sampling_.temperature != 1.0F) {
      for (float& score : scores_)
        score = narrow(score / sampling_.temperature);
    }
    // Top-k keeps every score tied with the k-th largest. A k-entry min-heap
    // finds that score in one pass; the codec window has 32k rows and k is
    // small, so collecting and partitioning every row cost most of a step.
    const std::size_t k = sampling_.top_k;
    top_.clear();
    for (std::uint32_t i = 0; i < n; ++i) {
      const float score = scores_[i];
      if (score == kNegativeInfinity)
        continue;
      if (top_.size() < k) {
        top_.push_back(score);
        std::push_heap(top_.begin(), top_.end(), std::greater<>());
      } else if (score > top_.front()) {
        std::pop_heap(top_.begin(), top_.end(), std::greater<>());
        top_.back() = score;
        std::push_heap(top_.begin(), top_.end(), std::greater<>());
      }
    }
    const float threshold =
        top_.size() < k || top_.empty() ? kNegativeInfinity : top_.front();
    candidates_.clear();
    for (std::uint32_t i = 0; i < n; ++i) {
      if (scores_[i] != kNegativeInfinity && scores_[i] >= threshold)
        candidates_.push_back({i, scores_[i]});
    }
    const auto descending = [](const Candidate& a, const Candidate& b) {
      return a.score > b.score || (a.score == b.score && a.index < b.index);
    };
    std::sort(candidates_.begin(), candidates_.end(), descending);
    const auto probabilities = [&](std::vector<double>* output) {
      const double maximum = candidates_.front().score;
      double total = 0.0;
      output->resize(candidates_.size());
      for (std::size_t i = 0; i < candidates_.size(); ++i) {
        (*output)[i] =
            std::exp(static_cast<double>(candidates_[i].score) - maximum);
        total += (*output)[i];
      }
      for (double& value : *output)
        value = static_cast<double>(narrow(static_cast<float>(value / total)));
    };
    std::vector<double> weights;
    if (sampling_.top_p < 1.0F && !candidates_.empty()) {
      probabilities(&weights);
      const std::size_t keep = legacy_bf16_ ? 3 : 1;
      double cumulative = 0.0;
      std::size_t kept = candidates_.size();
      for (std::size_t i = 0; i < candidates_.size(); ++i) {
        cumulative = static_cast<double>(
            narrow(static_cast<float>(cumulative + weights[i])));
        const double before = static_cast<double>(
            narrow(static_cast<float>(cumulative - weights[i])));
        if (i >= keep && before > static_cast<double>(sampling_.top_p)) {
          kept = i;
          break;
        }
      }
      candidates_.resize(kept);
    }
    probabilities(&weights);
    double total = 0.0;
    for (const double weight : weights)
      total += weight;
    const double draw = std::generate_canonical<double, 53>(generator_) * total;
    double cumulative = 0.0;
    chosen = candidates_.back().index;
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
      cumulative += weights[i];
      if (draw < cumulative) {
        chosen = candidates_[i].index;
        break;
      }
    }
  }
  ++step_;
  if (chosen != end_index_)
    history_.push_back(chosen);
  return chosen + window_.begin;
}

}  // namespace gufo::models::yue2
