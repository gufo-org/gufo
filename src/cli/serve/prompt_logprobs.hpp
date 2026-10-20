#ifndef GUFO_SERVER_PROMPT_LOGPROBS_HPP_
#define GUFO_SERVER_PROMPT_LOGPROBS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace gufo::server {

/// Largest `prompt_logprobs` a request may ask for, as in vLLM.
inline constexpr std::size_t kMaxPromptLogprobs = 20;

struct TokenLogprob {
  std::uint32_t token{0};
  double logprob{0.0};
};

/// The model's next-token distribution at one teacher-forced position:
/// log-softmax of the raw logits, without sampling controls.
struct PromptTokenLogprobs {
  /// The token the history actually continues with.
  std::uint32_t token{0};
  double logprob{0.0};
  /// One-based rank of `token`; ties rank by lower token ID first.
  std::size_t rank{0};
  /// Logits that were NaN or infinite; they carry no probability.
  std::size_t nonfinite{0};
  /// Highest-probability tokens, most probable first.
  std::vector<TokenLogprob> top;
};

/// Normalizes `logits` in double precision and reports `target` and the
/// `top_k` most probable tokens. Non-finite logits are excluded.
[[nodiscard]] inline PromptTokenLogprobs ScorePromptToken(
    std::span<const float> logits, std::uint32_t target, std::size_t top_k) {
  if (logits.empty())
    throw std::runtime_error("prompt scoring has no logits");
  if (target >= logits.size())
    throw std::invalid_argument("prompt token " + std::to_string(target) +
                                " is outside the vocabulary of " +
                                std::to_string(logits.size()));
  if (top_k > kMaxPromptLogprobs)
    throw std::invalid_argument("prompt_logprobs exceeds the maximum of " +
                                std::to_string(kMaxPromptLogprobs));
  PromptTokenLogprobs result;
  result.token = target;
  double max = -std::numeric_limits<double>::infinity();
  for (const float logit : logits) {
    if (std::isfinite(logit))
      max = std::max(max, static_cast<double>(logit));
    else
      ++result.nonfinite;
  }
  if (!std::isfinite(max))
    throw std::runtime_error("prompt scoring logits are all non-finite");
  double sum = 0.0;
  for (const float logit : logits) {
    if (std::isfinite(logit))
      sum += std::exp(static_cast<double>(logit) - max);
  }
  const double normalizer = max + std::log(sum);
  const float target_logit = logits[target];
  const bool target_finite = std::isfinite(target_logit);
  result.logprob = target_finite
                       ? static_cast<double>(target_logit) - normalizer
                       : -std::numeric_limits<double>::infinity();
  // Ordered as the rank below: higher logit first, then lower token ID.
  const auto before = [&](std::uint32_t a, std::uint32_t b) {
    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
  };
  std::size_t rank = 1;
  std::vector<std::uint32_t> top;
  top.reserve(top_k + 1);
  for (std::uint32_t token = 0; token < logits.size(); ++token) {
    if (!std::isfinite(logits[token]))
      continue;
    if (!target_finite || before(token, target))
      ++rank;
    if (top_k == 0)
      continue;
    if (top.size() < top_k) {
      top.push_back(token);
      std::push_heap(top.begin(), top.end(), before);
    } else if (before(token, top.front())) {
      std::pop_heap(top.begin(), top.end(), before);
      top.back() = token;
      std::push_heap(top.begin(), top.end(), before);
    }
  }
  // A non-finite target ranks after every finite token.
  result.rank = rank;
  std::sort_heap(top.begin(), top.end(), before);
  result.top.reserve(top.size());
  for (const auto token : top)
    result.top.push_back(
        {.token = token,
         .logprob = static_cast<double>(logits[token]) - normalizer});
  return result;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_PROMPT_LOGPROBS_HPP_
