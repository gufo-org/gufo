#ifndef GUFO_TESTS_QWEN27B_SAMPLING_CASES_HPP_
#define GUFO_TESTS_QWEN27B_SAMPLING_CASES_HPP_

#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::test {

struct SamplingCase {
  std::string_view name;
  sampling::SamplingConfig config;
};

// Individual strategies make failures diagnosable; the operator test also
// combines filters in a Cartesian matrix. Model tests reuse these cases.
inline std::vector<SamplingCase> QwenSamplingCases() {
  std::vector<SamplingCase> cases{
      {"greedy", {}},
      {"temperature-cold", {.temperature = 0.05F}},
      {"temperature", {.temperature = 1.0F}},
      {"temperature-hot", {.temperature = 2.0F}},
      {"top-k-one", {.temperature = 0.8F, .top_k = 1}},
      {"top-k", {.temperature = 0.8F, .top_k = 3}},
      {"top-p", {.temperature = 0.8F, .top_p = 0.7F}},
      {"min-p", {.temperature = 0.8F, .min_p = 0.3F}},
      {"min-p-one", {.temperature = 0.8F, .min_p = 1.0F}},
      {"top-k-floor", {.temperature = 0.8F, .top_k = 1, .min_keep = 3}},
      {"top-p-floor", {.temperature = 0.8F, .top_p = 0.01F, .min_keep = 3}},
      {"min-p-floor", {.temperature = 0.8F, .min_p = 1.0F, .min_keep = 3}},
      {"repeat-penalty", {.temperature = 0.8F, .repeat_penalty = 1.5F}},
      {"repeat-reward", {.temperature = 0.8F, .repeat_penalty = 0.7F}},
      {"frequency-penalty", {.temperature = 0.8F, .frequency_penalty = 0.4F}},
      {"frequency-reward", {.temperature = 0.8F, .frequency_penalty = -0.4F}},
      {"presence-penalty", {.temperature = 0.8F, .presence_penalty = 0.4F}},
      {"presence-reward", {.temperature = 0.8F, .presence_penalty = -0.4F}},
      {"history-one",
       {.temperature = 0.8F, .repeat_penalty = 1.5F, .repeat_last_n = 1}},
      {"history-disabled",
       {.temperature = 0.8F,
        .repeat_penalty = 1.5F,
        .repeat_last_n = 0,
        .frequency_penalty = 0.4F,
        .presence_penalty = 0.4F}},
      {"greedy-penalties",
       {.repeat_penalty = 1.5F,
        .repeat_last_n = 3,
        .frequency_penalty = 0.4F,
        .presence_penalty = -0.4F}},
      {"all-filters",
       {.temperature = 0.8F,
        .top_k = 4,
        .top_p = 0.7F,
        .min_p = 0.3F,
        .min_keep = 2,
        .repeat_penalty = 1.2F,
        .repeat_last_n = 3,
        .frequency_penalty = 0.2F,
        .presence_penalty = -0.1F}},
      {"all-floors",
       {.temperature = 0.8F,
        .top_k = 1,
        .top_p = 0.01F,
        .min_p = 1.0F,
        .min_keep = 3}},
  };
  for (auto& test : cases) {
    test.config.seed = 73;
  }
  cases[2].config.seed = 0;
  cases[3].config.seed = 808;
  return cases;
}

}  // namespace gufo::test

#endif
