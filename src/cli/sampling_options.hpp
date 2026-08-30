#ifndef GUFO_CLI_SAMPLING_OPTIONS_HPP_
#define GUFO_CLI_SAMPLING_OPTIONS_HPP_

#include <stdexcept>
#include <string_view>

#include "src/cli/arg_parser.hpp"
#include "src/core/sampling.hpp"

namespace gufo::cli {

/// Registers the complete model-independent sampling control surface.
///
/// New text-generation CLIs should use this helper instead of defining a
/// model-specific subset. Short aliases can be disabled when a parent command
/// already owns -t or -s.
inline void RegisterSamplingOptions(ArgParser& parser,
                                    sampling::SamplingConfig* config,
                                    std::string_view group = "Sampling",
                                    bool register_short_aliases = true) {
  if (config == nullptr) {
    throw std::invalid_argument("sampling CLI config must not be null");
  }
  parser.AddOption(register_short_aliases ? "-t" : "", "--temperature", "T",
                   "Randomness scale; 0.0 selects greedy argmax (default: 0.0)",
                   group, &config->temperature);
  parser.AddOption(
      "", "--top-k", "K",
      "Keep only the K highest-logit tokens (default: 0 = disabled)", group,
      &config->top_k);
  parser.AddOption(
      "", "--top-p", "P",
      "Keep the smallest token set whose cumulative probability reaches P "
      "(default: 1.0 = disabled)",
      group, &config->top_p);
  parser.AddOption(
      "", "--min-p", "P",
      "Keep tokens with probability at least P times the best token "
      "(default: 0.0 = disabled)",
      group, &config->min_p);
  parser.AddOption("", "--min-keep", "N",
                   "Minimum candidates retained by enabled sampling filters "
                   "(default: 0)",
                   group, &config->min_keep);
  parser.AddOption(
      register_short_aliases ? "-s" : "", "--seed", "N",
      "RNG seed for reproducible generation (default: -1 = random)", group,
      &config->seed);
  parser.AddOption("", "--repeat-penalty", "N",
                   "Multiplicative penalty for recently used tokens "
                   "(default: 1.0 = disabled)",
                   group, &config->repeat_penalty);
  parser.AddOption("", "--repeat-last-n", "N",
                   "Trailing token window used by repetition penalties "
                   "(default: 64; 0 = disabled)",
                   group, &config->repeat_last_n);
  parser.AddOption("", "--frequency-penalty", "N",
                   "Penalty multiplied by each token's recent occurrence count "
                   "(default: 0.0 = disabled)",
                   group, &config->frequency_penalty);
  parser.AddOption("", "--presence-penalty", "N",
                   "One-time penalty for tokens present in the recent window "
                   "(default: 0.0 = disabled)",
                   group, &config->presence_penalty);
}

}  // namespace gufo::cli

#endif  // GUFO_CLI_SAMPLING_OPTIONS_HPP_
