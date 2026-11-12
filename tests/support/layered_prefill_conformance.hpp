#ifndef GUFO_TESTS_SUPPORT_LAYERED_PREFILL_CONFORMANCE_HPP_
#define GUFO_TESTS_SUPPORT_LAYERED_PREFILL_CONFORMANCE_HPP_

// Shared conformance check for runners whose scheduling profile splits prompt
// passes into layer groups (SetPrefillYield). Every such runner must pass it.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gufo/ext/text.hpp"

namespace gufo::test {

struct LayeredPrefillCase {
  /// Prefilled in passes of the profile's width, split by layers.
  std::vector<ext::TextRunnerToken> prompt;
  /// The other session, decoding between the prompt's layer groups.
  std::vector<ext::TextRunnerToken> decoder_prompt;
  /// Further sessions decoding beside the first. With any, every decode step
  /// batches all unfinished decoders, alone and inside the yields, as the
  /// host does for a profile whose rounds are wider than one session; their
  /// lone reference is the same batches without the prompt.
  std::vector<std::vector<ext::TextRunnerToken>> more_decoder_prompts;
  std::size_t prompt_output_tokens{4};
  std::size_t decoder_output_tokens{16};
  /// Top log-probabilities compared at every output position.
  std::uint32_t top_logprobs{8};
  /// Layer groups per pass; zero takes the profile's maximum.
  std::size_t layer_groups{0};
  /// Plan the cache's prompt checkpoints, as serving does, so passes may
  /// capture them in place (PrefillThrough). Nothing is reused across runs.
  bool cache_prompt{false};
};

/// Runs `runner` through three fresh state pools, greedily: the prompt alone,
/// the decoders alone, then the decoders decoding one step in each yield of
/// the prompt's layered passes. Returns every difference in tokens or
/// log-probabilities against the lone runs; empty means bit-identical.
[[nodiscard]] std::vector<std::string> CheckLayeredPrefill(
    const std::shared_ptr<ext::TextModelRunner>& runner,
    const LayeredPrefillCase& test);

}  // namespace gufo::test

#endif  // GUFO_TESTS_SUPPORT_LAYERED_PREFILL_CONFORMANCE_HPP_
