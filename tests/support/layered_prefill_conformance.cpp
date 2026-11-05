#include "tests/support/layered_prefill_conformance.hpp"

#include <algorithm>
#include <exception>
#include <optional>
#include <sstream>
#include <utility>

#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::test {
namespace {

using server::TextRunnerPool;

struct Output {
  std::vector<ext::TextDecodeSelection> selections;
  bool stopped{false};
};

sampling::SamplingConfig Greedy(std::uint32_t top_logprobs) {
  sampling::SamplingConfig config;
  config.temperature = 0.0F;
  config.logprobs = top_logprobs;
  return config;
}

TextRunnerPool::Request Acquire(TextRunnerPool& pool,
                                std::vector<ext::TextRunnerToken> prompt,
                                const sampling::SamplingConfig& sampling) {
  // Nothing is reused or persisted, so every run computes its whole prompt.
  return pool.Acquire(std::move(prompt), sampling, {}, {},
                      /*reuse_prompt=*/false, 0, /*stop_at_eos=*/false,
                      /*persist=*/false);
}

void PrefillAll(TextRunnerPool::Request& request, std::size_t width,
                std::size_t groups = 1, const ext::PrefillYield& yield = {}) {
  while (!request.prefill_complete())
    (void)request.Prefill(width, groups, yield);
}

// One decode step, until `limit` tokens or a stop.
void Step(TextRunnerPool::Request& request, Output& output, std::size_t limit) {
  if (output.stopped || output.selections.size() >= limit)
    return;
  auto step = request.DecodeStep(limit - output.selections.size());
  for (auto& selection : step.selections)
    output.selections.push_back(std::move(selection));
  output.stopped = step.stop;
}

void Finish(TextRunnerPool::Request& request, Output& output,
            std::size_t limit) {
  while (!output.stopped && output.selections.size() < limit)
    Step(request, output, limit);
}

std::string Describe(const std::optional<sampling::TokenLogprobs>& logprobs) {
  if (!logprobs)
    return "none";
  std::ostringstream text;
  text.precision(17);
  text << logprobs->selected.token << ':' << logprobs->selected.logprob;
  for (const auto& entry : logprobs->top)
    text << ' ' << entry.token << ':' << entry.logprob;
  return text.str();
}

void Compare(const char* name, const Output& expected, const Output& actual,
             std::vector<std::string>& failures) {
  if (expected.stopped != actual.stopped ||
      expected.selections.size() != actual.selections.size())
    failures.push_back(std::string(name) + ": produced " +
                       std::to_string(actual.selections.size()) +
                       " tokens, expected " +
                       std::to_string(expected.selections.size()));
  const auto count =
      std::min(expected.selections.size(), actual.selections.size());
  for (std::size_t index = 0; index < count; ++index) {
    const auto& want = expected.selections[index];
    const auto& got = actual.selections[index];
    const auto want_logprobs = Describe(want.logprobs);
    const auto got_logprobs = Describe(got.logprobs);
    if (want.token != got.token || want_logprobs != got_logprobs) {
      failures.push_back(
          std::string(name) + " token " + std::to_string(index) + ": got " +
          std::to_string(got.token) + " [" + got_logprobs + "], expected " +
          std::to_string(want.token) + " [" + want_logprobs + "]");
      return;
    }
  }
}

}  // namespace

std::vector<std::string> CheckLayeredPrefill(
    const std::shared_ptr<ext::TextModelRunner>& runner,
    const LayeredPrefillCase& test) {
  const auto profile = runner->Descriptor().scheduling;
  if (!profile || profile->max_layer_groups < 2)
    return {"the runner's scheduling profile does not split passes"};
  if (test.prompt.size() <= profile->pass_tokens)
    return {"the prompt must span more than one pass"};
  const std::size_t width = profile->pass_tokens;
  const std::size_t groups =
      test.layer_groups == 0 ? profile->max_layer_groups : test.layer_groups;
  const auto sampling = Greedy(test.top_logprobs);
  std::vector<std::string> failures;
  try {
    Output prompt_alone;
    {
      TextRunnerPool pool(runner, 1);
      auto request = Acquire(pool, test.prompt, sampling);
      PrefillAll(request, width);
      Finish(request, prompt_alone, test.prompt_output_tokens);
    }
    Output decoder_alone;
    {
      TextRunnerPool pool(runner, 1);
      auto request = Acquire(pool, test.decoder_prompt, sampling);
      PrefillAll(request, width);
      Finish(request, decoder_alone, test.decoder_output_tokens);
    }

    Output prompt_split;
    Output decoder_split;
    std::size_t yields = 0;
    std::size_t yielded_tokens = 0;
    {
      TextRunnerPool pool(runner, 2);
      auto decoder = Acquire(pool, test.decoder_prompt, sampling);
      PrefillAll(decoder, width);
      Step(decoder, decoder_split, test.decoder_output_tokens);
      const auto before = decoder_split.selections.size();
      auto request = Acquire(pool, test.prompt, sampling);
      PrefillAll(request, width, groups, [&] {
        ++yields;
        Step(decoder, decoder_split, test.decoder_output_tokens);
      });
      yielded_tokens = decoder_split.selections.size() - before;
      Finish(request, prompt_split, test.prompt_output_tokens);
      Finish(decoder, decoder_split, test.decoder_output_tokens);
    }
    if (yields == 0)
      failures.push_back("the runner never yielded between layer groups");
    else if (yielded_tokens == 0)
      failures.push_back("the decoder produced no token inside a yield");
    Compare("prompt", prompt_alone, prompt_split, failures);
    Compare("decoder", decoder_alone, decoder_split, failures);
  } catch (const std::exception& error) {
    failures.push_back(std::string("run failed: ") + error.what());
  }
  return failures;
}

}  // namespace gufo::test
