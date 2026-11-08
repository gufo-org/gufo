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
                                const sampling::SamplingConfig& sampling,
                                bool cache_prompt) {
  // Every run has a fresh pool and persists nothing, so it computes its
  // whole prompt.
  return pool.Acquire(std::move(prompt), sampling, {}, {}, cache_prompt, 0,
                      /*stop_at_eos=*/false, /*persist=*/false);
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

void Rethrow(const std::exception_ptr& failure) {
  if (failure)
    std::rethrow_exception(failure);
}

// One step of every unfinished decoder, batched as the host batches a round.
void StepAll(TextRunnerPool& pool,
             std::vector<TextRunnerPool::Request>& requests,
             std::vector<Output>& outputs, std::size_t limit) {
  std::vector<std::size_t> active;
  for (std::size_t index = 0; index < requests.size(); ++index)
    if (!outputs[index].stopped && outputs[index].selections.size() < limit)
      active.push_back(index);
  if (active.empty())
    return;
  const auto plan = pool.SelectDecodePlan(active.size());
  const auto& capabilities = pool.runner().Descriptor().capabilities;
  if (active.size() == 1 || plan.kind != ext::TextExecutionPlanKind::kBatched ||
      (capabilities.multi_token_decode &&
       !capabilities.batched_multi_token_decode)) {
    for (const auto index : active)
      Step(requests[index], outputs[index], limit);
    return;
  }
  std::vector<TextRunnerPool::Request*> batch;
  for (const auto index : active)
    batch.push_back(&requests[index]);
  if (capabilities.batched_multi_token_decode) {
    std::vector<std::size_t> budgets;
    for (const auto index : active)
      budgets.push_back(limit - outputs[index].selections.size());
    auto steps = pool.DecodeBatch(batch, budgets, plan);
    for (std::size_t lane = 0; lane < active.size(); ++lane) {
      Rethrow(steps[lane].failure);
      auto& output = outputs[active[lane]];
      for (auto& selection : steps[lane].selections)
        output.selections.push_back(std::move(selection));
      output.stopped = steps[lane].stop;
    }
    return;
  }
  // Single-token decoders: select every lane, then advance them together.
  std::vector<TextRunnerPool::Request*> advancing;
  for (const auto index : active) {
    auto selection = requests[index].SelectNext();
    if (selection.stop) {
      outputs[index].stopped = true;
      continue;
    }
    outputs[index].selections.push_back(std::move(selection));
    advancing.push_back(&requests[index]);
  }
  if (advancing.size() == 1) {
    advancing.front()->Advance();
  } else if (!advancing.empty()) {
    for (const auto& failure :
         pool.AdvanceBatch(advancing, pool.SelectDecodePlan(advancing.size())))
      Rethrow(failure);
  }
}

bool Unfinished(const std::vector<Output>& outputs, std::size_t limit) {
  return std::ranges::any_of(outputs, [&](const Output& output) {
    return !output.stopped && output.selections.size() < limit;
  });
}

std::size_t Selections(const std::vector<Output>& outputs) {
  std::size_t count = 0;
  for (const auto& output : outputs)
    count += output.selections.size();
  return count;
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
      auto request = Acquire(pool, test.prompt, sampling, test.cache_prompt);
      PrefillAll(request, width);
      Finish(request, prompt_alone, test.prompt_output_tokens);
    }
    std::vector<std::vector<ext::TextRunnerToken>> decoder_prompts{
        test.decoder_prompt};
    decoder_prompts.insert(decoder_prompts.end(),
                           test.more_decoder_prompts.begin(),
                           test.more_decoder_prompts.end());
    const std::size_t limit = test.decoder_output_tokens;
    const auto start_decoders = [&](TextRunnerPool& pool) {
      std::vector<TextRunnerPool::Request> decoders;
      for (const auto& prompt : decoder_prompts) {
        decoders.push_back(Acquire(pool, prompt, sampling, test.cache_prompt));
        PrefillAll(decoders.back(), width);
      }
      return decoders;
    };

    std::vector<Output> decoders_alone(decoder_prompts.size());
    {
      TextRunnerPool pool(runner, decoder_prompts.size());
      auto decoders = start_decoders(pool);
      while (Unfinished(decoders_alone, limit))
        StepAll(pool, decoders, decoders_alone, limit);
    }

    Output prompt_split;
    std::vector<Output> decoders_split(decoder_prompts.size());
    std::size_t yields = 0;
    std::size_t yielded_tokens = 0;
    {
      TextRunnerPool pool(runner, decoder_prompts.size() + 1);
      auto decoders = start_decoders(pool);
      StepAll(pool, decoders, decoders_split, limit);
      const auto before = Selections(decoders_split);
      auto request = Acquire(pool, test.prompt, sampling, test.cache_prompt);
      PrefillAll(request, width, groups, [&] {
        ++yields;
        StepAll(pool, decoders, decoders_split, limit);
      });
      yielded_tokens = Selections(decoders_split) - before;
      Finish(request, prompt_split, test.prompt_output_tokens);
      while (Unfinished(decoders_split, limit))
        StepAll(pool, decoders, decoders_split, limit);
    }
    if (yields == 0)
      failures.push_back("the runner never yielded between layer groups");
    else if (yielded_tokens == 0)
      failures.push_back("the decoder produced no token inside a yield");
    Compare("prompt", prompt_alone, prompt_split, failures);
    for (std::size_t index = 0; index < decoder_prompts.size(); ++index)
      Compare(("decoder " + std::to_string(index)).c_str(),
              decoders_alone[index], decoders_split[index], failures);
  } catch (const std::exception& error) {
    failures.push_back(std::string("run failed: ") + error.what());
  }
  return failures;
}

}  // namespace gufo::test
