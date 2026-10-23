// Teacher-forced prompt scoring: the log-softmax helper and the scheduler
// path that reads the frontier's logits before advancing each history token.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/prompt_logprobs.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/text_model_runner.hpp"

namespace {

using gufo::server::kMaxPromptLogprobs;
using gufo::server::PromptTokenLogprobs;
using gufo::server::ScorePromptToken;
using gufo::server::TextDecodeSelection;
using gufo::server::TextExecutionPlan;
using gufo::server::TextGenerationScheduler;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillStep;
using gufo::server::TextRequestMetadata;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerPool;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;
using gufo::server::TextScoring;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

bool Near(double a, double b) {
  return std::abs(a - b) < 1e-9;
}

template<typename Exception, typename Function>
bool Throws(Function function) {
  try {
    function();
  } catch (const Exception&) {
    return true;
  }
  return false;
}

void TestLogSoftmax() {
  const std::vector<float> logits{1.0F, 3.0F, 2.0F, 3.0F, -1.0F};
  double sum = 0.0;
  for (const float logit : logits)
    sum += std::exp(static_cast<double>(logit));
  const double log_sum = std::log(sum);

  const auto scored = ScorePromptToken(logits, 2, 3);
  Require(scored.token == 2, "the scored token is the history's");
  Require(Near(scored.logprob, 2.0 - log_sum), "target log-probability");
  Require(scored.rank == 3, "two tokens outrank the target");
  Require(scored.nonfinite == 0, "all logits are finite");
  Require(scored.top.size() == 3, "top-k has k entries");
  // Ties rank by lower token ID first.
  Require(scored.top[0].token == 1 && scored.top[1].token == 3 &&
              scored.top[2].token == 2,
          "top-k is most probable first, ties by token ID");
  Require(Near(scored.top[0].logprob, 3.0 - log_sum),
          "top-k log-probabilities share the normalizer");
  double total = 0.0;
  for (const float logit : logits)
    total += std::exp(static_cast<double>(logit) - log_sum);
  Require(Near(total, 1.0), "the distribution is normalized");

  const auto tied = ScorePromptToken(logits, 3, 0);
  Require(tied.rank == 2 && tied.top.empty(),
          "a tied target ranks after the lower ID; k=0 lists none");
  const auto all = ScorePromptToken(logits, 4, logits.size());
  Require(all.top.size() == logits.size() && all.rank == 5,
          "k larger than the vocabulary lists the whole vocabulary");

  // Large logits stay finite: the normalizer subtracts the maximum first.
  const std::vector<float> large{1000.0F, 999.0F};
  const auto stable = ScorePromptToken(large, 1, 1);
  Require(std::isfinite(stable.logprob) &&
              Near(stable.logprob, -std::log1p(std::exp(1.0))),
          "log-softmax is stable for large logits");
}

void TestNonFiniteAndRejection() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> logits{0.0F, nan, 0.0F, inf};
  const auto scored = ScorePromptToken(logits, 0, 4);
  Require(scored.nonfinite == 2, "non-finite logits are counted");
  Require(Near(scored.logprob, std::log(0.5)),
          "non-finite logits carry no probability");
  Require(scored.top.size() == 2, "non-finite logits are never listed");
  const auto lost = ScorePromptToken(logits, 1, 1);
  Require(std::isinf(lost.logprob) && lost.logprob < 0 && lost.rank == 3,
          "a non-finite target ranks after every finite token");

  const std::vector<float> ok{0.0F, 1.0F};
  Require(Throws<std::invalid_argument>(
              [&] { (void)ScorePromptToken(ok, 0, kMaxPromptLogprobs + 1); }),
          "prompt_logprobs above 20 is rejected");
  Require(
      Throws<std::invalid_argument>([&] { (void)ScorePromptToken(ok, 2, 1); }),
      "a token outside the vocabulary is rejected");
  Require(Throws<std::runtime_error>(
              [&] { (void)ScorePromptToken(std::span<const float>{}, 0, 1); }),
          "empty logits are rejected");
  const std::vector<float> none{nan, inf};
  Require(
      Throws<std::runtime_error>([&] { (void)ScorePromptToken(none, 0, 1); }),
      "all non-finite logits are rejected");
}

constexpr std::size_t kVocab = 16;

/// Logits that depend on the whole history, so a scored position that saw
/// the wrong prefix has the wrong distribution.
std::vector<float> HistoryLogits(std::span<const TextRunnerToken> history) {
  std::size_t mix = history.size();
  for (const auto token : history)
    mix = mix * 31 + token;
  std::vector<float> logits(kVocab);
  for (std::size_t i = 0; i < kVocab; ++i)
    logits[i] = static_cast<float>((mix + i * 7) % 11) * 0.5F;
  return logits;
}

class HistoryState final : public TextRunnerState {
public:
  void Invalidate() noexcept override {
    tokens.clear();
    logits.clear();
  }
  std::vector<TextRunnerToken> tokens;
  std::vector<float> logits;
};

struct Calls {
  std::mutex mutex;
  std::vector<std::size_t> prefills;
  std::vector<TextRunnerToken> advances;
  std::size_t selections{0};
};

class HistoryRunner final : public TextModelRunner {
public:
  explicit HistoryRunner(std::shared_ptr<Calls> calls, bool logits = true)
      : calls_(std::move(calls)), logits_(logits) {}

  TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "history",
            .state_abi = "history-v1",
            .max_context = 32,
            .capabilities = TextRunnerCapabilities{.incremental_prefill = true,
                                                   .prefix_reuse = true},
            .persistence = std::nullopt};
  }
  TextRunnerResourceClaim ResourceClaim() const override { return {}; }
  std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {TextExecutionPlan{}};
  }
  std::vector<TextRunnerToken> Tokenize(std::string_view) const override {
    return {1};
  }
  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const gufo::server::ChatRequest&) const override {
    return std::nullopt;
  }
  std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    return std::string(tokens.size(), 'x');
  }
  std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<HistoryState>();
  }
  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t max_input_tokens) const override {
    auto& history = dynamic_cast<HistoryState&>(state);
    if (offset != history.tokens.size())
      throw std::logic_error("prefill offset is not the frontier");
    const auto count = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, std::size_t{4}});
    history.tokens.insert(history.tokens.end(), prompt.begin() + offset,
                          prompt.begin() + offset + count);
    {
      const std::lock_guard<std::mutex> lock(calls_->mutex);
      calls_->prefills.push_back(count);
    }
    return {.consumed_tokens = count,
            .decode_ready = history.tokens.size() == prompt.size()};
  }
  TextDecodeSelection SelectNext(
      TextRunnerState& state,
      gufo::sampling::SamplerState& sampler) const override {
    {
      const std::lock_guard<std::mutex> lock(calls_->mutex);
      ++calls_->selections;
    }
    auto& history = dynamic_cast<HistoryState&>(state);
    const auto token = sampler.Sample(HistoryLogits(history.tokens));
    return {.token = token, .piece = "x"};
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    dynamic_cast<HistoryState&>(state).tokens.push_back(token);
    const std::lock_guard<std::mutex> lock(calls_->mutex);
    calls_->advances.push_back(token);
  }
  std::span<const float> FrontierLogits(TextRunnerState& state) const override {
    if (!logits_)
      return {};
    auto& history = dynamic_cast<HistoryState&>(state);
    history.logits = HistoryLogits(history.tokens);
    return history.logits;
  }
  std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    return dynamic_cast<const HistoryState&>(state).tokens.size();
  }

private:
  std::shared_ptr<Calls> calls_;
  bool logits_;
};

struct Harness {
  explicit Harness(bool logits = true, std::size_t sessions = 1)
      : calls(std::make_shared<Calls>()),
        pool(std::make_shared<TextRunnerPool>(
            std::make_shared<HistoryRunner>(calls, logits), sessions)),
        scheduler(pool) {}

  TextGenerationScheduler::Request Score(std::vector<TextRunnerToken> history,
                                         std::size_t start, std::size_t top_k,
                                         bool cache_prompt = true) {
    std::vector<TextRunnerToken> prompt(history.begin(),
                                        history.begin() + start);
    TextScoring scoring{.tokens = {history.begin() + start, history.end()},
                        .top_k = top_k};
    gufo::sampling::SamplingConfig sampling;
    sampling.temperature = 1.0F;
    TextRequestMetadata metadata;
    metadata.cache_prompt = cache_prompt;
    metadata.scoring = std::move(scoring);
    return scheduler.Submit(std::move(prompt), history.size() - start, sampling,
                            {}, false, std::move(metadata));
  }

  std::shared_ptr<Calls> calls;
  std::shared_ptr<TextRunnerPool> pool;
  TextGenerationScheduler scheduler;
};

void TestSchedulerScoresTheGivenHistory() {
  const std::vector<TextRunnerToken> history{3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
  const std::size_t start = 3;
  Harness harness;
  auto request = harness.Score(history, start, 5);
  const auto result = request.Wait();

  Require(result.prompt_logprobs.size() == history.size() - start,
          "one entry per scored position");
  for (std::size_t i = 0; i < result.prompt_logprobs.size(); ++i) {
    const std::size_t position = start + i;
    // Position p is predicted from tokens [0, p): the offset is exact.
    const auto logits = HistoryLogits(std::span(history).first(position));
    const auto expected = ScorePromptToken(logits, history[position], 5);
    const auto& actual = result.prompt_logprobs[i];
    Require(actual.token == history[position],
            "position " + std::to_string(position) + " scores its own token");
    Require(
        Near(actual.logprob, expected.logprob) && actual.rank == expected.rank,
        "position " + std::to_string(position) +
            " uses the logits of exactly the preceding history");
    Require(actual.top.size() == 5, "top-k per position");
    for (std::size_t k = 0; k < 5; ++k)
      Require(actual.top[k].token == expected.top[k].token &&
                  Near(actual.top[k].logprob, expected.top[k].logprob),
              "top-k matches the reference distribution");
  }
  const std::lock_guard<std::mutex> lock(harness.calls->mutex);
  Require(harness.calls->selections == 0, "scoring never samples");
  const std::vector<TextRunnerToken> advanced(history.begin() + start,
                                              history.end() - 1);
  Require(harness.calls->advances == advanced,
          "every scored token but the last is advanced, in order");
  Require(result.tokens.empty() && result.completion_tokens == 0 &&
              result.text.empty(),
          "scoring generates nothing");
}

void TestScoringNeverReusesAPrefix() {
  const std::vector<TextRunnerToken> history{7, 7, 2, 8, 1, 8, 2, 8};
  Harness harness;
  const auto first = harness.Score(history, 4, 2).Wait();
  // Asked to reuse, the second request still prefills from the start.
  const auto second = harness.Score(history, 4, 2, true).Wait();
  Require(second.cached_prompt_tokens == 0 && !second.cache_hit,
          "a scored request never restores a cached prefix");
  std::size_t prefilled = 0;
  {
    const std::lock_guard<std::mutex> lock(harness.calls->mutex);
    for (const auto count : harness.calls->prefills)
      prefilled += count;
  }
  Require(prefilled == 8, "both requests prefill their whole prompt");
  for (std::size_t i = 0; i < first.prompt_logprobs.size(); ++i)
    Require(
        first.prompt_logprobs[i].logprob == second.prompt_logprobs[i].logprob &&
            first.prompt_logprobs[i].rank == second.prompt_logprobs[i].rank,
        "repeated scoring is bit-identical");
}

void TestScoringInterleavesWithGeneration() {
  Harness harness(true, 2);
  gufo::sampling::SamplingConfig greedy;
  auto generation =
      harness.scheduler.Submit({2, 4, 6}, 6, greedy, {}, false, {});
  auto scoring = harness.Score({5, 4, 3, 2, 1, 0, 1, 2, 3}, 2, 1);
  const auto generated = generation.Wait();
  const auto scored = scoring.Wait();
  Require(generated.tokens.size() == 6, "a generating peer keeps decoding");
  Require(scored.prompt_logprobs.size() == 7, "the scored peer completes");
}

void TestSchedulerRejections() {
  Harness harness;
  Require(Throws<std::invalid_argument>([&] {
            (void)harness.Score({1, 2, 3}, 1, kMaxPromptLogprobs + 1);
          }),
          "the scheduler rejects prompt_logprobs above 20");
  // 32-token context: 33 tokens need 32 positions of state, 34 need 33.
  std::vector<TextRunnerToken> fits(33, 1);
  (void)harness.Score(fits, 1, 0).Wait();
  std::vector<TextRunnerToken> too_long(34, 1);
  Require(
      Throws<std::length_error>([&] { (void)harness.Score(too_long, 1, 0); }),
      "a history beyond the context is rejected");
  TextRequestMetadata stops;
  stops.stop_sequences = {"x"};
  stops.scoring = TextScoring{.tokens = {1}};
  Require(Throws<std::invalid_argument>([&] {
            (void)harness.scheduler.Submit({1}, 1, {}, {}, false,
                                           std::move(stops));
          }),
          "scoring refuses stop sequences");

  Harness blind(false);
  bool failed = false;
  try {
    (void)blind.Score({1, 2, 3}, 1, 1).Wait();
  } catch (const std::invalid_argument& error) {
    failed = std::string(error.what()).find("logits") != std::string::npos;
  }
  Require(failed, "a runner without logits fails the request clearly");
}

}  // namespace

int main() {
  TestLogSoftmax();
  TestNonFiniteAndRejection();
  TestSchedulerScoresTheGivenHistory();
  TestScoringNeverReusesAPrefix();
  TestScoringInterleavesWithGeneration();
  TestSchedulerRejections();
  std::puts("prompt_scoring_test: ok");
  return 0;
}
