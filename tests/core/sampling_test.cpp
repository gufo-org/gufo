#include "src/core/sampling.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json_constraint.hpp"

namespace {

using gufo::sampling::ConstraintVocabulary;
using gufo::sampling::JsonConstraint;
using gufo::sampling::ReasoningBudget;
using gufo::sampling::SamplerState;
using gufo::sampling::SamplingConfig;
using gufo::sampling::TokenConstraint;
using gufo::sampling::TokenId;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

// Greedy logits whose argmax is `winner`; token 3 stands for `</think>`.
std::vector<float> Logits(TokenId winner) {
  std::vector<float> logits(8, 0.0F);
  logits[winner] = 5.0F;
  return logits;
}

std::shared_ptr<const ReasoningBudget> Budget(
    std::size_t tokens, std::vector<TokenId> forced = {3},
    std::vector<TokenId> prompt = {}) {
  auto budget = std::make_shared<ReasoningBudget>();
  budget->tokens = tokens;
  budget->forced = std::move(forced);
  budget->ends = {3};
  budget->prompt = std::move(prompt);
  return budget;
}

void TestReasoningBudgetForcesTheEnd() {
  SamplingConfig config;
  config.reasoning_budget = Budget(2);
  SamplerState sampler(config);
  Expect(!config.can_use_unmodified_argmax(),
         "a budget keeps requests off the unmodified argmax path");
  Expect(!sampler.ForcedToken() && sampler.CanSelectArgmax(1) &&
             !sampler.NeedsConstraintMask(),
         "the budget does not restrict reasoning before it is spent");
  sampler.Accept(sampler.Sample(Logits(1)));
  sampler.Accept(sampler.Sample(Logits(2)));
  Expect(sampler.ForcedToken() == TokenId{3},
         "two reasoning tokens exhaust a budget of two");
  Expect(sampler.NeedsConstraintMask(),
         "GPU sampling paths defer the forced end to the host sampler");
  Expect(!sampler.CanSelectArgmax(1) && sampler.CanSelectArgmax(3),
         "only the reasoning end is a selectable argmax");
  Expect(sampler.Sample(Logits(1)) == 3 &&
             sampler.Distribution(Logits(1)).best_token() == 3,
         "sampling and distributions return the forced end");
  const std::vector<TokenId> ids{5, 3, 1};
  const std::vector<float> compact{1.0F, 0.0F, 2.0F};
  Expect(sampler.Distribution(compact, ids).best_token() == 1,
         "compact distributions index the forced end among their IDs");
  // An MTP draft shortlist may not contain the forced end. Its proposal then
  // follows the shortlist; target verification rejects it and emits the end.
  const std::vector<TokenId> shortlist{5, 4, 1};
  Expect(sampler.Distribution(compact, shortlist).best_token() == 2,
         "a shortlist without the forced end keeps its own distribution");
  auto copy = sampler;
  copy.Accept(3);
  Expect(!copy.ForcedToken() && !copy.NeedsConstraintMask() &&
             sampler.ForcedToken(),
         "copies advance independently, as speculative verification needs");
  Expect(copy.Sample(Logits(6)) == 6, "the answer is sampled normally");
  copy.ResetHistory({});
  Expect(!copy.ForcedToken(), "a restart resets the reasoning budget");
}

void TestEarlyEndAndNoBudget() {
  SamplingConfig config;
  config.reasoning_budget = Budget(4);
  SamplerState sampler(config);
  sampler.Accept(std::vector<TokenId>{1, 3, 1, 1, 1, 1});
  Expect(!sampler.ForcedToken(),
         "a reasoning end within the budget closes the phase for good");
  SamplerState unlimited(SamplingConfig{});
  unlimited.Accept(std::vector<TokenId>(100, 1));
  Expect(!unlimited.ForcedToken(), "requests without a budget are unchanged");
  config.reasoning_budget = Budget(0);
  Expect(SamplerState(config).ForcedToken() == TokenId{3},
         "a zero budget closes reasoning immediately");
}

// Qwen as llama.cpp runs it: the prompt ends with "<think>\n", the forced end
// is "\n</think>" (tokens 4, 3) and token 7 (a tool call) also ends reasoning.
void TestReasoningBudgetMatchesLlamaCpp() {
  SamplingConfig config;
  for (const std::size_t tokens : {1U, 2U, 4U}) {
    config.reasoning_budget = Budget(tokens, {4, 3}, {4});
    SamplerState sampler(config);
    std::vector<TokenId> emitted;
    for (int step = 0; step < 8; ++step) {
      const auto token = sampler.Sample(Logits(1));
      sampler.Accept(token);
      emitted.push_back(token);
    }
    std::vector<TokenId> expected(tokens - 1, 1);
    expected.insert(expected.end(), {4, 3});
    expected.resize(8, 1);
    Expect(emitted == expected,
           "the prompt newline counts and the newline end is forced");
  }
  config.reasoning_budget = Budget(0, {4, 3}, {4});
  SamplerState zero(config);
  Expect(zero.ForcedToken() == TokenId{3},
         "a zero budget's prompt newline stands for the forced newline");
  zero.Accept(3);
  Expect(!zero.ForcedToken(), "the forced end completes reasoning");

  auto ending = std::make_shared<ReasoningBudget>(*Budget(2, {4, 3}));
  ending->ends.push_back(7);
  config.reasoning_budget = ending;
  SamplerState tool(config);
  tool.Accept(7);
  tool.Accept(std::vector<TokenId>(4, 1));
  Expect(!tool.ForcedToken(), "a tool call ends reasoning before the budget");

  // Token 5 ends inside a UTF-8 character and token 6 completes it.
  auto utf8 = std::make_shared<ReasoningBudget>(*Budget(2));
  utf8->partial_utf8 = std::make_shared<std::vector<bool>>(
      std::vector<bool>{false, false, false, false, false, true, false, false});
  config.reasoning_budget = utf8;
  SamplerState split(config);
  split.Accept(std::vector<TokenId>{1, 5, 5});
  Expect(!split.ForcedToken(), "a spent budget waits for a whole character");
  split.Accept(6);
  Expect(split.ForcedToken() == TokenId{3},
         "the end is forced after the character completes");
}

void TestReasoningBudgetKeepsLiteralAnswer() {
  SamplingConfig config;
  config.reasoning_budget = Budget(2, {4, 3}, {4});
  for (const bool natural_end : {false, true}) {
    SamplerState sampler(config);
    if (natural_end) {
      sampler.Accept(3);
    } else {
      sampler.Accept(std::vector<TokenId>{1, 4, 3});
    }
    // Token 0 stands for a literal <think> inside answer/tool data. Gufo's
    // framing recognizes reasoning markup only in the initial phase.
    sampler.Accept(std::vector<TokenId>{6, 0, 1, 2, 1, 2, 1, 2});
    Expect(!sampler.ForcedToken() && sampler.Sample(Logits(6)) == 6,
           "literal answer tokens must not restart a completed budget");
  }

  const std::vector<TokenId> thinking{1};
  Expect(gufo::sampling::BuildDistribution(Logits(1), config, {}, thinking)
                 .best_token() == 4,
         "distribution construction enforces an exhausted reasoning budget");
  const std::vector<TokenId> answer{1, 4, 3, 6, 0, 1, 2};
  Expect(gufo::sampling::BuildDistribution(Logits(1), config, {}, answer)
                 .best_token() == 1,
         "distribution construction preserves literal answer content");
}

// With a tool or format grammar, the forced end must be a legal grammar step
// and the answer after it stays constrained.
void TestReasoningBudgetWithGrammar() {
  const std::vector<std::string> pieces{
      "thinking", "</think>", "{\"x\":", "true}", "prose", ""};
  auto binding = std::make_shared<TokenConstraint>();
  binding->grammar = JsonConstraint::WithReasoning(JsonConstraint::Object());
  binding->vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t i) {
        return ConstraintVocabulary::Piece{pieces[i], i == 5};
      });
  SamplingConfig config;
  config.constraint = binding;
  auto budget = std::make_shared<ReasoningBudget>();
  budget->tokens = 1;
  budget->forced = {1};
  budget->ends = {1};
  config.reasoning_budget = budget;
  SamplerState sampler(config);
  sampler.Accept(sampler.Sample(std::vector<float>{5, 0, 0, 0, 0, 0}));
  const auto forced = sampler.Sample(std::vector<float>{5, 0, 0, 0, 4, 0});
  Expect(forced == 1, "the grammar request also closes at its budget");
  sampler.Accept(forced);
  Expect(!sampler.ForcedToken() &&
             sampler.Sample(std::vector<float>{5, 0, 1, 0, 4, 0}) == 2,
         "after the forced end the answer still follows the grammar");
}

void TestCompactTargetContract() {
  using gufo::sampling::LogitCandidate;
  const std::vector<float> logits{0.0F, 0.75F, -1.0F, 0.75F, 3.0F, 0.1F};
  // Deliberately unordered IDs. A frequency penalty can promote a token
  // whose raw logit is far below the original top-k.
  const std::vector<LogitCandidate> candidates{
      {4, 3.0F}, {2, -1.0F}, {3, 0.75F}, {1, 0.75F}};
  SamplingConfig config{.temperature = 0.7F,
                        .top_k = 2,
                        .top_p = 0.8F,
                        .frequency_penalty = -5.0F};
  SamplerState state(config);
  state.Accept(2);
  const auto expected = state.Distribution(logits);
  const auto actual = state.DistributionFromCandidates(candidates);
  Expect(expected.entries().size() == actual.entries().size(),
         "compact target retains full top-k support");
  for (std::size_t i = 0; i < expected.entries().size(); ++i)
    Expect(expected.entries()[i].token == actual.entries()[i].token &&
               std::bit_cast<std::uint64_t>(expected.entries()[i].value) ==
                   std::bit_cast<std::uint64_t>(actual.entries()[i].value),
           "compact target retains exact FP64 probabilities");
  for (unsigned mode = 0; mode < 3; ++mode) {
    auto unsupported = config;
    if (mode == 0)
      unsupported.temperature = 0;
    else if (mode == 1)
      unsupported.top_k = 0;
    else
      unsupported.reasoning_budget = Budget(0);
    bool refused = false;
    try {
      (void)SamplerState(unsupported).DistributionFromCandidates(candidates);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    Expect(refused, "compact target refuses unsupported selection semantics");
  }
}

// The head a device would report: the exact sequential host softmax sum and
// every finite token at least as large as the `size`-th best, shuffled.
gufo::sampling::LogitHead HeadOf(const std::vector<float>& logits,
                                 float temperature, std::size_t size,
                                 std::uint64_t* rng) {
  gufo::sampling::LogitHead head;
  std::vector<float> finite;
  for (const float x : logits)
    if (std::isfinite(x))
      finite.push_back(x);
  std::ranges::sort(finite, std::greater<>());
  head.maximum = finite.front();
  head.temperature = temperature;
  for (const float x : logits)
    if (std::isfinite(x))
      head.softmax_sum +=
          std::exp((static_cast<double>(x) - head.maximum) / temperature);
  const float floor = finite[std::min(size, finite.size()) - 1];
  for (std::size_t i = 0; i < logits.size(); ++i)
    if (std::isfinite(logits[i]) && logits[i] >= floor)
      head.candidates.push_back({static_cast<TokenId>(i), logits[i]});
  for (std::size_t i = head.candidates.size(); i > 1; --i)
    std::swap(head.candidates[i - 1],
              head.candidates[gufo::sampling::NextRandom(rng) % i]);
  return head;
}

void TestLogitHead() {
  std::uint64_t rng = 7;
  const auto normal = [&rng] {
    const double u = (gufo::sampling::Uniform(&rng) + 1e-12);
    const double v = gufo::sampling::Uniform(&rng);
    return std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v);
  };
  const std::vector<SamplingConfig> configs{
      {.temperature = 1.0F, .top_p = 0.95F, .seed = 1},
      {.temperature = 0.7F, .top_p = 0.9F, .min_p = 0.05F, .seed = 2},
      {.temperature = 0.6F, .top_p = 0.5F, .min_keep = 3, .seed = 3},
      {.temperature = 1.0F, .top_k = 40, .seed = 4},
      {.temperature = 1.3F,
       .top_k = 40,
       .top_p = 0.9F,
       .min_keep = 50,
       .seed = 5},
      // Read the full row: greedy, penalties, temperature alone.
      {.temperature = 0.0F, .seed = 6},
      {.temperature = 1.0F,
       .top_p = 0.95F,
       .seed = 7,
       .presence_penalty = 0.5F},
      {.temperature = 1.0F, .min_p = 0.1F, .seed = 8},
  };
  for (const auto& config : configs) {
    SamplerState row_sampler(config), head_sampler(config);
    for (int step = 0; step < 300; ++step) {
      // Sharp, moderate and flat rows over 4096 tokens, with ties and
      // non-finite entries.
      const double spread = step % 3 == 0 ? 6.0 : step % 3 == 1 ? 2.0 : 0.3;
      std::vector<float> logits(4096);
      for (auto& x : logits)
        x = static_cast<float>(spread * normal());
      logits[step % 4096] += 8.0F;
      logits[(step * 7) % 4096] = logits[(step * 13 + 1) % 4096];
      logits[(step * 31 + 5) % 4096] = -INFINITY;
      logits[(step * 17 + 9) % 4096] = NAN;
      // Heads of 1, 64 and 4096 tokens: too small ones fall back.
      const std::size_t size = step % 5 == 0 ? 1 : step % 5 < 3 ? 64 : 4096;
      const auto head =
          HeadOf(logits, std::max(config.temperature, 1e-3F), size, &rng);
      if (step % 11 == 0) {
        // A pending draw ignores both row and head.
        row_sampler.DeferSample(3);
        head_sampler.DeferSample(3);
      }
      const auto expected = row_sampler.Sample(logits);
      const auto actual = head_sampler.Sample(logits, head);
      Expect(expected == actual &&
                 row_sampler.rng_state() == head_sampler.rng_state(),
             "a logit head draws the full row's token with the same RNG");
      row_sampler.Accept(expected);
      head_sampler.Accept(actual);
    }
  }
  // A head whose sum claims twice the mass narrows top-p, so the draws
  // differ when the head is read.
  SamplerState row_sampler(configs[0]), head_sampler(configs[0]);
  std::size_t differ = 0;
  for (int step = 0; step < 100; ++step) {
    std::vector<float> logits(4096);
    for (auto& x : logits)
      x = static_cast<float>(2.0 * normal());
    auto head = HeadOf(logits, 1.0F, 4096, &rng);
    head.softmax_sum *= 0.5;
    differ += row_sampler.Sample(logits) != head_sampler.Sample(logits, head);
  }
  Expect(differ > 10, "a covering head is read instead of the row");
  // A head summed at another temperature is ignored.
  SamplerState plain(configs[0]), stale(configs[0]);
  for (int step = 0; step < 50; ++step) {
    std::vector<float> logits(4096);
    for (auto& x : logits)
      x = static_cast<float>(2.0 * normal());
    auto head = HeadOf(logits, 0.5F, 4096, &rng);
    Expect(plain.Sample(logits) == stale.Sample(logits, head) &&
               plain.rng_state() == stale.rng_state(),
           "a head at another temperature falls back to the row");
  }
  const std::vector<float> logits{0.0F, 2.0F, 1.0F, 2.0F};
  gufo::sampling::LogitHead wrong{.maximum = 2.0F,
                                  .temperature = 1.0F,
                                  .softmax_sum = 1.0,
                                  .candidates = {{2, 1.0F}}};
  bool refused = false;
  try {
    SamplerState sampler({.temperature = 1.0F, .top_k = 1});
    (void)sampler.Sample(logits, wrong);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  Expect(refused, "a head whose maximum is not its best logit is refused");
}

}  // namespace

void TestRawLogprobs() {
  using gufo::sampling::CollectLogprobs;
  const std::vector<float> logits{1000, 999, 998, -1000};
  const auto result = CollectLogprobs(logits, 3, 2);
  const double denominator = std::log(1.0 + std::exp(-1.0) + std::exp(-2.0));
  Expect(std::abs(result->selected.logprob - (-2000.0 - denominator)) < 1e-12,
         "stable log-softmax retains very unlikely selected tokens");
  Expect(result->top.size() == 2 && result->top[0].token == 0 &&
             result->top[1].token == 1,
         "alternatives come from the complete raw vocabulary");
  Expect(!CollectLogprobs({}, 999, std::nullopt),
         "disabled collection does no work");
  Expect(CollectLogprobs(logits, 0, 0)->top.empty(),
         "zero requests selected token only");
  const auto tied = CollectLogprobs(std::vector<float>{0, 0, 0}, 2, 20);
  Expect(tied->top.size() == 3 && tied->top[0].token == 0 &&
             std::abs(tied->selected.logprob + std::log(3.0)) < 1e-12,
         "ties have deterministic token order and exact normalization");
  const auto twenty = CollectLogprobs(std::vector<float>(32, 0), 31, 20);
  Expect(twenty->top.size() == 20 && twenty->top.back().token == 19 &&
             twenty->selected.token == 31,
         "twenty alternatives retain an independently selected token");
  const std::vector<std::string> pieces{"prose", "{", "}", ""};
  auto binding = std::make_shared<TokenConstraint>();
  binding->grammar = JsonConstraint::Object();
  binding->vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t i) {
        return ConstraintVocabulary::Piece{pieces[i], i == 3};
      });
  SamplingConfig constrained;
  constrained.constraint = binding;
  constrained.logprobs = 2;
  SamplerState grammar(constrained);
  const auto selected = grammar.Sample(logits);
  const auto raw = CollectLogprobs(logits, selected, constrained.logprobs);
  Expect(selected == 1 && raw->top[0].token == 0 &&
             std::abs(raw->selected.logprob + 1 + denominator) < 1e-12,
         "grammar masks change selection but not raw probabilities");
  for (const float temperature : {0.F, 0.7F}) {
    SamplingConfig config;
    config.temperature = temperature;
    config.top_k = 2;
    config.presence_penalty = 0.5;
    config.seed = 42;
    SamplerState plain(config);
    config.logprobs = 2;
    SamplerState observed(config);
    for (int i = 0; i < 16; ++i) {
      const auto a = plain.Sample(logits), b = observed.Sample(logits);
      const auto probabilities = CollectLogprobs(logits, b, config.logprobs);
      Expect(a == b && plain.rng_state() == observed.rng_state(),
             "observation preserves selection and RNG");
      Expect(std::abs(probabilities->top[0].logprob + denominator) < 1e-12,
             "raw logprobs ignore temperature, truncation and penalties");
      plain.Accept(a);
      observed.Accept(b);
    }
  }
  for (const auto invalid :
       {std::vector<float>{INFINITY}, std::vector<float>{NAN},
        std::vector<float>{-INFINITY}}) {
    bool failed = false;
    try {
      (void)CollectLogprobs(invalid, 0, 0);
    } catch (const std::runtime_error&) {
      failed = true;
    }
    Expect(failed, "invalid raw distributions fail explicitly");
  }
}

int main() {
  TestRawLogprobs();
  TestReasoningBudgetForcesTheEnd();
  TestEarlyEndAndNoBudget();
  TestReasoningBudgetMatchesLlamaCpp();
  TestReasoningBudgetKeepsLiteralAnswer();
  TestReasoningBudgetWithGrammar();
  TestCompactTargetContract();
  TestLogitHead();
  std::cout << "sampling_test passed\n";
  return 0;
}
