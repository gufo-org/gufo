#include "src/core/sampling.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
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

}  // namespace

int main() {
  TestReasoningBudgetForcesTheEnd();
  TestEarlyEndAndNoBudget();
  TestReasoningBudgetMatchesLlamaCpp();
  TestReasoningBudgetKeepsLiteralAnswer();
  TestReasoningBudgetWithGrammar();
  std::cout << "sampling_test passed\n";
  return 0;
}
