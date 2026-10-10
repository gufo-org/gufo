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

void TestReasoningBudgetForcesTheEnd() {
  SamplingConfig config;
  config.reasoning_budget = 2;
  config.reasoning_end = 3;
  SamplerState sampler(config);
  Expect(!config.can_use_unmodified_argmax(),
         "a budget keeps requests off the unmodified argmax path");
  Expect(!sampler.ForcedToken() && sampler.CanSelectArgmax(1),
         "the budget does not restrict reasoning before it is spent");
  sampler.Accept(sampler.Sample(Logits(1)));
  sampler.Accept(sampler.Sample(Logits(2)));
  Expect(sampler.ForcedToken() == TokenId{3},
         "two reasoning tokens exhaust a budget of two");
  Expect(!sampler.CanSelectArgmax(1) && sampler.CanSelectArgmax(3),
         "only the reasoning end is a selectable argmax");
  Expect(sampler.Sample(Logits(1)) == 3 &&
             sampler.Distribution(Logits(1)).best_token() == 3,
         "sampling and distributions return the forced end");
  const std::vector<TokenId> ids{5, 3, 1};
  const std::vector<float> compact{1.0F, 0.0F, 2.0F};
  Expect(sampler.Distribution(compact, ids).best_token() == 1,
         "compact distributions index the forced end among their IDs");
  auto copy = sampler;
  copy.Accept(3);
  Expect(!copy.ForcedToken() && sampler.ForcedToken(),
         "copies advance independently, as speculative verification needs");
  Expect(copy.Sample(Logits(6)) == 6, "the answer is sampled normally");
  copy.ResetHistory({});
  Expect(!copy.ForcedToken(), "a restart resets the reasoning budget");
}

void TestEarlyEndAndNoBudget() {
  SamplingConfig config;
  config.reasoning_budget = 4;
  config.reasoning_end = 3;
  SamplerState sampler(config);
  sampler.Accept(std::vector<TokenId>{1, 3, 1, 1, 1, 1});
  Expect(!sampler.ForcedToken(),
         "a reasoning end within the budget closes the phase for good");
  SamplerState unlimited(SamplingConfig{});
  unlimited.Accept(std::vector<TokenId>(100, 1));
  Expect(!unlimited.ForcedToken(), "requests without a budget are unchanged");
  config.reasoning_budget = 0;
  Expect(SamplerState(config).ForcedToken() == TokenId{3},
         "a zero budget closes reasoning immediately");
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
  config.reasoning_budget = 1;
  config.reasoning_end = 1;
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
  TestReasoningBudgetWithGrammar();
  std::cout << "sampling_test passed\n";
  return 0;
}
