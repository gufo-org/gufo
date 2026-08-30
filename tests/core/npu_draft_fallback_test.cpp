#include <cassert>
#include <iostream>
#include <vector>

#include "src/core/heterogeneous/npu_drafter.hpp"

void TestNpuDraftFallback() {
  gufo::heterogeneous::NpuDrafterConfig config;
  config.mode = gufo::heterogeneous::NpuDraftMode::kMTP;
  config.mtp_model_path.clear();
  config.dflash_model_path.clear();
  config.enable_xrt = false;
  config.max_draft_tokens = 4;
  config.vocab_size = 152064;

  gufo::heterogeneous::NpuDraftBackend drafter(config);
  assert(drafter.Name() == "NpuXdna2MtpDraftBackend");
  assert(!drafter.IsNpuActive());
  assert(!drafter.HasModel());
  assert(!drafter.GetStatusMessage().empty());

  std::cout << "NPU draft fallback status: " << drafter.GetStatusMessage()
            << "\n";

  const std::vector<gufo::tokenization::TokenId> prompt = {10, 20, 30, 40,
                                                           50, 60, 10, 20};
  const auto proposal = drafter.Propose(prompt, 8, 4);
  assert(proposal.tokens ==
         std::vector<gufo::tokenization::TokenId>({30, 40, 50, 60}));
  assert(proposal.start_pos == 8);

  const std::vector<gufo::tokenization::TokenId> accepted = {
      proposal.tokens[0], proposal.tokens[1]};
  drafter.AcceptFeedback(accepted, 999);

  const std::vector<gufo::tokenization::TokenId> no_match = {100, 101, 102};
  const auto empty_proposal = drafter.Propose(no_match, 3, 4);
  assert(empty_proposal.tokens.empty());

  drafter.Reset();
  std::cout << "TestNpuDrafterInitialization passed.\n";
}

int main() {
  TestNpuDraftFallback();
  std::cout << "NPU draft fallback test passed.\n";
  return 0;
}
