// Layered prefill on real weights: a prompt's passes split into layer groups,
// with other sessions decoding between them, must leave every token and
// log-probability of all of them unchanged (gufo::test::CheckLayeredPrefill).
//
// qwen27b_layered_prefill_test [MODEL.gguf [DFLASH.gguf]]
// Without arguments, GUFO_QWEN27B_MODEL and GUFO_QWEN27B_DFLASH_MODEL name
// the weights; the test is skipped without a target model and runs AR only
// without a draft.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/runner.hpp"
#include "tests/support/layered_prefill_conformance.hpp"

namespace {

constexpr std::uint32_t kContext = 16384;

// Varied text, so DFlash drafts are rejected as well as accepted.
std::string Ledger(std::size_t entries) {
  std::string text = "Inventory ledger.\n";
  for (std::size_t i = 0; i < entries; ++i)
    text += "Entry " + std::to_string(i) + " holds " +
            std::to_string(i * 7919 % 1009) + " units of part " +
            std::to_string(i * 31 % 97) + ".\n";
  return text + "Which part does entry 42 hold? Answer briefly.";
}

std::shared_ptr<gufo::ext::TextModelRunner> Create(const std::string& model,
                                                   const std::string& draft) {
  gufo::ext::TextRunnerOptions options;
  options.model_path = model;
  options.max_context = kContext;
  options.session_count = 4;
  if (!draft.empty()) {
    options.speculative.backend = gufo::ext::TextSpeculativeBackend::kDFlash;
    options.speculative.draft_model_path = draft;
  }
  std::string error;
  auto runner = gufo::models::qwen::CreateQwenTextRunner(options, &error);
  if (!runner)
    throw std::runtime_error(error);
  // Sessions allocate full-context K/V, including DFlash draft rows, when they
  // are created, and the pool admits them through per_request_state_bytes.
  // Charging prompt length again refuses cached follow-ups under low host
  // headroom without any new allocation (#1157).
  const auto claim = runner->ResourceClaim();
  if (claim.per_request_state_bytes.value_or(0) == 0 ||
      claim.prompt_state_bytes_per_token ||
      claim.prompt_admission_budget_bytes ||
      claim.prompt_admission_available_bytes)
    throw std::runtime_error(
        "Qwen must claim its session state and leave prompt state unclaimed");
  return runner;
}

bool Check(const std::string& name,
           const std::shared_ptr<gufo::ext::TextModelRunner>& runner,
           const gufo::test::LayeredPrefillCase& test) {
  const auto failures = gufo::test::CheckLayeredPrefill(runner, test);
  for (const auto& failure : failures)
    std::cerr << name << ": " << failure << '\n';
  std::cout << name << ',' << (failures.empty() ? "PASS" : "FAIL") << '\n';
  return failures.empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, draft;
  if (argc > 3) {
    std::cerr << "Usage: qwen27b_layered_prefill_test [MODEL.gguf "
                 "[DFLASH.gguf]]\n";
    return 2;
  }
  if (argc > 1) {
    model = argv[1];
    draft = argc > 2 ? argv[2] : "";
  } else {
    const char* target = std::getenv("GUFO_QWEN27B_MODEL");
    const char* head = std::getenv("GUFO_QWEN27B_DFLASH_MODEL");
    if (!target || !*target) {
      std::cerr << "GUFO_QWEN27B_MODEL is not set; skipping\n";
      return 77;
    }
    model = target;
    draft = head ? head : "";
  }
  try {
    struct Mode {
      std::string name;
      std::shared_ptr<gufo::ext::TextModelRunner> runner;
    };
    std::vector<Mode> modes{{"AR", Create(model, {})}};
    if (!draft.empty())
      modes.push_back({"DFlash2", Create(model, draft)});
    const auto& tokens = *modes.front().runner;

    gufo::test::LayeredPrefillCase test;
    // Three passes: two full 4,096-row ones and a tail.
    test.prompt = tokens.Tokenize(Ledger(600));
    if (test.prompt.size() <= 2 * 4096)
      throw std::runtime_error("layered prompt is shorter than two passes");
    test.decoder_prompt = tokens.Tokenize(
        "Write a C function that reverses a singly linked list in place, "
        "then explain its complexity.");
    // Enough tokens to decode in every yield of every pass.
    test.decoder_output_tokens = 64;
    test.prompt_output_tokens = 8;

    bool passed = true;
    for (const auto& mode : modes) {
      // Three decoders batched in every round, as the profile lets the host
      // run them.
      auto batched = test;
      batched.layer_groups = 16;
      batched.more_decoder_prompts = {
          tokens.Tokenize(
              "List the planets of the solar system with one fact each."),
          tokens.Tokenize("Translate 'the quick brown fox jumps over the lazy "
                          "dog' into French, German and Italian.")};
      passed = Check(mode.name + ",batched_3", mode.runner, batched) && passed;
      // With the cache, passes also plan prompt checkpoints, as serving does.
      for (const bool cache_prompt : {false, true}) {
        for (const std::size_t groups : {16u, 3u}) {
          test.layer_groups = groups;
          test.cache_prompt = cache_prompt;
          passed = Check(mode.name + "," + std::to_string(groups) + "_groups" +
                             (cache_prompt ? "_cached" : ""),
                         mode.runner, test) &&
                   passed;
        }
      }
    }
    return passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen27b_layered_prefill_test: " << failure.what() << '\n';
    return 1;
  }
}
