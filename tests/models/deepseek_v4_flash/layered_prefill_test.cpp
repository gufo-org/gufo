// Layered prefill on real weights: a prompt's passes split into layer groups,
// with another session decoding between them, must leave every token and
// log-probability of both unchanged (gufo::test::CheckLayeredPrefill).
//
// ds4_layered_prefill_test [MODEL.gguf [DSPARK.gguf]]
// Without arguments, GUFO_DEEPSEEK_V4_FLASH_MODEL and
// GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL name the weights; the test is skipped
// without a target and runs autoregressive decoding only without DSpark.
// Each mode loads the model on its own: two copies do not fit beside each
// other.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "src/models/deepseek_v4_flash/runner.hpp"
#include "tests/support/layered_prefill_conformance.hpp"

namespace {

constexpr std::uint32_t kContext = 16384;

// Varied text, so DSpark drafts are rejected as well as accepted.
std::string Ledger(std::size_t entries) {
  std::string text = "Inventory ledger.\n";
  for (std::size_t i = 0; i < entries; ++i)
    text += "Entry " + std::to_string(i) + " holds " +
            std::to_string(i * 7919 % 1009) + " units of part " +
            std::to_string(i * 31 % 97) + ".\n";
  return text + "Which part does entry 42 hold? Answer briefly.";
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

std::shared_ptr<gufo::ext::TextModelRunner> Create(const std::string& model,
                                                   const std::string& dspark) {
  gufo::ext::TextRunnerOptions options;
  options.model_path = model;
  options.max_context = kContext;
  options.session_count = 4;
  if (!dspark.empty()) {
    options.speculative.backend = gufo::ext::TextSpeculativeBackend::kDSpark;
    options.speculative.draft_model_path = dspark;
  }
  std::string error;
  auto runner = gufo::models::deepseek_v4_flash::CreateDeepSeekTextRunner(
      options, &error);
  if (!runner)
    throw std::runtime_error(error);
  return runner;
}

int main(int argc, char** argv) {
  std::string model, dspark;
  if (argc > 3) {
    std::cerr << "Usage: ds4_layered_prefill_test [MODEL.gguf [DSPARK.gguf]]\n";
    return 2;
  }
  if (argc > 1) {
    model = argv[1];
    dspark = argc > 2 ? argv[2] : "";
  } else {
    const char* target = std::getenv("GUFO_DEEPSEEK_V4_FLASH_MODEL");
    const char* support = std::getenv("GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL");
    if (!target || !*target) {
      std::cerr << "GUFO_DEEPSEEK_V4_FLASH_MODEL is not set; skipping\n";
      return 77;
    }
    model = target;
    dspark = support ? support : "";
  }
  try {
    bool passed = true;
    for (const std::string mode : {"AR", "DSpark"}) {
      if (mode == "DSpark" && dspark.empty())
        continue;
      const auto runner = Create(model, mode == "DSpark" ? dspark : "");

      gufo::test::LayeredPrefillCase test;
      // Three passes: two full 4,096-row ones and a tail.
      test.prompt = runner->Tokenize(Ledger(800));
      if (test.prompt.size() <= 2 * 4096)
        throw std::runtime_error("layered prompt has " +
                                 std::to_string(test.prompt.size()) +
                                 " tokens, fewer than two passes");
      test.decoder_prompt = runner->Tokenize(
          "Write a C function that reverses a singly linked list in place, "
          "then explain its complexity.");
      // Enough tokens to decode in every yield of every pass.
      test.decoder_output_tokens = 64;
      test.prompt_output_tokens = 8;

      // Three decoders batched in every round, as the profile lets the host
      // run them: batched AR decode and batched DSpark verification.
      auto batched = test;
      batched.layer_groups = 16;
      batched.more_decoder_prompts = {
          runner->Tokenize(
              "List the planets of the solar system with one fact each."),
          runner->Tokenize("Translate 'the quick brown fox jumps over the "
                           "lazy dog' into French, German and Italian.")};
      passed = Check(mode + ",batched_3", runner, batched) && passed;
      // With the cache, passes also plan prompt checkpoints, as serving does.
      for (const bool cache_prompt : {false, true}) {
        for (const std::size_t groups : {16u, 3u}) {
          test.layer_groups = groups;
          test.cache_prompt = cache_prompt;
          passed = Check(mode + "," + std::to_string(groups) + "_groups" +
                             (cache_prompt ? "_cached" : ""),
                         runner, test) &&
                   passed;
        }
      }
    }
    return passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "ds4_layered_prefill_test: " << failure.what() << '\n';
    return 1;
  }
}
