// Layered prefill on real weights: a prompt's passes split into layer groups,
// with another session decoding between them, must leave every token and
// log-probability of both unchanged (gufo::test::CheckLayeredPrefill).

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/runner.hpp"
#include "tests/support/layered_prefill_conformance.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

constexpr std::uint32_t kContext = 16384;

std::vector<gufo::ext::TextRunnerToken> Tokens(const qfn::Model& model,
                                               const std::string& text) {
  const auto ids = model.Tokenize(text);
  return {ids.begin(), ids.end()};
}

// Varied text, so MTP drafts are rejected as well as accepted.
std::string Ledger(std::size_t entries) {
  std::string text = "Inventory ledger.\n";
  for (std::size_t i = 0; i < entries; ++i)
    text += "Entry " + std::to_string(i) + " holds " +
            std::to_string(i * 7919 % 1009) + " units of part " +
            std::to_string(i * 31 % 97) + ".\n";
  return text + "Which part does entry 42 hold? Answer briefly.";
}

bool Check(const char* name,
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
  std::string first, mtp;
  if (argc == 5 && std::string_view(argv[1]) == "--model" &&
      std::string_view(argv[3]) == "--mtp-model") {
    first = argv[2];
    mtp = argv[4];
  } else if (argc == 1) {
    // ctest names the model through the environment and skips without it.
    const char* model = std::getenv("QWEN38_FLASH_NEXT_MODEL");
    const char* head = std::getenv("QWEN38_FLASH_NEXT_MTP_MODEL");
    if (!model || !*model || !head || !*head) {
      std::cerr << "QWEN38_FLASH_NEXT_MODEL and QWEN38_FLASH_NEXT_MTP_MODEL "
                   "are not set; skipping\n";
      return 77;
    }
    first = model;
    mtp = head;
  } else {
    std::cerr << "Usage: layered_prefill_test [--model FIRST.gguf "
                 "--mtp-model MTP.gguf]\n";
    return 2;
  }
  try {
    std::string error;
    auto model = qfn::Model::Load(
        first,
        {.max_context = kContext, .mtp_model_path = mtp, .max_draft_tokens = 7},
        &error);
    if (!model)
      throw std::runtime_error(error);

    gufo::test::LayeredPrefillCase test;
    // Three passes: two full 4,096-row ones and a tail.
    test.prompt = Tokens(*model, Ledger(600));
    if (test.prompt.size() <= 2 * model->PrefillCapacity())
      throw std::runtime_error("layered prompt is shorter than two passes");
    test.decoder_prompt = Tokens(
        *model,
        "Write a C function that reverses a singly linked list in place, "
        "then explain its complexity.");
    // Enough tokens to decode in every yield of every pass.
    test.decoder_output_tokens = 64;
    test.prompt_output_tokens = 8;

    gufo::ext::TextSpeculativeConfig speculative;
    speculative.backend = gufo::ext::TextSpeculativeBackend::kMtp;
    const auto ar = qfn::CreateQwenFlashNextTextRunner(model, kContext);
    const auto with_mtp =
        qfn::CreateQwenFlashNextTextRunner(model, kContext, speculative);
    bool passed = true;
    // Three decoders batched in every round, as the profile lets the host
    // run them: batched AR and batched MTP chains inside the hook.
    {
      auto batched = test;
      batched.layer_groups = 8;
      batched.more_decoder_prompts = {
          Tokens(*model,
                 "List the planets of the solar system with one fact each."),
          Tokens(*model,
                 "Translate 'the quick brown fox jumps over the lazy "
                 "dog' into French, German and Italian.")};
      passed = Check("AR,batched_3", ar, batched) && passed;
      passed = Check("MTP,batched_3", with_mtp, batched) && passed;
    }
    // With the cache, passes also capture prompt checkpoints in place.
    for (const bool cache_prompt : {false, true}) {
      for (const std::size_t groups : {8u, 3u}) {
        test.layer_groups = groups;
        test.cache_prompt = cache_prompt;
        const auto suffix = std::to_string(groups) + "_groups" +
                            (cache_prompt ? "_cached" : "");
        passed = Check(("AR," + suffix).c_str(), ar, test) && passed;
        passed = Check(("MTP," + suffix).c_str(), with_mtp, test) && passed;
      }
    }
    return passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "layered_prefill_test: " << failure.what() << '\n';
    return 1;
  }
}
