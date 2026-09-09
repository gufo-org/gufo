#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <span>
#include <string>
#include <string_view>

#include "src/cli/bench/bench.hpp"
#include "src/cli/prompt/prompt.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void CheckBackend(std::initializer_list<const char*> arguments,
                  std::string_view backend) {
  const std::span<const char* const> args(arguments.begin(), arguments.size());
  std::string error;
  const auto prompt = gufo::cli::ParsePromptOptions(args, &error);
  Expect(prompt.has_value(), error);
  const auto bench = gufo::cli::ParseBenchOptions(args, &error);
  Expect(bench.has_value(), error);
  Expect(prompt->speculative_backend == backend &&
             bench->speculative_backend == backend,
         "prompt and benchmark select the same DSpark mode");
  Expect(prompt->dspark_model_path == "support.gguf" &&
             bench->dspark_model_path == "support.gguf",
         "both commands retain the support artifact");
}

}  // namespace

int main() {
  CheckBackend({"--dspark-model", "support.gguf"}, "dspark");
  CheckBackend({"--speculative", "dspark", "--dspark-model", "support.gguf"},
               "dspark");
  CheckBackend({"--speculative", "off", "--dspark-model", "support.gguf"}, "");
  CheckBackend({"--dspark-model", "support.gguf", "--speculative", "off"}, "");

  const char* removed_policy[] = {"--draft-policy", "fixed"};
  Expect(!gufo::cli::ParsePromptOptions(removed_policy),
         "prompt rejects the removed policy selector");
  Expect(!gufo::cli::ParseBenchOptions(removed_policy),
         "benchmark rejects the removed policy selector");
  const char* args[] = {"--dspark-model", "support.gguf", "--draft-tokens",
                        "3"};
  const auto prompt = gufo::cli::ParsePromptOptions(args);
  const auto bench = gufo::cli::ParseBenchOptions(args);
  Expect(
      prompt && bench && prompt->draft_tokens == 3 && bench->draft_tokens == 3,
      "both commands retain the requested draft budget");
  std::cout << "DS4 CLI option contract passed\n";
}
