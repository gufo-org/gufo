#include "src/cli/bench/bench.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::abort();
  }
}

void TestDefaultOptions() {
  const std::array<const char*, 0> args{};
  const auto options = gufo::cli::ParseBenchOptions(args);
  Expect(options.has_value(), "default options parse");
  Expect(options->n_prompts == std::vector<std::size_t>{2048},
         "default prefill is 2048 tokens");
  Expect(options->n_depths == std::vector<std::size_t>{0},
         "default depth is zero");
  Expect(options->repetitions == 1, "default is one repetition");
  Expect(options->draft_tokens == 7, "default draft ceiling is seven");
  Expect(options->min_draft_tokens == 1, "default minimum draft is one");
}

void TestDepthOptions() {
  const std::array<const char*, 8> args = {
      "--n-prompt",    "2048",      "--n-gen",
      "128",           "--n-depth", "4096,8192,12288,16384",
      "--repetitions", "1"};
  const auto options = gufo::cli::ParseBenchOptions(args);
  Expect(options.has_value(), "depth options parse");
  Expect(options->n_prompts == std::vector<std::size_t>{2048},
         "prompt length parsed");
  Expect(options->n_gens == std::vector<std::size_t>{128},
         "generation length parsed");
  Expect(
      options->n_depths == std::vector<std::size_t>{4096, 8192, 12288, 16384},
      "depth list parsed");
  Expect(options->repetitions == 1, "repetition count parsed");
}

void TestHybridMtpOptions() {
  const std::array<const char*, 10> args = {"--speculative",
                                            "mtp-npu",
                                            "--mtp-model",
                                            "mtp.gguf",
                                            "--spec-draft-n-max",
                                            "2",
                                            "--spec-draft-n-min",
                                            "2",
                                            "--n-gen",
                                            "128"};
  const auto options = gufo::cli::ParseBenchOptions(args);
  Expect(options.has_value(), "hybrid MTP options parse");
  Expect(options->speculative_backend == "mtp-npu", "hybrid MTP mode parsed");
  Expect(options->mtp_model_path == "mtp.gguf", "MTP model path parsed");
  Expect(options->draft_tokens == 2, "draft token count parsed");
  Expect(options->min_draft_tokens == 2, "minimum draft count parsed");
}

void TestInvalidDepth() {
  std::string error;
  const std::array<const char*, 2> args = {"--n-depth", "invalid"};
  Expect(!gufo::cli::ParseBenchOptions(args, &error).has_value(),
         "invalid depth rejected");
  Expect(!error.empty(), "invalid depth reports an error");

  const std::array<const char*, 4> range_args = {"--draft-tokens", "3",
                                                 "--min-draft-tokens", "4"};
  Expect(!gufo::cli::ParseBenchOptions(range_args, &error).has_value(),
         "invalid draft range rejected");

  for (const char* flag : {"--spec-draft-p-min", "--draft-p-min"}) {
    const std::array<const char*, 2> removed = {flag, "0.75"};
    Expect(!gufo::cli::ParseBenchOptions(removed, &error).has_value(),
           "removed confidence policy is rejected");
  }
  const std::array<const char*, 4> fixed_block = {"--speculative", "dflash2",
                                                  "--spec-draft-n-min", "2"};
  Expect(!gufo::cli::ParseBenchOptions(fixed_block, &error).has_value() &&
             error.find("min-draft-tokens") != std::string::npos,
         "DFlash rejects an unused adaptive draft floor");
  for (const char* policy : {"fixed", "adaptive", "unknown"}) {
    const std::array<const char*, 4> args = {"--speculative", "dflash2",
                                             "--draft-policy", policy};
    const auto parsed = gufo::cli::ParseBenchOptions(args, &error);
    Expect(parsed.has_value() == (std::string_view(policy) != "unknown"),
           "DFlash benchmark validates its controller");
    if (parsed)
      Expect(parsed->draft_policy == policy,
             "DFlash benchmark retains the requested controller");
  }
  const std::array<const char*, 2> policy_without_backend = {"--draft-policy",
                                                             "adaptive"};
  Expect(!gufo::cli::ParseBenchOptions(policy_without_backend, &error),
         "a DFlash controller requires its backend");
}

void TestInvalidWorkload() {
  for (const char* option : {"-p", "-n", "-d"}) {
    for (const char* count : {"32x", "32,bad", "32,", ",32", "-1"}) {
      const char* args[] = {option, count};
      Expect(!gufo::cli::ParseBenchOptions(args),
             "malformed workload is rejected");
    }
  }
  const char* no_runs[] = {"-r", "0"};
  Expect(!gufo::cli::ParseBenchOptions(no_runs),
         "zero repetitions are rejected");
  const char* empty[] = {"-p", "0", "-n", "0"};
  Expect(!gufo::cli::ParseBenchOptions(empty), "empty benchmark is rejected");
  const char* decode[] = {"-p", "0", "-n", "64"};
  const auto options = gufo::cli::ParseBenchOptions(decode);
  Expect(options && options->n_prompts.empty() && options->n_gens.size() == 1,
         "zero still disables an individual workload");
}

}  // namespace

int main() {
  TestDefaultOptions();
  TestDepthOptions();
  TestHybridMtpOptions();
  TestInvalidDepth();
  TestInvalidWorkload();
  std::cout << "All benchmark CLI tests passed.\n";
  return 0;
}
