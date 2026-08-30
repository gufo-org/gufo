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
  Expect(options->n_depths == std::vector<std::size_t>{0},
         "default depth is zero");
  Expect(options->repetitions == 1, "default is one repetition");
  Expect(options->draft_tokens == 7, "default draft ceiling is seven");
  Expect(options->draft_policy == "auto", "default draft policy is auto");
  Expect(options->min_draft_tokens == 1, "default minimum draft is one");
  Expect(options->draft_p_min == 0.0F,
         "default draft confidence threshold is disabled");
  Expect(options->qwen_backend == "hip", "default Qwen backend is HIP");
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
  const std::array<const char*, 14> args = {"--speculative",
                                            "mtp-npu",
                                            "--mtp-model",
                                            "mtp.gguf",
                                            "--spec-draft-n-max",
                                            "2",
                                            "--draft-policy",
                                            "fixed",
                                            "--spec-draft-n-min",
                                            "2",
                                            "--spec-draft-p-min",
                                            "0.75",
                                            "--n-gen",
                                            "128"};
  const auto options = gufo::cli::ParseBenchOptions(args);
  Expect(options.has_value(), "hybrid MTP options parse");
  Expect(options->speculative_backend == "mtp-npu", "hybrid MTP mode parsed");
  Expect(options->mtp_model_path == "mtp.gguf", "MTP model path parsed");
  Expect(options->draft_tokens == 2, "draft token count parsed");
  Expect(options->draft_policy == "fixed", "fixed draft policy parsed");
  Expect(options->min_draft_tokens == 2, "minimum draft count parsed");
  Expect(options->draft_p_min > 0.74F && options->draft_p_min < 0.76F,
         "draft confidence threshold parsed");
}

void TestNativeHrxBackendOption() {
  const std::array<const char*, 2> args = {"--qwen-backend", "hrx-native"};
  const auto options = gufo::cli::ParseBenchOptions(args);
  Expect(options.has_value(), "native HRX backend option parses");
  Expect(options->qwen_backend == "hrx-native",
         "native HRX backend option is retained");
}

void TestNativeHrxValidationOption() {
  const std::array<const char*, 4> args = {"--qwen-backend", "hrx-native",
                                           "--validate-hrx", "3"};
  const auto options = gufo::cli::ParseBenchOptions(args);
  Expect(options.has_value(), "native HRX validation option parses");
  Expect(options->validate_hrx_tokens == 3,
         "native HRX validation token count is retained");

  std::string error;
  const std::array<const char*, 2> missing_backend = {"--validate-hrx", "3"};
  Expect(!gufo::cli::ParseBenchOptions(missing_backend, &error).has_value(),
         "native HRX validation rejects the HIP-only package route");

  const std::array<const char*, 4> zero_tokens = {
      "--qwen-backend", "hrx-native", "--validate-hrx", "0"};
  Expect(!gufo::cli::ParseBenchOptions(zero_tokens, &error).has_value(),
         "native HRX validation rejects zero decode tokens");
}

void TestInvalidDepth() {
  std::string error;
  const std::array<const char*, 2> args = {"--n-depth", "invalid"};
  Expect(!gufo::cli::ParseBenchOptions(args, &error).has_value(),
         "invalid depth rejected");
  Expect(!error.empty(), "invalid depth reports an error");

  const std::array<const char*, 2> policy_args = {"--draft-policy", "unknown"};
  Expect(!gufo::cli::ParseBenchOptions(policy_args, &error).has_value(),
         "invalid draft policy rejected");

  const std::array<const char*, 4> range_args = {"--draft-tokens", "3",
                                                 "--min-draft-tokens", "4"};
  Expect(!gufo::cli::ParseBenchOptions(range_args, &error).has_value(),
         "invalid draft range rejected");

  const std::array<const char*, 2> probability_args = {"--spec-draft-p-min",
                                                       "1.1"};
  Expect(!gufo::cli::ParseBenchOptions(probability_args, &error).has_value(),
         "invalid draft confidence threshold rejected");

  const std::array<const char*, 2> backend_args = {"--qwen-backend", "unknown"};
  Expect(!gufo::cli::ParseBenchOptions(backend_args, &error).has_value(),
         "invalid Qwen backend rejected");
}

}  // namespace

int main() {
  TestDefaultOptions();
  TestDepthOptions();
  TestHybridMtpOptions();
  TestNativeHrxBackendOption();
  TestNativeHrxValidationOption();
  TestInvalidDepth();
  std::cout << "All benchmark CLI tests passed.\n";
  return 0;
}
