#include "src/cli/bench/bench.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/bench_yarn.hpp"

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
  Expect(options->model_path.empty(), "no implicit model path");
  Expect(options->n_prompts == std::vector<std::size_t>{2048},
         "default prefill is 2048 tokens");
  Expect(options->n_depths == std::vector<std::size_t>{0},
         "default depth is zero");
  Expect(options->repetitions == 1, "default is one repetition");
  Expect(options->draft_tokens == 7, "default draft ceiling is seven");
  Expect(options->min_draft_tokens == 1, "default minimum draft is one");
  Expect(options->sampling.temperature == 0.0F && options->sampling.seed == 0,
         "benchmark sampling defaults remain greedy");
}

void TestDs4SamplingOptions() {
  const std::array<const char*, 4> args = {"--temperature", "0.6", "--seed",
                                           "7"};
  const auto options = gufo::cli::ParseBenchOptions(args);
  Expect(options && options->sampling.temperature == 0.6F &&
             options->sampling.seed == 7,
         "DS4 benchmark retains temperature and seed");
  for (const char* value : {"-1", "nan", "inf"}) {
    const std::array<const char*, 2> invalid = {"--temperature", value};
    Expect(!gufo::cli::ParseBenchOptions(invalid),
           "invalid benchmark temperature is rejected");
  }
  const std::array<const char*, 2> invalid_seed = {"--seed", "-1"};
  Expect(!gufo::cli::ParseBenchOptions(invalid_seed),
         "negative benchmark seed is rejected");
}

void TestFlashMtpSamplingOptions() {
  const char* args[] = {"--speculative",
                        "mtp",
                        "--mtp-model",
                        "mtp.gguf",
                        "--draft-tokens",
                        "3",
                        "--temperature",
                        "0.8",
                        "--seed",
                        "73",
                        "--top-k",
                        "40",
                        "--top-p",
                        "0.9",
                        "--min-p",
                        "0.01",
                        "--min-keep",
                        "2",
                        "--repeat-penalty",
                        "1.1",
                        "--repeat-last-n",
                        "16",
                        "--frequency-penalty",
                        "0.2",
                        "--presence-penalty",
                        "0.1"};
  const auto options = gufo::cli::ParseBenchOptions(args);
  Expect(options.has_value(), "Flash MTP sampling options parse");
  const auto& s = options->sampling;
  Expect(options->draft_tokens == 3 && s.temperature == 0.8F && s.seed == 73 &&
             s.top_k == 40 && s.top_p == 0.9F && s.min_p == 0.01F &&
             s.min_keep == 2 && s.repeat_penalty == 1.1F &&
             s.repeat_last_n == 16 && s.frequency_penalty == 0.2F &&
             s.presence_penalty == 0.1F,
         "Flash MTP benchmark retains every sampling control");
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

  const std::array<const char*, 4> unsupported_floor = {
      "--speculative", "dflash2", "--min-draft-tokens", "2"};
  Expect(!gufo::cli::ParseBenchOptions(unsupported_floor, &error).has_value() &&
             error.find("min-draft-tokens") != std::string::npos,
         "DFlash2 rejects an unsupported minimum draft length");
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

// Flash-Next bench follows serve's --enable-yarn rule on the context it
// reserves (depth + prompt or generation + 1) instead of enabling YaRN
// silently.
void TestFlashNextYarnRule() {
  using gufo::models::qwen38_flash_next::FlashNextBenchContext;
  using gufo::models::qwen38_flash_next::ResolveFlashNextBenchYarn;

  const std::array<const char*, 0> none{};
  auto defaults = gufo::cli::ParseBenchOptions(none);
  Expect(defaults && !defaults->enable_yarn, "YaRN is off by default");
  Expect(FlashNextBenchContext(*defaults) == 4096,
         "default workload reserves the 4096-token floor");

  const std::array<const char*, 7> within = {
      "--n-depth", "260000", "--n-prompt",   "2047",
      "--n-gen",   "128",    "--enable-yarn"};
  auto options = gufo::cli::ParseBenchOptions(within);
  Expect(options && options->enable_yarn, "--enable-yarn parses");
  Expect(FlashNextBenchContext(*options) == 262048,
         "depth + prompt + 1 sizes the benchmark context");
  std::optional<gufo::models::qwen38_flash_next::RopeScaling> scaling;
  std::string error;
  Expect(!ResolveFlashNextBenchYarn(*options, 262144, &scaling, &error) &&
             error ==
                 "--enable-yarn has no effect: context 262048 is within "
                 "the native 262144 tokens",
         "--enable-yarn is rejected when the benchmark fits natively");
  options->enable_yarn = false;
  error.clear();
  Expect(ResolveFlashNextBenchYarn(*options, 262144, &scaling, &error) &&
             !scaling.has_value(),
         "a benchmark within the native context leaves YaRN off");

  const std::array<const char*, 6> beyond = {
      "--n-depth", "260000", "--n-prompt", "4096", "--n-gen", "0"};
  options = gufo::cli::ParseBenchOptions(beyond);
  Expect(options && FlashNextBenchContext(*options) == 264097,
         "a deep benchmark reserves past the native context");
  Expect(!ResolveFlashNextBenchYarn(*options, 262144, &scaling, &error) &&
             error ==
                 "context above the native 262144 tokens needs static YaRN, "
                 "which applies to every request; pass --enable-yarn "
                 "(requested 264097)",
         "a benchmark past the native context requires --enable-yarn");
  options->enable_yarn = true;
  Expect(ResolveFlashNextBenchYarn(*options, 262144, &scaling, &error) &&
             scaling && scaling->original_context == 262144 &&
             gufo::models::qwen38_flash_next::ScaledContextLength(
                 *scaling, 262144) >= 264097,
         "--enable-yarn derives a factor covering the benchmark context");

  const std::array<const char*, 6> generation = {
      "--n-depth", "262000", "--n-prompt", "0", "--n-gen", "512"};
  options = gufo::cli::ParseBenchOptions(generation);
  Expect(options && FlashNextBenchContext(*options) == 262513,
         "depth + generation + 1 sizes the benchmark context too");
  Expect(!ResolveFlashNextBenchYarn(*options, 262144, &scaling, &error),
         "depth + generation past the native context requires the flag");
}

}  // namespace

int main() {
  TestDefaultOptions();
  TestDs4SamplingOptions();
  TestFlashMtpSamplingOptions();
  TestDepthOptions();
  TestInvalidDepth();
  TestInvalidWorkload();
  TestFlashNextYarnRule();
  std::cout << "All benchmark CLI tests passed.\n";
  return 0;
}
