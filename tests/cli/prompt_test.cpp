#include "src/cli/prompt/prompt.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <span>
#include <string>

void TestDefaultOptions() {
  const std::array<const char*, 2> args = {"Hello", "world"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->prompt_text == "Hello world");
  assert(opt->max_tokens == 128);
  assert(opt->sampling.temperature == 0.0F);
  assert(opt->use_chat_template);
  assert(opt->reasoning_mode == "off");
  assert(opt->reasoning_effort == "auto");
  assert(opt->preserve_thinking == "auto");
  assert(!opt->verbose);
  assert(opt->draft_tokens == 7);
  assert(opt->min_draft_tokens == 1);
}

void TestExplicitFlags() {
  const std::array<const char*, 10> args = {
      "--model", "model.gguf", "-n", "256",  "--temperature",
      "0.7",     "--raw",      "-v", "Test", "prompt"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->model_path == "model.gguf");
  assert(opt->max_tokens == 256);
  assert(opt->sampling.temperature > 0.69F &&
         opt->sampling.temperature < 0.71F);
  assert(!opt->use_chat_template);
  assert(opt->verbose);
  assert(opt->prompt_text == "Test prompt");
}

void TestHybridMtpFlags() {
  const std::array<const char*, 9> args = {"--speculative",
                                           "mtp-npu",
                                           "--mtp-model",
                                           "mtp.gguf",
                                           "--spec-draft-n-max",
                                           "2",
                                           "--spec-draft-n-min",
                                           "2",
                                           "Prompt"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->speculative_backend == "mtp-npu");
  assert(opt->mtp_model_path == "mtp.gguf");
  assert(opt->draft_tokens == 2);
  assert(opt->min_draft_tokens == 2);
}

void TestInvalidFlags() {
  std::string err;
  const std::array<const char*, 1> args1 = {"--model"};
  assert(!gufo::cli::ParsePromptOptions(args1, &err).has_value());
  assert(!err.empty());

  const std::array<const char*, 2> args2 = {"-n", "not_a_number"};
  assert(!gufo::cli::ParsePromptOptions(args2, &err).has_value());

  const std::array<const char*, 2> args3 = {"-t", "invalid_float"};
  assert(!gufo::cli::ParsePromptOptions(args3, &err).has_value());

  const std::array<const char*, 4> args5 = {"--draft-tokens", "3",
                                            "--min-draft-tokens", "4"};
  assert(!gufo::cli::ParsePromptOptions(args5, &err).has_value());

  const std::array<const char*, 2> args6 = {"--top-p", "0"};
  assert(!gufo::cli::ParsePromptOptions(args6, &err).has_value());

  for (const char* flag : {"--spec-draft-p-min", "--draft-p-min"}) {
    const std::array<const char*, 2> removed = {flag, "0.75"};
    assert(!gufo::cli::ParsePromptOptions(removed, &err).has_value());
  }
  const std::array<const char*, 6> fixed_block = {
      "--speculative",      "dflash2", "--dflash-model", "draft.gguf",
      "--min-draft-tokens", "2"};
  assert(!gufo::cli::ParsePromptOptions(fixed_block, &err).has_value());
  assert(err.find("fixed blocks") != std::string::npos);

  const std::array<const char*, 2> args8 = {"--chat-template", "qwen"};
  assert(!gufo::cli::ParsePromptOptions(args8, &err).has_value());

  const std::array<const char*, 2> args9 = {"--reasoning-budget", "1024"};
  assert(!gufo::cli::ParsePromptOptions(args9, &err).has_value());

  for (const auto* backend : {"dflash2", "mtp"}) {
    const std::array<const char*, 2> missing_path = {"--speculative", backend};
    assert(!gufo::cli::ParsePromptOptions(missing_path, &err).has_value());
    assert(err.find("requires --") != std::string::npos);
  }
  const std::array<const char*, 5> cpu_spec = {
      "--cpu", "--speculative", "dflash2", "--dflash-model", "draft.gguf"};
  assert(!gufo::cli::ParsePromptOptions(cpu_spec, &err).has_value());
  assert(err.find("ROCm") != std::string::npos);
  const std::array<const char*, 3> cpu_ar = {"--cpu", "--speculative", "off"};
  assert(gufo::cli::ParsePromptOptions(cpu_ar, &err).has_value());
}

void TestSamplingAndReasoningFlags() {
  const std::array<const char*, 29> args = {"--model",
                                            "model.gguf",
                                            "--top-p",
                                            "0.9",
                                            "--top-k",
                                            "50",
                                            "--min-p",
                                            "0.05",
                                            "--min-keep",
                                            "3",
                                            "-s",
                                            "42",
                                            "--repeat-penalty",
                                            "1.15",
                                            "--repeat-last-n",
                                            "128",
                                            "--frequency-penalty",
                                            "0.25",
                                            "--presence-penalty",
                                            "0.5",
                                            "--think",
                                            "on",
                                            "--reasoning-effort",
                                            "high",
                                            "--preserve-thinking",
                                            "off",
                                            "--no-display-prompt",
                                            "-p",
                                            "Explicit prompt text"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->model_path == "model.gguf");
  assert(opt->sampling.top_p > 0.89F && opt->sampling.top_p < 0.91F);
  assert(opt->sampling.top_k == 50);
  assert(opt->sampling.min_p > 0.04F && opt->sampling.min_p < 0.06F);
  assert(opt->sampling.min_keep == 3);
  assert(opt->sampling.seed == 42);
  assert(opt->sampling.repeat_penalty > 1.14F &&
         opt->sampling.repeat_penalty < 1.16F);
  assert(opt->sampling.repeat_last_n == 128);
  assert(opt->sampling.frequency_penalty > 0.24F &&
         opt->sampling.frequency_penalty < 0.26F);
  assert(opt->sampling.presence_penalty > 0.49F &&
         opt->sampling.presence_penalty < 0.51F);
  assert(opt->reasoning_mode == "on");
  assert(opt->reasoning_effort == "high");
  assert(opt->preserve_thinking == "off");
  assert(!opt->display_prompt);
  assert(opt->prompt_text == "Explicit prompt text");
}

int main() {
  TestDefaultOptions();
  TestExplicitFlags();
  TestSamplingAndReasoningFlags();
  TestHybridMtpFlags();
  TestInvalidFlags();
  std::cout << "All prompt CLI tests passed.\n";
  return 0;
}
