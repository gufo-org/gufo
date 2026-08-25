#include "src/cli/prompt/prompt.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <span>
#include <string>

void TestDefaultOptions() {
  const std::array<const char*, 2> args = {"Hello", "world"};
  const auto opt = strix::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->prompt_text == "Hello world");
  assert(opt->max_tokens == 128);
  assert(opt->temperature == 0.0F);
  assert(opt->use_chat_template);
  assert(!opt->verbose);
  assert(opt->draft_tokens == 7);
  assert(opt->draft_policy == "rolling");
  assert(opt->min_draft_tokens == 1);
}

void TestExplicitFlags() {
  const std::array<const char*, 10> args = {
      "--model", "model.gguf", "-n", "256",  "--temperature",
      "0.7",     "--raw",      "-v", "Test", "prompt"};
  const auto opt = strix::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->model_path == "model.gguf");
  assert(opt->max_tokens == 256);
  assert(opt->temperature > 0.69F && opt->temperature < 0.71F);
  assert(!opt->use_chat_template);
  assert(opt->verbose);
  assert(opt->prompt_text == "Test prompt");
}

void TestHybridMtpFlags() {
  const std::array<const char*, 11> args = {
      "--speculative",      "mtp-npu", "--mtp-model",    "mtp.gguf",
      "--draft-tokens",     "2",       "--draft-policy", "fixed",
      "--min-draft-tokens", "2",       "Prompt"};
  const auto opt = strix::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->speculative_backend == "mtp-npu");
  assert(opt->mtp_model_path == "mtp.gguf");
  assert(opt->draft_tokens == 2);
  assert(opt->draft_policy == "fixed");
  assert(opt->min_draft_tokens == 2);
}

void TestInvalidFlags() {
  std::string err;
  const std::array<const char*, 1> args1 = {"--model"};
  assert(!strix::cli::ParsePromptOptions(args1, &err).has_value());
  assert(!err.empty());

  const std::array<const char*, 2> args2 = {"-n", "not_a_number"};
  assert(!strix::cli::ParsePromptOptions(args2, &err).has_value());

  const std::array<const char*, 2> args3 = {"-t", "invalid_float"};
  assert(!strix::cli::ParsePromptOptions(args3, &err).has_value());

  const std::array<const char*, 2> args4 = {"--draft-policy", "unknown"};
  assert(!strix::cli::ParsePromptOptions(args4, &err).has_value());

  const std::array<const char*, 4> args5 = {"--draft-tokens", "3",
                                            "--min-draft-tokens", "4"};
  assert(!strix::cli::ParsePromptOptions(args5, &err).has_value());
}

void TestSamplingAndReasoningFlags() {
  const std::array<const char*, 23> args = {"--model",
                                            "model.gguf",
                                            "--top-p",
                                            "0.9",
                                            "--top-k",
                                            "50",
                                            "--min-p",
                                            "0.05",
                                            "-s",
                                            "42",
                                            "--repeat-penalty",
                                            "1.15",
                                            "--repeat-last-n",
                                            "128",
                                            "--think",
                                            "on",
                                            "--reasoning-budget",
                                            "1024",
                                            "--chat-template",
                                            "qwen",
                                            "--no-display-prompt",
                                            "-p",
                                            "Explicit prompt text"};
  const auto opt = strix::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->model_path == "model.gguf");
  assert(opt->top_p > 0.89F && opt->top_p < 0.91F);
  assert(opt->top_k == 50);
  assert(opt->min_p > 0.04F && opt->min_p < 0.06F);
  assert(opt->seed == 42);
  assert(opt->repeat_penalty > 1.14F && opt->repeat_penalty < 1.16F);
  assert(opt->repeat_last_n == 128);
  assert(opt->reasoning_mode == "on");
  assert(opt->reasoning_budget == 1024);
  assert(opt->chat_template == "qwen");
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
