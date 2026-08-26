#include "src/server/prompt_cli.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <span>
#include <string>

void TestDefaultOptions() {
  const std::array<const char*, 2> args = {"Hello", "world"};
  const auto opt = strix::server::ParsePromptOptions(args);
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
  const auto opt = strix::server::ParsePromptOptions(args);
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
  const auto opt = strix::server::ParsePromptOptions(args);
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
  assert(!strix::server::ParsePromptOptions(args1, &err).has_value());
  assert(!err.empty());

  const std::array<const char*, 2> args2 = {"-n", "not_a_number"};
  assert(!strix::server::ParsePromptOptions(args2, &err).has_value());

  const std::array<const char*, 2> args3 = {"-t", "invalid_float"};
  assert(!strix::server::ParsePromptOptions(args3, &err).has_value());

  const std::array<const char*, 2> args4 = {"--draft-policy", "unknown"};
  assert(!strix::server::ParsePromptOptions(args4, &err).has_value());

  const std::array<const char*, 4> args5 = {"--draft-tokens", "3",
                                            "--min-draft-tokens", "4"};
  assert(!strix::server::ParsePromptOptions(args5, &err).has_value());
}

int main() {
  TestDefaultOptions();
  TestExplicitFlags();
  TestHybridMtpFlags();
  TestInvalidFlags();
  std::cout << "All prompt CLI tests passed.\n";
  return 0;
}
