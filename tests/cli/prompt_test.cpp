#include "src/cli/prompt/prompt.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

void TestDefaultOptions() {
  const std::array<const char*, 2> args = {"Hello", "world"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->prompt_text == "Hello world");
  assert(opt->max_tokens == 128);
  assert(opt->sampling.temperature == 0.0F);
  assert(!opt->sampling_supplied.temperature && !opt->sampling_supplied.top_k);
  assert(opt->use_chat_template);
  assert(opt->system_prompt.empty());
  assert(opt->reasoning_mode == "auto");
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
  assert(opt->sampling_supplied.temperature && !opt->sampling_supplied.top_p);
  assert(!opt->use_chat_template);
  assert(opt->verbose);
  assert(opt->prompt_text == "Test prompt");
  const char* neutral[] = {"--temperature=0", "--top-k=0",
                           "--presence-penalty=0", "-s=0", "Hello"};
  const auto zero = gufo::cli::ParsePromptOptions(neutral);
  assert(zero);
  const auto sampling = gufo::sampling::ResolveTextSampling(
      gufo::sampling::TextModelPreset::kQwen38, false, zero->sampling,
      zero->sampling_supplied);
  assert(sampling.temperature == 0 && sampling.top_k == 0 &&
         sampling.presence_penalty == 0 && sampling.seed == 0);
  assert(sampling.top_p == 0.8F);
}

void TestFlashMtpFlags() {
  const char* args[] = {"--speculative",  "mtp", "--mtp-model",   "mtp.gguf",
                        "--draft-tokens", "3",   "--temperature", "0.7",
                        "--seed",         "1",   "Prompt"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->speculative_backend == "mtp");
  assert(opt->draft_tokens == 3);
  assert(opt->sampling.temperature == 0.7F && opt->sampling.seed == 1);
}

void TestImageFlags() {
  const char* args[] = {"--image", "a.png",           "--image",
                        "b.jpg",   "--add-vision-id", "Compare"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt && opt->add_vision_id);
  assert((opt->image_paths == std::vector<std::string>{"a.png", "b.jpg"}));
  assert(!gufo::cli::ParsePromptOptions(std::array{"Compare"})->add_vision_id);
  std::vector<const char*> many;
  for (unsigned i = 0; i < 17; ++i)
    many.insert(many.end(), {"--image", "a.png"});
  many.push_back("Compare");
  const auto parsed = gufo::cli::ParsePromptOptions(many);
  assert(parsed && parsed->image_paths.size() == 17);
  assert(!gufo::cli::ParsePromptOptions(std::array{"--image", ""}));
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

  const std::array<const char*, 2> args6 = {"--top-p", "-0.01"};
  assert(!gufo::cli::ParsePromptOptions(args6, &err).has_value());
  const auto zero_top_p =
      gufo::cli::ParsePromptOptions(std::array{"--top-p", "0"}, &err);
  assert(zero_top_p && zero_top_p->sampling.top_p == 0.0F);

  const std::array<const char*, 6> unsupported_floor = {
      "--speculative",      "dflash2", "--dflash-model", "draft.gguf",
      "--min-draft-tokens", "2"};
  assert(!gufo::cli::ParsePromptOptions(unsupported_floor, &err).has_value());
  assert(err.find("min-draft-tokens") != std::string::npos);
  for (const char* policy : {"fixed", "adaptive", "unknown"}) {
    const std::array<const char*, 6> args = {"--speculative",  "dflash2",
                                             "--dflash-model", "draft.gguf",
                                             "--draft-policy", policy};
    const auto parsed = gufo::cli::ParsePromptOptions(args, &err);
    assert(parsed.has_value() == (std::string_view(policy) != "unknown"));
    if (parsed)
      assert(parsed->draft_policy == policy);
  }
  const std::array<const char*, 2> policy_without_backend = {"--draft-policy",
                                                             "adaptive"};
  assert(!gufo::cli::ParsePromptOptions(policy_without_backend, &err));

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
  const auto resolved = gufo::sampling::ResolveTextSampling(
      gufo::sampling::TextModelPreset::kQwen38, false, opt->sampling,
      opt->sampling_supplied);
  assert(resolved.temperature == 0.7F);
  assert(resolved.top_p == opt->sampling.top_p &&
         resolved.top_k == opt->sampling.top_k &&
         resolved.min_p == opt->sampling.min_p &&
         resolved.min_keep == opt->sampling.min_keep &&
         resolved.seed == opt->sampling.seed &&
         resolved.repeat_penalty == opt->sampling.repeat_penalty &&
         resolved.repeat_last_n == opt->sampling.repeat_last_n &&
         resolved.frequency_penalty == opt->sampling.frequency_penalty &&
         resolved.presence_penalty == opt->sampling.presence_penalty);
  assert(opt->reasoning_mode == "on");
  assert(opt->reasoning_effort == "high");
  assert(opt->preserve_thinking == "off");
  assert(!opt->display_prompt);
  assert(opt->prompt_text == "Explicit prompt text");
}

// Expected values are written out, independent of PresetDefaults.
void TestPresetFixtures() {
  using gufo::sampling::ResolveTextSampling;
  using gufo::sampling::TextModelPreset;
  const gufo::sampling::SamplingConfig configured;
  const gufo::sampling::SamplingOverrides none;
  struct Row {
    TextModelPreset preset;
    std::optional<bool> thinking;
    float temperature;
    float top_p;
    int top_k;
    float presence;
  };
  const Row rows[] = {
      {TextModelPreset::kQwen38, true, 1.0F, 0.95F, 20, 0.0F},
      {TextModelPreset::kQwen38, std::nullopt, 1.0F, 0.95F, 20, 0.0F},
      {TextModelPreset::kQwen38, false, 0.7F, 0.8F, 20, 1.5F},
      {TextModelPreset::kDeepSeekV4Flash, true, 1.0F, 0.95F, 0, 0.0F},
      {TextModelPreset::kDeepSeekV4Flash, std::nullopt, 1.0F, 0.95F, 0, 0.0F},
      {TextModelPreset::kDeepSeekV4Flash, false, 1.0F, 0.95F, 0, 0.0F},
  };
  for (const auto& row : rows) {
    const auto r =
        ResolveTextSampling(row.preset, row.thinking, configured, none);
    assert(r.temperature == row.temperature && r.top_p == row.top_p &&
           r.top_k == row.top_k && r.presence_penalty == row.presence &&
           r.min_p == 0.0F && r.min_keep == 0 && r.seed == -1 &&
           r.repeat_penalty == 1.0F && r.repeat_last_n == 64 &&
           r.frequency_penalty == 0.0F);
  }
  const auto unspecified = ResolveTextSampling(TextModelPreset::kUnspecified,
                                               true, configured, none);
  assert(unspecified.temperature == 0.0F && unspecified.top_p == 1.0F);
}

void TestDefaultsValidation() {
  gufo::sampling::TextSamplingDefaults defaults;
  defaults.thinking_enabled.temperature = 0.0F;
  gufo::sampling::ValidateTextSamplingDefaults(defaults);
  for (auto* values :
       {&defaults.thinking_unspecified, &defaults.thinking_enabled,
        &defaults.thinking_disabled}) {
    const auto valid = *values;
    values->temperature = -1.0F;
    bool rejected = false;
    try {
      gufo::sampling::ValidateTextSamplingDefaults(defaults);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
    *values = valid;
  }
}

void TestProviderDefaults() {
  using gufo::sampling::ResolveTextSampling;
  gufo::sampling::TextSamplingDefaults provider;
  provider.thinking_enabled = {
      .temperature = 0.6F, .top_k = 64, .top_p = 0.9F, .min_p = 0.05F};
  provider.thinking_disabled = {.temperature = 0.0F, .repeat_penalty = 1.2F};
  const char* args[] = {"--top-k=0", "--top-p=1", "Hello"};
  const auto cli = gufo::cli::ParsePromptOptions(args);
  assert(cli);
  const auto on = ResolveTextSampling(provider, true, cli->sampling,
                                      cli->sampling_supplied);
  assert(on.temperature == 0.6F && on.top_k == 0 && on.top_p == 1.0F &&
         on.min_p == 0.05F);
  const auto off = ResolveTextSampling(provider, false, cli->sampling,
                                       cli->sampling_supplied);
  assert(off.temperature == 0.0F && off.repeat_penalty == 1.2F &&
         off.top_k == 0 && off.top_p == 1.0F && off.min_p == 0.0F);
  const auto unset = ResolveTextSampling(provider, std::nullopt, cli->sampling,
                                         cli->sampling_supplied);
  assert(unset.temperature == 0.0F && unset.top_k == 0);
  // No provider defaults: the command line is used unchanged.
  const auto plain = ResolveTextSampling(std::nullopt, true, cli->sampling,
                                         cli->sampling_supplied);
  assert(plain.temperature == cli->sampling.temperature);
  // The caller's constraint survives resolution.
  auto constraint = std::make_shared<gufo::sampling::TokenConstraint>();
  auto configured = cli->sampling;
  configured.constraint = constraint;
  assert(ResolveTextSampling(provider, true, configured, {}).constraint ==
         constraint);
}

int main() {
  TestPresetFixtures();
  TestProviderDefaults();
  TestDefaultsValidation();
  TestDefaultOptions();
  TestExplicitFlags();
  TestSamplingAndReasoningFlags();
  TestFlashMtpFlags();
  TestImageFlags();
  TestInvalidFlags();
  std::cout << "All prompt CLI tests passed.\n";
  return 0;
}
