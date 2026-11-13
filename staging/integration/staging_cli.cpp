#include "staging/integration/staging_cli.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/text.hpp"
#include "src/cli/arg_parser.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/image.hpp"
#include "src/core/reasoning.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/vision/prompt.hpp"
#include "staging/models/gemma4/cli_runner.hpp"
#include "staging/models/glm5_next/cli_runner.hpp"
#include "staging/models/ling3_0/cli_runner.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include "staging/models/flint/cli_runner.hpp"
#include "staging/models/nemotron_h_moe/cli_runner.hpp"
#include "staging/models/qwen36_a3b/cli_runner.hpp"
#endif

namespace gufo::cli {
namespace {

#if !defined(ENGINE_ENABLE_HIP)
void PrintModelLoadTime(std::chrono::steady_clock::time_point start,
                        bool success = true) {
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  auto& output = success ? std::cout : std::cerr;
  output << "[Model Load]: " << load_seconds << " s";
  if (!success) {
    output << " (failed)";
  }
  output << '\n';
}

#endif

std::optional<ReasoningEffort> ParseEffort(std::string_view value) {
  if (value == "minimal") {
    return ReasoningEffort::kMinimal;
  }
  if (value == "low") {
    return ReasoningEffort::kLow;
  }
  if (value == "medium") {
    return ReasoningEffort::kMedium;
  }
  if (value == "high") {
    return ReasoningEffort::kHigh;
  }
  if (value == "xhigh") {
    return ReasoningEffort::kXHigh;
  }
  if (value == "max") {
    return ReasoningEffort::kMax;
  }
  return std::nullopt;
}

constexpr std::uint32_t kDefaultContext = 4096;

ReasoningOptions PromptReasoningOptions(const ext::TextPromptOptions& opt) {
  ReasoningOptions reasoning;
  if (opt.reasoning_mode == "on") {
    reasoning.enabled = true;
  } else if (opt.reasoning_mode == "off") {
    reasoning.enabled = false;
  }
  if (opt.reasoning_effort != "auto") {
    reasoning.effort = ParseEffort(opt.reasoning_effort);
    reasoning.enabled = true;
  }
  if (opt.preserve_thinking == "on") {
    reasoning.preserve_thinking = true;
  } else if (opt.preserve_thinking == "off") {
    reasoning.preserve_thinking = false;
  }
  return reasoning;
}

void AttachImages(const ext::TextPromptOptions& options,
                  tokenization::ChatMessage& message) {
  for (const auto& path : options.image_paths) {
    message.images.push_back(
        {0, std::make_shared<const std::vector<std::uint8_t>>(
                core::ReadImageFile(path))});
  }
}

std::shared_ptr<const models::qwen::vision::Prompt> PrepareVision(
    const ext::TextPromptOptions& opt,
    const tokenization::QwenTokenizer& tokenizer,
    std::span<const tokenization::ChatMessage> messages,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder) {
  if (!encoder) {
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  }
  const auto reasoning = PromptReasoningOptions(opt);
  return std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, messages, {},
          tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id),
          encoder->identity(), kDefaultContext));
}

#if !defined(ENGINE_ENABLE_HIP)
int RejectOnCpu(std::string_view name,
                std::chrono::steady_clock::time_point start) {
  std::cerr << name << " requires ENGINE_ENABLE_HIP=ON\n";
  PrintModelLoadTime(start, false);
  return 1;
}
#endif

}  // namespace

std::optional<int> StagingRunPrompt(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point model_load_start) {
#if defined(ENGINE_ENABLE_HIP)
  if (models::gemma4::IsGemma4(reader)) {
    return models::gemma4::RunGemma4Prompt(opt, reader, model_load_start);
  }
  if (models::ling3_0::IsLing3Model(reader)) {
    if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
      std::cerr << "Ling 3.0 Flash does not support image input\n";
      return 2;
    }
    return models::ling3_0::RunLing3Prompt(opt, reader, model_load_start,
                                           PromptReasoningOptions(opt));
  }
  if (models::qwen36_a3b::IsQwen36A3B(reader)) {
    std::string rendered_prompt = opt.prompt_text;
    std::vector<tokenization::ChatMessage> messages;
    if (opt.use_chat_template) {
      if (!opt.system_prompt.empty()) {
        messages.push_back(
            {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
      }
      messages.push_back(
          {tokenization::ChatRole::kUser, opt.prompt_text, "", ""});
      const auto reasoning = PromptReasoningOptions(opt);
      const auto rendered = tokenization::QwenChatTemplate::Render(
          messages, {}, tokenization::ResolveQwenChatOptions(reasoning,
                                                             opt.add_vision_id));
      if (!rendered.has_value()) {
        std::cerr << "Error formatting chat template.\n";
        return 1;
      }
      rendered_prompt = *rendered;
    }
    return models::qwen36_a3b::RunQwen36A3BPrompt(
        opt, reader, model_load_start, rendered_prompt, messages, AttachImages,
        PrepareVision);
  }
#else
  if (models::gemma4::IsGemma4(reader)) {
    return RejectOnCpu("Gemma-4-26B-A4B", model_load_start);
  }
  if (models::ling3_0::IsLing3Model(reader)) {
    return RejectOnCpu("Ling 3.0 Flash", model_load_start);
  }
#endif
  return std::nullopt;
}

std::optional<int> StagingRunChat(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point model_load_start) {
#if defined(ENGINE_ENABLE_HIP)
  if (models::gemma4::IsGemma4(reader)) {
    return models::gemma4::RunGemma4Chat(opt, reader, model_load_start);
  }
  if (models::ling3_0::IsLing3Model(reader)) {
    if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
      std::cerr << "Ling 3.0 Flash does not support image input\n";
      return 2;
    }
    return models::ling3_0::RunLing3Chat(opt, reader, model_load_start,
                                         PromptReasoningOptions(opt));
  }
  if (models::qwen36_a3b::IsQwen36A3B(reader)) {
    return models::qwen36_a3b::RunQwen36A3BChat(opt, reader, model_load_start);
  }
#else
  if (models::gemma4::IsGemma4(reader)) {
    return RejectOnCpu("Gemma-4-26B-A4B", model_load_start);
  }
  if (models::ling3_0::IsLing3Model(reader)) {
    return RejectOnCpu("Ling 3.0 Flash", model_load_start);
  }
#endif
  return std::nullopt;
}

std::optional<int> StagingRunBenchmark(
    const ext::TextBenchOptions& opt,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
#if defined(ENGINE_ENABLE_HIP)
  if (models::qwen36_a3b::IsQwen36A3B(*reader)) {
    return models::qwen36_a3b::RunQwen36A3BBenchmark(opt, reader,
                                                     model_load_start);
  }
  if (models::gemma4::IsGemma4(*reader)) {
    return models::gemma4::RunGemma4Benchmark(opt, reader, model_load_start);
  }
  if (models::ling3_0::IsLing3Model(*reader)) {
    return models::ling3_0::RunLing3Benchmark(opt, reader, model_load_start);
  }
  if (models::nemotron_h_moe::IsNemotronHBolt(*reader)) {
    return models::nemotron_h_moe::RunNemotronBenchmark(opt, reader,
                                                        model_load_start);
  }
  if (models::flint::IsFlint(*reader)) {
    return models::flint::RunFlintBenchmark(opt, reader, model_load_start);
  }
  if (models::glm5_next::IsGlm5Next(*reader)) {
    return models::glm5_next::RunGlm5NextBenchmark(opt, reader,
                                                   model_load_start);
  }
#else
  (void)opt;
  (void)reader;
  (void)model_load_start;
#endif
  return std::nullopt;
}

void RegisterStagingTextOptions(ArgParser& parser,
                                ext::TextPromptOptions& opt) {
  parser.AddOption("", "--attn-window", "N",
                   "Qwen3.6-35B-A3B prefill attention: sliding-window tokens "
                   "(0 = dense, default)",
                   "Model", &opt.attn_window);
  parser.AddOption("", "--attn-sink", "N",
                   "Qwen3.6-35B-A3B prefill attention: always-attended "
                   "initial tokens when --attn-window is set (default 0)",
                   "Model", &opt.attn_sink);
}

}  // namespace gufo::cli
