#include "src/cli/prompt/prompt.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/sampling_options.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/heterogeneous/npu_drafter.hpp"
#include "src/core/speculative/draft_heads.hpp"
#include "src/core/speculative/prompt_lookup_backend.hpp"
#include "src/core/speculative/self_speculative.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#endif

namespace gufo::cli {

static void PrintTextHelp(std::string_view program_name,
                          std::string_view command) {
  PromptOptions opt;
  gufo::cli::ArgParser parser(
      std::string(program_name) + " " + std::string(command),
      command == "chat" ? "Interactive conversation."
                        : "Execute one prompt request and exit.");
  parser.AddOption("-m", "--model", "PATH", "Path to GGUF model file", "Model",
                   &opt.model_path);
  if (command == "prompt") {
    parser.AddOption("-p", "--prompt", "TEXT", "Direct input prompt text",
                     "Prompt", &opt.prompt_text);
    parser.AddOption("-f", "--file", "PATH",
                     "File path to read input prompt text from", "Prompt",
                     &opt.prompt_file);
  }
  parser.AddOption("", "--system", "PROMPT",
                   "System role instructions prepended to the prompt "
                   "(default: helpful assistant)",
                   "Prompt", &opt.system_prompt);
  if (command == "prompt") {
    parser.AddInverseFlag("", "--raw",
                          "Disable chat template framing and pass raw tokens",
                          "Prompt", &opt.use_chat_template);
    parser.AddInverseFlag(
        "", "--no-display-prompt",
        "Suppress echoing the prompt before generated response", "Prompt",
        &opt.display_prompt);
  }
  parser.AddOption("-n", "--max-tokens", "N",
                   "Maximum number of new tokens to generate (default: 128)",
                   "Sampling", &opt.max_tokens);
  RegisterSamplingOptions(parser, &opt.sampling);
  parser.AddOption("", "--think", "MODE",
                   "Reasoning mode: on, off, or auto (default: off)",
                   "Reasoning", &opt.reasoning_mode);
  parser.AddOption("", "--reasoning-effort", "LEVEL",
                   "Effort: auto, minimal, low, medium, high, xhigh, or max",
                   "Reasoning", &opt.reasoning_effort);
  parser.AddOption("", "--preserve-thinking", "MODE",
                   "Replay prior reasoning: on, off, or auto", "Reasoning",
                   &opt.preserve_thinking);
  parser.AddOption(
      "", "--speculative", "MODE",
      "Draft backend: dspark (DeepSeek V4 Flash), dflash, dflash2, "
      "mtp, mtp-npu, npu, pld, self, or off",
      "Speculative", &opt.speculative_backend);
  parser.AddOption("", "--dflash-model", "PATH",
                   "Path to quantized Qwen DFlash/DFlash-2 GGUF file",
                   "Speculative", &opt.dflash_model_path);
  parser.AddOption("", "--dspark-model", "PATH",
                   "Path to the DeepSeek V4 Flash DSpark support GGUF file",
                   "Speculative", &opt.dspark_model_path);
  parser.AddOption("", "--mtp-model", "PATH",
                   "Path to quantized Qwen MTP draft head GGUF file",
                   "Speculative", &opt.mtp_model_path);
  parser.AddOption("-d", "--draft-tokens", "N",
                   "Maximum speculative draft tokens evaluated per step "
                   "(default: 7)",
                   "Speculative", &opt.draft_tokens);
  parser.AddOption("", "--spec-draft-n-max", "N",
                   "llama.cpp-compatible alias for --draft-tokens",
                   "Speculative", &opt.draft_tokens);

  parser.AddOption("", "--min-draft-tokens", "N",
                   "Adaptive draft floor (default: 1)", "Speculative",
                   &opt.min_draft_tokens);
  parser.AddOption("", "--spec-draft-n-min", "N",
                   "llama.cpp-compatible alias for --min-draft-tokens",
                   "Speculative", &opt.min_draft_tokens);
  parser.AddOption(
      "", "--spec-draft-p-min", "P",
      "Stop at the first draft token below confidence P; 0 disables "
      "(default: 0)",
      "Speculative", &opt.draft_p_min);
  parser.AddOption("", "--draft-p-min", "P", "Alias for --spec-draft-p-min",
                   "Speculative", &opt.draft_p_min);
  parser.AddFlag("", "--cpu",
                 "Force CPU OpenMP execution fallback instead of GPU ROCm",
                 "Hardware", &opt.force_cpu);
  parser.AddFlag("-v", "--verbose",
                 "Print detailed timing, latency breakdown, and tok/s metrics",
                 "General", &opt.verbose);
  parser.PrintHelp();
}

void PrintPromptHelp(std::string_view program_name) {
  PrintTextHelp(program_name, "prompt");
}

void PrintChatHelp(std::string_view program_name) {
  PrintTextHelp(program_name, "chat");
}

namespace {

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

bool IsDeepSeekV4Flash(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "deepseek4";
}

std::optional<ReasoningEffort> ParseReasoningEffort(std::string_view value) {
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

ReasoningOptions PromptReasoningOptions(const PromptOptions& options) {
  ReasoningOptions reasoning;
  if (options.reasoning_mode == "on") {
    reasoning.enabled = true;
  } else if (options.reasoning_mode == "off") {
    reasoning.enabled = false;
  }
  if (options.reasoning_effort != "auto") {
    reasoning.effort = ParseReasoningEffort(options.reasoning_effort);
    reasoning.enabled = true;
  }
  if (options.preserve_thinking == "on") {
    reasoning.preserve_thinking = true;
  } else if (options.preserve_thinking == "off") {
    reasoning.preserve_thinking = false;
  }
  return reasoning;
}

tokenization::QwenReasoningEffort QwenEffort(
    std::optional<ReasoningEffort> effort) {
  switch (effort.value_or(ReasoningEffort::kXHigh)) {
    case ReasoningEffort::kMinimal:
    case ReasoningEffort::kLow:
      return tokenization::QwenReasoningEffort::kLow;
    case ReasoningEffort::kMedium:
      return tokenization::QwenReasoningEffort::kMedium;
    case ReasoningEffort::kHigh:
    case ReasoningEffort::kXHigh:
    case ReasoningEffort::kMax:
      return tokenization::QwenReasoningEffort::kXHigh;
  }
  return tokenization::QwenReasoningEffort::kXHigh;
}

#if defined(ENGINE_ENABLE_HIP)
constexpr std::uint32_t kDefaultContext = 4096;

std::shared_ptr<models::deepseek_v4_flash::Model> LoadDeepSeekModel(
    const PromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  if (opt.force_cpu) {
    std::cerr << "DeepSeek V4 Flash is supported only by the ROCm backend\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  std::string template_error;
  if (!models::deepseek_v4_flash::ValidateGgufTemplate(reader,
                                                       &template_error)) {
    std::cerr << "Unsupported DeepSeek chat template: " << template_error
              << '\n';
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  const bool dspark_requested = opt.speculative_backend == "dspark";
  if (!opt.speculative_backend.empty() && !dspark_requested) {
    std::cerr << "DeepSeek V4 Flash supports only --speculative dspark\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  if (opt.speculative_backend == "dspark" && opt.dspark_model_path.empty()) {
    std::cerr << "--speculative dspark requires --dspark-model\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  if (dspark_requested &&
      (opt.min_draft_tokens != 1 || opt.draft_p_min != 0.0F)) {
    std::cerr << "DSpark uses model-owned adaptive drafting; "
                 "--min-draft-tokens and --spec-draft-p-min are unsupported\n";
    return nullptr;
  }

  std::string error;
  auto model = models::deepseek_v4_flash::Model::Load(
      opt.model_path,
      models::deepseek_v4_flash::ModelOptions{
          .max_context = kDefaultContext,
          .prefill_chunk = 2048,
          .power_percent = 100,
          .dspark_model_path = dspark_requested ? opt.dspark_model_path : "",
      },
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating DeepSeek V4 Flash model: " << error << '\n';
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  PrintModelLoadTime(load_start);

  return model;
}

int GenerateDeepSeekResponse(
    const PromptOptions& opt,
    const std::shared_ptr<models::deepseek_v4_flash::Model>& model,
    models::deepseek_v4_flash::Session& session,
    std::span<const int> prompt_tokens, std::string* reply = nullptr) {
  std::string error;
  const bool dspark_requested = model->HasDspark();
  const auto emit = [&](int token) {
    const auto piece = model->DecodeToken(token);
    if (reply != nullptr)
      reply->append(piece);
    std::cout << piece << std::flush;
  };
  if (prompt_tokens.empty()) {
    std::cerr << "DeepSeek V4 Flash prompt produced no tokens\n";
    return 1;
  }
  if (prompt_tokens.size() >= kDefaultContext ||
      opt.max_tokens >= kDefaultContext - prompt_tokens.size()) {
    std::cerr << "DeepSeek V4 Flash prompt and output exceed the 4096-token "
                 "CLI context\n";
    return 1;
  }

  if (!session.Sync(prompt_tokens, &error)) {
    std::cerr << "DeepSeek V4 Flash prefill failed: " << error << '\n';
    return 1;
  }

  if (opt.verbose) {
    std::cout << "[Engine]: DeepSeek V4 Flash ROCm (gfx1151)\n"
              << "Model: " << model->ModelName() << '\n'
              << "Prompt tokens: " << prompt_tokens.size() << '\n'
              << "Max tokens: " << opt.max_tokens << '\n'
              << "--- Generation Output ---\n";
  }

  std::vector<sampling::TokenId> sampling_history;
  sampling_history.reserve(prompt_tokens.size());
  for (const int token : prompt_tokens) {
    if (token < 0) {
      std::cerr << "DeepSeek V4 Flash produced an invalid prompt token\n";
      return 1;
    }
    sampling_history.push_back(static_cast<sampling::TokenId>(token));
  }
  sampling::SamplerState sampler(opt.sampling, sampling_history);

  /*
   * DSpark drives generation only for greedy decoding. Its verification rule is
   * a greedy prefix match, so a sampled request keeps the ordinary decode path
   * rather than silently changing the distribution.
   */
  const bool use_dspark = dspark_requested && session.HasDspark() &&
                          opt.sampling.can_use_unmodified_argmax();
  if (dspark_requested && !use_dspark) {
    std::cerr << "[Speculative]: dspark needs greedy sampling; using "
                 "autoregressive decode\n";
  }

  const auto generation_start = std::chrono::steady_clock::now();
  std::size_t generated = 0;
  if (use_dspark) {
    std::vector<int> emitted;
    bool stop = false;
    while (generated < opt.max_tokens && !stop) {
      if (!session.DsparkStep(opt.max_tokens - generated, opt.draft_tokens,
                              &emitted, &error)) {
        std::cerr << "\nDeepSeek V4 Flash speculative decode failed: " << error
                  << '\n';
        return 1;
      }
      if (emitted.empty())
        break;
      for (const int token : emitted) {
        if (model->IsStopToken(token) || generated >= opt.max_tokens) {
          stop = true;
          break;
        }
        sampler.Accept(static_cast<sampling::TokenId>(token));
        emit(token);
        ++generated;
      }
    }
  } else {
    for (; generated < opt.max_tokens; ++generated) {
      const auto logits = session.CopyLogits(&error);
      if (logits.empty()) {
        std::cerr << "\nDeepSeek V4 Flash token selection failed: " << error
                  << '\n';
        return 1;
      }
      const int token = static_cast<int>(sampler.Sample(logits));
      if (model->IsStopToken(token)) {
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(token));
      emit(token);
      if (generated + 1 < opt.max_tokens && !session.Evaluate(token, &error)) {
        std::cerr << "\nDeepSeek V4 Flash decode failed: " << error << '\n';
        return 1;
      }
    }
  }
  std::cout << '\n';

  if (use_dspark) {
    const auto stats = session.DsparkStatistics();
    const double support_acceptance =
        stats.support_drafted != 0
            ? static_cast<double>(stats.support_accepted) /
                  static_cast<double>(stats.support_drafted)
            : 0.0;
    const double positional_acceptance =
        stats.support_drafted != 0
            ? static_cast<double>(stats.positional_accepted) /
                  static_cast<double>(stats.support_drafted)
            : 0.0;
    const double full_block_rate =
        stats.steps != 0 ? static_cast<double>(stats.full_blocks) /
                               static_cast<double>(stats.steps)
                         : 0.0;
    std::cerr << "[Speculative]: acceptance=" << support_acceptance
              << " drafted=" << stats.support_drafted
              << " accepted=" << stats.support_accepted
              << " verification_steps=" << stats.steps
              << " skipped=" << stats.skipped
              << " positional_acceptance=" << positional_acceptance
              << " positional_accepted=" << stats.positional_accepted
              << " full_block_rate=" << full_block_rate
              << " full_blocks=" << stats.full_blocks
              << " anchors=" << stats.anchors
              << " verifier_rows=" << stats.verifier_rows << '\n';
  }

  if (opt.verbose && generated > 0) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      generation_start)
            .count();
    std::cout << "Generated " << generated << " tokens on ROCm in " << seconds
              << "s (" << static_cast<double>(generated) / seconds
              << " tok/s)\n";
  }
  return 0;
}
int RunDeepSeekPrompt(const PromptOptions& opt, const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start) {
  auto model = LoadDeepSeekModel(opt, reader, load_start);
  if (!model)
    return 1;
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (!session) {
    std::cerr << "DeepSeek session creation failed: " << error << '\n';
    return 1;
  }
  std::vector<int> prompt_tokens;
  if (opt.use_chat_template) {
    const auto reasoning = PromptReasoningOptions(opt);
    const std::vector<models::deepseek_v4_flash::ChatMessage> messages = {
        {.role = "system",
         .content = opt.system_prompt,
         .reasoning_content = {},
         .tool_calls = {}},
        {.role = "user",
         .content = opt.prompt_text,
         .reasoning_content = {},
         .tool_calls = {}},
    };
    prompt_tokens = model->EncodeChat(
        messages,
        models::deepseek_v4_flash::ChatTemplateOptions{
            .enable_thinking = reasoning.enabled.value_or(false),
            .reasoning_effort =
                reasoning.effort.value_or(ReasoningEffort::kLow),
            .preserve_thinking = reasoning.preserve_thinking.value_or(false),
        });
  } else {
    prompt_tokens = model->Tokenize(opt.prompt_text);
  }
  return GenerateDeepSeekResponse(opt, model, *session, prompt_tokens);
}

int RunDeepSeekChat(const PromptOptions& opt, const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "DeepSeek interactive chat requires chat framing; use prompt "
                 "--raw for raw text\n";
    return 1;
  }
  auto model = LoadDeepSeekModel(opt, reader, load_start);
  if (!model)
    return 1;
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (!session) {
    std::cerr << "DeepSeek session creation failed: " << error << '\n';
    return 1;
  }
  std::vector<models::deepseek_v4_flash::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back({.role = "system",
                       .content = opt.system_prompt,
                       .reasoning_content = {},
                       .tool_calls = {}});
  }
  const auto reasoning = PromptReasoningOptions(opt);
  const models::deepseek_v4_flash::ChatTemplateOptions chat_options{
      .enable_thinking = reasoning.enabled.value_or(false),
      .reasoning_effort = reasoning.effort.value_or(ReasoningEffort::kLow),
      .preserve_thinking = reasoning.preserve_thinking.value_or(false),
  };
  std::cout << "=== Gufo Interactive Chat (DeepSeek V4 Flash) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit")
      break;
    if (input.empty())
      continue;
    history.push_back({.role = "user",
                       .content = input,
                       .reasoning_content = {},
                       .tool_calls = {}});
    const auto tokens = model->EncodeChat(history, chat_options);
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (GenerateDeepSeekResponse(opt, model, *session, tokens, &reply) != 0)
      return 1;
    models::deepseek_v4_flash::ChatMessage response{};
    response.role = "assistant";
    if (chat_options.enable_thinking) {
      constexpr std::string_view end = "</think>";
      const auto boundary = reply.find(end);
      response.reasoning_content = reply.substr(0, boundary);
      if (boundary != std::string::npos) {
        response.content = reply.substr(boundary + end.size());
      }
    } else {
      response.content = std::move(reply);
    }
    history.push_back(std::move(response));
    std::cout << '\n';
  }
  return 0;
}

#endif

}  // namespace

std::optional<PromptOptions> ParsePromptOptions(
    std::span<const char* const> args, std::string* error_msg) {
  PromptOptions opt;
  gufo::cli::ArgParser parser("gufo prompt",
                              "Execute one prompt request and exit.");
  // Model
  parser.AddOption("-m", "--model", "PATH", "Path to GGUF model file", "Model",
                   &opt.model_path);

  // Prompt & Formatting
  parser.AddOption("-p", "--prompt", "TEXT", "Direct input prompt text",
                   "Prompt", &opt.prompt_text);
  parser.AddOption("-f", "--file", "PATH",
                   "File path to read input prompt text from", "Prompt",
                   &opt.prompt_file);
  parser.AddOption("", "--system", "PROMPT",
                   "System role instructions prepended to the prompt "
                   "(default: helpful assistant)",
                   "Prompt", &opt.system_prompt);
  parser.AddInverseFlag("", "--raw",
                        "Disable chat template framing and pass raw tokens",
                        "Prompt", &opt.use_chat_template);
  parser.AddInverseFlag("", "--no-display-prompt",
                        "Suppress echoing the prompt before generated response",
                        "Prompt", &opt.display_prompt);

  // Sampling
  parser.AddOption("-n", "--max-tokens", "N",
                   "Maximum number of new tokens to generate (default: 128)",
                   "Sampling", &opt.max_tokens);
  RegisterSamplingOptions(parser, &opt.sampling);

  // Reasoning
  parser.AddOption("", "--think", "MODE",
                   "Reasoning mode: on, off, or auto (default: off)",
                   "Reasoning", &opt.reasoning_mode);
  parser.AddOption("", "--reasoning-effort", "LEVEL",
                   "Effort: auto, minimal, low, medium, high, xhigh, or max",
                   "Reasoning", &opt.reasoning_effort);
  parser.AddOption("", "--preserve-thinking", "MODE",
                   "Replay prior reasoning: on, off, or auto", "Reasoning",
                   &opt.preserve_thinking);

  // Speculative & Hardware
  bool speculative_explicit = false;
  const auto parse_speculative_backend =
      [&opt, &speculative_explicit](std::string_view, std::string_view value,
                                    std::string*) -> bool {
    speculative_explicit = true;
    if (value == "none" || value == "off" || value == "false" ||
        value == "disabled") {
      opt.speculative_backend.clear();
    } else {
      opt.speculative_backend = value;
    }
    return true;
  };
  parser.AddCustomOption(
      "", "--speculative", "MODE",
      "Draft backend: dspark (DeepSeek V4 Flash), dflash, dflash2, mtp, "
      "mtp-npu, npu, pld, self, or off",
      "Speculative", parse_speculative_backend);
  parser.AddCustomOption("", "--speculative-decoding", "MODE",
                         "Alias for --speculative", "Speculative",
                         parse_speculative_backend);
  parser.AddOption("", "--dflash-model", "PATH",
                   "Path to quantized Qwen DFlash/DFlash-2 GGUF file",
                   "Speculative", &opt.dflash_model_path);
  parser.AddOption("", "--dspark-model", "PATH",
                   "Path to the DeepSeek V4 Flash DSpark support GGUF file",
                   "Speculative", &opt.dspark_model_path);
  parser.AddOption("", "--mtp-model", "PATH",
                   "Path to quantized Qwen MTP draft head GGUF file",
                   "Speculative", &opt.mtp_model_path);
  parser.AddCustomOption(
      "-d", "--draft-tokens", "N",
      "Maximum speculative draft tokens evaluated per step (default: 7)",
      "Speculative",
      [&opt](std::string_view, std::string_view value,
             std::string* error) -> bool {
        std::uint32_t count = 0;
        const auto [ptr, ec] =
            std::from_chars(value.data(), value.data() + value.size(), count);
        if (ec != std::errc{} || ptr != value.data() + value.size() ||
            count == 0) {
          if (error != nullptr) {
            *error = "Invalid integer for draft-tokens: " + std::string(value);
          }
          return false;
        }
        opt.draft_tokens = count;
        return true;
      });

  parser.AddCustomOption(
      "", "--min-draft-tokens", "N", "Adaptive draft floor (default: 1)",
      "Speculative",
      [&opt](std::string_view, std::string_view value,
             std::string* error) -> bool {
        std::uint32_t count = 0;
        const auto [ptr, ec] =
            std::from_chars(value.data(), value.data() + value.size(), count);
        if (ec != std::errc{} || ptr != value.data() + value.size() ||
            count == 0) {
          if (error != nullptr) {
            *error =
                "Invalid integer for min-draft-tokens: " + std::string(value);
          }
          return false;
        }
        opt.min_draft_tokens = count;
        return true;
      });
  parser.AddOption("", "--spec-draft-n-max", "N",
                   "llama.cpp-compatible alias for --draft-tokens",
                   "Speculative", &opt.draft_tokens);
  parser.AddOption("", "--spec-draft-n-min", "N",
                   "llama.cpp-compatible alias for --min-draft-tokens",
                   "Speculative", &opt.min_draft_tokens);
  parser.AddOption(
      "", "--spec-draft-p-min", "P",
      "Stop at the first draft token below confidence P; 0 disables "
      "(default: 0)",
      "Speculative", &opt.draft_p_min);
  parser.AddOption("", "--draft-p-min", "P", "Alias for --spec-draft-p-min",
                   "Speculative", &opt.draft_p_min);
  parser.AddFlag("", "--cpu",
                 "Force CPU OpenMP execution fallback instead of GPU ROCm",
                 "Hardware", &opt.force_cpu);
  parser.AddFlag("-v", "--verbose",
                 "Print detailed timing, latency breakdown, and tok/s metrics",
                 "General", &opt.verbose);

  parser.JoinPositionals(&opt.prompt_text);

  if (!parser.Parse(args, error_msg)) {
    return std::nullopt;
  }
  if (parser.IsHelpRequested()) {
    return std::nullopt;
  }
  try {
    opt.sampling.Validate();
  } catch (const std::invalid_argument& exception) {
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return std::nullopt;
  }
  if (!speculative_explicit && !opt.dspark_model_path.empty()) {
    opt.speculative_backend = "dspark";
  }

  if (opt.draft_tokens == 0 || opt.min_draft_tokens == 0 ||
      opt.min_draft_tokens > opt.draft_tokens) {
    if (error_msg != nullptr) {
      *error_msg = "min-draft-tokens cannot exceed draft-tokens";
    }
    return std::nullopt;
  }
  if (!std::isfinite(opt.draft_p_min) || opt.draft_p_min < 0.0F ||
      opt.draft_p_min > 1.0F) {
    if (error_msg != nullptr) {
      *error_msg = "spec-draft-p-min must be in [0, 1]";
    }
    return std::nullopt;
  }
  if (opt.reasoning_mode != "on" && opt.reasoning_mode != "off" &&
      opt.reasoning_mode != "auto") {
    if (error_msg != nullptr) {
      *error_msg = "--think must be on, off, or auto";
    }
    return std::nullopt;
  }
  if (opt.reasoning_effort != "auto" &&
      !ParseReasoningEffort(opt.reasoning_effort).has_value()) {
    if (error_msg != nullptr) {
      *error_msg =
          "--reasoning-effort must be auto, minimal, low, medium, high, "
          "xhigh, or max";
    }
    return std::nullopt;
  }
  if (opt.reasoning_mode == "off" && opt.reasoning_effort != "auto") {
    if (error_msg != nullptr) {
      *error_msg = "--reasoning-effort cannot be set while --think is off";
    }
    return std::nullopt;
  }
  if (opt.preserve_thinking != "on" && opt.preserve_thinking != "off" &&
      opt.preserve_thinking != "auto") {
    if (error_msg != nullptr) {
      *error_msg = "--preserve-thinking must be on, off, or auto";
    }
    return std::nullopt;
  }
  return opt;
}

int RunPrompt(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParsePromptOptions(args, &parse_err);
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintPromptHelp("gufo");
      return 2;
    }
    PrintPromptHelp("gufo");
    return 0;
  }

  auto opt = *opt_res;

  if (!opt.prompt_file.empty()) {
    std::ifstream file(opt.prompt_file);
    if (!file.is_open()) {
      std::cerr << "Error: could not open prompt file '" << opt.prompt_file
                << "'\n";
      return 1;
    }
    std::string file_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    if (!opt.prompt_text.empty()) {
      opt.prompt_text = file_content + "\n" + opt.prompt_text;
    } else {
      opt.prompt_text = std::move(file_content);
    }
  }

  if (opt.prompt_text.empty() && opt.model_path.empty()) {
    PrintPromptHelp("gufo");
    return 0;
  }

  if (opt.model_path.empty()) {
    std::cout
        << "gufo prompt: prompt received: \"" << opt.prompt_text << "\"\n"
        << "(Specify --model <PATH.gguf> to execute local model generation)\n";
    return 0;
  }

  const auto model_load_start = std::chrono::steady_clock::now();
  std::string err;
  auto reader_owner = gufo::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader_owner) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const std::shared_ptr<const gufo::core::GgufReader> reader(
      std::move(reader_owner));

#if defined(ENGINE_ENABLE_HIP)
  if (IsDeepSeekV4Flash(*reader)) {
    return RunDeepSeekPrompt(opt, *reader, model_load_start);
  }
#else
  if (IsDeepSeekV4Flash(*reader)) {
    std::cerr << "DeepSeek V4 Flash requires ENGINE_ENABLE_HIP=ON\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
#endif

  std::string rendered_prompt = opt.prompt_text;
  if (opt.use_chat_template) {
    std::vector<tokenization::ChatMessage> messages;
    if (!opt.system_prompt.empty()) {
      messages.push_back(
          {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
    }
    messages.push_back(
        {tokenization::ChatRole::kUser, opt.prompt_text, "", ""});
    const auto reasoning = PromptReasoningOptions(opt);
    const auto rendered = tokenization::QwenChatTemplate::Render(
        messages,
        tokenization::ChatTemplateOptions{
            .add_generation_prompt = true,
            .enable_thinking = reasoning.enabled.value_or(false),
            .reasoning_effort = QwenEffort(reasoning.effort),
            .preserve_thinking = reasoning.preserve_thinking.value_or(true),
        });
    if (rendered.has_value()) {
      rendered_prompt = *rendered;
    }
  }

#if defined(ENGINE_ENABLE_HIP)
  int dev_count = 0;
  if (!opt.force_cpu && hipGetDeviceCount(&dev_count) == hipSuccess &&
      dev_count > 0) {
    auto gpu_exec = gufo::hip::QwenGpuExecutor::CreateFromGguf(reader, &err);
    if (gpu_exec) {
      PrintModelLoadTime(model_load_start);
      const auto prompt_tokens =
          gpu_exec->GetTokenizer().Encode(rendered_prompt);

      if (opt.verbose) {
        const auto& config = gpu_exec->GetConfig();
        std::cout << "[Engine]: AMD Strix Halo gfx1151 GPU Executor\n"
                  << "Model: " << config.architecture << " ("
                  << config.num_layers
                  << " layers, hidden=" << config.hidden_size
                  << ", heads=" << config.num_attention_heads << ")\n"
                  << "Prompt tokens (" << prompt_tokens.size() << "): ";
        for (const auto t : prompt_tokens) {
          std::cout << "[" << t << ": '"
                    << gpu_exec->GetTokenizer().DecodeToken(t) << "'] ";
        }
        std::cout << "\nMax tokens: " << opt.max_tokens << "\n"
                  << "--- Generation Output ---\n";
      }

      models::GenerationOptions gen_opts;
      gen_opts.max_new_tokens = opt.max_tokens;
      gen_opts.sampling = opt.sampling;

      auto start_time = std::chrono::steady_clock::now();
      std::size_t generated_count = 0;

      if (!opt.speculative_backend.empty()) {
        const auto& config = gpu_exec->GetConfig();
        std::unique_ptr<speculative::IDraftBackend> draft_backend;
        if (opt.speculative_backend == "dflash" ||
            opt.speculative_backend == "dflash2" ||
            opt.speculative_backend == "dflash-2") {
          std::string dflash_path = opt.dflash_model_path;
          if (dflash_path.empty()) {
            if (const char* environment = std::getenv("GUFO_DFLASH_MODEL");
                environment != nullptr) {
              dflash_path = environment;
            }
          }
          if (dflash_path.empty()) {
            dflash_path = opt.model_path;
          }
          hip::QwenDFlashGpuDraftConfig cfg{
              .max_context = gpu_exec->GetMaxContext(),
              .max_draft_tokens = static_cast<std::uint32_t>(opt.draft_tokens),
              .draft_p_min = opt.draft_p_min,
          };
          draft_backend = hip::QwenDFlashGpuDraftBackend::CreateFromGguf(
              dflash_path, gpu_exec->GetSharedModel(), cfg, &err);
          if (draft_backend == nullptr) {
            std::cerr << "Failed to initialize DFlash backend: " << err << '\n';
            return 1;
          }
        } else if (opt.speculative_backend == "npu") {
          heterogeneous::NpuDrafterConfig cfg;
          cfg.max_draft_tokens = opt.draft_tokens;
          cfg.vocab_size = config.vocab_size;
          draft_backend = std::make_unique<heterogeneous::NpuDraftBackend>(cfg);
        } else if (opt.speculative_backend == "pld" ||
                   opt.speculative_backend == "lookup") {
          speculative::PromptLookupConfig cfg;
          cfg.max_draft_tokens = opt.draft_tokens;
          draft_backend =
              std::make_unique<speculative::PromptLookupDraftBackend>(cfg);
        } else if (opt.speculative_backend == "mtp" ||
                   opt.speculative_backend == "mtp-npu") {
          std::string mtp_path = opt.mtp_model_path;
          if (mtp_path.empty()) {
            if (const char* environment = std::getenv("GUFO_MTP_MODEL");
                environment != nullptr) {
              mtp_path = environment;
            }
          }
          hip::QwenMtpGpuDraftConfig cfg{
              .max_context = gpu_exec->GetMaxContext(),
              .max_draft_tokens = static_cast<std::uint32_t>(opt.draft_tokens),
              .execution_mode =
                  opt.speculative_backend == "mtp-npu"
                      ? hip::QwenMtpExecutionMode::kHybridNpuEhProj
                      : hip::QwenMtpExecutionMode::kGpu,
          };
          draft_backend = hip::QwenMtpGpuDraftBackend::CreateFromGguf(
              mtp_path, gpu_exec->GetSharedModel(), cfg, &err);
          if (draft_backend == nullptr) {
            std::cerr << "Failed to initialize MTP backend: " << err << '\n';
            return 1;
          }
        } else if (opt.speculative_backend == "self") {
          speculative::SelfSpeculativeConfig cfg;
          cfg.total_layers = config.num_layers;
          cfg.exit_layer = std::max<std::uint32_t>(4U, config.num_layers / 4);
          cfg.draft_step_count = opt.draft_tokens;
          draft_backend =
              std::make_unique<speculative::SelfSpeculativeBackend>(cfg);
        } else {
          std::cerr << "Unknown speculative backend: "
                    << opt.speculative_backend << '\n';
          return 1;
        }

        if (draft_backend) {
          speculative::SpeculativeOptions s_opts;
          s_opts.max_draft_tokens = opt.draft_tokens;
          s_opts.min_draft_tokens = opt.min_draft_tokens;
          s_opts.initial_draft_tokens = opt.draft_tokens;
          const bool block_diffusion_draft =
              opt.speculative_backend == "dflash" ||
              opt.speculative_backend == "dflash2" ||
              opt.speculative_backend == "dflash-2";
          s_opts.enable_adaptive_draft_length = !block_diffusion_draft;
          if (opt.speculative_backend == "dflash" ||
              opt.speculative_backend == "dflash2" ||
              opt.speculative_backend == "dflash-2") {
            s_opts.use_batched_verification = true;
            s_opts.use_batched_lm_head = true;
            s_opts.target_bf16_from_layer = 48;
          } else if (opt.speculative_backend == "mtp" ||
                     opt.speculative_backend == "mtp-npu") {
            s_opts.use_batched_verification = true;
            s_opts.use_batched_lm_head = true;
            s_opts.target_bf16_from_layer = 0;
            s_opts.target_fp32_from_layer = 63;
          }
          speculative::SpeculativeVerifier spec_verifier(
              *gpu_exec, std::move(draft_backend), s_opts);

          start_time = std::chrono::steady_clock::now();
          (void)spec_verifier.Generate(
              prompt_tokens, gen_opts,
              [&](tokenization::TokenId, std::string_view piece) -> bool {
                std::cout << piece << std::flush;
                ++generated_count;
                return true;
              });
          if (opt.verbose) {
            const auto& stats = spec_verifier.GetStats();
            std::cerr << "\n[Speculative]: acceptance="
                      << stats.AcceptanceRate()
                      << " drafted=" << stats.total_draft_tokens
                      << " accepted=" << stats.total_accepted_tokens
                      << " verification_steps="
                      << stats.total_verification_steps << '\n';
          }
        }
      } else {
        start_time = std::chrono::steady_clock::now();
        gpu_exec->Generate(
            prompt_tokens, gen_opts,
            [&](tokenization::TokenId, std::string_view piece) -> bool {
              std::cout << piece << std::flush;
              ++generated_count;
              return true;
            });
      }

      std::cout << "\n";

      if (opt.verbose && generated_count > 0) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time);
        const double sec = static_cast<double>(elapsed.count()) / 1000.0;
        const double tok_per_sec =
            (sec > 0.0) ? (static_cast<double>(generated_count) / sec) : 0.0;
        std::cout << "Generated " << generated_count << " tokens on GPU in "
                  << sec << "s (" << tok_per_sec << " tok/s)\n";
      }
      return 0;
    }
  }
#endif

  auto generator = models::QwenGenerator::CreateFromGguf(*reader, &err);
  if (!generator) {
    std::cerr << "Error creating Qwen generator: " << err << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  const auto prompt_tokens = generator->GetTokenizer().Encode(rendered_prompt);

  if (opt.verbose) {
    const auto& config = generator->GetConfig();
    std::cout << "[Engine]: CPU (OpenMP Multi-Threaded)\n"
              << "Model: " << config.architecture << " (" << config.num_layers
              << " layers, hidden=" << config.hidden_size
              << ", heads=" << config.num_attention_heads << ")\n"
              << "Prompt tokens: " << prompt_tokens.size() << "\n"
              << "Max tokens: " << opt.max_tokens << "\n"
              << "Temperature: " << opt.sampling.temperature << "\n"
              << "--- Generation Output ---\n";
  }

  models::GenerationOptions gen_opts;
  gen_opts.max_new_tokens = opt.max_tokens;
  gen_opts.sampling = opt.sampling;

  const auto start_time = std::chrono::steady_clock::now();
  std::size_t generated_count = 0;

  const auto tokens = generator->Generate(
      prompt_tokens, gen_opts,
      [&](tokenization::TokenId, std::string_view piece) -> bool {
        std::cout << piece << std::flush;
        ++generated_count;
        return true;
      });

  std::cout << "\n";

  if (opt.verbose && generated_count > 0) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    const double sec = static_cast<double>(elapsed.count()) / 1000.0;
    const double tok_per_sec =
        (sec > 0.0) ? (static_cast<double>(generated_count) / sec) : 0.0;
    std::cout << "Generated " << generated_count << " tokens on CPU in " << sec
              << "s (" << tok_per_sec << " tok/s)\n";
  }

  return 0;
}

int RunChat(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParsePromptOptions(args, &parse_err);
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintChatHelp("gufo");
      return 2;
    }
    PrintChatHelp("gufo");
    return 0;
  }

  const auto& opt = *opt_res;
  if (opt.model_path.empty()) {
    std::cout << "gufo chat: interactive conversation mode\n"
              << "(Specify --model <PATH.gguf> to load model weights)\n";
    return 0;
  }

  const auto model_load_start = std::chrono::steady_clock::now();
  std::string err;
  const auto reader = gufo::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }

#if defined(ENGINE_ENABLE_HIP)
  if (IsDeepSeekV4Flash(*reader)) {
    return RunDeepSeekChat(opt, *reader, model_load_start);
  }
#endif

  auto generator = models::QwenGenerator::CreateFromGguf(*reader, &err);
  if (!generator) {
    std::cerr << "Error creating Qwen generator: " << err << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  std::cout << "=== Gufo Interactive Chat ("
            << generator->GetConfig().architecture << ") ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";

  std::vector<tokenization::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back(
        {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
  }

  std::string user_input;
  while (true) {
    std::cout << ">>> User: ";
    if (!std::getline(std::cin, user_input)) {
      break;
    }
    if (user_input == "exit" || user_input == "quit") {
      break;
    }
    if (user_input.empty()) {
      continue;
    }

    history.push_back({tokenization::ChatRole::kUser, user_input, "", ""});
    const auto reasoning = PromptReasoningOptions(opt);
    const auto rendered_prompt = tokenization::QwenChatTemplate::Render(
        history,
        tokenization::ChatTemplateOptions{
            .add_generation_prompt = true,
            .enable_thinking = reasoning.enabled.value_or(false),
            .reasoning_effort = QwenEffort(reasoning.effort),
            .preserve_thinking = reasoning.preserve_thinking.value_or(true),
        });
    if (!rendered_prompt.has_value()) {
      std::cerr << "Error formatting chat template.\n";
      continue;
    }

    const auto prompt_tokens =
        generator->GetTokenizer().Encode(*rendered_prompt);

    std::cout << "<<< Assistant: ";
    std::string assistant_reply;

    models::GenerationOptions gen_opts;
    gen_opts.max_new_tokens = opt.max_tokens > 0 ? opt.max_tokens : 256;
    gen_opts.sampling = opt.sampling;

    const auto reply_tokens = generator->Generate(
        prompt_tokens, gen_opts,
        [&](tokenization::TokenId, std::string_view piece) -> bool {
          std::cout << piece << std::flush;
          assistant_reply += piece;
          return true;
        });

    std::cout << "\n\n";
    history.push_back(
        {tokenization::ChatRole::kAssistant, assistant_reply, "", ""});
  }

  return 0;
}

}  // namespace gufo::cli
