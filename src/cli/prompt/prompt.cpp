#include "src/cli/prompt/prompt.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/registry.hpp"
#include "gufo/ext/text.hpp"
#include "src/cli/arg_parser.hpp"
#include "src/cli/sampling_options.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"

namespace gufo::cli {

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

void RegisterImageOptions(ArgParser& parser, PromptOptions& opt) {
  parser.AddFlag("", "--add-vision-id",
                 "Prefix image inputs with Picture N: in the chat template",
                 "Prompt", &opt.add_vision_id);
  parser.AddOption("", "--mmproj", "PATH",
                   "Qwen BF16 vision sidecar (auto-discovered beside model)",
                   "Model", &opt.vision_model_path);
  parser.AddCustomOption(
      "", "--image", "PATH",
      "PNG/JPEG attached before text in the first user turn; repeat for "
      "multiple images",
      "Prompt",
      [&opt](std::string_view, std::string_view value, std::string* error) {
        if (value.empty()) {
          if (error)
            *error = "--image requires a file path";
          return false;
        }
        opt.image_paths.emplace_back(value);
        return true;
      });
}

void RegisterTextOptions(ArgParser& parser, PromptOptions& opt,
                         bool& speculative_explicit, std::string_view command) {
  // Model
  parser.AddOption("-m", "--model", "PATH", "Path to GGUF model file", "Model",
                   &opt.model_path);
  RegisterImageOptions(parser, opt);

  // Prompt & Formatting
  if (command == "prompt") {
    parser.AddOption("-p", "--prompt", "TEXT", "Direct input prompt text",
                     "Prompt", &opt.prompt_text);
    parser.AddOption("-f", "--file", "PATH",
                     "File path to read input prompt text from", "Prompt",
                     &opt.prompt_file);
  }
  parser.AddOption("", "--system", "PROMPT",
                   "Optional system role instructions prepended to the prompt",
                   "Prompt", &opt.system_prompt);
  parser.AddOption("-n", "--max-tokens", "N",
                   "Maximum tokens to generate (default: 128)", "Sampling",
                   &opt.max_tokens);

  RegisterSamplingOptions(parser, &opt.sampling, "Sampling", true, true);

  parser.AddOption("", "--think", "MODE",
                   "Reasoning mode: on, off, or auto (default: model)",
                   "Reasoning", &opt.reasoning_mode);
  parser.AddOption("", "--reasoning-effort", "auto|minimal|low|medium|high",
                   "Reasoning effort budget selector (default: auto)",
                   "Reasoning", &opt.reasoning_effort);
  parser.AddOption("", "--preserve-thinking", "auto|off|on",
                   "Preserve thinking trace tokens across turns "
                   "(default: auto)",
                   "Reasoning", &opt.preserve_thinking);

  if (command == "prompt") {
    parser.AddInverseFlag("", "--raw",
                          "Disable chat template framing and pass raw tokens",
                          "Prompt", &opt.use_chat_template);
    parser.AddInverseFlag(
        "", "--no-display-prompt",
        "Suppress echoing the prompt before generated response", "Prompt",
        &opt.display_prompt);
  }

  // Speculative
  parser.AddCustomOption(
      "", "--speculative", "none|mtp|dflash2|dspark",
      "Speculative decoding backend (default: none)", "Speculative",
      [&opt, &speculative_explicit](std::string_view flag, std::string_view val,
                                    std::string* error) -> bool {
        speculative_explicit = true;
        if (val == "none" || val == "off") {
          opt.speculative_backend.clear();
          return true;
        }
        if (val == "mtp" || val == "dflash2" || val == "dspark") {
          opt.speculative_backend = std::string(val);
          return true;
        }
        if (error) {
          *error = "Unknown speculative backend '" + std::string(val) +
                   "' for " + std::string(flag) +
                   " (must be 'none', 'off', 'mtp', 'dflash2', or 'dspark')";
        }
        return false;
      });

  parser.AddOption("", "--mtp-model", "PATH",
                   "Path to GGUF model containing MTP draft weights",
                   "Speculative", &opt.mtp_model_path);

  parser.AddOption("", "--dflash-model", "PATH",
                   "Path to GGUF model containing DFlash draft weights",
                   "Speculative", &opt.dflash_model_path);

  parser.AddOption("", "--draft-policy", "POLICY",
                   "DFlash2 draft length policy: fixed or adaptive "
                   "(default: adaptive)",
                   "Speculative", &opt.draft_policy);

  parser.AddOption("", "--dspark-model", "PATH",
                   "Path to GGUF model containing DSpark support weights",
                   "Speculative", &opt.dspark_model_path);

  parser.AddOption("", "--draft-tokens", "N",
                   "Maximum draft tokens per speculative step (default: 7)",
                   "Speculative", &opt.draft_tokens);

  parser.AddOption("", "--min-draft-tokens", "N",
                   "Minimum draft tokens per speculative step (default: 1)",
                   "Speculative", &opt.min_draft_tokens);

  // Execution
  parser.AddFlag("-v", "--verbose", "Enable verbose diagnostic output",
                 "General", &opt.verbose);
  parser.AddFlag("", "--cpu", "Force CPU execution even when GPU is available",
                 "Execution", &opt.force_cpu);

  if (command == "prompt") {
    parser.JoinPositionals(&opt.prompt_text);
  }
}

std::optional<PromptOptions> ParseTextOptions(std::span<const char* const> args,
                                              std::string* error_msg,
                                              std::string_view command) {
  ArgParser parser("gufo " + std::string(command),
                   command == "chat" ? "Interactive conversation."
                                     : "Execute one prompt request and exit.");
  PromptOptions opt;
  bool speculative_explicit = false;
  RegisterTextOptions(parser, opt, speculative_explicit, command);

  if (!parser.Parse(args, error_msg)) {
    return std::nullopt;
  }
  if (parser.IsHelpRequested()) {
    return std::nullopt;
  }
  opt.sampling_supplied = SamplingOptionsSupplied(parser);
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

  if (opt.force_cpu && !opt.speculative_backend.empty()) {
    if (error_msg != nullptr) {
      *error_msg =
          "speculative decoding requires the ROCm backend; remove --cpu";
    }
    return std::nullopt;
  }

  if (opt.speculative_backend == "mtp" && opt.mtp_model_path.empty()) {
    if (error_msg != nullptr)
      *error_msg = "MTP requires --mtp-model";
    return std::nullopt;
  }
  if (opt.speculative_backend == "dflash2" && opt.dflash_model_path.empty()) {
    if (error_msg != nullptr)
      *error_msg = "DFlash2 requires --dflash-model";
    return std::nullopt;
  }
  if (opt.speculative_backend == "dspark" && opt.dspark_model_path.empty()) {
    if (error_msg != nullptr)
      *error_msg = "DSpark requires --dspark-model";
    return std::nullopt;
  }
  if (opt.draft_tokens == 0 || opt.min_draft_tokens == 0 ||
      opt.min_draft_tokens > opt.draft_tokens) {
    if (error_msg != nullptr) {
      *error_msg = "min-draft-tokens cannot exceed draft-tokens";
    }
    return std::nullopt;
  }

  if (!opt.draft_policy.empty() &&
      ((opt.draft_policy != "fixed" && opt.draft_policy != "adaptive") ||
       opt.speculative_backend != "dflash2")) {
    if (error_msg != nullptr)
      *error_msg = "--draft-policy requires DFlash2 and fixed or adaptive";
    return std::nullopt;
  }
  if (opt.min_draft_tokens != 1 && opt.speculative_backend == "dflash2") {
    if (error_msg != nullptr)
      *error_msg =
          "DFlash2 requires --min-draft-tokens 1; bound blocks with "
          "--draft-tokens";
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

}  // namespace

static void PrintTextHelp(std::string_view program_name,
                          std::string_view command) {
  PromptOptions opt;
  bool speculative_explicit = false;
  ArgParser parser(std::string(program_name) + " " + std::string(command),
                   command == "chat" ? "Interactive conversation."
                                     : "Execute one prompt request and exit.");
  RegisterTextOptions(parser, opt, speculative_explicit, command);
  parser.PrintHelp();
}

void PrintPromptHelp(std::string_view program_name) {
  PrintTextHelp(program_name, "prompt");
}

void PrintChatHelp(std::string_view program_name) {
  PrintTextHelp(program_name, "chat");
}

std::optional<PromptOptions> ParsePromptOptions(
    std::span<const char* const> args, std::string* error_msg) {
  return ParseTextOptions(args, error_msg, "prompt");
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

  ext::Registry registry;
  ext::RegisterBuiltExtensions(registry);
  const auto* provider = ext::ProbeTextProvider(*reader);
  if (provider == nullptr || provider->prompt_driver == nullptr) {
    std::cerr << "Error: unsupported model architecture for prompt\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  return provider->prompt_driver(opt, *reader, model_load_start);
}

int RunChat(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParseTextOptions(args, &parse_err, "chat");
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintChatHelp("gufo");
      return 2;
    }
    PrintChatHelp("gufo");
    return 0;
  }

  auto opt = *opt_res;
  if (opt.model_path.empty()) {
    std::cout << "gufo chat: interactive conversation mode\n"
              << "(Specify --model <PATH.gguf> to load model weights)\n";
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

  ext::Registry registry;
  ext::RegisterBuiltExtensions(registry);
  const auto* provider = ext::ProbeTextProvider(*reader);
  if (provider == nullptr || provider->chat_driver == nullptr) {
    std::cerr << "Error: unsupported model architecture for chat\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  return provider->chat_driver(opt, *reader, model_load_start);
}

}  // namespace gufo::cli
