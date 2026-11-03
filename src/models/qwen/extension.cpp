#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/qwen/probe.hpp"

#if !defined(ENGINE_ENABLE_HIP)
#include <chrono>
#include <iostream>

#include "src/families/qwen/text/chat_template.hpp"
#include "src/models/qwen/generator.hpp"

namespace gufo::models::qwen {
int RunQwenPromptCpu(const ext::TextPromptOptions& opt,
                     const core::GgufReader& reader,
                     std::chrono::steady_clock::time_point model_load_start) {
  std::string err;
  auto generator = models::QwenGenerator::CreateFromGguf(
      reader, &err, !opt.template_renderer);
  if (!generator) {
    std::cerr << "Error creating Qwen generator: " << err << "\n";
    return 1;
  }
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    model_load_start)
          .count();
  std::cout << "[Model Load]: " << load_seconds << " s\n";

  auto prompt_tokens = generator->GetTokenizer().Encode(opt.prompt_text);
  if (opt.template_renderer) {
    std::vector<tokenization::ChatMessage> messages;
    if (!opt.system_prompt.empty())
      messages.emplace_back(tokenization::ChatRole::kSystem, opt.system_prompt);
    messages.emplace_back(tokenization::ChatRole::kUser, opt.prompt_text);
    ReasoningOptions reasoning;
    if (opt.reasoning_mode != "auto")
      reasoning.enabled = opt.reasoning_mode == "on";
    if (opt.preserve_thinking != "auto")
      reasoning.preserve_thinking = opt.preserve_thinking == "on";
    for (const auto& [name, effort] :
         {std::pair{"minimal", ReasoningEffort::kMinimal},
          std::pair{"low", ReasoningEffort::kLow},
          std::pair{"medium", ReasoningEffort::kMedium},
          std::pair{"high", ReasoningEffort::kHigh},
          std::pair{"xhigh", ReasoningEffort::kXHigh},
          std::pair{"max", ReasoningEffort::kMax}}) {
      if (opt.reasoning_effort == name)
        reasoning.effort = effort;
    }
    auto options =
        tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id);
    options.jinja = opt.template_renderer;
    const auto tokens = tokenization::QwenChatTemplate::RenderAndTokenize(
        generator->GetTokenizer(), messages, {}, options, &err);
    if (!tokens) {
      std::cerr << "Error formatting chat template: " << err << '\n';
      return 1;
    }
    prompt_tokens = *tokens;
  }

  models::GenerationOptions gen_opts;
  gen_opts.max_new_tokens = opt.max_tokens;
  gen_opts.sampling = opt.sampling;

  const auto start = std::chrono::steady_clock::now();
  auto generated = generator->Generate(
      prompt_tokens, gen_opts,
      [](tokenization::TokenId, std::string_view token_str) {
        std::cout << token_str << std::flush;
        return true;
      });
  const double sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::cout << "\n\nGenerated " << generated.size() << " tokens on CPU in "
            << sec << "s (" << (sec > 0 ? generated.size() / sec : 0)
            << " tok/s)\n";
  return 0;
}
}  // namespace gufo::models::qwen

void qwen_register(gufo::ext::Registry& registry) {
  registry.AddTextProvider({
      .id = "qwen27-q4",
      .probe = &gufo::models::qwen::ProbeQwen,
      .probe_gguf = &gufo::models::qwen::ProbeQwenGguf,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = &gufo::models::qwen::RunQwenPromptCpu,
      .chat_driver = nullptr,
  });
}
#else
namespace gufo::models::qwen {}  // namespace gufo::models::qwen

void qwen_register(gufo::ext::Registry&) {}
#endif
