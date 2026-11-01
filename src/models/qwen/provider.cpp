#include <gufo/ext/generated/qwen_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/qwen/probe.hpp"
#include "src/models/qwen/runner.hpp"

namespace gufo::models::qwen {

#if defined(ENGINE_ENABLE_HIP)
int RunQwenBenchmark(const ext::TextBenchOptions& opt,
                     const std::shared_ptr<const core::GgufReader>& reader,
                     std::chrono::steady_clock::time_point model_load_start);

int RunQwenPrompt(const ext::TextPromptOptions& opt,
                  const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point model_load_start);

int RunQwenChat(const ext::TextPromptOptions& opt,
                const core::GgufReader& reader,
                std::chrono::steady_clock::time_point model_load_start);
#endif

}  // namespace gufo::models::qwen

void qwen_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_qwen::kProvider_qwen27_q4);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "qwen27-q4",
      .probe = &gufo::models::qwen::ProbeQwen,
      .probe_gguf = &gufo::models::qwen::ProbeQwenGguf,
      .create = &gufo::models::qwen::CreateQwenTextRunner,
      .bench_driver = &gufo::models::qwen::RunQwenBenchmark,
      .prompt_driver = &gufo::models::qwen::RunQwenPrompt,
      .chat_driver = &gufo::models::qwen::RunQwenChat,
  });
#endif
}
