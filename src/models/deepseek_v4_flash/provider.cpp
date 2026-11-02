#include <gufo/ext/generated/deepseek_v4_flash_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/deepseek_v4_flash/probe.hpp"
#include "src/models/deepseek_v4_flash/runner.hpp"

namespace gufo::models::deepseek_v4_flash {

#if defined(ENGINE_ENABLE_HIP)
int RunDeepSeekBenchmark(
    const ext::TextBenchOptions& opt,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start);

int RunDeepSeekPrompt(const ext::TextPromptOptions& opt,
                      const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point model_load_start);

int RunDeepSeekChat(const ext::TextPromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point model_load_start);
#endif

}  // namespace gufo::models::deepseek_v4_flash

void deepseek_v4_flash_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_deepseek_v4_flash::kProvider_deepseek4);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "deepseek4",
      .probe = &gufo::models::deepseek_v4_flash::ProbeDeepSeek,
      .probe_gguf = &gufo::models::deepseek_v4_flash::ProbeDeepSeekGguf,
      .create = &gufo::models::deepseek_v4_flash::CreateDeepSeekTextRunner,
      .bench_driver = &gufo::models::deepseek_v4_flash::RunDeepSeekBenchmark,
      .prompt_driver = &gufo::models::deepseek_v4_flash::RunDeepSeekPrompt,
      .chat_driver = &gufo::models::deepseek_v4_flash::RunDeepSeekChat,
  });
#endif
}
