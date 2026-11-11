#include <gufo/ext/generated/qwen38_flash_next_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/qwen38_flash_next/drivers.hpp"
#include "src/models/qwen38_flash_next/probe.hpp"
#include "src/models/qwen38_flash_next/runner.hpp"

void qwen38_flash_next_gyro_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_qwen38_flash_next::
          kProvider_flash_next_gyro_m);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "flash-next-gyro-m",
      .probe = &gufo::models::qwen38_flash_next::ProbeFlashNextGyro,
      .probe_gguf = &gufo::models::qwen38_flash_next::ProbeFlashNextGyroGguf,
      .create = &gufo::models::qwen38_flash_next::CreateQwenFlashNextTextRunner,
      .bench_driver =
          &gufo::models::qwen38_flash_next::RunQwen38FlashNextBenchmark,
      .prompt_driver = &gufo::models::qwen38_flash_next::RunFlashNextPrompt,
      .chat_driver = &gufo::models::qwen38_flash_next::RunFlashNextChat,
      .sampling_defaults =
          &gufo::models::qwen38_flash_next::FlashNextSamplingDefaults,
      // TQ2_T expert rows are this extension's storage layout: core maps them
      // without sizing them, and the trellis kernels read them directly.
      .accepts_unsized_tensors = true,
      .speculative_defaults = {{.backend = "mtp",
                                .max_draft_tokens = 7,
                                .min_draft_tokens = 1}},
  });
#endif
}
