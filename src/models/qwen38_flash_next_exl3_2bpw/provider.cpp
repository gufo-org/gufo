// Qwen3.8-Flash-Next from turboderp's EXL3 2.05 bpw checkpoint
// (2.05bpw_h4_ng4). The Flash-Next runtime binds the checkpoint directory,
// streams its n-gram table from disk and serves its MTP block for drafts.
#include <gufo/ext/generated/qwen38_flash_next_exl3_2bpw_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <string_view>

#include "src/models/qwen38_flash_next/probe.hpp"
#include "src/models/qwen38_flash_next/runner.hpp"

namespace {

bool Probe(std::string_view model_root) {
  return gufo::models::qwen38_flash_next::ProbeFlashNextExl3(model_root, 1.5,
                                                             2.5);
}

}  // namespace

void qwen38_flash_next_exl3_2bpw_provider_register(
    gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_qwen38_flash_next_exl3_2bpw::
          kProvider_flash_next_exl3_2bpw);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "flash-next-exl3-2bpw",
      .probe = &Probe,
      .create = &gufo::models::qwen38_flash_next::CreateQwenFlashNextTextRunner,
      .speculative_defaults = {{.backend = "mtp",
                                .max_draft_tokens = 7,
                                .min_draft_tokens = 1}},
  });
#endif
}
