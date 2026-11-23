#include <gufo/ext/generated/inkling2_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/inkling2/probe.hpp"
#include "src/models/inkling2/runner.hpp"

void inkling2_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_inkling2::kProvider_inkling2);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "inkling2",
      .probe = &gufo::models::inkling2::ProbeInkling2,
      .probe_gguf = &gufo::models::inkling2::ProbeInkling2Gguf,
      .create = &gufo::models::inkling2::CreateInkling2TextRunner,
      // NVFP4 (ggml type 40) is unsized to core; the extension sizes it.
      .accepts_unsized_tensors = true,
      // The verification width is the runner's own (adaptive) choice, up to
      // the drafter's block of seven proposals.
      .speculative_defaults = {{.backend = "dspark",
                                .max_draft_tokens = 7,
                                .min_draft_tokens = 1}},
  });
#endif
}
