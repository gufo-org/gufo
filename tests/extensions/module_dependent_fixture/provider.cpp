#include <gufo/ext/families/qwen/models/flash_next/weights.hpp>
#include <gufo/ext/families/qwen/text/chat_template.hpp>
#include <gufo/ext/registry.hpp>

#include "gufo/ext/generated/module_dependent_fixture_constants.hpp"

// Registers only when the dependency's and the family's code resolve from the
// loading executable: the module links neither library.
void module_dependent_fixture_provider_register(gufo::ext::Registry& registry) {
  namespace flash_next = gufo::models::qwen38_flash_next;
  if (flash_next::ExpandedQ8_0Tensors(flash_next::LayerWeights{}) == 0 &&
      gufo::tokenization::GenerationPrompt(false).find("assistant") !=
          std::string_view::npos) {
    registry.AddProvider(
        gufo::ext::generated::extension_module_dependent_fixture::
            kProvider_module_dependent_fixture);
  }
}
