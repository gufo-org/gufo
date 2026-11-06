#include <gufo/ext/embeddings.hpp>
#include <gufo/ext/generated/template_model_constants.hpp>
#include <gufo/ext/registry.hpp>

namespace {

void RegisterTemplateProvider(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_template_model::kProvider_template_model);
}

}  // namespace

void template_model_provider_register(gufo::ext::Registry& registry) {
  RegisterTemplateProvider(registry);
}

void template_model_register(gufo::ext::Registry& registry) {
  (void)registry;
}
