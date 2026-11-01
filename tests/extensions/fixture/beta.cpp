#include "gufo/ext/generated/beta_constants.hpp"

void beta_register(gufo::ext::Registry&) {}
void beta_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_beta::kProvider_beta);
}
