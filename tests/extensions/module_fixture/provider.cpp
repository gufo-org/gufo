#include <cstdint>
#include <gufo/ext/crypto.hpp>
#include <gufo/ext/registry.hpp>
#include <span>
#include <string_view>

#include "gufo/ext/generated/module_fixture_constants.hpp"

// Registers only when a core service resolves from the loading executable:
// the module carries no copy of it.
void module_fixture_provider_register(gufo::ext::Registry& registry) {
  constexpr std::string_view kText = "gufo";
  const auto digest = gufo::crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(kText.data()), kText.size()));
  if (digest.starts_with("1b443830")) {
    registry.AddProvider(gufo::ext::generated::extension_module_fixture::
                             kProvider_module_fixture);
  }
}
