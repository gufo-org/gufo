#include <cstdint>
#include <gufo/ext/crypto.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/registry.hpp>
#include <span>
#include <string_view>

#include "gufo/ext/generated/module_fixture_constants.hpp"

// Registers only when core services resolve from the loading executable: the
// module carries no copy of them.
void module_fixture_provider_register(gufo::ext::Registry& registry) {
  constexpr std::string_view kText = "gufo";
  const auto digest = gufo::crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(kText.data()), kText.size()));
  if (digest.starts_with("1b443830") &&
      gufo::json::parse(R"({"a":[1]})")["a"].size() == 1) {
    registry.AddProvider(gufo::ext::generated::extension_module_fixture::
                             kProvider_module_fixture);
  }
}
