#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gufo/ext/crypto.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/safetensors.hpp>
#include <span>
#include <string_view>

#include "gufo/ext/generated/module_fixture_constants.hpp"

// Registers only when core services resolve from the loading executable: the
// module carries no copy of them.
void module_fixture_provider_register(gufo::ext::Registry& registry) {
  constexpr std::string_view kText = "gufo";
  const auto digest = gufo::crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(kText.data()), kText.size()));
  constexpr std::string_view kHeader =
      R"({"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})";
  std::array<std::byte, 8 + kHeader.size() + 1> file{};
  file[0] = static_cast<std::byte>(kHeader.size());
  std::memcpy(file.data() + 8, kHeader.data(), kHeader.size());
  if (digest.starts_with("1b443830") &&
      gufo::json::parse(R"({"a":[1]})")["a"].size() == 1 &&
      gufo::safetensors::Parse(file).tensors.size() == 1) {
    registry.AddProvider(gufo::ext::generated::extension_module_fixture::
                             kProvider_module_fixture);
  }
}
