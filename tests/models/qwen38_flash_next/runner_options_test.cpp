#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "gufo/ext/generated/qwen38_flash_next_constants.hpp"
#include "src/models/qwen38_flash_next/runner.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

static_assert(qfn::DecodeConcurrencyForSessions(0) == 1);
static_assert(qfn::DecodeConcurrencyForSessions(1) == 1);
static_assert(qfn::DecodeConcurrencyForSessions(2) == 2);
static_assert(qfn::DecodeConcurrencyForSessions(4) == 4);
static_assert(qfn::DecodeConcurrencyForSessions(8) == 8);
static_assert(qfn::DecodeConcurrencyForSessions(9) == 8);
static_assert(qfn::DecodeConcurrencyForSessions(
                  std::numeric_limits<std::size_t>::max()) == 8);

namespace {
void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void RejectedVisionSettings(const gufo::ext::TextRunnerOptions& options,
                            std::string_view diagnostic) {
  bool rejected = false;
  try {
    (void)qfn::VisionMemoryOnDemand(options);
  } catch (const std::invalid_argument& error) {
    rejected = std::string_view(error.what()).find(diagnostic) !=
               std::string_view::npos;
  }
  Expect(rejected, "invalid vision settings are rejected");
}
}  // namespace

int main() {
  // The metadata is generated from the real provider declaration. Simulate
  // HIP availability without loading the provider or touching a device.
  gufo::ext::Registry registry;
  registry.BeginExtension({"qwen38_flash_next", "test", ""});
  auto provider = gufo::ext::generated::extension_qwen38_flash_next::
      kProvider_flash_next_q4;
  provider.available = true;
  provider.unavailable_reason.clear();
  registry.AddProvider(std::move(provider));
  gufo::ext::ProviderInfo unrelated;
  unrelated.id = "unrelated-provider";
  registry.AddProvider(std::move(unrelated));
  gufo::ext::TextRunnerOptions options;
  Expect(!qfn::VisionMemoryOnDemand(options),
         "default admission is conservative");
  options.vision_model_path = "projector.gguf";
  Expect(!qfn::VisionMemoryOnDemand(options),
         "projector presence does not enable on-demand admission");
  bool rejected = false;
  const std::array<std::string_view, 2> on_demand_arguments{
      "--vision-memory-policy", "on-demand"};
  auto parsed = gufo::ext::ParseOptions(registry, "serve", on_demand_arguments);
  gufo::ext::ValidateSelectedOptions(
      registry, "serve", registry.Provider("flash-next-q4"), parsed);
  options.provider_options = parsed;
  options.vision_model_path = "projector.gguf";
  Expect(qfn::VisionMemoryOnDemand(options),
         "explicit on-demand uses no unused fixed startup pool");
  options.vision_model_path.clear();
  RejectedVisionSettings(options, "--mmproj");
  options.vision_model_path = "projector.gguf";
  for (const auto policy : {"", "automatic", "fixed", "ON-DEMAND"}) {
    options.provider_options["--vision-memory-policy"] = std::string(policy);
    RejectedVisionSettings(options, "conservative or on-demand");
  }

  options.provider_options["--vision-memory-policy"] =
      std::string("conservative");
  options.vision_model_path.clear();
  Expect(!qfn::VisionMemoryOnDemand(options),
         "explicit conservative admission needs no vision projector");

  rejected = false;
  try {
    gufo::ext::ValidateSelectedOptions(
        registry, "serve", registry.Provider("unrelated-provider"), parsed);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected, "unsupported providers reject the on-demand policy");
  for (const auto command : {"prompt", "bench"}) {
    rejected = false;
    try {
      (void)gufo::ext::ParseOptions(registry, command, on_demand_arguments);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Expect(rejected, "on-demand policy is confined to the serving runner");
  }
}
