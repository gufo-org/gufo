#include "src/cli/speculative_settings.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace gufo::cli {
namespace {

/// The host's settings for a backend no provider declares.
ext::TextSpeculativeDefaults HostDefaults(std::string_view backend) {
  ext::TextSpeculativeDefaults defaults{.backend = std::string(backend)};
  if (backend == "dflash2") {
    defaults.draft_policies = {"fixed", "adaptive"};
  }
  return defaults;
}

ext::TextSpeculativeDefaults DefaultsFor(const ext::TextProvider& provider,
                                         std::string_view backend) {
  const auto found = std::ranges::find(provider.speculative_defaults, backend,
                                       &ext::TextSpeculativeDefaults::backend);
  return found != provider.speculative_defaults.end() ? *found
                                                      : HostDefaults(backend);
}

bool Accepts(const ext::TextSpeculativeDefaults& defaults,
             std::string_view policy) {
  return std::ranges::find(defaults.draft_policies, policy) !=
         defaults.draft_policies.end();
}

void CheckRange(std::uint32_t max_draft_tokens,
                std::uint32_t min_draft_tokens) {
  if (min_draft_tokens > max_draft_tokens) {
    throw std::invalid_argument("min-draft-tokens cannot exceed draft-tokens");
  }
}

}  // namespace

SpeculativeFlags SpeculativeFlags::Given(const ArgParser& parser,
                                         std::uint32_t max_draft_tokens,
                                         std::uint32_t min_draft_tokens,
                                         const std::string& draft_policy) {
  SpeculativeFlags flags;
  if (parser.WasSupplied("--draft-tokens")) {
    flags.max_draft_tokens = max_draft_tokens;
  }
  if (parser.WasSupplied("--min-draft-tokens")) {
    flags.min_draft_tokens = min_draft_tokens;
  }
  if (parser.WasSupplied("--draft-policy")) {
    flags.draft_policy = draft_policy;
  }
  return flags;
}

void CheckSpeculativeFlags(const ext::Registry& registry,
                           std::string_view backend,
                           const SpeculativeFlags& flags) {
  if (flags.max_draft_tokens && flags.min_draft_tokens) {
    CheckRange(*flags.max_draft_tokens, *flags.min_draft_tokens);
  }
  if (!flags.draft_policy) {
    return;
  }
  const auto& policy = *flags.draft_policy;
  if (backend.empty()) {
    throw std::invalid_argument("--draft-policy requires --speculative");
  }
  bool accepted = Accepts(HostDefaults(backend), policy);
  for (const auto& info : registry.Providers()) {
    const auto* provider = ext::FindTextProvider(info.id);
    if (info.available && provider != nullptr) {
      accepted = accepted || Accepts(DefaultsFor(*provider, backend), policy);
    }
  }
  if (!accepted) {
    throw std::invalid_argument("--draft-policy " + policy +
                                " is not supported by --speculative " +
                                std::string(backend));
  }
}

SpeculativeSettings ResolveSpeculativeSettings(
    const ext::TextProvider& provider, std::string_view backend,
    const SpeculativeFlags& flags) {
  const auto defaults =
      backend.empty() ? HostDefaults(backend) : DefaultsFor(provider, backend);
  SpeculativeSettings settings{
      .max_draft_tokens =
          flags.max_draft_tokens.value_or(defaults.max_draft_tokens),
      .min_draft_tokens =
          flags.min_draft_tokens.value_or(defaults.min_draft_tokens),
      .draft_policy = flags.draft_policy.value_or(defaults.draft_policy),
  };
  if (flags.draft_policy &&
      (backend.empty() || !Accepts(defaults, *flags.draft_policy))) {
    throw std::invalid_argument(
        "provider " + provider.id + " does not support --draft-policy " +
        *flags.draft_policy + " for --speculative " + std::string(backend));
  }
  CheckRange(settings.max_draft_tokens, settings.min_draft_tokens);
  return settings;
}

ext::TextSpeculativeConfig SpeculativeConfigFor(
    std::string_view backend, const std::string& mtp_model_path,
    const std::string& dflash_model_path,
    const std::string& dspark_model_path) {
  ext::TextSpeculativeConfig config;
  if (backend.empty() || backend == "off") {
    return config;
  }
  if (backend == "mtp") {
    config.backend = ext::TextSpeculativeBackend::kMtp;
    config.draft_model_path = mtp_model_path;
  } else if (backend == "dflash2") {
    config.backend = ext::TextSpeculativeBackend::kDFlash;
    config.draft_model_path = dflash_model_path;
  } else if (backend == "dspark") {
    config.backend = ext::TextSpeculativeBackend::kDSpark;
    config.draft_model_path = dspark_model_path;
  } else {
    throw std::invalid_argument("speculative backend '" + std::string(backend) +
                                "' is not supported");
  }
  return config;
}

void RequireDraftModel(std::string_view backend,
                       const std::string& mtp_model_path,
                       const std::string& dflash_model_path,
                       const std::string& dspark_model_path) {
  const auto config = SpeculativeConfigFor(
      backend, mtp_model_path, dflash_model_path, dspark_model_path);
  if (config.backend == ext::TextSpeculativeBackend::kDisabled ||
      !config.draft_model_path.empty()) {
    return;
  }
  throw std::invalid_argument(backend == "mtp" ? "MTP requires --mtp-model"
                              : backend == "dflash2"
                                  ? "DFlash2 requires --dflash-model"
                                  : "DSpark requires --dspark-model");
}

void ApplySpeculativeSettings(const SpeculativeSettings& settings,
                              ext::TextSpeculativeConfig* config) {
  config->max_draft_tokens = settings.max_draft_tokens;
  config->min_draft_tokens = settings.min_draft_tokens;
  config->draft_policy = settings.draft_policy;
  if (config->backend == ext::TextSpeculativeBackend::kDFlash) {
    config->dflash_policy =
        ext::ParseTextDFlashDraftPolicy(settings.draft_policy);
  }
}

}  // namespace gufo::cli
