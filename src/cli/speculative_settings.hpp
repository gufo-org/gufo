#ifndef GUFO_CLI_SPECULATIVE_SETTINGS_HPP_
#define GUFO_CLI_SPECULATIVE_SETTINGS_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "gufo/ext/registry.hpp"
#include "gufo/ext/text.hpp"
#include "src/cli/arg_parser.hpp"

namespace gufo::cli {

/// The draft settings given on the command line; unset ones take the selected
/// provider's `TextSpeculativeDefaults`.
struct SpeculativeFlags {
  std::optional<std::uint32_t> max_draft_tokens;
  std::optional<std::uint32_t> min_draft_tokens;
  std::optional<std::string> draft_policy;

  /// The values of `--draft-tokens`, `--min-draft-tokens` and
  /// `--draft-policy` that `parser` was given.
  static SpeculativeFlags Given(const ArgParser& parser,
                                std::uint32_t max_draft_tokens,
                                std::uint32_t min_draft_tokens,
                                const std::string& draft_policy);
};

struct SpeculativeSettings {
  std::uint32_t max_draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  std::string draft_policy;
};

/// Before a model is selected: throws std::invalid_argument when the flags
/// cannot hold for any available provider (a policy without `backend`, or one
/// no provider accepts for it; a minimum above an explicit maximum).
void CheckSpeculativeFlags(const ext::Registry& registry,
                           std::string_view backend,
                           const SpeculativeFlags& flags);

/// The settings `provider` runs `backend` with. Throws std::invalid_argument
/// when it does not accept the given policy or the minimum exceeds the limit.
/// An empty backend (no speculation) resolves to the host's values.
SpeculativeSettings ResolveSpeculativeSettings(
    const ext::TextProvider& provider, std::string_view backend,
    const SpeculativeFlags& flags);

/// The configuration a runner loads with for `backend` ("" or "off" disables
/// speculation), drafting from the model the backend's own option names; the
/// draft settings stay the host's until `ApplySpeculativeSettings`. Throws
/// std::invalid_argument for an unknown backend or a missing draft model.
ext::TextSpeculativeConfig SpeculativeConfigFor(
    std::string_view backend, const std::string& mtp_model_path,
    const std::string& dflash_model_path, const std::string& dspark_model_path);

/// Stores `settings` in `config`, including the DFlash2 policy they name.
void ApplySpeculativeSettings(const SpeculativeSettings& settings,
                              ext::TextSpeculativeConfig* config);

}  // namespace gufo::cli

#endif  // GUFO_CLI_SPECULATIVE_SETTINGS_HPP_
