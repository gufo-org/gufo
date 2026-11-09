#ifndef GUFO_CLI_PROVIDER_OPTIONS_HPP_
#define GUFO_CLI_PROVIDER_OPTIONS_HPP_

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/options.hpp"
#include "gufo/ext/registry.hpp"
#include "src/cli/arg_parser.hpp"

namespace gufo::cli {

/// The options providers declare in their manifests for one command, from
/// parsing to delivery to the selected provider.
///
/// Options no core parser owns are registered on the command's parser, listed
/// under "Provider options" and parsed by `ext::ParseOptions`; the selected
/// provider must declare every one given. Declared spellings the command's
/// parser already owns (`--mtp-model`, ...) keep their core handling and are
/// not delivered.
class ProviderOptions {
public:
  ProviderOptions(
      const ext::Registry& registry, std::string command,
      ext::Capability capability = ext::Capability::kTextGeneration);
  ProviderOptions(const ProviderOptions&) = delete;
  ProviderOptions& operator=(const ProviderOptions&) = delete;

  /// Registers the declared options on `parser`; call after the core options.
  void Register(ArgParser& parser);
  /// Parses the captured values after `parser.Parse`. Throws
  /// std::invalid_argument for type errors and repeated options. Returns only
  /// explicit provider values, for commands that select a provider separately.
  const ext::ParsedOptions& Parse();
  /// The explicit values for `provider_id`. Throws std::invalid_argument when
  /// a given option belongs to another provider or another command.
  [[nodiscard]] ext::ParsedOptions Select(std::string_view provider_id) const;

private:
  struct Declared {
    ext::OptionType type;
    std::string help;
    std::vector<std::string> providers;
  };

  const ext::Registry& registry_;
  std::string command_;
  std::map<std::string, Declared, std::less<>> declared_;
  std::vector<std::string> tokens_;
  ext::ParsedOptions values_;
};

}  // namespace gufo::cli

#endif  // GUFO_CLI_PROVIDER_OPTIONS_HPP_
