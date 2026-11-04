#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace gufo::ext {
class Registry;
struct ProviderInfo;
enum class OptionType { kBoolean, kInteger, kNumber, kString, kPath };
using OptionValue = std::variant<bool, std::int64_t, double, std::string>;
struct OptionSpec {
  std::string name;
  OptionType type = OptionType::kString;
  std::vector<std::string> commands;
  std::string help;
};
using ParsedOptions = std::map<std::string, OptionValue, std::less<>>;

bool IsCoreOption(std::string_view name);
// True when the option is declared for the command. A declaration for the bare
// `serve` command applies to every `serve <modality>` command, matching the Nix
// serving helper; a modality-specific spelling applies only to that modality.
bool OptionApplies(const OptionSpec& option, std::string_view command);
// Parse only explicit values: provider defaults/validation run after selection.
// Boolean flags accept --flag, --flag=true and --flag=false. Other types accept
// --name=value or --name value. Duplicate options and positional tokens fail.
// Parsing the union never makes another provider's option valid on selection.
ParsedOptions ParseOptions(const Registry& registry, std::string_view command,
                           std::span<const std::string_view> arguments,
                           std::span<const OptionSpec> core_options = {});
void ValidateSelectedOptions(const Registry& registry, std::string_view command,
                             const ProviderInfo& provider,
                             const ParsedOptions& options);
}  // namespace gufo::ext
