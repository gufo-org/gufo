#include "gufo/ext/options.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "gufo/ext/registry.hpp"

namespace gufo::ext {
namespace {
bool Applies(const OptionSpec& option, std::string_view command) {
  return std::ranges::find(option.commands, command) != option.commands.end();
}
OptionValue ParseValue(OptionType type, std::string_view value,
                       const std::string& name) {
  if (type == OptionType::kString || type == OptionType::kPath) {
    return std::string(value);
  }
  if (type == OptionType::kBoolean) {
    if (value == "true")
      return true;
    if (value == "false")
      return false;
  } else if (type == OptionType::kInteger) {
    std::int64_t result{};
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), result);
    if (error == std::errc{} && end == value.data() + value.size())
      return result;
  } else if (type == OptionType::kNumber) {
    double result{};
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), result);
    if (error == std::errc{} && end == value.data() + value.size() &&
        std::isfinite(result))
      return result;
  }
  throw std::invalid_argument("invalid value for " + name + ": " +
                              std::string(value));
}
}  // namespace
bool IsCoreOption(std::string_view name) {
#define GUFO_CORE_OPTION(value) \
  if (name == value)            \
    return true;
#include "gufo/ext/core_options.def"
#undef GUFO_CORE_OPTION
  return false;
}
ParsedOptions ParseOptions(const Registry& registry, std::string_view command,
                           std::span<const std::string_view> arguments,
                           std::span<const OptionSpec> core_options) {
  std::map<std::string, OptionType, std::less<>> specs;
  auto add = [&](const OptionSpec& option) {
    if (!Applies(option, command))
      return;
    const auto [it, inserted] = specs.emplace(option.name, option.type);
    if (!inserted && it->second != option.type) {
      throw std::invalid_argument("conflicting option types: " + option.name);
    }
  };
  for (const auto& option : core_options) {
    if (!IsCoreOption(option.name)) {
      throw std::invalid_argument("unknown core option: " + option.name);
    }
    add(option);
  }
  // Provider-valued controls remain core-owned and parse without a model load.
  specs.emplace("--provider", OptionType::kString);
  specs.emplace("--speculative", OptionType::kString);
  for (const auto& provider : registry.Providers()) {
    if (provider.available) {
      for (const auto& option : provider.options)
        add(option);
    }
  }
  ParsedOptions parsed;
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    auto argument = arguments[index];
    const auto equals = argument.find('=');
    const std::string name(argument.substr(0, equals));
    const auto spec = specs.find(name);
    if (spec == specs.end())
      throw std::invalid_argument("unknown option: " + name);
    std::string_view value;
    if (equals != argument.npos) {
      value = argument.substr(equals + 1);
    } else if (spec->second == OptionType::kBoolean) {
      value = "true";
    } else {
      if (++index == arguments.size() || arguments[index].starts_with("--")) {
        throw std::invalid_argument("missing value for " + name);
      }
      value = arguments[index];
    }
    if (!parsed.emplace(name, ParseValue(spec->second, value, name)).second) {
      throw std::invalid_argument("duplicate option: " + name);
    }
  }
  if (const auto found = parsed.find("--provider"); found != parsed.end()) {
    const auto& selected =
        registry.Provider(std::get<std::string>(found->second));
    if (!selected.available) {
      throw std::invalid_argument("provider " + selected.id + ": " +
                                  selected.unavailable_reason);
    }
  }
  if (const auto found = parsed.find("--speculative"); found != parsed.end()) {
    const auto& value = std::get<std::string>(found->second);
    const bool supported =
        value == "off" ||
        std::ranges::any_of(registry.Providers(), [&](const auto& provider) {
          return provider.available &&
                 std::ranges::find(provider.speculative, value) !=
                     provider.speculative.end();
        });
    if (!supported)
      throw std::invalid_argument("unknown --speculative value: " + value);
  }
  return parsed;
}
void ValidateSelectedOptions(const Registry& registry, std::string_view command,
                             const ProviderInfo& provider,
                             const ParsedOptions& options) {
  // Verify that the selection came from this registry, rather than accepting
  // invented capabilities on a caller-authored ProviderInfo copy.
  const auto& selected = registry.Provider(provider.id);
  if (!selected.available) {
    throw std::invalid_argument("provider " + selected.id + ": " +
                                selected.unavailable_reason);
  }
  for (const auto& [name, value] : options) {
    if (name == "--provider" && std::get<std::string>(value) != selected.id) {
      throw std::invalid_argument(
          "--provider does not match selected provider " + selected.id);
    }
    if (name == "--speculative") {
      const auto& spelling = std::get<std::string>(value);
      if (spelling != "off" &&
          std::ranges::find(selected.speculative, spelling) ==
              selected.speculative.end()) {
        throw std::invalid_argument(
            "provider " + selected.id +
            " does not support --speculative=" + spelling);
      }
    }
    if (!IsCoreOption(name) &&
        !std::ranges::any_of(selected.options, [&](const auto& option) {
          return option.name == name && Applies(option, command);
        })) {
      throw std::invalid_argument("provider " + selected.id +
                                  " does not support " + name + " for " +
                                  std::string(command));
    }
  }
}
}  // namespace gufo::ext
