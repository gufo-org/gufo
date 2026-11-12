#include "src/cli/provider_options.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace gufo::cli {
namespace {

std::string_view ValueHint(ext::OptionType type) {
  switch (type) {
    case ext::OptionType::kInteger:
      return "N";
    case ext::OptionType::kNumber:
      return "X";
    case ext::OptionType::kPath:
      return "PATH";
    default:
      return "VALUE";
  }
}

}  // namespace

ProviderOptions::ProviderOptions(const ext::Registry& registry,
                                 std::string command,
                                 ext::Capability capability)
    : registry_(registry), command_(std::move(command)) {
  // Options for another capability belong to its own command path.
  for (const auto& provider : registry_.Providers()) {
    if (!provider.available || provider.capability != capability) {
      continue;
    }
    for (const auto& option : provider.options) {
      if (!ext::OptionApplies(option, command_)) {
        continue;
      }
      auto& declared = declared_[option.name];
      declared.type = option.type;
      if (declared.help.empty()) {
        declared.help = option.help;
      }
      declared.providers.push_back(provider.id);
    }
  }
}

void ProviderOptions::Register(ArgParser& parser) {
  for (const auto& [name, declared] : declared_) {
    if (parser.FindOption(name) != nullptr) {
      continue;  // core-owned spelling: core parses and applies it
    }
    std::string providers;
    for (const auto& id : declared.providers) {
      providers += (providers.empty() ? "" : ", ") + id;
    }
    const std::string description =
        declared.help.empty() ? "(" + providers + ")"
                              : declared.help + " (" + providers + ")";
    const bool boolean = declared.type == ext::OptionType::kBoolean;
    // A bare boolean flag is `--name`; every other value is passed inline so
    // ext::ParseOptions never reads a following token as its value.
    auto capture = [this, &name, boolean](
                       std::string_view, std::string_view value, std::string*) {
      tokens_.push_back(
          value.empty() && boolean ? name : name + "=" + std::string(value));
      return true;
    };
    if (boolean) {
      parser.AddValueFlag(name, description, "Provider options", capture);
    } else {
      parser.AddCustomOption("", name, ValueHint(declared.type), description,
                             "Provider options", capture);
    }
  }
}

const ext::ParsedOptions& ProviderOptions::Parse() {
  const std::vector<std::string_view> tokens(tokens_.begin(), tokens_.end());
  values_ = ext::ParseOptions(registry_, command_, tokens);
  return values_;
}

ext::ParsedOptions ProviderOptions::Select(std::string_view provider_id) const {
  if (values_.empty()) {
    return {};
  }
  const auto providers = registry_.Providers();
  const auto found =
      std::ranges::find(providers, provider_id, &ext::ProviderInfo::id);
  if (found == providers.end()) {
    throw std::invalid_argument("provider " + std::string(provider_id) +
                                " does not support " + values_.begin()->first +
                                " for " + command_);
  }
  ext::ValidateSelectedOptions(registry_, command_, *found, values_);
  return values_;
}

}  // namespace gufo::cli
