#ifndef STRIX_CLI_ARG_PARSER_HPP_
#define STRIX_CLI_ARG_PARSER_HPP_

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace strix::cli {

/// A lightweight, type-safe C++20 command-line argument parser.
class ArgParser {
public:
  struct Option {
    std::string short_name;
    std::string long_name;
    std::string value_hint;
    std::string description;
    std::string group;
    bool is_flag = false;
    std::function<bool(std::string_view, std::string_view, std::string*)>
        parse_fn;
  };

  explicit ArgParser(std::string_view program_name,
                     std::string_view description = "")
      : program_name_(program_name), description_(description) {}

  /// Register a boolean switch flag (e.g. `--verbose` or `-v`).
  ArgParser& AddFlag(std::string_view short_name, std::string_view long_name,
                     std::string_view description, std::string_view group,
                     bool* target) {
    Option opt;
    opt.short_name = std::string(short_name);
    opt.long_name = std::string(long_name);
    opt.description = std::string(description);
    opt.group = std::string(group);
    opt.is_flag = true;
    opt.parse_fn = [target](std::string_view, std::string_view,
                            std::string*) -> bool {
      if (target != nullptr) {
        *target = true;
      }
      return true;
    };
    options_.push_back(std::move(opt));
    return *this;
  }

  /// Register a boolean flag that sets false (e.g. `--raw` or
  /// `--no-something`).
  ArgParser& AddInverseFlag(std::string_view short_name,
                            std::string_view long_name,
                            std::string_view description,
                            std::string_view group, bool* target) {
    Option opt;
    opt.short_name = std::string(short_name);
    opt.long_name = std::string(long_name);
    opt.description = std::string(description);
    opt.group = std::string(group);
    opt.is_flag = true;
    opt.parse_fn = [target](std::string_view, std::string_view,
                            std::string*) -> bool {
      if (target != nullptr) {
        *target = false;
      }
      return true;
    };
    options_.push_back(std::move(opt));
    return *this;
  }

  /// Register a typed option (e.g. `--model <PATH>`, `--temperature <T>`,
  /// etc.).
  template<typename T>
  ArgParser& AddOption(std::string_view short_name, std::string_view long_name,
                       std::string_view value_hint,
                       std::string_view description, std::string_view group,
                       T* target) {
    Option opt;
    opt.short_name = std::string(short_name);
    opt.long_name = std::string(long_name);
    opt.value_hint = std::string(value_hint);
    opt.description = std::string(description);
    opt.group = std::string(group);
    opt.is_flag = false;
    opt.parse_fn = [target](std::string_view flag_name, std::string_view val,
                            std::string* err) -> bool {
      return ParseValue(val, target, std::string(flag_name), err);
    };
    options_.push_back(std::move(opt));
    return *this;
  }

  /// Register a custom parse callback option.
  ArgParser& AddCustomOption(
      std::string_view short_name, std::string_view long_name,
      std::string_view value_hint, std::string_view description,
      std::string_view group,
      std::function<bool(std::string_view, std::string_view, std::string*)>
          custom_parser) {
    Option opt;
    opt.short_name = std::string(short_name);
    opt.long_name = std::string(long_name);
    opt.value_hint = std::string(value_hint);
    opt.description = std::string(description);
    opt.group = std::string(group);
    opt.is_flag = false;
    opt.parse_fn = std::move(custom_parser);
    options_.push_back(std::move(opt));
    return *this;
  }

  /// Set the positional arguments receiver.
  ArgParser& SetPositionalHandler(
      std::function<bool(std::string_view, std::string*)> handler) {
    positional_handler_ = std::move(handler);
    return *this;
  }

  /// Convenience: collect all positional arguments into a
  /// std::vector<std::string>.
  ArgParser& CollectPositionals(std::vector<std::string>* target) {
    positional_handler_ = [target](std::string_view val, std::string*) {
      if (target != nullptr) {
        target->emplace_back(val);
      }
      return true;
    };
    return *this;
  }

  /// Convenience: join all positional arguments with spaces into a std::string.
  ArgParser& JoinPositionals(std::string* target) {
    positional_handler_ = [target](std::string_view val, std::string*) {
      if (target != nullptr) {
        if (!target->empty()) {
          target->push_back(' ');
        }
        target->append(val);
      }
      return true;
    };
    return *this;
  }

  /// Parse the supplied arguments span. Returns true on success.
  [[nodiscard]] bool Parse(std::span<const char* const> args,
                           std::string* error_msg = nullptr) const {
    for (std::size_t i = 0; i < args.size(); ++i) {
      const std::string_view arg = args[i];

      if (arg == "-h" || arg == "--help") {
        help_requested_ = true;
        return true;
      }

      // Check if it matches an option
      const Option* matched = nullptr;
      for (const auto& opt : options_) {
        if ((!opt.short_name.empty() && arg == opt.short_name) ||
            (!opt.long_name.empty() && arg == opt.long_name)) {
          matched = &opt;
          break;
        }
      }

      if (matched != nullptr) {
        if (matched->is_flag) {
          if (!matched->parse_fn(arg, "", error_msg)) {
            return false;
          }
        } else {
          if (i + 1 >= args.size()) {
            if (error_msg != nullptr) {
              *error_msg = "Missing argument for " + std::string(arg);
            }
            return false;
          }
          const std::string_view val = args[++i];
          if (!matched->parse_fn(arg, val, error_msg)) {
            return false;
          }
        }
      } else if (!arg.empty() && arg[0] == '-' && arg.size() > 1) {
        // Unknown flag
        if (error_msg != nullptr) {
          *error_msg = "Unknown option: " + std::string(arg);
        }
        return false;
      } else {
        // Positional argument
        if (positional_handler_) {
          if (!positional_handler_(arg, error_msg)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  [[nodiscard]] bool IsHelpRequested() const { return help_requested_; }

  /// Generate grouped formatted help string.
  [[nodiscard]] std::string FormatHelp() const {
    std::ostringstream out;
    out << "Usage: " << program_name_ << " [OPTIONS]";
    if (positional_handler_) {
      out << " [ARGS...]";
    }
    out << "\n\n";

    if (!description_.empty()) {
      out << description_ << "\n\n";
    }

    // Collect options by group
    std::vector<std::string> group_order;
    std::map<std::string, std::vector<const Option*>> groups;

    for (const auto& opt : options_) {
      const std::string& grp = opt.group.empty() ? "Options" : opt.group;
      if (groups.find(grp) == groups.end()) {
        group_order.push_back(grp);
      }
      groups[grp].push_back(&opt);
    }

    for (const auto& grp_name : group_order) {
      out << grp_name << ":\n";
      for (const auto* opt : groups.at(grp_name)) {
        std::string flag_str = "  ";
        if (!opt->short_name.empty()) {
          flag_str += opt->short_name;
          if (!opt->long_name.empty()) {
            flag_str += ", ";
          }
        } else {
          flag_str += "    ";
        }
        if (!opt->long_name.empty()) {
          flag_str += opt->long_name;
        }
        if (!opt->value_hint.empty()) {
          flag_str += " <" + opt->value_hint + ">";
        }

        out << std::left << std::setw(28) << flag_str << " " << opt->description
            << "\n";
      }
      out << "\n";
    }

    out << "General:\n";
    out << std::left << std::setw(28) << "  -h, --help" << " Print help\n";

    return out.str();
  }

  void PrintHelp(std::ostream& os = std::cout) const { os << FormatHelp(); }

private:
  // --- Type parsing helper implementations ---

  static bool ParseValue(std::string_view val, std::string* target,
                         const std::string&, std::string*) {
    if (target != nullptr) {
      *target = std::string(val);
    }
    return true;
  }

  static bool ParseValue(std::string_view val, std::filesystem::path* target,
                         const std::string&, std::string*) {
    if (target != nullptr) {
      *target = std::filesystem::path(val);
    }
    return true;
  }

  template<typename T>
    requires std::is_integral_v<T> && (!std::is_same_v<T, bool>)
  static bool ParseValue(std::string_view val, T* target,
                         const std::string& opt_name, std::string* err) {
    if (val.empty()) {
      if (err)
        *err = "Empty value for " + opt_name;
      return false;
    }
    T num = 0;
    const auto res = std::from_chars(val.data(), val.data() + val.size(), num);
    if (res.ec != std::errc{} || res.ptr != val.data() + val.size()) {
      if (err) {
        *err = "Invalid integer for " + opt_name + ": " + std::string(val);
      }
      return false;
    }
    if (target != nullptr) {
      *target = num;
    }
    return true;
  }

  static bool ParseValue(std::string_view val, float* target,
                         const std::string& opt_name, std::string* err) {
    try {
      std::size_t idx = 0;
      const std::string s(val);
      const float f = std::stof(s, &idx);
      if (idx != s.size()) {
        if (err)
          *err = "Invalid float for " + opt_name + ": " + s;
        return false;
      }
      if (target != nullptr) {
        *target = f;
      }
      return true;
    } catch (...) {
      if (err)
        *err = "Invalid float for " + opt_name + ": " + std::string(val);
      return false;
    }
  }

  static bool ParseValue(std::string_view val, double* target,
                         const std::string& opt_name, std::string* err) {
    try {
      std::size_t idx = 0;
      const std::string s(val);
      const double d = std::stod(s, &idx);
      if (idx != s.size()) {
        if (err)
          *err = "Invalid float for " + opt_name + ": " + s;
        return false;
      }
      if (target != nullptr) {
        *target = d;
      }
      return true;
    } catch (...) {
      if (err)
        *err = "Invalid float for " + opt_name + ": " + std::string(val);
      return false;
    }
  }

  // Comma-separated integers list: e.g. "4096,8192,12288"
  template<typename T>
    requires std::is_integral_v<T>
  static bool ParseValue(std::string_view val, std::vector<T>* target,
                         const std::string& opt_name, std::string* err) {
    std::vector<T> result;
    std::size_t start = 0;
    while (start < val.size()) {
      auto comma = val.find(',', start);
      if (comma == std::string_view::npos) {
        comma = val.size();
      }
      const std::string_view token = val.substr(start, comma - start);
      if (token.empty()) {
        if (err)
          *err = "Empty element in list for " + opt_name;
        return false;
      }
      T num = 0;
      const auto res =
          std::from_chars(token.data(), token.data() + token.size(), num);
      if (res.ec != std::errc{} || res.ptr != token.data() + token.size()) {
        if (err)
          *err = "Invalid integer in list for " + opt_name + ": " +
                 std::string(token);
        return false;
      }
      result.push_back(num);
      start = comma + 1;
    }
    if (target != nullptr) {
      *target = std::move(result);
    }
    return true;
  }

  std::string program_name_;
  std::string description_;
  std::vector<Option> options_;
  std::function<bool(std::string_view, std::string*)> positional_handler_;
  mutable bool help_requested_ = false;
};

}  // namespace strix::cli

#endif  // STRIX_CLI_ARG_PARSER_HPP_
