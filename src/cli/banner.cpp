#include "src/cli/banner.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>

namespace gufo::cli {

namespace {

/// True when the environment advertises a UTF-8 locale, so the banner can use
/// the rounded eyes and the middle dot without risking mojibake.
bool SupportsUnicode() {
  for (const char* name : {"LC_ALL", "LC_CTYPE", "LANG"}) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
      continue;
    const std::string_view locale(value);
    return locale.find("UTF-8") != std::string_view::npos ||
           locale.find("utf8") != std::string_view::npos;
  }
  return false;
}

/// True for invocations whose output should stay unadorned: `--json` keeps
/// structured output machine-readable, and help pages read better without the
/// banner above them.
bool SuppressesBanner(std::span<const char* const> options) {
  return std::ranges::any_of(options, [](const char* option) {
    const std::string_view value(option);
    return value == "--json" || value == "--help" || value == "-h";
  });
}

}  // namespace

void PrintStartupBanner(std::string_view version,
                        std::span<const char* const> options) {
  static bool printed = false;
  if (printed)
    return;
  printed = true;

  if (::isatty(STDERR_FILENO) == 0 || SuppressesBanner(options))
    return;

  const bool color = std::getenv("NO_COLOR") == nullptr;
  const bool unicode = SupportsUnicode();
  const char* eyes = unicode ? "(◉,◉)" : "(o,o)";
  const char* separator = unicode ? " · " : " - ";
  // Logo orange, matching the eyes in assets/gufo-logo.jpg.
  const char* tint = color ? "\033[38;2;232;140;42m" : "";
  const char* bold = color ? "\033[1m" : "";
  const char* reset = color ? "\033[0m" : "";

  std::clog << "  " << tint << ",___," << reset << "   " << bold << "gufo"
            << reset << ' ' << version << '\n'
            << "  " << tint << eyes << reset << "   Fast LLM & DiT inference"
            << separator << "gfx1151\n"
            << std::flush;
}

}  // namespace gufo::cli
