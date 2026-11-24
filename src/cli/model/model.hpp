#pragma once
#include <span>
#include <string>
#include <vector>

namespace gufo::cli {
// `gufo model <COMMAND>`: list, install, update, remove and verify the
// weights that extensions offer.
int RunModel(std::span<const char* const> args);

// Replaces the value of `--model`/`-m` with an installed model's path when it
// names an installed model rather than an existing file. Returns the
// arguments unchanged otherwise; throws when the value names a known model
// that is not installed.
std::vector<std::string> ResolveModelArguments(
    std::span<const char* const> args);
}  // namespace gufo::cli
