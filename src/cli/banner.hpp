#ifndef GUFO_CLI_BANNER_HPP_
#define GUFO_CLI_BANNER_HPP_

#include <span>
#include <string_view>

namespace gufo::cli {

/// Writes the two-line startup banner to stderr once per interactive
/// invocation. Skipped when stderr is not a terminal or when `options`
/// requests machine-readable output, so redirected and structured output
/// stay clean.
void PrintStartupBanner(std::string_view version,
                        std::span<const char* const> options);

}  // namespace gufo::cli

#endif  // GUFO_CLI_BANNER_HPP_
