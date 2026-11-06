#ifndef GUFO_CLI_BENCH_HPP_
#define GUFO_CLI_BENCH_HPP_

#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "gufo/ext/text.hpp"

namespace gufo::cli {

using BenchOptions = ext::TextBenchOptions;

void PrintBenchHelp(std::string_view program_name);
std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg = nullptr);
int RunBench(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_BENCH_HPP_
