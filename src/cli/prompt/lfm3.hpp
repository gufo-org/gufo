#ifndef GUFO_CLI_LFM3_HPP_
#define GUFO_CLI_LFM3_HPP_
#include "src/cli/bench/bench.hpp"
#include "src/cli/prompt/prompt.hpp"
namespace gufo::cli {
int RunLfm3Prompt(const PromptOptions& options, bool interactive);
int RunLfm3Bench(const BenchOptions& options);
}  // namespace gufo::cli
#endif
