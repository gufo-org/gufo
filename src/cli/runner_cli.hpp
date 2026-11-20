#ifndef GUFO_CLI_RUNNER_CLI_HPP_
#define GUFO_CLI_RUNNER_CLI_HPP_

#include <chrono>

#include "gufo/ext/text.hpp"

namespace gufo::cli {

// `prompt`, `chat` and `bench` for a text provider that has a runner
// (`create`) but no drivers of its own, such as a model directory a
// provider's `probe` claims: they run the server's backend in-process.
[[nodiscard]] int RunRunnerPrompt(const ext::TextPromptOptions& options,
                                  std::chrono::steady_clock::time_point start);
[[nodiscard]] int RunRunnerChat(const ext::TextPromptOptions& options,
                                std::chrono::steady_clock::time_point start);
[[nodiscard]] int RunRunnerBench(const ext::TextBenchOptions& options,
                                 std::chrono::steady_clock::time_point start);

}  // namespace gufo::cli

#endif
