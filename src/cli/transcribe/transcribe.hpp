#ifndef GUFO_CLI_TRANSCRIBE_HPP_
#define GUFO_CLI_TRANSCRIBE_HPP_

#include <span>
#include <string_view>

namespace gufo::cli {

void PrintTranscribeHelp(std::string_view program_name);
[[nodiscard]] int RunTranscribe(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_TRANSCRIBE_HPP_
